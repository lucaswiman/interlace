"""Redis client monkey-patching for key-level conflict detection.

Intercepts Redis command execution on major Python Redis clients to
extract key-level read/write sets.  Reports each key as a separate
resource to the I/O reporter, suppressing the coarser endpoint-level
socket I/O reports.

Follows the same monkey-patching pattern as ``_sql_cursor.py``.

Supported sync clients:

* **redis-py** (``redis.Redis``, ``redis.StrictRedis``)

The interception hooks into the low-level ``execute_command`` method
that all high-level Redis methods funnel through, so every Redis
operation is captured regardless of whether the user calls ``r.get()``,
``r.set()``, ``r.hset()``, etc.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._io_detection import _io_tls, get_io_reporter
from frontrun._redis_parsing import parse_redis_access

_suppress_tids: set[int] = set()
_suppress_lock = _rt.lock()

# When True, Redis interception forces scheduling points even without
# the IO reporter.  Used during counterexample reproduction to enforce
# the DPOR schedule at Redis command boundaries.  See defect #9.
_redis_replay_mode = False


@contextlib.contextmanager
def _suppress_endpoint_io() -> Generator[None, None, None]:
    """Temporarily suppress endpoint-level I/O for the current thread."""
    tid = threading.get_native_id()
    _io_tls._redis_suppress = True
    with _suppress_lock:
        _suppress_tids.add(tid)
    try:
        yield
    finally:
        with _suppress_lock:
            _suppress_tids.discard(tid)
        _io_tls._redis_suppress = False


def is_redis_tid_suppressed(tid: int) -> bool:
    """Check if a thread ID is currently suppressed (for LD_PRELOAD bridge)."""
    with _suppress_lock:
        return tid in _suppress_tids


# ---------------------------------------------------------------------------
# Resource ID construction
# ---------------------------------------------------------------------------


def _redis_resource_id(key: str, *, db_scope: str | None = None) -> str:
    """Build a resource ID for a Redis key."""
    resource = f"redis:{key}"
    if db_scope is not None:
        resource = f"{resource}:db={db_scope}"
    return resource


def _get_redis_db_scope(client: Any) -> str | None:
    """Extract a stable database scope from a Redis client object."""
    # redis-py exposes connection_pool.connection_kwargs
    pool = getattr(client, "connection_pool", None)
    if pool is not None:
        kwargs = getattr(pool, "connection_kwargs", {})
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 6379)
        db = kwargs.get("db", 0)
        return f"redis:{host}:{port}/{db}"
    return None


# ---------------------------------------------------------------------------
# DPOR context helpers (mirrors _sql_cursor.py)
# ---------------------------------------------------------------------------


def _get_dpor_context() -> tuple[Any, int] | None:
    """Return (scheduler, thread_id) if DPOR is active, else ``None``."""
    from frontrun._io_detection import get_dpor_scheduler, get_dpor_thread_id

    scheduler = get_dpor_scheduler()
    if scheduler is None:
        return None
    thread_id = get_dpor_thread_id()
    if thread_id is None:
        return None
    return scheduler, thread_id


# ---------------------------------------------------------------------------
# Core interception
# ---------------------------------------------------------------------------


def _report_redis_access(
    cmd_name: str,
    cmd_args: tuple[object, ...],
    *,
    client: Any = None,
) -> bool:
    """Parse a Redis command and report key accesses to the per-thread reporter.

    Returns ``True`` if any Redis-level reporting was performed (which means
    endpoint-level I/O should be suppressed for the subsequent Redis call).
    """
    reporter = get_io_reporter()
    if reporter is None:
        return False

    access = parse_redis_access(cmd_name, cmd_args)

    # Transaction control — no key-level reporting needed.
    if access.is_transaction_control and not access.read_keys and not access.write_keys:
        return True  # Still suppress endpoint I/O for protocol overhead.

    if not access.read_keys and not access.write_keys:
        return False

    db_scope = _get_redis_db_scope(client) if client is not None else None

    for key in access.read_keys:
        res_id = _redis_resource_id(key, db_scope=db_scope)
        reporter(res_id, "read")

    for key in access.write_keys:
        res_id = _redis_resource_id(key, db_scope=db_scope)
        reporter(res_id, "write")

    return True


def _intercept_execute_command(
    original_method: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Intercept redis.Redis.execute_command to report key-level accesses."""
    if not args:
        return original_method(self, *args, **kwargs)

    cmd_name = str(args[0])
    cmd_args = args[1:]

    reported = _report_redis_access(cmd_name, cmd_args, client=self)

    # In replay mode (defect #9 fix), force a scheduling point even
    # without IO reporting so the replay scheduler can enforce the
    # interleaving at Redis command boundaries.  Only do this for
    # data commands (those that have keys), not connection-setup
    # commands (AUTH, SELECT, CLIENT SETNAME, etc.) which didn't
    # create scheduling points during exploration.
    needs_scheduling_point = reported
    if not needs_scheduling_point and _redis_replay_mode:
        access = parse_redis_access(cmd_name, cmd_args)
        needs_scheduling_point = bool(access.read_keys or access.write_keys)

    # Force a DPOR scheduling point so the engine can interleave between
    # Redis operations.
    dpor_ctx = None
    resource_id = ""
    if needs_scheduling_point:
        dpor_ctx = _get_dpor_context()
        if dpor_ctx is not None:
            # Build a structured resource ID for IO-anchored replay.
            db_scope = _get_redis_db_scope(self) or ""
            first_key = str(cmd_args[0]) if cmd_args else ""
            resource_id = f"redis:{cmd_name}:{first_key}:{db_scope}"
            dpor_ctx[0].before_io(dpor_ctx[1], resource_id)

    try:
        if reported:
            with _suppress_endpoint_io():
                result = original_method(self, *args, **kwargs)
        else:
            result = original_method(self, *args, **kwargs)
    finally:
        # after_io runs inside a finally so the IO trace is recorded
        # even if the command raises.  During replay, this atomically
        # switches threads under the scheduler's condition lock.
        if dpor_ctx is not None:
            dpor_ctx[0].after_io(dpor_ctx[1], resource_id)

    return result


def _intercept_pipeline_execute(
    original_method: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Intercept redis.Pipeline.execute to report all queued commands."""
    # Pipeline command_stack contains the queued commands.
    command_stack = getattr(self, "command_stack", [])
    reported = False
    for cmd in command_stack:
        # redis-py Pipeline stores commands as PipelineCommand or tuples.
        if hasattr(cmd, "args"):
            cmd_args_full = cmd.args
        elif isinstance(cmd, (list, tuple)):
            cmd_args_full = cmd[0] if cmd and isinstance(cmd[0], (list, tuple)) else cmd
        else:
            continue
        if cmd_args_full:
            cmd_name = str(cmd_args_full[0])
            cmd_cmd_args = tuple(cmd_args_full[1:])
            if _report_redis_access(cmd_name, cmd_cmd_args, client=self):
                reported = True

    # In replay mode (defect #9 fix), force a scheduling point even
    # without IO reporting so the replay scheduler can enforce the
    # interleaving at Redis command boundaries.  Without this, pipeline
    # commands (used by get_many/set_many in e.g. Flask-Caching @memoize)
    # silently skip their scheduling points during replay, causing the
    # schedule to misalign.  See defect #10.
    needs_scheduling_point = reported or _redis_replay_mode

    dpor_ctx = None
    resource_id = ""
    if needs_scheduling_point:
        dpor_ctx = _get_dpor_context()
        if dpor_ctx is not None:
            resource_id = "redis:PIPELINE"
            dpor_ctx[0].before_io(dpor_ctx[1], resource_id)

    try:
        if reported:
            with _suppress_endpoint_io():
                result = original_method(self, *args, **kwargs)
        else:
            result = original_method(self, *args, **kwargs)
    finally:
        if dpor_ctx is not None:
            dpor_ctx[0].after_io(dpor_ctx[1], resource_id)

    return result


# ---------------------------------------------------------------------------
# Global patching state
# ---------------------------------------------------------------------------

_redis_patched = False
_PATCHES: list[tuple[Any, str, Any]] = []
_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}


def _patch_redis_py() -> None:
    """Patch redis-py ``Redis.execute_command`` and ``Pipeline.execute``."""
    try:
        import redis as redis_lib  # type: ignore[import-untyped]
    except ImportError:
        return

    # Patch Redis.execute_command (the central command dispatch).
    client_cls = redis_lib.Redis
    key = (client_cls, "execute_command")
    if key not in _ORIGINAL_METHODS:
        original = getattr(client_cls, "execute_command", None)
        if original is not None:
            _ORIGINAL_METHODS[key] = original

            def _make_patched_exec(orig: Any) -> Any:
                def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return _intercept_execute_command(orig, self, *args, **kwargs)

                _patched.__name__ = orig.__name__
                _patched.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
                return _patched

            setattr(client_cls, "execute_command", _make_patched_exec(original))
            _PATCHES.append((client_cls, "execute_command", original))

    # Also patch StrictRedis if it's a separate class.
    strict_cls = getattr(redis_lib, "StrictRedis", None)
    if strict_cls is not None and strict_cls is not client_cls:
        key = (strict_cls, "execute_command")
        if key not in _ORIGINAL_METHODS:
            original = getattr(strict_cls, "execute_command", None)
            if original is not None:
                _ORIGINAL_METHODS[key] = original

                def _make_patched_strict(orig: Any) -> Any:
                    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                        return _intercept_execute_command(orig, self, *args, **kwargs)

                    _patched.__name__ = orig.__name__
                    _patched.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
                    return _patched

                setattr(strict_cls, "execute_command", _make_patched_strict(original))
                _PATCHES.append((strict_cls, "execute_command", original))

    # Patch Pipeline.execute for pipelined commands.
    pipeline_cls = getattr(redis_lib.client, "Pipeline", None)
    if pipeline_cls is None:
        pipeline_cls = getattr(redis_lib, "Pipeline", None)
    if pipeline_cls is not None:
        key = (pipeline_cls, "execute")
        if key not in _ORIGINAL_METHODS:
            original = getattr(pipeline_cls, "execute", None)
            if original is not None:
                _ORIGINAL_METHODS[key] = original

                def _make_patched_pipe(orig: Any) -> Any:
                    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                        return _intercept_pipeline_execute(orig, self, *args, **kwargs)

                    _patched.__name__ = orig.__name__
                    _patched.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
                    return _patched

                setattr(pipeline_cls, "execute", _make_patched_pipe(original))
                _PATCHES.append((pipeline_cls, "execute", original))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch_redis() -> None:
    """Monkey-patch Redis clients for key-level conflict detection."""
    global _redis_patched  # noqa: PLW0603
    if _redis_patched:
        return
    _patch_redis_py()
    _redis_patched = True


def unpatch_redis() -> None:
    """Restore original Redis client methods."""
    global _redis_patched  # noqa: PLW0603
    if not _redis_patched:
        return
    for obj, attr, original in _PATCHES:
        setattr(obj, attr, original)
    _PATCHES.clear()
    _ORIGINAL_METHODS.clear()
    _redis_patched = False


def set_redis_replay_mode(enabled: bool) -> None:
    """Enable/disable Redis replay mode for counterexample reproduction.

    When enabled, Redis command interception creates scheduling points
    even without the IO reporter, so the replay scheduler can enforce
    the DPOR schedule at Redis command boundaries.  See defect #9.
    """
    global _redis_replay_mode  # noqa: PLW0603
    _redis_replay_mode = enabled
