"""Async Redis client monkey-patching for key-level conflict detection.

Async counterpart to ``_redis_client.py``.  Intercepts async Redis command
execution to extract key-level read/write sets.

Supported async clients:

* **redis-py async** (``redis.asyncio.Redis``) — the standard async interface
  since redis-py 4.2+.  ``aioredis`` was merged into redis-py and uses
  the same ``redis.asyncio`` module.
* **coredis** — alternative async Redis client.
"""

from __future__ import annotations

from typing import Any

from frontrun._io_detection import get_dpor_context as _get_dpor_context
from frontrun._redis_client import (
    _report_pipeline_commands,
    _report_redis_access,
    _suppress_endpoint_io,
)

# Lazy imports to avoid circular dependency — resolved at first use.
_in_scheduler_pause = None
_scheduler_var_ref = None


def _get_in_scheduler_pause() -> Any:
    global _in_scheduler_pause  # noqa: PLW0603
    if _in_scheduler_pause is None:
        from frontrun.async_dpor import _in_scheduler_pause as _isp

        _in_scheduler_pause = _isp
    return _in_scheduler_pause


def _get_scheduler_var() -> Any:
    global _scheduler_var_ref  # noqa: PLW0603
    if _scheduler_var_ref is None:
        from frontrun.async_dpor import _scheduler_var, _task_id_var

        _scheduler_var_ref = (_scheduler_var, _task_id_var)
    return _scheduler_var_ref


# ---------------------------------------------------------------------------
# Async interception
# ---------------------------------------------------------------------------


async def _intercept_execute_command_async(
    original_method: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Async version of ``_intercept_execute_command``."""
    if not args:
        return await original_method(self, *args, **kwargs)

    cmd_name = str(args[0])
    cmd_args = args[1:]

    reported = _report_redis_access(cmd_name, cmd_args, client=self)

    if reported:
        # Force a real DPOR scheduling point so the engine can interleave
        # between Redis operations.  The I/O was added to _pending_io by
        # _report_redis_access; on_proceed will flush it to the engine.
        dpor_ctx = _get_dpor_context()
        if dpor_ctx is not None:
            scheduler = dpor_ctx[0]
            refs = _get_scheduler_var()
            task_id = refs[1].get()
            if task_id is not None and hasattr(scheduler, "pause"):
                await scheduler.pause(task_id)

    if reported:
        # Suppress AutoPauseCoroutine scheduling during Redis network I/O.
        # Without this, every socket yield creates an empty DPOR step,
        # pushing adjacent Redis commands far apart in the trace and
        # preventing DPOR from exploring the gap between e.g. EXISTS and SET.
        pause_var = _get_in_scheduler_pause()
        depth = pause_var.get()
        pause_var.set(depth + 1)
        try:
            with _suppress_endpoint_io():
                return await original_method(self, *args, **kwargs)
        finally:
            pause_var.set(depth)
    return await original_method(self, *args, **kwargs)


async def _intercept_pipeline_execute_async(
    original_method: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Async version of ``_intercept_pipeline_execute``."""
    reported = _report_pipeline_commands(self)

    if reported:
        dpor_ctx = _get_dpor_context()
        if dpor_ctx is not None:
            scheduler = dpor_ctx[0]
            refs = _get_scheduler_var()
            task_id = refs[1].get()
            if task_id is not None and hasattr(scheduler, "pause"):
                await scheduler.pause(task_id)

    if reported:
        pause_var = _get_in_scheduler_pause()
        depth = pause_var.get()
        pause_var.set(depth + 1)
        try:
            with _suppress_endpoint_io():
                return await original_method(self, *args, **kwargs)
        finally:
            pause_var.set(depth)
    return await original_method(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Global patching state
# ---------------------------------------------------------------------------

_redis_async_patched = False
_ASYNC_PATCHES: list[tuple[Any, str, Any]] = []
_ASYNC_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}


# ---------------------------------------------------------------------------
# redis.asyncio patching (redis-py 4.2+ / aioredis merged)
# ---------------------------------------------------------------------------


def _patch_redis_asyncio() -> None:
    """Patch ``redis.asyncio.Redis.execute_command`` and ``Pipeline.execute``."""
    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]
    except ImportError:
        return

    client_cls = aioredis.Redis
    key = (client_cls, "execute_command")
    if key not in _ASYNC_ORIGINAL_METHODS:
        original = getattr(client_cls, "execute_command", None)
        if original is not None:
            _ASYNC_ORIGINAL_METHODS[key] = original

            def _make_patched(orig: Any) -> Any:
                async def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return await _intercept_execute_command_async(orig, self, *args, **kwargs)

                _patched.__name__ = orig.__name__
                _patched.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
                return _patched

            setattr(client_cls, "execute_command", _make_patched(original))
            _ASYNC_PATCHES.append((client_cls, "execute_command", original))

    # Patch async Pipeline.
    pipeline_cls = getattr(aioredis, "Pipeline", None)
    if pipeline_cls is None:
        # redis.asyncio.client.Pipeline
        client_mod = getattr(aioredis, "client", None)
        if client_mod is not None:
            pipeline_cls = getattr(client_mod, "Pipeline", None)
    if pipeline_cls is not None:
        key = (pipeline_cls, "execute")
        if key not in _ASYNC_ORIGINAL_METHODS:
            original = getattr(pipeline_cls, "execute", None)
            if original is not None:
                _ASYNC_ORIGINAL_METHODS[key] = original

                def _make_patched_pipe(orig: Any) -> Any:
                    async def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                        return await _intercept_pipeline_execute_async(orig, self, *args, **kwargs)

                    _patched.__name__ = orig.__name__
                    _patched.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
                    return _patched

                setattr(pipeline_cls, "execute", _make_patched_pipe(original))
                _ASYNC_PATCHES.append((pipeline_cls, "execute", original))


# ---------------------------------------------------------------------------
# coredis patching
# ---------------------------------------------------------------------------


def _patch_coredis() -> None:
    """Patch ``coredis.Redis.execute_command``."""
    try:
        import coredis  # type: ignore[import-untyped]
    except ImportError:
        return

    client_cls = coredis.Redis
    key = (client_cls, "execute_command")
    if key not in _ASYNC_ORIGINAL_METHODS:
        original = getattr(client_cls, "execute_command", None)
        if original is not None:
            _ASYNC_ORIGINAL_METHODS[key] = original

            def _make_patched(orig: Any) -> Any:
                async def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return await _intercept_execute_command_async(orig, self, *args, **kwargs)

                _patched.__name__ = orig.__name__
                _patched.__qualname__ = getattr(orig, "__qualname__", orig.__name__)
                return _patched

            setattr(client_cls, "execute_command", _make_patched(original))
            _ASYNC_PATCHES.append((client_cls, "execute_command", original))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch_redis_async() -> None:
    """Monkey-patch async Redis clients for key-level conflict detection."""
    global _redis_async_patched  # noqa: PLW0603
    if _redis_async_patched:
        return
    _patch_redis_asyncio()
    _patch_coredis()
    _redis_async_patched = True


def unpatch_redis_async() -> None:
    """Restore original async Redis client methods."""
    global _redis_async_patched  # noqa: PLW0603
    if not _redis_async_patched:
        return
    for obj, attr, original in _ASYNC_PATCHES:
        setattr(obj, attr, original)
    _ASYNC_PATCHES.clear()
    _ASYNC_ORIGINAL_METHODS.clear()
    _redis_async_patched = False
