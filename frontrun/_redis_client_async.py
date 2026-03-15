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

from frontrun._redis_client import (
    _get_dpor_context,
    _report_redis_access,
    _suppress_endpoint_io,
)

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

    # Force a DPOR scheduling point so the engine can interleave between
    # Redis operations.
    if reported:
        dpor_ctx = _get_dpor_context()
        if dpor_ctx is not None:
            dpor_ctx[0].report_and_wait(None, dpor_ctx[1])

    if reported:
        with _suppress_endpoint_io():
            return await original_method(self, *args, **kwargs)
    return await original_method(self, *args, **kwargs)


async def _intercept_pipeline_execute_async(
    original_method: Any,
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Async version of ``_intercept_pipeline_execute``."""
    command_stack = getattr(self, "command_stack", [])
    reported = False
    for cmd in command_stack:
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

    if reported:
        dpor_ctx = _get_dpor_context()
        if dpor_ctx is not None:
            dpor_ctx[0].report_and_wait(None, dpor_ctx[1])

    if reported:
        with _suppress_endpoint_io():
            return await original_method(self, *args, **kwargs)
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
