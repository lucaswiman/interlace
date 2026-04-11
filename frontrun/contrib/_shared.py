"""Shared helpers for contrib adapter dispatch."""

from __future__ import annotations

from typing import Any


def dispatch_threads_or_tasks(
    sync_impl: Any,
    async_impl: Any,
    /,
    *args: Any,
    api_name: str = "dispatcher",
    **kwargs: Any,
) -> Any:
    """Dispatch to the sync or async contrib implementation.

    Exactly one of ``threads=`` or ``tasks=`` must be present. The async branch
    returns the awaitable from ``async_impl`` unchanged.
    """
    has_threads = "threads" in kwargs
    has_tasks = "tasks" in kwargs
    if has_threads == has_tasks:
        raise TypeError(f"{api_name}() requires exactly one of 'threads' or 'tasks'")
    if has_tasks:
        return async_impl(*args, **kwargs)
    return sync_impl(*args, **kwargs)
