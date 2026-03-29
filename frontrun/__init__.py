"""
Frontrun: Deterministic concurrency testing for Python.

Trace markers (sync)::

    from frontrun.common import Schedule, Step
    from frontrun import TraceExecutor

Async trace markers (pass a dict of task names to coroutines)::

    from frontrun import TraceExecutor
    from frontrun.common import Schedule, Step
    executor = TraceExecutor(schedule)
    executor.run({"task1": coro_factory1, "task2": coro_factory2})

DPOR (Dynamic Partial Order Reduction) systematic exploration::

    from frontrun.dpor import explore_dpor

Async DPOR systematic exploration::

    from frontrun.async_dpor import explore_async_dpor, await_point

Bytecode exploration::

    from frontrun import explore_interleavings

Async shuffler exploration::

    from frontrun import explore_interleavings

Contrib helpers (use threads= for sync, tasks= for async)::

    from frontrun.contrib.django import django_dpor
    from frontrun.contrib.sqlalchemy import sqlalchemy_dpor, get_connection, get_async_connection
"""

from importlib.metadata import version as _metadata_version
from typing import Any

from frontrun.common import NondeterministicSQLError
from frontrun.trace_markers import TraceExecutor

try:
    __version__: str = _metadata_version("frontrun")
except Exception:
    __version__ = "0.0.0"


def explore_interleavings(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to sync or async interleaving exploration.

    Use ``threads=[...]`` for threaded code or ``tasks=[...]`` for async code.
    The async form returns an awaitable.
    """
    has_threads = "threads" in kwargs
    has_tasks = "tasks" in kwargs
    if has_threads == has_tasks:
        raise TypeError("explore_interleavings() requires exactly one of 'threads' or 'tasks'")

    if has_tasks:
        from frontrun.async_shuffler import explore_interleavings as _async_explore_interleavings

        return _async_explore_interleavings(*args, **kwargs)

    from frontrun.bytecode import explore_interleavings as _sync_explore_interleavings

    return _sync_explore_interleavings(*args, **kwargs)


__all__ = ["NondeterministicSQLError", "TraceExecutor", "__version__", "explore_interleavings"]
