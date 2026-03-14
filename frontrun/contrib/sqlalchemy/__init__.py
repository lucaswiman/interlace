"""SQLAlchemy helpers for DPOR integration testing (sync and async).

Use ``sqlalchemy_dpor`` for both sync and async code:

Sync (threads)::

    result = sqlalchemy_dpor(
        engine=engine,
        setup=_State,
        threads=[thread_a, thread_b],
        invariant=_invariant,
    )

Async (tasks)::

    result = await sqlalchemy_dpor(
        engine=async_engine,
        setup=_State,
        tasks=[task_a, task_b],
        invariant=_invariant,
    )

Inside sync threads, use :func:`get_connection`.
Inside async tasks, use :func:`get_async_connection`.
"""

from typing import Any


def sqlalchemy_dpor(*args: Any, **kwargs: Any) -> Any:
    """Run DPOR with per-thread/task SQLAlchemy connection management.

    Use ``threads=[...]`` for sync code or ``tasks=[...]`` for async code.
    The async form returns an awaitable.
    """
    has_threads = "threads" in kwargs
    has_tasks = "tasks" in kwargs
    if has_threads == has_tasks:
        raise TypeError("sqlalchemy_dpor() requires exactly one of 'threads' or 'tasks'")

    if has_tasks:
        from frontrun.contrib.sqlalchemy._async import async_sqlalchemy_dpor

        return async_sqlalchemy_dpor(*args, **kwargs)

    from frontrun.contrib.sqlalchemy._sync import sqlalchemy_dpor as _sync_sqlalchemy_dpor

    return _sync_sqlalchemy_dpor(*args, **kwargs)


# Keep async_sqlalchemy_dpor importable for backward compatibility.
from frontrun.contrib.sqlalchemy._async import async_sqlalchemy_dpor, get_async_connection  # noqa: E402, F811
from frontrun.contrib.sqlalchemy._sync import get_connection  # noqa: E402

__all__ = ["async_sqlalchemy_dpor", "get_async_connection", "get_connection", "sqlalchemy_dpor"]
