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

from frontrun.contrib._shared import dispatch_threads_or_tasks
from frontrun.contrib.sqlalchemy._async import async_sqlalchemy_dpor, get_async_connection
from frontrun.contrib.sqlalchemy._sync import get_connection
from frontrun.contrib.sqlalchemy._sync import sqlalchemy_dpor as _sync_sqlalchemy_dpor


def sqlalchemy_dpor(*args: Any, **kwargs: Any) -> Any:
    """Run DPOR with per-thread/task SQLAlchemy connection management.

    Use ``threads=[...]`` for sync code or ``tasks=[...]`` for async code.
    The async form returns an awaitable.
    """
    return dispatch_threads_or_tasks(
        _sync_sqlalchemy_dpor,
        async_sqlalchemy_dpor,
        *args,
        api_name="sqlalchemy_dpor",
        **kwargs,
    )


__all__ = ["async_sqlalchemy_dpor", "get_async_connection", "get_connection", "sqlalchemy_dpor"]
