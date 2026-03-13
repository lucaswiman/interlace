"""Async SQLAlchemy helper for async DPOR integration testing.

Wraps ``explore_async_dpor`` to handle per-task async connection management
and optional lock_timeout injection automatically.

Example::

    from frontrun.contrib.async_sqlalchemy import async_sqlalchemy_dpor, get_connection

    result = await async_sqlalchemy_dpor(
        engine=async_engine,
        setup=_State,
        tasks=[task_a, task_b],
        invariant=_invariant,
        lock_timeout=500,
    )
    assert result.property_holds, result.explanation

Inside a task function, retrieve the per-task connection with::

    conn = get_connection()
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from frontrun.common import InterleavingResult

T = TypeVar("T")

_current_connection: contextvars.ContextVar[Any] = contextvars.ContextVar("_async_sa_connection")


def get_connection() -> Any:
    """Return the per-task async SQLAlchemy connection set by ``async_sqlalchemy_dpor``."""
    return _current_connection.get()


async def async_sqlalchemy_dpor(
    engine: Any,
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool],
    *,
    lock_timeout: int | None = None,
    detect_sql: bool = True,
    **kwargs: Any,
) -> InterleavingResult:
    """Run ``explore_async_dpor`` with per-task async SQLAlchemy connection management.

    Args:
        engine: An async SQLAlchemy ``AsyncEngine`` instance.
        setup: Called once per execution to create fresh shared state.
        tasks: List of async callables, each receiving the shared state.
            Each task gets its own async connection.
            Use :func:`get_connection` inside the task to access it.
        invariant: Predicate over shared state after all tasks complete.
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            task's connection.
        detect_sql: Passed through to ``explore_async_dpor`` (default True).
        **kwargs: Forwarded verbatim to ``explore_async_dpor``.
    """
    from frontrun.async_dpor import explore_async_dpor

    def wrapped_setup() -> T:
        engine.sync_engine.dispose()
        return setup()

    def wrap_task(fn: Callable[[T], Coroutine[Any, Any, None]]) -> Callable[[T], Coroutine[Any, Any, None]]:
        async def wrapper(state: T) -> None:
            async with engine.connect() as conn:
                if lock_timeout is not None:
                    await conn.exec_driver_sql(f"SET lock_timeout = '{int(lock_timeout)}ms'")
                token = _current_connection.set(conn)
                try:
                    await fn(state)
                finally:
                    _current_connection.reset(token)

        return wrapper

    wrapped_tasks = [wrap_task(fn) for fn in tasks]

    return await explore_async_dpor(
        setup=wrapped_setup,
        tasks=wrapped_tasks,
        invariant=invariant,
        detect_sql=detect_sql,
        **kwargs,
    )
