"""Async Django helper for async DPOR integration testing.

Wraps ``explore_async_dpor`` to handle per-task Django async database
connection management and optional lock_timeout injection automatically.

Prefer the unified ``django_dpor`` dispatcher (pass ``tasks=`` for async)::

    from frontrun.contrib.django import django_dpor

    result = await django_dpor(
        setup=_State,
        tasks=[task_a, task_b],
        invariant=_invariant,
        lock_timeout=500,
    )
    assert result.property_holds, result.explanation
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from frontrun.common import InterleavingResult
from frontrun.contrib.django._sync import DJANGO_TRACE_PACKAGES

T = TypeVar("T")


async def async_django_dpor(
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool],
    *,
    db_alias: str = "default",
    lock_timeout: int | None = None,
    detect_sql: bool = True,
    trace_packages: list[str] | None = None,
    **kwargs: Any,
) -> InterleavingResult:
    """Run ``explore_async_dpor`` with per-task Django async connection management.

    Each task closes the shared Django connection and opens a fresh one.

    By default, ``trace_packages`` is set to :data:`~frontrun.contrib.django._sync.DJANGO_TRACE_PACKAGES`
    so that code inside ``django_*`` apps and ``django.contrib.sites``
    submodules is traced.  Pass an explicit list (or ``[]``) to override.

    Args:
        setup: Called once per execution to create fresh shared state.
        tasks: List of async callables, each receiving the shared state.
        invariant: Predicate over shared state after all tasks complete.
        db_alias: Django database alias to use (default ``"default"``).
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            task's connection.
        detect_sql: Passed through to ``explore_async_dpor`` (default True).
        trace_packages: Package name patterns (fnmatch syntax) to trace.
            Defaults to :data:`~frontrun.contrib.django._sync.DJANGO_TRACE_PACKAGES`.
            Pass ``[]`` to disable extra tracing beyond user code.
        **kwargs: Forwarded verbatim to ``explore_async_dpor``.
    """
    from django.db import connections  # type: ignore[import-not-found]

    from frontrun.async_dpor import explore_async_dpor

    if trace_packages is None:
        trace_packages = list(DJANGO_TRACE_PACKAGES)

    def wrapped_setup() -> T:
        connections.close_all()
        return setup()

    def wrap_task(fn: Callable[[T], Coroutine[Any, Any, None]]) -> Callable[[T], Coroutine[Any, Any, None]]:
        async def wrapper(state: T) -> None:
            conn = connections[db_alias]
            conn.close()
            conn.ensure_connection()
            if lock_timeout is not None:
                cursor = conn.cursor()
                cursor.execute(f"SET lock_timeout = '{int(lock_timeout)}ms'")
                cursor.close()
            try:
                await fn(state)
            finally:
                conn.close()

        return wrapper

    wrapped_tasks = [wrap_task(fn) for fn in tasks]

    return await explore_async_dpor(
        setup=wrapped_setup,
        tasks=wrapped_tasks,
        invariant=invariant,
        detect_sql=detect_sql,
        trace_packages=trace_packages,
        **kwargs,
    )
