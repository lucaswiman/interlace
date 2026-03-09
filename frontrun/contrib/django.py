"""Django helper for DPOR integration testing.

Wraps ``explore_dpor`` to handle per-thread Django database connection
management and optional lock_timeout injection automatically.

Example::

    from frontrun.contrib.django import django_dpor

    result = django_dpor(
        setup=_State,
        threads=[thread_a, thread_b],
        invariant=_invariant,
        lock_timeout=500,  # optional, milliseconds
    )
    assert result.property_holds, result.explanation
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def django_dpor(
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    *,
    db_alias: str = "default",
    lock_timeout: int | None = None,
    detect_io: bool = True,
    **kwargs: Any,
) -> Any:
    """Run ``explore_dpor`` with per-thread Django connection management.

    Each thread closes the shared Django connection and opens a fresh one,
    ensuring DPOR sees independent connections per thread.

    Args:
        setup: Called once per execution to create fresh shared state.
            All Django DB connections are closed before this runs.
        threads: List of callables, each receiving the shared state.
        invariant: Predicate over shared state after all threads complete.
        db_alias: Django database alias to use (default ``"default"``).
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            thread's connection before running the thread.
        detect_io: Passed through to ``explore_dpor`` (default True).
        **kwargs: Forwarded verbatim to ``explore_dpor``.
    """
    from django.db import connections  # type: ignore[import-not-found]

    from frontrun.dpor import explore_dpor

    def wrapped_setup() -> T:
        connections.close_all()
        return setup()

    def wrap_thread(fn: Callable[[T], None]) -> Callable[[T], None]:
        def wrapper(state: T) -> None:
            conn = connections[db_alias]
            conn.close()
            conn.ensure_connection()
            if lock_timeout is not None:
                with conn.cursor() as cursor:
                    cursor.execute(f"SET lock_timeout = '{lock_timeout}ms'")
            try:
                fn(state)
            finally:
                conn.close()

        return wrapper

    wrapped_threads = [wrap_thread(fn) for fn in threads]

    return explore_dpor(
        setup=wrapped_setup,
        threads=wrapped_threads,
        invariant=invariant,
        detect_io=detect_io,
        **kwargs,
    )
