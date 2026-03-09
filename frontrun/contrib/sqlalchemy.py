"""SQLAlchemy helper for DPOR integration testing.

Wraps ``explore_dpor`` to handle per-thread connection management and
optional lock_timeout injection automatically.

Example::

    from frontrun.contrib.sqlalchemy import sqlalchemy_dpor

    result = sqlalchemy_dpor(
        engine=engine,
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


def sqlalchemy_dpor(
    engine: Any,
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    *,
    lock_timeout: int | None = None,
    detect_io: bool = True,
    **kwargs: Any,
) -> Any:
    """Run ``explore_dpor`` with per-thread SQLAlchemy connection management.

    Args:
        engine: A SQLAlchemy ``Engine`` instance.
        setup: Called once per execution to create fresh shared state.
            Any open connections on the engine are disposed before this runs.
        threads: List of callables, each receiving the shared state.
            Each thread gets its own connection scoped to its execution.
        invariant: Predicate over shared state after all threads complete.
        lock_timeout: If set, execute ``SET lock_timeout = <N>ms`` on each
            thread's connection before running the thread.  Converts
            C-level row-lock blocking into a fast PostgreSQL error.
        detect_io: Passed through to ``explore_dpor`` (default True).
        **kwargs: Forwarded verbatim to ``explore_dpor``.
    """
    from frontrun.dpor import explore_dpor

    def wrapped_setup() -> T:
        engine.dispose()
        return setup()

    def wrap_thread(fn: Callable[[T], None]) -> Callable[[T], None]:
        def wrapper(state: T) -> None:
            with engine.connect() as conn:
                if lock_timeout is not None:
                    conn.exec_driver_sql(f"SET lock_timeout = '{lock_timeout}ms'")
                # Expose the connection on the state object so thread fns can use it.
                _prev = getattr(state, "_dpor_conn", None)
                state._dpor_conn = conn  # type: ignore[attr-defined]
                try:
                    fn(state)
                finally:
                    state._dpor_conn = _prev  # type: ignore[attr-defined]

        return wrapper

    wrapped_threads = [wrap_thread(fn) for fn in threads]

    return explore_dpor(
        setup=wrapped_setup,
        threads=wrapped_threads,
        invariant=invariant,
        detect_io=detect_io,
        **kwargs,
    )
