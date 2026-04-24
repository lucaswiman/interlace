"""Unified entry point for frontrun interleaving exploration.

This module provides :func:`explore`, a single function that dispatches to the
appropriate underlying implementation based on worker type and strategy.

Examples::

    from frontrun import explore

    # Sync DPOR (default)
    result = explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    result.assert_holds()

    # Async — detected automatically from coroutine function
    async def worker(state): ...
    result = await explore(setup=make_state, workers=worker, count=2, invariant=...)

    # Strategy selection
    result = explore(..., strategy="dpor")    # default
    result = explore(..., strategy="random")  # formerly explore_interleavings
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from frontrun.common import InterleavingResult

T = TypeVar("T")


def explore(
    setup: Callable[[], Any],
    workers: Callable[[Any], Any] | list[Callable[[Any], Any]] | tuple[Callable[[Any], Any], ...],
    invariant: Callable[[Any], bool],
    *,
    count: int | None = None,
    strategy: str = "dpor",
    # DPOR-specific kwargs
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    timeout_per_run: float = 5.0,
    stop_on_first: bool = True,
    detect_io: bool = True,
    deadlock_timeout: float = 5.0,
    reproduce_on_failure: int = 10,
    total_timeout: float | None = None,
    warn_nondeterministic_sql: bool = True,
    lock_timeout: int | None = None,
    trace_packages: list[str] | None = None,
    track_dunder_dict_accesses: bool = False,
    search: str | None = None,
    patch_sleep: bool = True,
    serializable_invariant: Callable[[Any], Any] | bool = False,
    error_on_any_race: bool = False,
    # Random-specific kwargs
    max_attempts: int = 200,
    max_ops: int | None = None,
    seed: int | None = None,
    debug: bool = False,
    # Async DPOR specific kwargs (kept for passthrough)
    detect_sql: bool = False,
) -> Any:
    """Explore thread/task interleavings for concurrency bugs.

    A unified entry point that dispatches to the appropriate underlying
    implementation based on worker type (sync vs async) and strategy.

    Args:
        setup: Creates fresh shared state for each execution.
        workers: A callable (when ``count`` is provided) or a list/tuple of
            callables. Sync callables run as threads; async callables (coroutine
            functions) run as asyncio tasks.
        invariant: Predicate over shared state; must be True after all
            workers complete.
        count: When ``workers`` is a single callable, replicate it this many
            times. Must be positive. Cannot be used when ``workers`` is a
            list/tuple.
        strategy: ``"dpor"`` (default) for systematic DPOR exploration, or
            ``"random"`` for random schedule sampling (formerly
            ``explore_interleavings``).
        max_executions: Safety limit on total executions (DPOR only).
        preemption_bound: Limit on preemptions per execution (DPOR only).
        max_branches: Maximum scheduling points per execution (DPOR only).
        timeout_per_run: Timeout for each individual run.
        stop_on_first: Stop on first invariant violation (DPOR only).
        detect_io: Detect socket/file I/O operations as resource accesses.
            For async DPOR, also activates Redis key-level patching.
        deadlock_timeout: Seconds to wait before declaring a deadlock.
        reproduce_on_failure: Replay counterexample this many times.
        total_timeout: Maximum total exploration time in seconds.
        warn_nondeterministic_sql: Raise on nondeterministic SQL INSERT.
        lock_timeout: Auto-set PostgreSQL lock_timeout (milliseconds).
        trace_packages: Package patterns to trace in addition to user code.
        track_dunder_dict_accesses: Report ``obj.__dict__`` accesses (DPOR).
        search: Wakeup-tree traversal strategy (DPOR only).
        patch_sleep: Replace ``time.sleep`` / ``asyncio.sleep`` with no-op.
        serializable_invariant: Check serializability against sequential runs.
        error_on_any_race: Treat unsynchronized races as failures (DPOR only).
        max_attempts: Random schedule samples to try (random strategy only).
        max_ops: Maximum schedule length per attempt (random strategy only).
        seed: RNG seed for reproducibility (random strategy only).
        debug: Enable debug output (random strategy only).
        detect_sql: Patch async SQL drivers (async DPOR only).

    Returns:
        :class:`~frontrun.common.InterleavingResult` (sync) or a coroutine
        that resolves to one (async workers).

    Raises:
        ValueError: If ``count`` and a list of workers are both provided,
            ``count <= 0``, or ``strategy`` is unrecognised.
    """
    # --- Resolve worker list ---
    worker_list = _resolve_workers(workers, count)

    # --- Validate strategy ---
    if strategy not in ("dpor", "random"):
        raise ValueError(f"explore(): unknown strategy={strategy!r}; must be 'dpor' or 'random'")

    # --- Detect async workers ---
    is_async = any(inspect.iscoroutinefunction(w) for w in worker_list)

    if is_async:
        return _explore_async(
            setup=setup,
            workers=worker_list,
            invariant=invariant,
            strategy=strategy,
            max_executions=max_executions,
            preemption_bound=preemption_bound,
            max_branches=max_branches,
            timeout_per_run=timeout_per_run,
            stop_on_first=stop_on_first,
            detect_io=detect_io,
            deadlock_timeout=deadlock_timeout,
            reproduce_on_failure=reproduce_on_failure,
            total_timeout=total_timeout,
            warn_nondeterministic_sql=warn_nondeterministic_sql,
            lock_timeout=lock_timeout,
            trace_packages=trace_packages,
            patch_sleep=patch_sleep,
            serializable_invariant=serializable_invariant,
            error_on_any_race=error_on_any_race,
            max_attempts=max_attempts,
            max_ops=max_ops,
            seed=seed,
            detect_sql=detect_sql,
        )
    else:
        return _explore_sync(
            setup=setup,
            workers=worker_list,
            invariant=invariant,
            strategy=strategy,
            max_executions=max_executions,
            preemption_bound=preemption_bound,
            max_branches=max_branches,
            timeout_per_run=timeout_per_run,
            stop_on_first=stop_on_first,
            detect_io=detect_io,
            deadlock_timeout=deadlock_timeout,
            reproduce_on_failure=reproduce_on_failure,
            total_timeout=total_timeout,
            warn_nondeterministic_sql=warn_nondeterministic_sql,
            lock_timeout=lock_timeout,
            trace_packages=trace_packages,
            track_dunder_dict_accesses=track_dunder_dict_accesses,
            search=search,
            patch_sleep=patch_sleep,
            serializable_invariant=serializable_invariant,
            error_on_any_race=error_on_any_race,
            max_attempts=max_attempts,
            max_ops=max_ops,
            seed=seed,
            debug=debug,
        )


def _resolve_workers(
    workers: Callable[[Any], Any] | list[Callable[[Any], Any]] | tuple[Callable[[Any], Any], ...],
    count: int | None,
) -> list[Callable[[Any], Any]]:
    """Expand workers + count into a concrete list."""
    if isinstance(workers, (list, tuple)):
        if count is not None:
            raise ValueError(
                "explore(): 'count' cannot be used when 'workers' is already a list or tuple. "
                "Either pass a single callable with count=N, or pass a list without count."
            )
        worker_list = list(workers)
    else:
        if count is not None:
            if count <= 0:
                raise ValueError(f"explore(): count must be a positive integer, got {count!r}")
            worker_list = [workers] * count
        else:
            worker_list = [workers]

    if not worker_list:
        raise ValueError("explore(): workers list is empty")

    return worker_list


def _explore_sync(
    setup: Callable[[], Any],
    workers: list[Callable[[Any], Any]],
    invariant: Callable[[Any], bool],
    strategy: str,
    **kwargs: Any,
) -> InterleavingResult:
    """Dispatch to the sync DPOR or random implementation."""
    if strategy == "dpor":
        from frontrun._dpor_runtime.explore import _explore_dpor

        dpor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "max_executions",
                "preemption_bound",
                "max_branches",
                "timeout_per_run",
                "stop_on_first",
                "detect_io",
                "deadlock_timeout",
                "reproduce_on_failure",
                "total_timeout",
                "warn_nondeterministic_sql",
                "lock_timeout",
                "trace_packages",
                "track_dunder_dict_accesses",
                "search",
                "patch_sleep",
                "serializable_invariant",
                "error_on_any_race",
            )
        }
        return _explore_dpor(setup=setup, threads=workers, invariant=invariant, **dpor_kwargs)
    else:  # random
        from frontrun.bytecode import explore_random as _explore_random

        _random_sync_keys = {
            "max_attempts",
            "max_ops",
            "timeout_per_run",
            "seed",
            "debug",
            "detect_io",
            "deadlock_timeout",
            "reproduce_on_failure",
            "total_timeout",
            "warn_nondeterministic_sql",
            "trace_packages",
            "patch_sleep",
            "serializable_invariant",
            "error_on_any_race",
        }
        random_kwargs = {k: v for k, v in kwargs.items() if k in _random_sync_keys and v is not None}
        return _explore_random(setup=setup, threads=workers, invariant=invariant, **random_kwargs)


async def _explore_async(
    setup: Callable[[], Any],
    workers: list[Callable[[Any], Any]],
    invariant: Callable[[Any], bool],
    strategy: str,
    **kwargs: Any,
) -> InterleavingResult:
    """Dispatch to the async DPOR or random implementation."""
    if strategy == "dpor":
        from frontrun.async_dpor import _explore_async_dpor

        # detect_io in async DPOR enables both SQL and Redis
        detect_io = kwargs.pop("detect_io", True)
        detect_sql = kwargs.pop("detect_sql", False) or detect_io
        detect_redis = detect_io  # detect_io now covers Redis in async too

        dpor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "max_executions",
                "preemption_bound",
                "max_branches",
                "timeout_per_run",
                "stop_on_first",
                "deadlock_timeout",
                "reproduce_on_failure",
                "total_timeout",
                "warn_nondeterministic_sql",
                "lock_timeout",
                "trace_packages",
                "patch_sleep",
                "serializable_invariant",
                "error_on_any_race",
            )
        }
        return await _explore_async_dpor(
            setup=setup,
            tasks=workers,
            invariant=invariant,
            detect_sql=detect_sql,
            detect_redis=detect_redis,
            **dpor_kwargs,
        )
    else:  # random
        from frontrun.async_shuffler import explore_async_random as _explore_async_random

        _async_random_keys = {
            "max_attempts",
            "max_ops",
            "timeout_per_run",
            "seed",
            "detect_sql",
            "deadlock_timeout",
            "trace_packages",
            "patch_sleep",
            "serializable_invariant",
            "error_on_any_race",
        }
        # Omit None values so underlying functions use their own defaults.
        random_kwargs = {k: v for k, v in kwargs.items() if k in _async_random_keys and v is not None}
        # detect_io maps to detect_sql for async random
        if "detect_io" in kwargs:
            random_kwargs.setdefault("detect_sql", kwargs["detect_io"])
        return await _explore_async_random(setup=setup, tasks=workers, invariant=invariant, **random_kwargs)
