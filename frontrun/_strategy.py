"""Strategy interface used by :func:`frontrun.explore`.

Each of the four exploration approaches (sync DPOR, sync random, async DPOR,
async random) is wrapped in a small adapter that conforms to the
:class:`Strategy` / :class:`AsyncStrategy` protocols below.  The
:func:`frontrun.explore` dispatcher looks adapters up in
:data:`STRATEGIES` / :data:`ASYNC_STRATEGIES` rather than threading a stack
of ``if strategy == ...`` branches; adding a fifth approach later is then a
matter of registering one more adapter.

Adapters live here (rather than next to each implementation) because some of
the implementation modules — ``async_dpor.py`` and ``_dpor_runtime/`` — are
owned by other concurrent agents and cannot be edited from this branch.

Each adapter owns its allowed-kwargs filter so the dispatcher does not need
to know which of the unified ``explore()`` keyword arguments apply to which
underlying implementation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from frontrun.common import InterleavingResult


def _select_kwargs(kwargs: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """Keep only *allowed* keys whose value is not ``None``.

    Filtering ``None`` lets the underlying implementation apply its own
    default rather than letting the dispatcher's ``None`` override it.
    """
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


@runtime_checkable
class Strategy(Protocol):
    """Synchronous exploration strategy.

    Attributes:
        name: Human-readable identifier (matches the ``strategy=`` kwarg of
            :func:`frontrun.explore`, e.g. ``"dpor"`` or ``"random"``).
        allowed_keys: The subset of :func:`frontrun.explore` keyword
            arguments this strategy understands.  The dispatcher filters
            unknown / ``None`` kwargs against this set before calling
            :meth:`run`.
    """

    name: str
    allowed_keys: frozenset[str]

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult: ...


@runtime_checkable
class AsyncStrategy(Protocol):
    """Asynchronous exploration strategy.

    Same shape as :class:`Strategy` but :meth:`run` returns an awaitable.
    """

    name: str
    allowed_keys: frozenset[str]

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> Awaitable[InterleavingResult]: ...


# ---------------------------------------------------------------------------
# Sync adapters
# ---------------------------------------------------------------------------


_DPOR_SYNC_KEYS: frozenset[str] = frozenset(
    {
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
    }
)

_RANDOM_SYNC_KEYS: frozenset[str] = frozenset(
    {
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
)


class _SyncDporStrategy:
    """Adapter routing sync DPOR through ``_dpor_runtime.explore._explore_dpor``."""

    name: str = "dpor"
    allowed_keys: frozenset[str] = _DPOR_SYNC_KEYS

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun._dpor_runtime.explore import _explore_dpor

        dpor_kwargs = _select_kwargs(kwargs, self.allowed_keys)
        return _explore_dpor(setup=setup, threads=workers, invariant=invariant, **dpor_kwargs)


class _SyncRandomStrategy:
    """Adapter routing sync random exploration through ``bytecode.explore_random``."""

    name: str = "random"
    allowed_keys: frozenset[str] = _RANDOM_SYNC_KEYS

    def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun.bytecode import explore_random as _explore_random

        random_kwargs = _select_kwargs(kwargs, self.allowed_keys)
        return _explore_random(setup=setup, threads=workers, invariant=invariant, **random_kwargs)


# ---------------------------------------------------------------------------
# Async adapters
# ---------------------------------------------------------------------------


_DPOR_ASYNC_KEYS: frozenset[str] = frozenset(
    {
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
    }
)

_RANDOM_ASYNC_KEYS: frozenset[str] = frozenset(
    {
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
)


class _AsyncDporStrategy:
    """Adapter routing async DPOR through ``async_dpor._explore_async_dpor``.

    ``detect_io`` is consumed here (not forwarded) and expanded into
    ``detect_sql`` + ``detect_redis`` because async DPOR exposes those as
    separate kwargs while the unified API treats them as one.
    """

    name: str = "dpor"
    allowed_keys: frozenset[str] = _DPOR_ASYNC_KEYS

    async def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun.async_dpor import _explore_async_dpor

        # detect_io in async DPOR enables both SQL and Redis
        detect_io = kwargs.pop("detect_io", True)
        detect_sql = kwargs.pop("detect_sql", False) or detect_io
        detect_redis = detect_io

        dpor_kwargs = _select_kwargs(kwargs, self.allowed_keys)
        return await _explore_async_dpor(
            setup=setup,
            tasks=workers,
            invariant=invariant,
            detect_sql=detect_sql,
            detect_redis=detect_redis,
            **dpor_kwargs,
        )


class _AsyncRandomStrategy:
    """Adapter routing async random through ``async_shuffler.explore_async_random``.

    ``detect_io`` collapses into the underlying ``detect_sql`` kwarg here.
    """

    name: str = "random"
    allowed_keys: frozenset[str] = _RANDOM_ASYNC_KEYS

    async def run(
        self,
        *,
        setup: Callable[[], Any],
        workers: list[Callable[[Any], Any]],
        invariant: Callable[[Any], bool],
        **kwargs: Any,
    ) -> InterleavingResult:
        from frontrun.async_shuffler import explore_async_random as _explore_async_random

        detect_io = kwargs.pop("detect_io", True)
        detect_sql_explicit = kwargs.pop("detect_sql", False)
        random_kwargs = _select_kwargs(kwargs, self.allowed_keys)
        random_kwargs["detect_sql"] = detect_sql_explicit or detect_io
        return await _explore_async_random(setup=setup, tasks=workers, invariant=invariant, **random_kwargs)


# ---------------------------------------------------------------------------
# Registries — keyed by the user-facing ``strategy=`` value
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, Strategy] = {
    "dpor": _SyncDporStrategy(),
    "random": _SyncRandomStrategy(),
}

ASYNC_STRATEGIES: dict[str, AsyncStrategy] = {
    "dpor": _AsyncDporStrategy(),
    "random": _AsyncRandomStrategy(),
}


__all__ = [
    "ASYNC_STRATEGIES",
    "STRATEGIES",
    "AsyncStrategy",
    "Strategy",
]
