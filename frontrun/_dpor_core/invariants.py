"""Pure invariant / serializability / race helpers shared by sync and async DPOR.

The functions in this module never touch threading, asyncio, sys.settrace,
sys.monitoring, ContextVars, or any scheduler internals.  They take raw
inputs (state, baseline sets, counts, callables) and return strings,
booleans, or sets.

Both ``frontrun/_dpor_runtime/explore.py`` and ``frontrun/async_dpor.py``
import from here so the two drivers stay in lockstep on:

* the shape of the serializability baseline,
* the wording of the unsynchronized-race failure explanation, and
* the trace-filter teardown that runs when baseline computation raises.

The actual ``check_invariant`` and ``check_serializability_violation``
helpers continue to live in :mod:`frontrun.common`; both drivers (and
the random/marker explorers) already share them from there.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from frontrun._tracing import set_active_trace_filter as _set_active_trace_filter
from frontrun.common import (
    compute_serializable_states,
    compute_serializable_states_async,
    resolve_serializable_hash_fn,
)


def compute_serializable_baseline_sync(
    setup: Callable[[], Any],
    threads: list[Callable[[Any], None]],
    serializable_invariant: Callable[[Any], Any] | bool,
) -> tuple[set[Any] | None, Callable[[Any], Any]]:
    """Compute the serializable-states baseline for the sync DPOR driver.

    Returns ``(serial_valid_states, serial_hash_fn)``.  When
    *serializable_invariant* is ``False``, the baseline is ``None`` and
    the hash function defaults to :func:`repr`.

    On any exception during baseline computation the active trace filter
    is cleared (mirroring the explicit ``_set_active_trace_filter(None)``
    that the driver previously did inline) before re-raising.
    """
    if serializable_invariant is False:
        return None, repr
    try:
        hash_fn: Callable[[Any], Any] = resolve_serializable_hash_fn(serializable_invariant) or repr
        valid = compute_serializable_states(setup, threads, state_hash=hash_fn)
    except BaseException:
        _set_active_trace_filter(None)
        raise
    return valid, hash_fn


async def compute_serializable_baseline_async(
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Any]],
    serializable_invariant: Callable[[Any], Any] | bool,
) -> tuple[set[Any] | None, Callable[[Any], Any]]:
    """Async counterpart to :func:`compute_serializable_baseline_sync`.

    Mirrors the sync version exactly except that it awaits each task in
    every permutation (via :func:`compute_serializable_states_async`).
    """
    if serializable_invariant is False:
        return None, repr
    try:
        hash_fn: Callable[[Any], Any] = resolve_serializable_hash_fn(serializable_invariant) or repr
        valid = await compute_serializable_states_async(setup, tasks, state_hash=hash_fn)
    except BaseException:
        _set_active_trace_filter(None)
        raise
    return valid, hash_fn


def format_race_failure_explanation(
    execution_num: int,
    num_races: int,
    *,
    actor_plural: str = "threads",
) -> str:
    """Build the explanation string for an ``error_on_any_race`` failure.

    *actor_plural* is ``"threads"`` for the sync driver and ``"tasks"``
    for the async driver — the only place the two messages differed
    before extraction.
    """
    return (
        f"Unsynchronized race detected in execution {execution_num}.\n"
        f"{num_races} race(s) found between {actor_plural} on shared objects."
    )
