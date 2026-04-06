"""Shared data structures for frontrun."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frontrun._sql_anomaly import SqlAnomaly


class NondeterministicSQLError(Exception):
    """Raised when SQL INSERT statements are detected during exploration.

    Autoincrement/SERIAL/IDENTITY columns assign IDs based on execution
    order, making test results non-deterministic across interleavings.
    Pre-allocate rows with explicit IDs in your test setup instead.

    Pass ``warn_nondeterministic_sql=False`` to suppress this check if
    you understand the implications.
    """


@dataclass
class Step:
    """Represents a single step in the execution schedule.

    Attributes:
        execution_name: The name of the execution unit (thread/task) that should execute this step
        marker_name: The marker name that identifies this synchronization point
    """

    execution_name: str
    marker_name: str

    def __repr__(self):
        return f"Step({self.execution_name!r}, {self.marker_name!r})"


class Schedule:
    """Defines the execution order for tasks at synchronization points.

    A schedule is a linear sequence of steps that specify which task should
    execute which marker in order.
    """

    def __init__(self, steps: list[Step]):
        """Initialize a schedule with a list of steps.

        Args:
            steps: Ordered list of Step objects defining the execution sequence
        """
        self.steps = steps
        self._validate()

    def _validate(self):
        """Validate that the schedule is well-formed."""
        if not self.steps:
            raise ValueError("Schedule must contain at least one step")

    def __repr__(self):
        return f"Schedule({self.steps!r})"


@dataclass
class InterleavingResult:
    """Result of exploring interleavings.

    Returned by :func:`~frontrun.bytecode.explore_interleavings`,
    :func:`~frontrun.async_shuffler.explore_interleavings`, and
    :func:`~frontrun.dpor.explore_dpor`.

    Attributes:
        property_holds: True if the invariant held under all tested interleavings.
        counterexample: First schedule that violated the invariant (if any).
        num_explored: How many interleavings were tested.
        unique_interleavings: Number of distinct schedule orderings observed.
            Provides a lower bound on interleaving-space coverage.  Relevant
            for random bytecode exploration; DPOR always explores distinct
            interleavings so this equals ``num_explored``.
        failures: All failing (execution_number, schedule) pairs.  Only
            populated by DPOR when ``stop_on_first=False``.
        explanation: Human-readable explanation of the race condition, showing
            interleaved source lines and the conflict pattern. None if no
            race was found.
        reproduction_attempts: Number of times the counterexample schedule
            was re-run to test reproducibility.  0 if no counterexample.
        reproduction_successes: How many of those re-runs reproduced the
            invariant violation.
        sql_anomaly: Classified SQL isolation anomaly (if any SQL I/O events
            were recorded).  A :class:`~frontrun._sql_anomaly.SqlAnomaly`
            instance, or None if the failure did not involve SQL.
    """

    property_holds: bool
    counterexample: list[int] | Schedule | None = None
    num_explored: int = 0
    unique_interleavings: int = 0
    failures: list[tuple[int, list[int]]] = field(default_factory=list)
    explanation: str | None = None
    reproduction_attempts: int = 0
    reproduction_successes: int = 0
    sql_anomaly: SqlAnomaly | None = None
    races_detected: bool = False

    def __repr__(self) -> str:
        ce = self.counterexample
        if ce is not None and isinstance(ce, list) and len(ce) > 10:
            ce_repr = f"[{', '.join(map(str, ce[:5]))}, ...({len(ce)} steps)]"
        else:
            ce_repr = repr(ce)
        parts = [
            f"property_holds={self.property_holds}",
            f"counterexample={ce_repr}",
            f"num_explored={self.num_explored}",
        ]
        if self.races_detected:
            parts.append("races_detected=True")
        return f"InterleavingResult({', '.join(parts)})"


def compute_serializable_states(
    setup: Callable[[], Any],
    thread_funcs: list[Callable[[Any], None]],
    state_hash: Callable[[Any], Any] | None = None,
) -> set[Any]:
    """Compute the set of valid serializable states.

    Runs all N! sequential orderings of the thread functions and collects
    the hash of each resulting state.  An interleaved execution is
    *serializable* if its final state hash is in this set.

    Args:
        setup: Factory that creates fresh shared state.
        thread_funcs: Thread/task functions (each takes state as argument).
        state_hash: Hash function for state.  If None, uses ``repr()``.

    Returns:
        Set of valid state hashes.
    """
    if state_hash is None:
        state_hash = repr
    valid: set[Any] = set()
    for perm in permutations(range(len(thread_funcs))):
        s = setup()
        for i in perm:
            thread_funcs[i](s)
        valid.add(state_hash(s))
    return valid


async def compute_serializable_states_async(
    setup: Callable[[], Any],
    task_funcs: list[Callable[[Any], Any]],
    state_hash: Callable[[Any], Any] | None = None,
) -> set[Any]:
    """Async version of compute_serializable_states.

    Runs all N! sequential orderings of async task functions.
    """
    if state_hash is None:
        state_hash = repr
    valid: set[Any] = set()
    for perm in permutations(range(len(task_funcs))):
        s = setup()
        for i in perm:
            await task_funcs[i](s)
        valid.add(state_hash(s))
    return valid
