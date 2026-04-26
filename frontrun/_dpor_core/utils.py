"""Pure DPOR utility helpers shared by sync and async drivers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontrun._opcode_observer import StableObjectIds


def group_schedule_runs(schedule: list[int]) -> list[tuple[int, int]]:
    """Collapse a flat schedule into (actor_id, count) run-length pairs.

    Example::

        group_schedule_runs([0, 0, 1, 1, 1, 2])
        # → [(0, 2), (1, 3), (2, 1)]
    """
    runs: list[tuple[int, int]] = []
    for tid in schedule:
        if runs and runs[-1][0] == tid:
            runs[-1] = (tid, runs[-1][1] + 1)
        else:
            runs.append((tid, 1))
    return runs


def make_deadline(total_timeout: float | None) -> float | None:
    """Return ``time.monotonic() + total_timeout`` or ``None`` when no timeout."""
    return time.monotonic() + total_timeout if total_timeout is not None else None


def reset_execution_state(stable_ids: StableObjectIds) -> None:
    """Reset per-execution state shared by sync and async DPOR drivers.

    Clears the SQL insert tracker and resets the stable object ID map so
    IDs assigned in a previous execution do not bleed into the next.
    """
    from frontrun._sql_insert_tracker import clear_insert_tracker

    clear_insert_tracker()
    stable_ids.reset_for_execution()


def extend_replay_schedule(
    replay_schedule: list[int],
    replay_index: int,
    replay_max_ops: int,
    num_actors: int,
    actors_done: set[int],
) -> bool:
    """Extend *replay_schedule* with still-active actors when the schedule runs short.

    Used by replay schedulers (both sync and async) to avoid stalling when the
    original counterexample schedule is exhausted before all actors finish.

    Args:
        replay_schedule: The mutable schedule list to extend in-place.
        replay_index:    Current read position in *replay_schedule*.
        replay_max_ops:  Hard upper bound; refuse to extend beyond this.
        num_actors:      Total number of threads/tasks (range base for IDs).
        actors_done:     Set of actor IDs that have already finished.

    Returns:
        ``True`` if the schedule was extended, ``False`` if it could not be
        (either the cap was reached or all actors are done).
    """
    if replay_index >= replay_max_ops:
        return False
    active = [t for t in range(num_actors) if t not in actors_done]
    if not active:
        return False
    replay_schedule.extend(active)
    return True
