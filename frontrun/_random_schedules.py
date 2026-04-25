"""Shared random schedule generators for sync + async exploration drivers.

Both :mod:`frontrun.bytecode` (threaded opcode-level) and
:mod:`frontrun.async_shuffler` (await-point-level) drive their random search
loops with the same fair, round-robin schedule. This module owns the
schedule generation so both modes can stay in sync without code drift.
"""

from __future__ import annotations

import random
from typing import Any


def random_round_robin_schedule(rng: random.Random, num_actors: int, max_ops: int) -> list[int]:
    """Build a fair, round-robin schedule of actor ids.

    The schedule is a sequence of full permutations of ``range(num_actors)``,
    so every actor gets the same number of slots. ``num_rounds`` is drawn
    uniformly from ``[1, max(1, max_ops // num_actors)]``.
    """
    num_rounds = rng.randint(1, max(1, max_ops // num_actors))
    schedule: list[int] = []
    for _ in range(num_rounds):
        round_perm = list(range(num_actors))
        rng.shuffle(round_perm)
        schedule.extend(round_perm)
    return schedule


def fair_schedule_strategy(num_actors: int, max_ops: int) -> Any:
    """Hypothesis strategy producing fair, round-robin schedules.

    Mirrors :func:`random_round_robin_schedule` but draws permutations via
    ``hypothesis.strategies.permutations`` so generated schedules shrink
    sensibly. Returns the strategy lazily to avoid importing ``hypothesis``
    at module import time.
    """
    from hypothesis import strategies as st  # type: ignore[import-not-found]

    max_rounds = max(1, max_ops // num_actors)
    actors = list(range(num_actors))

    @st.composite  # type: ignore[attr-defined]
    def _fair_schedule(draw: st.DrawFn) -> list[int]:  # type: ignore[attr-defined,name-defined]
        num_rounds = draw(st.integers(min_value=1, max_value=max_rounds))  # type: ignore[attr-defined]
        schedule: list[int] = []
        for _ in range(num_rounds):
            schedule.extend(draw(st.permutations(actors)))  # type: ignore[attr-defined]
        return schedule

    return _fair_schedule()


__all__ = ["fair_schedule_strategy", "random_round_robin_schedule"]
