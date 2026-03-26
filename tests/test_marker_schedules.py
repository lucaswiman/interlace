"""Tests for property-based marker schedule generation.

Tests marker_schedule_strategy (Hypothesis strategy for generating valid
marker-level schedules) and explore_marker_interleavings (exhaustive or
random exploration of all marker-level interleavings).
"""

import pytest
from hypothesis import given, settings

from frontrun.common import Schedule, Step
from frontrun.trace_markers import (
    TraceExecutor,
    all_marker_schedules,
    explore_marker_interleavings,
    marker_schedule_strategy,
)


# ---------------------------------------------------------------------------
# Test subject: classic lost-update race
# ---------------------------------------------------------------------------


class BankAccount:
    def __init__(self, balance: int = 0):
        self.balance = balance

    def transfer(self, amount: int) -> None:
        current = self.balance  # frontrun: read_balance
        new_balance = current + amount
        self.balance = new_balance  # frontrun: write_balance


# ---------------------------------------------------------------------------
# marker_schedule_strategy — Hypothesis strategy
# ---------------------------------------------------------------------------


class TestMarkerScheduleStrategy:
    """marker_schedule_strategy generates valid Schedule objects."""

    def test_generates_schedule_objects(self):
        """Strategy produces Schedule instances."""
        strategy = marker_schedule_strategy(
            threads={"t1": ["read", "write"], "t2": ["read", "write"]},
        )
        example = strategy.example()
        assert isinstance(example, Schedule)

    def test_preserves_per_thread_order(self):
        """Each thread's markers appear in their declared order."""
        strategy = marker_schedule_strategy(
            threads={"t1": ["a", "b", "c"], "t2": ["x", "y"]},
        )
        for _ in range(50):
            schedule = strategy.example()
            t1_markers = [s.marker_name for s in schedule.steps if s.execution_name == "t1"]
            t2_markers = [s.marker_name for s in schedule.steps if s.execution_name == "t2"]
            assert t1_markers == ["a", "b", "c"], f"t1 order violated: {t1_markers}"
            assert t2_markers == ["x", "y"], f"t2 order violated: {t2_markers}"

    def test_total_step_count(self):
        """Schedule contains exactly the sum of all markers."""
        strategy = marker_schedule_strategy(
            threads={"t1": ["a", "b"], "t2": ["x", "y", "z"]},
        )
        for _ in range(20):
            schedule = strategy.example()
            assert len(schedule.steps) == 5

    def test_three_threads(self):
        """Works with three or more threads."""
        strategy = marker_schedule_strategy(
            threads={"t1": ["a"], "t2": ["b"], "t3": ["c"]},
        )
        schedule = strategy.example()
        assert len(schedule.steps) == 3
        names = {s.execution_name for s in schedule.steps}
        assert names == {"t1", "t2", "t3"}

    @given(
        schedule=marker_schedule_strategy(
            threads={"w1": ["read", "write"], "w2": ["read", "write"]},
        )
    )
    @settings(max_examples=100)
    def test_hypothesis_integration(self, schedule: Schedule):
        """Strategy works as a Hypothesis argument."""
        assert isinstance(schedule, Schedule)
        assert len(schedule.steps) == 4
        w1 = [s.marker_name for s in schedule.steps if s.execution_name == "w1"]
        w2 = [s.marker_name for s in schedule.steps if s.execution_name == "w2"]
        assert w1 == ["read", "write"]
        assert w2 == ["read", "write"]


# ---------------------------------------------------------------------------
# all_marker_schedules — exhaustive enumeration
# ---------------------------------------------------------------------------


class TestAllMarkerSchedules:
    """all_marker_schedules enumerates all valid interleavings."""

    def test_two_threads_two_markers_each(self):
        """C(4,2) = 6 valid interleavings for 2 threads with 2 markers each."""
        schedules = all_marker_schedules(
            threads={"t1": ["a", "b"], "t2": ["x", "y"]},
        )
        assert len(schedules) == 6
        # All should be distinct
        as_tuples = {tuple((s.execution_name, s.marker_name) for s in sched.steps) for sched in schedules}
        assert len(as_tuples) == 6

    def test_preserves_order(self):
        """Every enumerated schedule preserves per-thread marker order."""
        schedules = all_marker_schedules(
            threads={"t1": ["a", "b", "c"], "t2": ["x", "y"]},
        )
        for sched in schedules:
            t1 = [s.marker_name for s in sched.steps if s.execution_name == "t1"]
            t2 = [s.marker_name for s in sched.steps if s.execution_name == "t2"]
            assert t1 == ["a", "b", "c"]
            assert t2 == ["x", "y"]

    def test_single_thread(self):
        """Single thread = exactly one schedule."""
        schedules = all_marker_schedules(threads={"t1": ["a", "b", "c"]})
        assert len(schedules) == 1
        markers = [s.marker_name for s in schedules[0].steps]
        assert markers == ["a", "b", "c"]

    def test_three_threads(self):
        """Three threads with 1 marker each = 3! = 6 interleavings."""
        schedules = all_marker_schedules(
            threads={"t1": ["a"], "t2": ["b"], "t3": ["c"]},
        )
        assert len(schedules) == 6

    def test_three_threads_mixed_lengths(self):
        """Three threads: C(5,2)*C(3,1) = 10*3 = 30 for [2,2,1]."""
        schedules = all_marker_schedules(
            threads={"t1": ["a", "b"], "t2": ["x", "y"], "t3": ["z"]},
        )
        # Total = 5! / (2! * 2! * 1!) = 30
        assert len(schedules) == 30

    def test_returns_schedule_objects(self):
        """Returns list of Schedule instances."""
        schedules = all_marker_schedules(threads={"t1": ["a"], "t2": ["b"]})
        assert all(isinstance(s, Schedule) for s in schedules)


# ---------------------------------------------------------------------------
# explore_marker_interleavings — run schedules against real code
# ---------------------------------------------------------------------------


class TestExploreMarkerInterleavings:
    """explore_marker_interleavings runs marker-level schedules against real code."""

    def test_finds_lost_update_bug(self):
        """Exhaustive marker exploration finds the lost-update race."""
        result = explore_marker_interleavings(
            setup=lambda: BankAccount(balance=100),
            threads={
                "thread1": (lambda s: s.transfer(50), ["read_balance", "write_balance"]),
                "thread2": (lambda s: s.transfer(50), ["read_balance", "write_balance"]),
            },
            invariant=lambda s: s.balance == 200,
        )
        assert not result.property_holds
        assert result.num_explored > 0
        # The counterexample should be a Schedule
        assert result.counterexample is not None

    def test_correct_code_passes(self):
        """Code that is correct under all interleavings passes."""

        class SafeCounter:
            def __init__(self):
                self.value = 0
                import threading

                self.lock = threading.Lock()

            def increment(self):
                with self.lock:
                    temp = self.value  # frontrun: read
                    self.value = temp + 1  # frontrun: write

        result = explore_marker_interleavings(
            setup=SafeCounter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
        )
        assert result.property_holds
        assert result.num_explored == 6  # C(4,2) = 6

    def test_exhaustive_explores_all(self):
        """Exhaustive mode explores all valid interleavings."""
        explored_schedules: list[Schedule] = []

        class State:
            def __init__(self):
                self.log: list[str] = []

            def worker(self, name: str) -> None:
                self.log.append(f"{name}_a")  # frontrun: step_a
                self.log.append(f"{name}_b")  # frontrun: step_b

        def capture_invariant(s: State) -> bool:
            explored_schedules.append(s.log[:])  # type: ignore[arg-type]
            return True

        result = explore_marker_interleavings(
            setup=State,
            threads={
                "t1": (lambda s: s.worker("t1"), ["step_a", "step_b"]),
                "t2": (lambda s: s.worker("t2"), ["step_a", "step_b"]),
            },
            invariant=capture_invariant,
        )
        assert result.property_holds
        assert result.num_explored == 6  # C(4,2) = 6

    def test_stops_on_first_failure(self):
        """By default, stops after finding the first failure."""
        call_count = 0

        class Counter:
            def __init__(self):
                self.value = 0

            def inc(self):
                nonlocal call_count
                call_count += 1
                temp = self.value  # frontrun: read
                self.value = temp + 1  # frontrun: write

        result = explore_marker_interleavings(
            setup=Counter,
            threads={
                "t1": (lambda s: s.inc(), ["read", "write"]),
                "t2": (lambda s: s.inc(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
        )
        assert not result.property_holds
        # Should have stopped before exploring all 6
        assert result.num_explored < 6

    def test_counterexample_is_reproducible(self):
        """The counterexample schedule can be replayed to reproduce the bug."""
        result = explore_marker_interleavings(
            setup=lambda: BankAccount(balance=100),
            threads={
                "thread1": (lambda s: s.transfer(50), ["read_balance", "write_balance"]),
                "thread2": (lambda s: s.transfer(50), ["read_balance", "write_balance"]),
            },
            invariant=lambda s: s.balance == 200,
        )
        assert result.counterexample is not None

        # Replay the counterexample
        account = BankAccount(balance=100)
        executor = TraceExecutor(result.counterexample)
        executor.run("thread1", lambda: account.transfer(50))
        executor.run("thread2", lambda: account.transfer(50))
        executor.wait(timeout=5.0)
        # The counterexample should reproduce the invariant violation
        assert account.balance != 200
