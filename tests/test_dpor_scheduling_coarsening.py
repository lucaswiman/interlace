"""Tests for DPOR scheduling coarsening optimization.

Verifies that reducing scheduling points for non-shared-access opcodes
reduces the DPOR exploration space without missing real bugs or deadlocks.
"""

from __future__ import annotations

import threading

from frontrun.dpor import explore_dpor


class TestSchedulingCoarsening:
    """Scheduling coarsening should reduce exploration without missing bugs."""

    def test_dining_philosophers_two_with_shared_write_execution_count(self) -> None:
        """Two dining philosophers with shared write.

        With scheduling coarsening, the DPOR engine should need far fewer
        executions because non-shared opcodes (LOAD_FAST, BINARY_OP +,
        SWAP, COPY, etc.) are no longer scheduling points.

        Before optimization: ~100+ executions
        After optimization: should be significantly fewer
        """
        num_philosophers = 2

        class State:
            def __init__(self) -> None:
                self.forks = [threading.Lock() for _ in range(num_philosophers)]
                self.x = 0

        def make_philosopher(i: int):  # noqa: ANN202
            def philosopher(s: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                with s.forks[left]:
                    s.x += 1
                    with s.forks[right]:
                        pass

            return philosopher

        result = explore_dpor(
            setup=State,
            threads=[make_philosopher(i) for i in range(num_philosophers)],
            invariant=lambda s: True,
            max_executions=5000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=True,
        )

        assert not result.property_holds, "Deadlock should be found"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()
        # With scheduling coarsening, we should find the deadlock in
        # far fewer executions than the current ~100+
        assert result.num_explored <= 50, (
            f"Expected <=50 executions with scheduling coarsening, got {result.num_explored}"
        )

    def test_dining_philosophers_three_with_shared_write_execution_count(self) -> None:
        """Three dining philosophers with shared write.

        This is the key benchmark from the commit message. With 17 scheduling
        points per thread (one per opcode), the DPOR search tree explodes.
        Scheduling coarsening should dramatically reduce this.

        Before optimization: thousands of executions
        After optimization: should be hundreds or fewer
        """
        num_philosophers = 3

        class State:
            def __init__(self) -> None:
                self.forks = [threading.Lock() for _ in range(num_philosophers)]
                self.x = 0

        def make_philosopher(i: int):  # noqa: ANN202
            def philosopher(s: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                with s.forks[left]:
                    s.x += 1
                    with s.forks[right]:
                        pass

            return philosopher

        result = explore_dpor(
            setup=State,
            threads=[make_philosopher(i) for i in range(num_philosophers)],
            invariant=lambda s: True,
            max_executions=50000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=True,
        )

        assert not result.property_holds, "Deadlock should be found"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()
        # With scheduling coarsening, the 3-philosopher case should be
        # dramatically more efficient
        assert result.num_explored <= 500, (
            f"Expected <=500 executions with scheduling coarsening, got {result.num_explored}"
        )

    def test_dining_philosophers_two_exhaustive(self) -> None:
        """Two dining philosophers with shared write, exhaustive exploration."""
        num_philosophers = 2

        class State:
            def __init__(self) -> None:
                self.forks = [threading.Lock() for _ in range(num_philosophers)]
                self.x = 0

        def make_philosopher(i: int):  # noqa: ANN202
            def philosopher(s: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                with s.forks[left]:
                    s.x += 1
                    with s.forks[right]:
                        pass

            return philosopher

        result = explore_dpor(
            setup=State,
            threads=[make_philosopher(i) for i in range(num_philosophers)],
            invariant=lambda s: True,
            max_executions=5000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=False,
        )

        print(f"\nN=2 exhaustive: {result.num_explored} executions, failures={len(result.failures)}")

    def test_dining_philosophers_three_exhaustive(self) -> None:
        """Three dining philosophers with shared write, exhaustive exploration."""
        num_philosophers = 3

        class State:
            def __init__(self) -> None:
                self.forks = [threading.Lock() for _ in range(num_philosophers)]
                self.x = 0

        def make_philosopher(i: int):  # noqa: ANN202
            def philosopher(s: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                with s.forks[left]:
                    s.x += 1
                    with s.forks[right]:
                        pass

            return philosopher

        result = explore_dpor(
            setup=State,
            threads=[make_philosopher(i) for i in range(num_philosophers)],
            invariant=lambda s: True,
            max_executions=50000,
            preemption_bound=2,
            detect_io=False,
            deadlock_timeout=2.0,
            stop_on_first=False,
        )

        print(f"\nN=3 exhaustive: {result.num_explored} executions, failures={len(result.failures)}")

    def test_lost_update_still_detected(self) -> None:
        """Scheduling coarsening must not hide real races.

        The lost-update bug in a simple counter should still be found.
        """

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds
        assert len(result.failures) > 0

    def test_augmented_assignment_still_detected(self) -> None:
        """The += race should still be detected with coarser scheduling."""

        class Counter:
            def __init__(self) -> None:
                self.value = 0

            def increment(self) -> None:
                self.value += 1

        result = explore_dpor(
            setup=Counter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=500,
            preemption_bound=2,
        )

        assert not result.property_holds

    def test_lock_protected_still_correct(self) -> None:
        """Lock-protected code should still pass with coarser scheduling."""

        class LockedCounter:
            def __init__(self) -> None:
                self.value = 0
                self.lock = threading.Lock()

            def increment(self) -> None:
                with self.lock:
                    temp = self.value
                    self.value = temp + 1

        result = explore_dpor(
            setup=LockedCounter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            max_executions=50,
            preemption_bound=2,
        )

        assert result.property_holds
