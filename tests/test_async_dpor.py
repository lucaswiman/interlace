"""Tests for async DPOR (Dynamic Partial Order Reduction).

Verifies that explore_async_dpor systematically explores async
interleavings using the Rust DPOR engine, with await points as
scheduling granularity.
"""

from __future__ import annotations

import asyncio

from frontrun.cli import require_active


class TestAsyncDporBasic:
    """Basic async DPOR functionality tests."""

    def test_finds_lost_update(self) -> None:
        """DPOR should systematically find the lost-update race."""
        require_active("test_async_dpor_lost_update")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment],
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should find the lost update"
        assert result.num_explored >= 1

    def test_no_race_when_atomic(self) -> None:
        """DPOR should verify correctness when there's no race."""
        require_active("test_async_dpor_no_race")
        from frontrun.async_dpor import explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def safe_increment(counter: Counter) -> None:
            # Atomic: no await between read and write
            counter.value += 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[safe_increment, safe_increment],
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds, f"No race expected: {result.counterexample}"
        assert result.num_explored >= 1

    def test_three_tasks(self) -> None:
        """DPOR should handle three concurrent tasks."""
        require_active("test_async_dpor_three_tasks")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment, increment],
                invariant=lambda c: c.value == 3,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Should find lost update with 3 tasks"

    def test_multiple_await_points(self) -> None:
        """DPOR should explore interleavings with multiple await points per task."""
        require_active("test_async_dpor_multiple_awaits")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.log: list[str] = []

        async def task_a(state: State) -> None:
            state.log.append("a1")
            await await_point()
            state.log.append("a2")
            await await_point()
            state.log.append("a3")

        async def task_b(state: State) -> None:
            state.log.append("b1")
            await await_point()
            state.log.append("b2")

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[task_a, task_b],
                invariant=lambda s: True,  # always passes
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds
        # DPOR should have explored multiple distinct interleavings
        assert result.num_explored >= 1

    def test_stop_on_first(self) -> None:
        """stop_on_first=True should stop after finding the first violation."""
        require_active("test_async_dpor_stop_on_first")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment],
                invariant=lambda c: c.value == 2,
                stop_on_first=True,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert len(result.failures) == 1


class TestAsyncDporDeadlock:
    """Deadlocks should surface as property_holds=False, not silently time out.

    Mirrors the sync TestDeadlockAsInvariantViolation tests in test_dpor.py,
    adapted for async coroutines with asyncio.Lock as the locking primitive.
    """

    def test_two_coroutine_row_lock_deadlock(self) -> None:
        """Classic lock-order inversion: C1 locks row1→row2, C2 locks row2→row1.

        C1 acquires row1, C2 acquires row2, then C1 tries row2 (blocked)
        and C2 tries row1 (blocked).  DPOR should find the deadlocking
        interleaving and report it — not actually deadlock.
        """
        require_active("test_async_dpor_two_coroutine_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.row1 = asyncio.Lock()
                self.row2 = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.row1.acquire()
            try:
                await await_point()
                await state.row2.acquire()
                state.row2.release()
            finally:
                state.row1.release()

        async def coroutine2(state: State) -> None:
            await state.row2.acquire()
            try:
                await await_point()
                await state.row1.acquire()
                state.row1.release()
            finally:
                state.row2.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,  # no data invariant — deadlock itself is the bug
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_three_coroutine_directed_cycle_deadlock(self) -> None:
        """Three-coroutine deadlock: C1→row1→row2, C2→row2→row3, C3→row3→row1.

        Forms the directed cycle C1→C2→C3→C1 when each holds its first
        lock and waits for its second.
        """
        require_active("test_async_dpor_three_coroutine_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.row1 = asyncio.Lock()
                self.row2 = asyncio.Lock()
                self.row3 = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.row1.acquire()
            try:
                await await_point()
                await state.row2.acquire()
                state.row2.release()
            finally:
                state.row1.release()

        async def coroutine2(state: State) -> None:
            await state.row2.acquire()
            try:
                await await_point()
                await state.row3.acquire()
                state.row3.release()
            finally:
                state.row2.release()

        async def coroutine3(state: State) -> None:
            await state.row3.acquire()
            try:
                await await_point()
                await state.row1.acquire()
                state.row1.release()
            finally:
                state.row3.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2, coroutine3],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "3-way deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_combined_asyncio_lock_and_row_lock_deadlock(self) -> None:
        """Mixed resource deadlock: one coroutine holds a row lock and waits
        for an asyncio.Lock, the other holds the asyncio.Lock and waits for
        the row lock.

        This cross-resource deadlock is invisible to both the DB backend
        (which only sees row locks) and asyncio (which only sees asyncio.Lock).
        Only the DPOR scheduler can detect it.
        """
        require_active("test_async_dpor_combined_lock_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.row_lock = asyncio.Lock()  # simulated DB row lock
                self.app_lock = asyncio.Lock()  # application-level asyncio.Lock

        async def coroutine1(state: State) -> None:
            # Acquire the "row lock" first, then try the app lock
            await state.row_lock.acquire()
            try:
                await await_point()
                await state.app_lock.acquire()
                state.app_lock.release()
            finally:
                state.row_lock.release()

        async def coroutine2(state: State) -> None:
            # Acquire the app lock first, then try the row lock
            await state.app_lock.acquire()
            try:
                await await_point()
                await state.row_lock.acquire()
                state.row_lock.release()
            finally:
                state.app_lock.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Cross-resource deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_partial_deadlock_with_completing_coroutine(self) -> None:
        """Three coroutines: two deadlock while a third completes normally.

        C1 and C2 have lock-order inversion; C3 does independent work.
        The partial deadlock should still be detected even though C3 finishes.
        """
        require_active("test_async_dpor_partial_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()
                self.c3_done = False

        async def coroutine1(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await state.lock_b.acquire()
            try:
                await await_point()
                await state.lock_a.acquire()
                state.lock_a.release()
            finally:
                state.lock_b.release()

        async def coroutine3(state: State) -> None:
            await await_point()
            state.c3_done = True

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2, coroutine3],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Partial deadlock should still be detected"

    def test_no_deadlock_consistent_lock_order(self) -> None:
        """Consistent lock ordering should not be reported as a deadlock.

        Both coroutines acquire locks in the same order (lock_a then lock_b),
        so no cycle is possible.
        """
        require_active("test_async_dpor_no_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert result.property_holds, "Consistent lock order should not be reported as deadlock"
