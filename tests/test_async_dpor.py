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
