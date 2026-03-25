"""Tests for progress reporting callbacks in exploration functions."""

import asyncio

import pytest


def test_explore_dpor_on_progress_callback():
    """explore_dpor calls on_progress after each execution."""
    from frontrun.dpor import explore_dpor

    class Counter:
        def __init__(self):
            self.value = 0

        def increment(self):
            temp = self.value
            self.value = temp + 1

    progress_calls: list[tuple[int, int | None]] = []

    def on_progress(explored: int, total_estimate: int | None) -> None:
        progress_calls.append((explored, total_estimate))

    result = explore_dpor(
        setup=Counter,
        threads=[lambda c: c.increment(), lambda c: c.increment()],
        invariant=lambda c: True,
        on_progress=on_progress,
    )

    assert len(progress_calls) > 0
    assert len(progress_calls) == result.num_explored
    # Verify calls are sequential
    for i, (explored, _) in enumerate(progress_calls):
        assert explored == i + 1


def test_explore_interleavings_on_progress_callback():
    """explore_interleavings (bytecode) calls on_progress after each attempt."""
    from frontrun.bytecode import explore_interleavings

    class Counter:
        def __init__(self):
            self.value = 0

        def increment(self):
            self.value += 1

    progress_calls: list[tuple[int, int | None]] = []

    result = explore_interleavings(
        setup=Counter,
        threads=[lambda c: c.increment(), lambda c: c.increment()],
        invariant=lambda c: True,
        max_attempts=5,
        on_progress=lambda explored, total: progress_calls.append((explored, total)),
    )

    assert len(progress_calls) == 5
    assert len(progress_calls) == result.num_explored


@pytest.mark.asyncio
async def test_explore_async_dpor_on_progress_callback():
    """explore_async_dpor calls on_progress after each execution."""
    from frontrun.async_dpor import explore_async_dpor

    class Counter:
        def __init__(self):
            self.value = 0

    progress_calls: list[tuple[int, int | None]] = []

    async def increment(counter: Counter) -> None:
        temp = counter.value
        await asyncio.sleep(0)
        counter.value = temp + 1

    result = await explore_async_dpor(
        setup=Counter,
        tasks=[lambda c: increment(c), lambda c: increment(c)],
        invariant=lambda c: True,
        on_progress=lambda explored, total: progress_calls.append((explored, total)),
    )

    assert len(progress_calls) > 0
    assert len(progress_calls) == result.num_explored


@pytest.mark.asyncio
async def test_explore_async_interleavings_on_progress_callback():
    """async explore_interleavings calls on_progress after each attempt."""
    from frontrun.async_shuffler import explore_interleavings

    class Counter:
        def __init__(self):
            self.value = 0

    progress_calls: list[tuple[int, int | None]] = []

    async def increment(counter: Counter) -> None:
        counter.value += 1

    result = await explore_interleavings(
        setup=Counter,
        tasks=[lambda c: increment(c), lambda c: increment(c)],
        invariant=lambda c: True,
        max_attempts=5,
        on_progress=lambda explored, total: progress_calls.append((explored, total)),
    )

    assert len(progress_calls) == 5
    assert len(progress_calls) == result.num_explored


def test_on_progress_not_called_when_none():
    """When on_progress is None (default), no error occurs."""
    from frontrun.dpor import explore_dpor

    class State:
        pass

    def noop(state: State) -> None:
        pass

    # Should work without on_progress
    result = explore_dpor(
        setup=State,
        threads=[noop, noop],
        invariant=lambda s: True,
    )
    assert result.property_holds


def test_on_progress_stops_early_on_failure():
    """on_progress is called for each execution, including the failing one."""
    from frontrun.dpor import explore_dpor

    class Counter:
        def __init__(self):
            self.value = 0

        def increment(self):
            temp = self.value
            self.value = temp + 1

    progress_calls: list[tuple[int, int | None]] = []

    result = explore_dpor(
        setup=Counter,
        threads=[lambda c: c.increment(), lambda c: c.increment()],
        invariant=lambda c: c.value == 2,
        on_progress=lambda explored, total: progress_calls.append((explored, total)),
        stop_on_first=True,
    )

    # Should have been called for each explored execution
    assert len(progress_calls) == result.num_explored
    assert not result.property_holds
