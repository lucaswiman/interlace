from __future__ import annotations

import asyncio
import random
import threading
from typing import Any

import pytest


class _PauseRecorder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def pause(self, task_id: int) -> None:
        from frontrun._async_autopause import _in_scheduler_pause

        self.calls.append(task_id)
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            await asyncio.sleep(0)
        finally:
            _in_scheduler_pause.set(depth)


def test_wrap_auto_paused_tasks_inserts_scheduler_pause() -> None:
    from frontrun._async_autopause import wrap_auto_paused_tasks

    recorder = _PauseRecorder()
    events: list[str] = []

    async def task() -> None:
        events.append("before")
        await asyncio.sleep(0)
        events.append("after")

    wrapped = wrap_auto_paused_tasks({7: task}, recorder)
    asyncio.run(wrapped[7]())

    assert events == ["before", "after"]
    assert recorder.calls
    assert set(recorder.calls) == {7}


def test_dispatch_threads_or_tasks_selects_correct_impl() -> None:
    from frontrun.contrib._shared import dispatch_threads_or_tasks

    sync_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    async_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def sync_impl(*args: Any, **kwargs: Any) -> str:
        sync_calls.append((args, kwargs))
        return "sync"

    async def async_impl(*args: Any, **kwargs: Any) -> str:
        async_calls.append((args, kwargs))
        return "async"

    assert dispatch_threads_or_tasks(sync_impl, async_impl, setup=object, threads=[lambda _: None]) == "sync"
    assert sync_calls and not async_calls

    result = asyncio.run(
        dispatch_threads_or_tasks(
            sync_impl,
            async_impl,
            setup=object,
            tasks=[lambda _: _noop()],
        )
    )
    assert result == "async"
    assert len(async_calls) == 1


def test_dispatch_threads_or_tasks_requires_exactly_one_mode() -> None:
    from frontrun.contrib._shared import dispatch_threads_or_tasks

    def sync_impl(*args: Any, **kwargs: Any) -> None:
        return None

    async def async_impl(*args: Any, **kwargs: Any) -> None:
        return None

    with pytest.raises(TypeError, match="requires exactly one"):
        dispatch_threads_or_tasks(sync_impl, async_impl, setup=object)

    with pytest.raises(TypeError, match="requires exactly one"):
        dispatch_threads_or_tasks(
            sync_impl,
            async_impl,
            setup=object,
            threads=[lambda _: None],
            tasks=[lambda _: _noop()],
        )


async def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# Shared schedule generation helpers (sync + async random exploration)
# ---------------------------------------------------------------------------


def test_random_round_robin_schedule_is_fair_and_deterministic() -> None:
    from frontrun._random_schedules import random_round_robin_schedule

    rng = random.Random(0)
    schedule = random_round_robin_schedule(rng, num_actors=3, max_ops=12)

    # Every entry is a valid actor id.
    assert schedule, "schedule should not be empty"
    assert all(0 <= entry < 3 for entry in schedule)

    # The schedule is a sequence of full permutations of [0, 1, 2], so the
    # length is a multiple of num_actors and every actor appears equally
    # often -- this is the fairness property.
    assert len(schedule) % 3 == 0
    counts = [schedule.count(i) for i in range(3)]
    assert counts[0] == counts[1] == counts[2]

    # Same RNG seed -> same schedule (determinism).
    rng2 = random.Random(0)
    assert random_round_robin_schedule(rng2, num_actors=3, max_ops=12) == schedule


def test_random_round_robin_schedule_respects_max_ops_cap() -> None:
    from frontrun._random_schedules import random_round_robin_schedule

    rng = random.Random(123)
    schedule = random_round_robin_schedule(rng, num_actors=4, max_ops=4)
    # max_ops // num_actors == 1, so the schedule should be exactly one round.
    assert len(schedule) == 4
    assert sorted(schedule) == [0, 1, 2, 3]


def test_fair_schedule_strategy_generates_round_complete_schedules() -> None:
    pytest.importorskip("hypothesis")
    from hypothesis import HealthCheck, given, settings

    from frontrun._random_schedules import fair_schedule_strategy

    # Property: every generated schedule is built from full round-robin
    # permutations of [0, ..., num_actors - 1].
    @given(schedule=fair_schedule_strategy(num_actors=3, max_ops=15))
    @settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def _check(schedule: list[int]) -> None:
        assert schedule, "fair schedule should be non-empty"
        assert len(schedule) % 3 == 0, "fair schedule must be a sequence of complete rounds"
        for start in range(0, len(schedule), 3):
            assert sorted(schedule[start : start + 3]) == [0, 1, 2]

    _check()


# ---------------------------------------------------------------------------
# Shared marker-executor finalization (sync + async trace markers)
# ---------------------------------------------------------------------------


def _make_completed_coordinator(num_steps: int = 1) -> Any:
    from frontrun._marker_coordination import ThreadCoordinator
    from frontrun.common import Schedule, Step

    schedule = Schedule([Step(execution_name=f"t{i}", marker_name="m") for i in range(num_steps)])
    coord = ThreadCoordinator(schedule)
    coord.current_step = num_steps
    coord.completed = True
    return coord


def test_finalize_marker_executor_run_no_op_when_clean() -> None:
    from frontrun._marker_coordination import finalize_marker_executor_run

    coord = _make_completed_coordinator(num_steps=2)
    # Should not raise: no threads alive, no errors, schedule completed.
    finalize_marker_executor_run(
        threads=[],
        timeout=None,
        task_errors={},
        coordinator=coord,
        timeout_message=lambda alive: "unused",
    )


def test_finalize_marker_executor_run_raises_for_alive_threads() -> None:
    from frontrun._marker_coordination import finalize_marker_executor_run

    coord = _make_completed_coordinator()
    blocker = threading.Event()
    thread = threading.Thread(target=blocker.wait, name="stuck", daemon=True)
    thread.start()
    try:
        with pytest.raises(TimeoutError, match="custom-timeout-message: stuck"):
            finalize_marker_executor_run(
                threads=[thread],
                timeout=0.05,
                task_errors={},
                coordinator=coord,
                timeout_message=lambda alive: "custom-timeout-message: " + ", ".join(t.name for t in alive),
            )
    finally:
        blocker.set()
        thread.join(timeout=1.0)


def test_finalize_marker_executor_run_reraises_first_task_error() -> None:
    from frontrun._marker_coordination import finalize_marker_executor_run

    coord = _make_completed_coordinator()
    err1 = RuntimeError("first")
    err2 = RuntimeError("second")
    with pytest.raises(RuntimeError, match="first"):
        finalize_marker_executor_run(
            threads=[],
            timeout=None,
            task_errors={"a": err1, "b": err2},
            coordinator=coord,
            timeout_message=lambda alive: "unused",
        )


def test_finalize_marker_executor_run_detects_partial_schedule() -> None:
    from frontrun._marker_coordination import ThreadCoordinator, finalize_marker_executor_run
    from frontrun.common import Schedule, Step

    schedule = Schedule(
        [
            Step(execution_name="t1", marker_name="m1"),
            Step(execution_name="t2", marker_name="never_reached"),
        ]
    )
    coord = ThreadCoordinator(schedule)
    coord.current_step = 1  # one step consumed, second never reached
    coord.completed = False

    with pytest.raises(TimeoutError, match="Schedule incomplete.*never_reached"):
        finalize_marker_executor_run(
            threads=[],
            timeout=None,
            task_errors={},
            coordinator=coord,
            timeout_message=lambda alive: "unused",
        )
