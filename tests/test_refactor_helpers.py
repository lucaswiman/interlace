from __future__ import annotations

import asyncio
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
