"""Django helpers for DPOR integration testing (sync and async).

Use ``django_dpor`` for both sync and async code:

Sync (threads)::

    result = django_dpor(
        setup=_State,
        threads=[thread_a, thread_b],
        invariant=_invariant,
    )

Async (tasks)::

    result = await django_dpor(
        setup=_State,
        tasks=[task_a, task_b],
        invariant=_invariant,
    )
"""

from typing import Any

from frontrun.contrib._shared import dispatch_threads_or_tasks
from frontrun.contrib.django._async import async_django_dpor
from frontrun.contrib.django._sync import DJANGO_TRACE_PACKAGES
from frontrun.contrib.django._sync import django_dpor as _sync_django_dpor


def django_dpor(*args: Any, **kwargs: Any) -> Any:
    """Run DPOR with per-thread/task Django connection management.

    Use ``threads=[...]`` for sync code or ``tasks=[...]`` for async code.
    The async form returns an awaitable.
    """
    return dispatch_threads_or_tasks(_sync_django_dpor, async_django_dpor, *args, api_name="django_dpor", **kwargs)


__all__ = ["DJANGO_TRACE_PACKAGES", "async_django_dpor", "django_dpor"]
