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


def django_dpor(*args: Any, **kwargs: Any) -> Any:
    """Run DPOR with per-thread/task Django connection management.

    Use ``threads=[...]`` for sync code or ``tasks=[...]`` for async code.
    The async form returns an awaitable.
    """
    has_threads = "threads" in kwargs
    has_tasks = "tasks" in kwargs
    if has_threads == has_tasks:
        raise TypeError("django_dpor() requires exactly one of 'threads' or 'tasks'")

    if has_tasks:
        from frontrun.contrib.django._async import async_django_dpor

        return async_django_dpor(*args, **kwargs)

    from frontrun.contrib.django._sync import django_dpor as _sync_django_dpor

    return _sync_django_dpor(*args, **kwargs)


# Keep async_django_dpor importable for backward compatibility.
from frontrun.contrib.django._async import async_django_dpor  # noqa: E402, F811

__all__ = ["async_django_dpor", "django_dpor"]
