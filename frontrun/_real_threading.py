"""Saved references to real threading/queue primitives.

This module captures the original factories at import time, before
``_cooperative.patch_locks()`` monkey-patches the ``threading`` and
``queue`` modules.  Any module that needs a *real* (non-cooperative)
lock — including ``_deadlock.py`` and ``_cooperative.py`` — should
import from here.

This file is a leaf module with no ``frontrun`` imports, so it can
never participate in circular-import chains.
"""

import queue
import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

# Names are lowercase to match the existing internal API (``real_lock``,
# ``real_rlock``, …) and to avoid N813 lint violations when re-exported
# from ``_cooperative.py``.
lock = threading.Lock
rlock = threading.RLock
semaphore = threading.Semaphore
bounded_semaphore = threading.BoundedSemaphore
event = threading.Event
condition = threading.Condition
queue_ = queue.Queue
lifo_queue = queue.LifoQueue
priority_queue = queue.PriorityQueue

_factory_lock = lock()
_ConstructedT = TypeVar("_ConstructedT")


def _construct_with_real_threading(
    factory: Callable[..., _ConstructedT], *args: Any, **kwargs: Any
) -> _ConstructedT:
    """Instantiate *factory* while threading primitives point to the originals."""
    with _factory_lock:
        saved_lock = threading.Lock
        saved_rlock = threading.RLock
        saved_semaphore = threading.Semaphore
        saved_bounded = threading.BoundedSemaphore
        saved_event = threading.Event
        saved_condition = threading.Condition
        threading.Lock = lock
        threading.RLock = rlock
        threading.Semaphore = semaphore
        threading.BoundedSemaphore = bounded_semaphore
        threading.Event = event
        threading.Condition = condition
        try:
            return factory(*args, **kwargs)
        finally:
            threading.Lock = saved_lock
            threading.RLock = saved_rlock
            threading.Semaphore = saved_semaphore
            threading.BoundedSemaphore = saved_bounded
            threading.Event = saved_event
            threading.Condition = saved_condition


def make_event() -> threading.Event:
    return _construct_with_real_threading(event)


def make_queue(maxsize: int = 0) -> queue.Queue[Any]:
    return cast(queue.Queue[Any], _construct_with_real_threading(queue_, maxsize))


def make_lifo_queue(maxsize: int = 0) -> queue.LifoQueue[Any]:
    return cast(queue.LifoQueue[Any], _construct_with_real_threading(lifo_queue, maxsize))


def make_priority_queue(maxsize: int = 0) -> queue.PriorityQueue[Any]:
    return cast(queue.PriorityQueue[Any], _construct_with_real_threading(priority_queue, maxsize))
