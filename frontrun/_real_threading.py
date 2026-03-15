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


def _construct_with_real_threading(factory: object, *args: object, **kwargs: object) -> object:
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
            return factory(*args, **kwargs)  # type: ignore[operator]
        finally:
            threading.Lock = saved_lock
            threading.RLock = saved_rlock
            threading.Semaphore = saved_semaphore
            threading.BoundedSemaphore = saved_bounded
            threading.Event = saved_event
            threading.Condition = saved_condition


def make_event() -> threading.Event:
    return _construct_with_real_threading(event)


def make_queue(maxsize: int = 0) -> queue.Queue[object]:
    return _construct_with_real_threading(queue_, maxsize)


def make_lifo_queue(maxsize: int = 0) -> queue.LifoQueue[object]:
    return _construct_with_real_threading(lifo_queue, maxsize)


def make_priority_queue(maxsize: int = 0) -> queue.PriorityQueue[object]:
    return _construct_with_real_threading(priority_queue, maxsize)
