"""Saved references to real threading/queue primitives.

This module captures the original factories at import time, before
``_cooperative.patch_locks()`` monkey-patches the ``threading`` and
``queue`` modules.  Any module that needs a *real* (non-cooperative)
lock — including ``_deadlock.py`` and ``_cooperative.py`` — should
import from here.

This file is a leaf module with no ``frontrun`` imports, so it can
never participate in circular-import chains.
"""

import contextlib
import queue
import threading
from collections.abc import Iterator
from typing import Any, cast

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

# Attribute names on ``threading`` we swap to the real factories while
# constructing primitives.  Keeping this in one place avoids repeating
# the save/restore block across every ``make_*`` helper.
_PATCHED_ATTRS: tuple[tuple[str, Any], ...] = (
    ("Lock", lock),
    ("RLock", rlock),
    ("Semaphore", semaphore),
    ("BoundedSemaphore", bounded_semaphore),
    ("Event", event),
    ("Condition", condition),
)


@contextlib.contextmanager
def _with_real_threading_primitives() -> Iterator[None]:
    """Temporarily restore the real threading factories on ``threading``.

    Yields with ``threading.Lock`` / ``Event`` / etc. pointing at the
    originals captured at import time; restores the previous values on
    exit.  Serialised on ``_factory_lock`` so concurrent callers don't
    race on the module-level attributes.
    """
    with _factory_lock:
        saved = [(name, getattr(threading, name)) for name, _ in _PATCHED_ATTRS]
        for name, real in _PATCHED_ATTRS:
            setattr(threading, name, real)
        try:
            yield
        finally:
            for name, prev in saved:
                setattr(threading, name, prev)


def make_event() -> threading.Event:
    with _with_real_threading_primitives():
        return event()


def make_queue(maxsize: int = 0) -> queue.Queue[Any]:
    with _with_real_threading_primitives():
        return cast(queue.Queue[Any], queue_(maxsize))


def make_lifo_queue(maxsize: int = 0) -> queue.LifoQueue[Any]:
    with _with_real_threading_primitives():
        return cast(queue.LifoQueue[Any], lifo_queue(maxsize))


def make_priority_queue(maxsize: int = 0) -> queue.PriorityQueue[Any]:
    with _with_real_threading_primitives():
        return cast(queue.PriorityQueue[Any], priority_queue(maxsize))
