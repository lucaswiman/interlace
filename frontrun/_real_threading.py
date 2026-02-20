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
