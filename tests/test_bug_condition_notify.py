"""Test that CooperativeCondition.notify(1) wakes exactly one waiter.

Bug: CooperativeCondition uses a monotonically increasing _notify_count
to track notifications.  notify(n) increments by n, and each waiter spins
until _notify_count exceeds its snapshot.  But when multiple waiters hold
the same snapshot value, notify(1) wakes ALL of them instead of just one.

Example:
  - Waiters A, B, C all record snapshot = 5
  - Producer calls notify(1), bumping _notify_count to 6
  - All three see 6 > 5 and wake up, but only one should

This violates the threading.Condition contract where notify(1) should
wake at most one waiter.  In user code that relies on this (e.g.,
bounded producer-consumer queues), the bug causes spurious wakeups
that can lead to incorrect behavior during concurrency testing.
"""

import threading
import time

from frontrun._cooperative import (
    CooperativeCondition,
    CooperativeLock,
    set_context,
    clear_context,
)


class FakeScheduler:
    """Minimal scheduler stub for testing cooperative primitives."""

    _finished = False
    _error = None

    def wait_for_turn(self, thread_id: int) -> None:
        time.sleep(0.001)


def test_condition_notify_one_wakes_only_one():
    """Verify that notify(1) wakes exactly one waiter, not all.

    We directly test the _notify_count mechanism:
    - Set up 3 "waiters" that all record the same snapshot
    - Call notify(1)
    - Check how many would see the notification

    This is a unit test of the notification counting logic.
    """
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    # Simulate 3 waiters that all recorded the same snapshot
    # In real usage, all three call wait() and see notify_count_before_wait=0.
    # After notify(1), _notify_count becomes 1.
    # All three check: 1 > 0 → True → all wake up (BUG: should be just 1).

    # We can test this directly by checking the counting logic:
    initial_count = cond._notify_count

    # Simulate what 3 concurrent waiters would snapshot:
    snapshots = [initial_count, initial_count, initial_count]

    # Producer calls notify(1) — should wake exactly 1 waiter
    lock.acquire()
    cond.notify(1)
    lock.release()

    new_count = cond._notify_count

    # Count how many waiters would see the notification
    woken = sum(1 for snap in snapshots if new_count > snap)

    assert woken == 1, (
        f"notify(1) should wake exactly 1 waiter out of 3, but {woken} would "
        f"see the notification (snapshots={snapshots}, new_count={new_count}). "
        f"The _notify_count mechanism broadcasts to ALL waiters with the same "
        f"snapshot instead of waking exactly n."
    )


def test_condition_notify_two_wakes_exactly_two():
    """Verify that notify(2) wakes exactly two waiters, not all."""
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    initial_count = cond._notify_count
    snapshots = [initial_count] * 5  # 5 waiters

    lock.acquire()
    cond.notify(2)
    lock.release()

    new_count = cond._notify_count
    woken = sum(1 for snap in snapshots if new_count > snap)

    assert woken == 2, (
        f"notify(2) should wake exactly 2 waiters out of 5, but {woken} would "
        f"see the notification."
    )
