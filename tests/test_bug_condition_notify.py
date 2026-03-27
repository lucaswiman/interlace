"""Test that CooperativeCondition.notify(1) wakes exactly one waiter.

Bug: CooperativeCondition used a monotonically increasing _notify_count
to track notifications.  notify(n) incremented by n, and each waiter spun
until _notify_count exceeded its snapshot.  But when multiple waiters held
the same snapshot value, notify(1) woke ALL of them instead of just one.

Example:
  - Waiters A, B, C all record snapshot = 5
  - Producer calls notify(1), bumping _notify_count to 6
  - All three see 6 > 5 and wake up, but only one should

This violates the threading.Condition contract where notify(1) should
wake at most one waiter.  In user code that relies on this (e.g.,
bounded producer-consumer queues), the bug causes spurious wakeups
that can lead to incorrect behavior during concurrency testing.

Fix: Use a ticket-based system where each waiter gets a unique sequential
ticket. notify(n) advances a served counter by n.  A waiter wakes only
when its ticket < served.
"""

from frontrun._cooperative import (
    CooperativeCondition,
    CooperativeLock,
)


def test_condition_notify_one_wakes_only_one():
    """Verify that notify(1) wakes exactly one waiter, not all.

    We simulate the ticket/served mechanism:
    - 3 waiters take tickets 0, 1, 2 (_next_ticket becomes 3)
    - notify(1) advances _served to 1
    - Only ticket 0 satisfies ticket < served; tickets 1, 2 do not.
    """
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()

    # Simulate 3 waiters taking tickets while holding the lock.
    # In real usage, each wait() call takes a ticket before releasing.
    cond._waiters = 3
    cond._next_ticket = 3  # tickets 0, 1, 2 assigned
    waiter_tickets = [0, 1, 2]

    # Producer calls notify(1)
    cond.notify(1)

    served = cond._served
    lock.release()

    # Count how many waiters would wake: ticket < served
    woken = sum(1 for t in waiter_tickets if t < served)

    assert woken == 1, (
        f"notify(1) should wake exactly 1 waiter out of 3, but {woken} would "
        f"see the notification (tickets={waiter_tickets}, served={served})."
    )


def test_condition_notify_two_wakes_exactly_two():
    """Verify that notify(2) wakes exactly two waiters, not all."""
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()

    cond._waiters = 5
    cond._next_ticket = 5  # tickets 0..4
    waiter_tickets = list(range(5))

    cond.notify(2)

    served = cond._served
    lock.release()

    woken = sum(1 for t in waiter_tickets if t < served)

    assert woken == 2, (
        f"notify(2) should wake exactly 2 waiters out of 5, but {woken} would "
        f"see the notification (tickets={waiter_tickets}, served={served})."
    )


def test_condition_notify_all_wakes_all():
    """Verify that notify_all() wakes all waiters."""
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()

    cond._waiters = 4
    cond._next_ticket = 4
    waiter_tickets = list(range(4))

    cond.notify_all()

    served = cond._served
    lock.release()

    woken = sum(1 for t in waiter_tickets if t < served)

    assert woken == 4, f"notify_all() should wake all 4 waiters, but {woken} would see the notification."


def test_condition_notify_more_than_waiters():
    """Verify that notify(n) where n > waiters doesn't over-serve."""
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()

    cond._waiters = 2
    cond._next_ticket = 2
    waiter_tickets = [0, 1]

    # notify(5) but only 2 waiters exist
    cond.notify(5)

    served = cond._served
    lock.release()

    woken = sum(1 for t in waiter_tickets if t < served)

    assert woken == 2, f"notify(5) with only 2 waiters should wake 2, not {woken}. served={served} should be 2, not 5."
    assert served == 2, f"served should be capped at number of waiters (2), got {served}"


def test_condition_sequential_notify_accumulates():
    """Verify that multiple notify(1) calls accumulate correctly."""
    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()

    cond._waiters = 3
    cond._next_ticket = 3
    waiter_tickets = [0, 1, 2]

    cond.notify(1)  # serves ticket 0
    cond.notify(1)  # serves ticket 1

    served = cond._served
    lock.release()

    woken = sum(1 for t in waiter_tickets if t < served)

    assert woken == 2, f"Two notify(1) calls should wake 2 waiters, but {woken} would wake."
