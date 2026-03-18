"""Defect #11: DPOR cannot detect races between two separate lock objects.

Problem statement
-----------------

DPOR detects data races by finding **conflicts** — two threads accessing the
*same* shared object where at least one access is a write.  It then explores
alternate interleavings around those conflict points.  This works perfectly
for single-object races (lost updates, TOCTOU on one variable, etc.).

However, a common real-world race pattern involves a **compound operation**
that updates two separately-locked values in sequence, where a concurrent
reader observes the values *between* the two updates and sees an inconsistent
snapshot:

    # Thread A (writer — "observe"):
    with count_lock:                 # Lock 1
        count._value += 1
    # << race window: count incremented, sum not yet >>
    with sum_lock:                   # Lock 2
        sum._value += amount

    # Thread B (reader — "collect"):
    with count_lock:                 # Lock 1
        c = count._value             # sees count incremented
    with sum_lock:                   # Lock 2
        s = sum._value               # but sum not yet incremented → c > s

This is the pattern used by ``prometheus_client.Summary.observe()``.  A
stress test (100k iterations, 2 threads) confirms ``count > sum`` is
observable within ~1000 iterations.

Why DPOR misses this
--------------------

DPOR sees the following per-object conflicts:

  - ``count._value``: Thread A writes, Thread B reads → conflict ✓
  - ``sum._value``:   Thread A writes, Thread B reads → conflict ✓

Both objects have valid conflicts.  DPOR explores interleavings that
reorder Thread A's access to ``count._value`` relative to Thread B's
access to ``count._value``, and similarly for ``sum._value``.

But the **cross-object invariant** (``count <= sum`` after both threads
finish) requires DPOR to reason about the *gap between the release of
Lock 1 and the acquisition of Lock 2* in Thread A.  Thread B must
interleave in this exact gap — *after* Thread A releases ``count_lock``
but *before* Thread A acquires ``sum_lock``.

DPOR's partial-order reduction explores interleavings at conflict points
on *individual objects*.  The gap between two lock releases on *different*
objects is not a conflict point for either object individually, so DPOR
never inserts a scheduling decision there.  The race is invisible to
single-object conflict analysis.

What would be needed to fix this
--------------------------------

Detecting multi-lock atomicity violations requires one of:

1. **Lockset analysis** layered on top of DPOR — track which locks each
   thread holds at each scheduling point and detect "lock-set gaps" where
   a compound operation releases one lock before acquiring another.

2. **Atomicity inference** — recognize that ``count`` and ``sum`` are
   semantically related (e.g., they appear in the same invariant) and
   treat accesses to both as a single compound conflict.

3. **Exhaustive preemption** at lock boundaries — force a scheduling point
   at every lock release, not just at conflict points.  This is sound but
   exponentially expensive (it defeats the purpose of partial-order
   reduction).

All three are fundamental extensions to the DPOR algorithm, not code fixes.

Running
-------
::

    make test-3.14 PYTEST_ARGS="-v -k test_defect11_multi"
"""

from __future__ import annotations

import threading

import pytest

from frontrun.dpor import explore_dpor


class TestMultiLockRaces:
    """Defect #11: DPOR misses races that span two separately-locked objects.

    These tests demonstrate real race conditions (confirmed by stress testing)
    that DPOR's single-object conflict model cannot detect.  They are marked
    ``xfail`` because this is a known fundamental limitation.
    """

    @pytest.mark.xfail(
        reason=(
            "Defect #11: DPOR's single-object conflict model cannot detect "
            "races in compound operations that span two separate locks.  The "
            "race window exists between the release of lock 1 and the "
            "acquisition of lock 2, which is not a conflict point for either "
            "object individually."
        ),
        strict=True,
    )
    def test_prometheus_summary_observe_pattern(self) -> None:
        """Two separately-locked counters updated non-atomically.

        Mimics prometheus_client's Summary.observe():
          - Thread A increments count (lock 1), then increments sum (lock 2)
          - Thread B reads count (lock 1), then reads sum (lock 2)

        The race: Thread B reads count AFTER Thread A increments it, but
        reads sum BEFORE Thread A increments it → sees count > sum.

        DPOR should report property_holds=False (the race is real and
        confirmed by stress testing), but it reports True because it
        never explores the interleaving where Thread B runs between
        Thread A's two locked sections.
        """

        class State:
            def __init__(self) -> None:
                self.count_lock = threading.Lock()
                self.sum_lock = threading.Lock()
                self.count = 0
                self.total = 0
                self.observed_count: int | None = None
                self.observed_total: int | None = None

        def writer(state: State) -> None:
            # Increment count and sum in two separately-locked sections.
            # The race window is between these two blocks.
            with state.count_lock:
                state.count += 1
            # << Thread B can interleave here and see count=1, total=0 >>
            with state.sum_lock:
                state.total += 10

        def reader(state: State) -> None:
            # Read both values — if we interleave in the gap, we see
            # count=1 but total=0.
            with state.count_lock:
                state.observed_count = state.count
            with state.sum_lock:
                state.observed_total = state.total

        def invariant(state: State) -> bool:
            # If reader ran, count and total should be consistent:
            # either both 0 (reader ran first) or count=1 and total=10
            # (writer finished first).  count=1 and total=0 means the
            # reader interleaved between the two locked sections.
            if state.observed_count is None:
                return True  # reader didn't run yet
            if state.observed_count == 0:
                return True  # reader ran before writer
            # count > 0: writer's first lock section ran.
            # total should also be updated.
            return state.observed_total == state.observed_count * 10

        result = explore_dpor(
            setup=State,
            threads=[writer, reader],
            invariant=invariant,
            detect_io=False,
            max_executions=200,
            preemption_bound=None,  # Full exploration — no bound
            deadlock_timeout=10.0,
        )

        assert not result.property_holds, (
            f"DPOR should detect the multi-lock race but missed it.  "
            f"Explored {result.interleavings_explored} interleavings.  "
            f"The race exists: a reader can see count=1, total=0 when it "
            f"interleaves between the writer's two locked sections."
        )

    @pytest.mark.xfail(
        reason=(
            "Defect #11: DPOR's single-object conflict model cannot detect "
            "races in compound operations that update two objects under "
            "separate locks.  Both objects have valid conflicts individually, "
            "but the cross-object inconsistency window is invisible to DPOR."
        ),
        strict=True,
    )
    def test_transfer_between_two_locked_accounts(self) -> None:
        """Transfer between two separately-locked accounts is non-atomic.

        A common pattern in financial systems: debit one account (lock 1),
        then credit another (lock 2).  A concurrent reader can observe
        the intermediate state where money has been debited but not yet
        credited — the total across both accounts is temporarily wrong.
        """

        class State:
            def __init__(self) -> None:
                self.lock_a = threading.Lock()
                self.lock_b = threading.Lock()
                self.balance_a = 100
                self.balance_b = 100
                self.observed_total: int | None = None

        def transfer(state: State) -> None:
            # Debit A, then credit B — NOT atomic across both accounts.
            with state.lock_a:
                state.balance_a -= 50
            # << race window: A debited, B not yet credited >>
            with state.lock_b:
                state.balance_b += 50

        def auditor(state: State) -> None:
            # Read both balances — should always sum to 200.
            with state.lock_a:
                a = state.balance_a
            with state.lock_b:
                b = state.balance_b
            state.observed_total = a + b

        def invariant(state: State) -> bool:
            if state.observed_total is None:
                return True
            # Total should always be 200 (conservation of money).
            return state.observed_total == 200

        result = explore_dpor(
            setup=State,
            threads=[transfer, auditor],
            invariant=invariant,
            detect_io=False,
            max_executions=200,
            preemption_bound=None,
            deadlock_timeout=10.0,
        )

        assert not result.property_holds, (
            f"DPOR should detect the transfer atomicity violation but missed "
            f"it.  Explored {result.interleavings_explored} interleavings.  "
            f"The auditor can observe balance_a=50, balance_b=100 (total=150) "
            f"when it interleaves between the debit and credit."
        )
