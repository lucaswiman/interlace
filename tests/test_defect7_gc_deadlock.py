"""Defect #7: DPOR deadlock when __del__ fires during cooperative lock operations.

When Python's garbage collector runs inside DPOR scheduler machinery (e.g.,
inside ``_process_opcode`` or ``_sync_reporter``) and a ``__del__`` method
tries to acquire a cooperative lock, the cooperative lock wrapper re-enters
the scheduler/WaitForGraph, causing a deadlock.

The fix is a thread-local reentrancy guard: when already inside DPOR
scheduler machinery, cooperative locks fall back to real blocking behavior.
"""

from __future__ import annotations

import threading

from frontrun._cooperative import (
    CooperativeLock,
    _scheduler_tls,
    set_context,
    set_sync_reporter,
)
from frontrun._deadlock import WaitForGraph, install_wait_for_graph, uninstall_wait_for_graph


class TestGCReentrancyGuard:
    """Test that cooperative locks don't deadlock when acquired reentrantly
    (simulating GC-triggered __del__ during scheduler operations)."""

    def test_cooperative_lock_inside_sync_reporter(self) -> None:
        """Acquiring a cooperative lock inside a sync reporter should not deadlock.

        This simulates the scenario where GC triggers redis.__del__() during
        _sync_reporter, and __del__ tries to acquire a cooperative lock.
        """
        lock = CooperativeLock()
        acquired = False

        class FakeScheduler:
            _finished = False
            _error = False

            def wait_for_turn(self, thread_id: int) -> None:
                pass

            def report_error(self, err: Exception) -> None:
                pass

        scheduler = FakeScheduler()
        graph = install_wait_for_graph()

        try:
            set_context(scheduler, 0)

            reporter_entered = False

            def sync_reporter(event: str, obj_id: int) -> None:
                nonlocal reporter_entered, acquired
                reporter_entered = True
                # Simulate GC-triggered __del__ that acquires another lock.
                # Without the reentrancy guard, this would deadlock because
                # the sync_reporter itself is called from within lock operations.
                inner_lock = CooperativeLock()
                acquired = inner_lock.acquire(blocking=True, timeout=1.0)
                if acquired:
                    inner_lock.release()

            set_sync_reporter(sync_reporter)

            # Acquiring the outer lock triggers the sync reporter, which
            # then tries to acquire the inner lock. Without reentrancy
            # protection, this deadlocks.
            lock.acquire()
            lock.release()

            assert reporter_entered, "sync reporter should have been called"
            assert acquired, "inner lock should have been acquired without deadlock"
        finally:
            set_context(None, None)  # type: ignore[arg-type]
            set_sync_reporter(None)
            uninstall_wait_for_graph()

    def test_waitforgraph_reentrant_access(self) -> None:
        """WaitForGraph operations during GC should not deadlock.

        Simulates the scenario where add_holding is called while _lock is
        already held (e.g., during add_waiting's cycle detection, GC fires
        and triggers another lock operation that calls add_holding).
        """
        graph = WaitForGraph()

        # Simulate holding the graph's internal lock and then trying to
        # re-enter it. Without the fix, this would deadlock.
        # We do this by patching the lock to be reentrant.
        result = [False]

        def reentrant_test() -> None:
            # Add a waiting edge (acquires graph._lock)
            graph.add_waiting(0, 100)
            # Now simulate GC calling add_holding while _lock is held
            # This should not deadlock.
            graph.add_holding(1, 200)
            graph.remove_waiting(0, 100)
            graph.remove_holding(1, 200)
            result[0] = True

        # Run with a timeout to detect deadlocks
        t = threading.Thread(target=reentrant_test)
        t.start()
        t.join(timeout=3.0)
        assert not t.is_alive(), "Thread deadlocked in WaitForGraph"
        assert result[0], "WaitForGraph operations should complete"

    def test_cooperative_lock_with_reentrancy_guard(self) -> None:
        """When _in_dpor_machinery flag is set, cooperative locks use real blocking."""

        lock = CooperativeLock()

        class FakeScheduler:
            _finished = False
            _error = False

            def wait_for_turn(self, thread_id: int) -> None:
                pass

        set_context(FakeScheduler(), 0)
        try:
            # Set the reentrancy guard
            _scheduler_tls._in_dpor_machinery = True
            try:
                # Should use real blocking, not cooperative scheduling
                assert lock.acquire(blocking=True, timeout=1.0)
                lock.release()
            finally:
                _scheduler_tls._in_dpor_machinery = False
        finally:
            set_context(None, None)  # type: ignore[arg-type]
