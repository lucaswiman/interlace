"""Defect #11: GC destructor deadlock with CooperativeRLock.

CooperativeRLock.release() is missing the _in_dpor_machinery() reentrancy
guard that CooperativeLock.release() has.  When GC fires during
_report_and_wait (which holds scheduler._condition and sets
_in_dpor_machinery=True), redis.Redis.__del__ can trigger
connection_pool.disconnect() which uses an RLock.  The RLock.release()
calls _report("lock_release") which tries to acquire scheduler._condition,
causing a deadlock.

Fix: add the same guard to CooperativeRLock.release().
"""

from __future__ import annotations

from frontrun._cooperative import (
    CooperativeRLock,
    _scheduler_tls,
    set_context,
    set_sync_reporter,
)
from frontrun._deadlock import install_wait_for_graph, uninstall_wait_for_graph


class TestRLockGCReentrancyGuard:
    """CooperativeRLock must not deadlock when released during DPOR machinery."""

    def test_rlock_release_inside_dpor_machinery(self) -> None:
        """Releasing a CooperativeRLock while _in_dpor_machinery should not deadlock.

        Simulates: _report_and_wait holds scheduler._condition →
        GC fires → redis.__del__ → connection_pool.disconnect() →
        with self._lock (CooperativeRLock) → __exit__ → release() →
        _report("lock_release") → reporter() → with scheduler._condition
        → DEADLOCK.
        """
        rlock = CooperativeRLock()
        released_ok = False
        reporter_called = False

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
            set_context(scheduler, 0)  # type: ignore[arg-type]

            def reporter(event: str, obj_id: int) -> None:
                nonlocal reporter_called
                reporter_called = True

            set_sync_reporter(reporter)

            # Acquire the RLock normally
            rlock.acquire()

            # Simulate being inside DPOR machinery (as if _report_and_wait
            # holds scheduler._condition)
            _scheduler_tls._in_dpor_machinery = True

            # Release should NOT call _report() since we're in DPOR machinery.
            # Without the fix, this deadlocks (or in this test, calls reporter
            # which would try to acquire scheduler._condition).
            rlock.release()
            released_ok = True
        finally:
            _scheduler_tls._in_dpor_machinery = False
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            uninstall_wait_for_graph()

        assert released_ok, "RLock release deadlocked inside DPOR machinery"
        # The guard should skip _report() entirely
        assert not reporter_called, (
            "CooperativeRLock.release() should skip _report() when _in_dpor_machinery() "
            "is True, but reporter was called"
        )

    def test_rlock_release_normal_reports(self) -> None:
        """CooperativeRLock.release() should still call _report() normally."""
        rlock = CooperativeRLock()
        events: list[str] = []

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
            set_context(scheduler, 0)  # type: ignore[arg-type]

            def reporter(event: str, obj_id: int) -> None:
                events.append(event)

            set_sync_reporter(reporter)

            rlock.acquire()
            rlock.release()
        finally:
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            uninstall_wait_for_graph()

        assert "lock_release" in events, f"Expected lock_release event, got {events}"

    def test_rlock_reentrant_release_inner_during_machinery(self) -> None:
        """Reentrant RLock: inner release during machinery should not deadlock.

        Simulates nested acquisition where the outer release happens normally
        but the inner release happens during DPOR machinery.
        """
        rlock = CooperativeRLock()

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
            set_context(scheduler, 0)  # type: ignore[arg-type]
            set_sync_reporter(lambda event, obj_id: None)

            # Acquire twice (reentrant)
            rlock.acquire()
            rlock.acquire()

            # Release inner (just decrements count, doesn't release underlying lock)
            rlock.release()

            # Now simulate DPOR machinery
            _scheduler_tls._in_dpor_machinery = True

            # Release outer — this is the final release that touches the real lock
            rlock.release()
        finally:
            _scheduler_tls._in_dpor_machinery = False
            set_sync_reporter(None)
            set_context(None, None)  # type: ignore[arg-type]
            uninstall_wait_for_graph()

        # Should complete without deadlock
        assert not rlock._is_owned()
