"""Unit tests for _deadlock.py: DeadlockError, WaitForGraph kind param, format_cycle."""

from __future__ import annotations

import threading
import time
from typing import Any

from frontrun._deadlock import DeadlockError, WaitForGraph, format_cycle


class TestDeadlockError:
    def test_is_exception(self) -> None:
        err = DeadlockError("msg", "T1 -> L1 -> T2")
        assert isinstance(err, Exception)

    def test_not_timeout_error(self) -> None:
        err = DeadlockError("msg", "cycle")
        assert not isinstance(err, TimeoutError)

    def test_cycle_description(self) -> None:
        err = DeadlockError("deadlock!", "thread 0 -> row_lock x -> thread 1")
        assert err.cycle_description == "thread 0 -> row_lock x -> thread 1"
        assert str(err) == "deadlock!"


class TestWaitForGraphKind:
    def test_row_lock_cycle_detected(self) -> None:
        """T0 waits for row_lock 0 held by T1, T1 waits for row_lock 1 held by T0 → cycle."""
        g = WaitForGraph()
        # T0 holds row_lock 0
        g.add_holding(0, 0, kind="row_lock")
        # T1 holds row_lock 1
        g.add_holding(1, 1, kind="row_lock")
        # T0 waits for row_lock 1
        g.add_waiting(0, 1, kind="row_lock")
        # T1 waits for row_lock 0 → cycle
        cycle = g.add_waiting(1, 0, kind="row_lock")
        assert cycle is not None

    def test_no_cycle_non_overlapping(self) -> None:
        g = WaitForGraph()
        g.add_holding(0, 0, kind="row_lock")
        g.add_holding(1, 1, kind="row_lock")
        # T0 waits for row_lock 1, but T1 doesn't wait for anything
        cycle = g.add_waiting(0, 1, kind="row_lock")
        assert cycle is None

    def test_row_lock_and_lock_nodes_disjoint(self) -> None:
        """row_lock and lock nodes with same integer ID don't collide."""
        g = WaitForGraph()
        # ("lock", 0) and ("row_lock", 0) are different nodes
        g.add_holding(0, 0, kind="lock")
        g.add_holding(1, 0, kind="row_lock")
        # T0 waits for row_lock 0 → T1 holds row_lock 0 → T1 doesn't wait → no cycle
        cycle = g.add_waiting(0, 0, kind="row_lock")
        assert cycle is None

    def test_remove_waiting_row_lock(self) -> None:
        g = WaitForGraph()
        g.add_holding(0, 0, kind="row_lock")
        g.add_holding(1, 1, kind="row_lock")
        g.add_waiting(0, 1, kind="row_lock")
        g.remove_waiting(0, 1, kind="row_lock")
        # Now T1 waits for row_lock 0 — no cycle because T0 isn't waiting
        cycle = g.add_waiting(1, 0, kind="row_lock")
        assert cycle is None

    def test_remove_holding_row_lock(self) -> None:
        g = WaitForGraph()
        g.add_holding(0, 0, kind="row_lock")
        g.add_holding(1, 1, kind="row_lock")
        g.add_waiting(0, 1, kind="row_lock")
        g.remove_holding(1, 1, kind="row_lock")
        # T1 no longer holds row_lock 1 → T0 is waiting but the edge T1→T0 is gone
        # Now T1 waits for row_lock 0: T1 → row_lock(0) → T0 → row_lock(1)
        # But row_lock(1) has no outgoing edge anymore → no cycle
        cycle = g.add_waiting(1, 0, kind="row_lock")
        assert cycle is None

    def test_default_kind_lock_backward_compatible(self) -> None:
        """Default kind='lock' keeps existing cooperative lock callers working."""
        g = WaitForGraph()
        g.add_holding(0, 100)
        g.add_holding(1, 200)
        g.add_waiting(0, 200)
        cycle = g.add_waiting(1, 100)
        assert cycle is not None  # deadlock


class TestFormatCycleWithLockNames:
    def test_row_lock_with_names(self) -> None:
        cycle = [("thread", 0), ("row_lock", 7), ("thread", 1), ("row_lock", 3)]
        names = {7: "sql:users:(('id',42))", 3: "sql:orders:(('id',1))"}
        result = format_cycle(cycle, lock_names=names)
        assert "sql:users:(('id',42))" in result
        assert "sql:orders:(('id',1))" in result
        assert "thread 0" in result
        assert "thread 1" in result

    def test_row_lock_without_names_falls_back(self) -> None:
        cycle = [("thread", 0), ("row_lock", 7)]
        result = format_cycle(cycle)
        assert "0x7" in result

    def test_lock_kind(self) -> None:
        cycle = [("thread", 0), ("lock", 0xFF)]
        result = format_cycle(cycle)
        assert "lock 0xff" in result

    def test_thread_kind(self) -> None:
        cycle = [("thread", 3)]
        result = format_cycle(cycle)
        assert "thread 3" in result


class TestCooperativeLockDeadlock:
    """Cooperative lock deadlocks should raise DeadlockError, not TimeoutError."""

    def _setup(self) -> tuple[Any, Any]:
        """Install graph + scheduler mock, return (graph, fake_scheduler)."""
        from frontrun._deadlock import install_wait_for_graph

        graph = install_wait_for_graph()

        class FakeScheduler:
            _error: Exception | None = None
            _finished: bool = False

            def report_error(self, err: Exception) -> None:
                self._error = err

            def wait_for_turn(self, tid: int) -> None:
                pass

        return graph, FakeScheduler()

    def _teardown(self) -> None:
        from frontrun._deadlock import uninstall_wait_for_graph

        uninstall_wait_for_graph()

    def test_cooperative_lock_deadlock_raises_deadlock_error(self) -> None:
        """CooperativeLock: cycle detection raises DeadlockError not TimeoutError."""
        from frontrun._cooperative import CooperativeLock, set_context
        from frontrun._deadlock import DeadlockError, SchedulerAbort

        graph, sched = self._setup()
        try:
            lock_a = CooperativeLock()
            lock_b = CooperativeLock()

            caught: list[Exception] = []

            def t0() -> None:
                set_context(sched, 0)
                lock_a.acquire()
                # signal t1 that we hold lock_a — yield so t1 runs
                time.sleep(0.05)
                try:
                    lock_b.acquire()
                except (SchedulerAbort, DeadlockError):
                    pass
                except Exception as e:
                    caught.append(e)

            def t1() -> None:
                set_context(sched, 1)
                time.sleep(0.02)
                lock_b.acquire()
                try:
                    lock_a.acquire()  # should detect cycle → DeadlockError
                except (SchedulerAbort, DeadlockError):
                    pass
                except Exception as e:
                    caught.append(e)

            ta = threading.Thread(target=t0)
            tb = threading.Thread(target=t1)
            ta.start()
            tb.start()
            ta.join(timeout=3.0)
            tb.join(timeout=3.0)

            assert isinstance(sched._error, DeadlockError), (
                f"Expected DeadlockError on scheduler, got {sched._error!r}"
            )
            assert not isinstance(sched._error, TimeoutError)
        finally:
            self._teardown()

    def test_cooperative_rlock_deadlock_raises_deadlock_error(self) -> None:
        """CooperativeRLock: cycle detection raises DeadlockError not TimeoutError."""
        from frontrun._cooperative import CooperativeRLock, set_context
        from frontrun._deadlock import DeadlockError, SchedulerAbort

        _, sched = self._setup()
        try:
            lock_a = CooperativeRLock()
            lock_b = CooperativeRLock()

            def t0() -> None:
                set_context(sched, 0)
                lock_a.acquire()
                time.sleep(0.05)
                try:
                    lock_b.acquire()
                except (SchedulerAbort, DeadlockError):
                    pass

            def t1() -> None:
                set_context(sched, 1)
                time.sleep(0.02)
                lock_b.acquire()
                try:
                    lock_a.acquire()  # should detect cycle
                except (SchedulerAbort, DeadlockError):
                    pass

            ta = threading.Thread(target=t0)
            tb = threading.Thread(target=t1)
            ta.start()
            tb.start()
            ta.join(timeout=3.0)
            tb.join(timeout=3.0)

            assert isinstance(sched._error, DeadlockError), (
                f"Expected DeadlockError on scheduler, got {sched._error!r}"
            )
            assert not isinstance(sched._error, TimeoutError)
        finally:
            self._teardown()
