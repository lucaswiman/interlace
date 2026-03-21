"""Regression tests for sound I/O deferral while holding locks.

Problem statement
-----------------

Sync DPOR buffers I/O in ``pending_io`` and reports it to the Rust engine at
the next scheduling point. Historically that flush was gated only on
``lock_depth == 0``. That is unsound: if thread A holds an unrelated lock but
thread B is still runnable, deferring A's I/O moves the wakeup-tree insertion
point past a real race window.

The sound condition is scheduler-based, not lock-count-based:

  - If another thread is runnable, pending I/O must flush immediately.
  - Deferral is only safe while the current thread is inside a lock and no
    competing thread can run.
"""

from __future__ import annotations

from frontrun.dpor import DporScheduler, _dpor_tls


class _FakeExecution:
    def __init__(self, runnable: list[int]) -> None:
        self._runnable = list(runnable)

    def runnable_threads(self) -> list[int]:
        return list(self._runnable)


class _FakeEngine:
    def __init__(self) -> None:
        self.io_calls: list[tuple[int, int, str]] = []

    def schedule(self, execution: _FakeExecution) -> int | None:
        runnable = execution.runnable_threads()
        return runnable[0] if runnable else None

    def report_io_access(self, execution: _FakeExecution, thread_id: int, object_id: int, kind: str) -> None:
        self.io_calls.append((thread_id, object_id, kind))


class TestLockDepthIoWindow:
    def teardown_method(self) -> None:
        _dpor_tls.pending_io = []
        _dpor_tls.lock_depth = 0

    def test_flushes_pending_io_inside_lock_when_other_thread_is_runnable(self) -> None:
        engine = _FakeEngine()
        execution = _FakeExecution([0, 1])
        scheduler = DporScheduler(engine, execution, num_threads=2)

        _dpor_tls.pending_io = [(123, "write")]
        _dpor_tls.lock_depth = 1

        assert scheduler._report_and_wait(None, 0)

        assert engine.io_calls == [(0, 123, "write")], (
            "Pending I/O should flush immediately when another thread is runnable, "
            "even if the current thread still holds a lock."
        )
        assert _dpor_tls.pending_io == []

    def test_keeps_pending_io_buffered_while_inside_lock_and_no_other_thread_can_run(self) -> None:
        engine = _FakeEngine()
        execution = _FakeExecution([0])
        scheduler = DporScheduler(engine, execution, num_threads=1)

        _dpor_tls.pending_io = [(456, "read")]
        _dpor_tls.lock_depth = 1

        assert scheduler._report_and_wait(None, 0)

        assert engine.io_calls == [], (
            "Deferral should remain in place when the current thread is inside a "
            "lock and no competing thread is runnable."
        )
        assert _dpor_tls.pending_io == [(456, "read")]
