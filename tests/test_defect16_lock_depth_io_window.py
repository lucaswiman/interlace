"""Regression test for lock_depth deferring I/O past a real race window.

Problem statement
-----------------

`_dpor_tls.lock_depth` currently tracks only how many locks a thread holds.
When it is non-zero, sync DPOR buffers I/O reports in `pending_io` and does
not flush them to the Rust engine until the thread exits *all* locks.

That is safe only if every runnable competitor is blocked by one of those
locks. If thread A holds an unrelated lock `guard`, performs I/O on a shared
resource, and then performs more I/O before releasing `guard`, thread B can
still legally race on that shared resource even though it never touches
`guard`.

This test models exactly that shape:

  - Thread A: with guard: append "A1", append "A2"
  - Thread B: append "B" without taking guard

The interesting schedule is `A1, B, A2`.  It is a real execution because B is
not blocked on `guard`.  If DPOR defers A's first I/O report until after A
releases `guard`, the backtrack point moves too late and the `A1, B, A2`
ordering is missed.
"""

from __future__ import annotations

import threading
from pathlib import Path

from frontrun.dpor import explore_dpor


class TestLockDepthIoWindow:
    """DPOR must not hide races behind unrelated lock nesting depth."""

    def test_unrelated_lock_does_not_hide_io_interleaving(self, tmp_path: Path) -> None:
        log_path = tmp_path / "io-window.log"

        class State:
            def __init__(self) -> None:
                self.guard = threading.Lock()
                self.log_path = log_path

        def append_token(path: Path, token: str) -> None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(token)

        def thread_a(state: State) -> None:
            with state.guard:
                append_token(state.log_path, "A1\n")
                append_token(state.log_path, "A2\n")

        def thread_b(state: State) -> None:
            append_token(state.log_path, "B\n")

        def invariant(_state: State) -> bool:
            contents = log_path.read_text(encoding="utf-8")
            return contents != "A1\nB\nA2\n"

        result = explore_dpor(
            setup=State,
            threads=[thread_a, thread_b],
            invariant=invariant,
            detect_io=True,
            max_executions=100,
            preemption_bound=None,
            deadlock_timeout=5.0,
            stop_on_first=True,
            reproduce_on_failure=0,
        )

        assert not result.property_holds, (
            "DPOR should detect the real `A1, B, A2` file-I/O interleaving even "
            "though thread A holds an unrelated lock. If this stays green, "
            "lock_depth is still deferring A's I/O reports past the race window."
        )
