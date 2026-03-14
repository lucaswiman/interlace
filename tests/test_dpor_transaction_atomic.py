"""Tests for DPOR counterexample reproduction with transaction.atomic().

Bug 1: OpcodeScheduler doesn't skip scheduling points inside explicit SQL
transactions, so counterexample replay fails (schedule exhausted too early).

Bug 2: _process_opcode uses getattr(obj, attr) for LOAD_ATTR, which triggers
property descriptors. Properties that access the DB cause recursive lock deadlock.
"""

from __future__ import annotations

from typing import Any

import pytest

from frontrun._io_detection import _io_tls
from frontrun.bytecode import BytecodeShuffler, OpcodeScheduler

# ---------------------------------------------------------------------------
# Bug 1: OpcodeScheduler must respect _in_transaction
# ---------------------------------------------------------------------------


class TestOpcodeSchedulerTransactionAwareness:
    """OpcodeScheduler.wait_for_turn must skip scheduling points inside
    explicit SQL transactions, matching DporScheduler.report_and_wait behavior."""

    def test_transaction_opcodes_not_counted_as_schedule_steps(self) -> None:
        """When _in_transaction is True and _is_autobegin is False,
        OpcodeScheduler should NOT consume a schedule entry.

        This directly reproduces the mismatch: DPOR generates a schedule
        with N entries (skipping transaction body), but OpcodeScheduler
        consumes entries for ALL opcodes including transaction body,
        exhausting the schedule prematurely.
        """
        ops_outside_tx: list[str] = []
        ops_inside_tx: list[str] = []

        def thread_fn() -> None:
            # Phase 1: outside transaction — schedule entries should be consumed
            x = 1  # noqa: F841
            y = 2  # noqa: F841

            # Phase 2: inside explicit transaction — should NOT consume entries
            _io_tls._in_transaction = True
            _io_tls._is_autobegin = False
            a = 3  # noqa: F841
            b = 4  # noqa: F841
            _io_tls._in_transaction = False

            # Phase 3: back outside transaction — should consume entries again
            z = 5  # noqa: F841

        # Create a short schedule — if transaction opcodes are NOT skipped,
        # this schedule will be exhausted before the thread finishes and
        # _extend_schedule will kick in with round-robin.
        # If they ARE skipped, fewer entries are needed.
        schedule = [0] * 500  # generous schedule for a single thread
        scheduler = OpcodeScheduler(schedule, num_threads=1, max_ops=5000)
        runner = BytecodeShuffler(scheduler, detect_io=False)

        runner.run([thread_fn], timeout=5.0)

        # The key assertion: with transaction awareness, fewer schedule
        # entries should be consumed because transaction body opcodes
        # don't count. Without the fix, _index would be higher.
        consumed_with_awareness = scheduler._index

        # Now run without the transaction flag to get baseline
        scheduler2 = OpcodeScheduler([0] * 500, num_threads=1, max_ops=5000)
        runner2 = BytecodeShuffler(scheduler2, detect_io=False)

        def thread_fn_no_tx() -> None:
            x = 1  # noqa: F841
            y = 2  # noqa: F841
            # Same operations, but NOT inside a transaction
            a = 3  # noqa: F841
            b = 4  # noqa: F841
            z = 5  # noqa: F841

        runner2.run([thread_fn_no_tx], timeout=5.0)
        consumed_without_awareness = scheduler2._index

        # With transaction awareness, opcodes inside the transaction should
        # NOT be counted, so consumed_with_awareness < consumed_without_awareness.
        # The tx flag setting/clearing itself takes some opcodes, but the body
        # operations (a=3, b=4) should be skipped.
        assert consumed_with_awareness < consumed_without_awareness, (
            f"OpcodeScheduler consumed {consumed_with_awareness} entries with "
            f"transaction active, but {consumed_without_awareness} without. "
            f"Expected fewer entries when inside a transaction because those "
            f"opcodes should be skipped (matching DporScheduler behavior)."
        )


# ---------------------------------------------------------------------------
# Bug 2: Property access in _process_opcode causes recursive lock deadlock
# ---------------------------------------------------------------------------


class TestPropertyAccessDeadlock:
    """_process_opcode must not trigger property descriptors via getattr().

    When LOAD_ATTR encounters a property whose getter does I/O (e.g. a
    Django model property that queries the DB), getattr() triggers the
    getter. If the getter tries to acquire the scheduler lock (via
    cursor.execute → _intercept_execute → report_and_wait), we get a
    recursive lock deadlock because _process_opcode is already called
    from within the locked section of report_and_wait.
    """

    def test_load_attr_uses_safe_getattr(self) -> None:
        """Verify _process_opcode uses _safe_getattr instead of bare
        getattr for LOAD_ATTR, avoiding property descriptor triggers."""
        import inspect

        from frontrun.dpor import _process_opcode

        source = inspect.getsource(_process_opcode)

        # Find the LOAD_ATTR section
        load_attr_idx = source.index("LOAD_ATTR")
        # Find the next elif/else to bound the section
        next_section = source.index("elif op ==", load_attr_idx + 1)
        load_attr_section = source[load_attr_idx:next_section]

        # The LOAD_ATTR section must use _safe_getattr (which bypasses
        # property descriptors) instead of bare getattr(obj, attr).
        assert "_safe_getattr" in load_attr_section, (
            "_process_opcode LOAD_ATTR handler does not use _safe_getattr. "
            "Bare getattr(obj, attr) triggers property descriptors, causing "
            "recursive lock deadlock when a property does DB access."
        )

    def test_property_not_triggered_during_dpor(self) -> None:
        """End-to-end: accessing an object with a property must not trigger
        the property getter during DPOR shadow-stack tracking."""
        call_count = [0]

        class ModelWithDBProperty:
            def __init__(self) -> None:
                self.name = "test"

            @property
            def expensive(self) -> str:
                call_count[0] += 1
                return "computed"

        obj = ModelWithDBProperty()

        def thread_fn(_state: Any) -> None:
            # Access the property — the opcode tracer will see LOAD_ATTR
            # for 'expensive' and push a value onto the shadow stack.
            # With the fix, it uses object.__getattribute__ which returns
            # the property descriptor object, NOT triggering the getter.
            _ = obj.expensive  # noqa: F841

        # Run under DPOR to trigger _process_opcode
        from frontrun.dpor import explore_dpor

        call_count[0] = 0

        result = explore_dpor(
            setup=lambda: None,
            threads=[thread_fn],
            invariant=lambda _: True,
            max_executions=1,
            reproduce_on_failure=0,
        )

        # The getter WILL be called by the actual thread execution (that's
        # correct — the thread does `_ = obj.expensive`). But it should
        # NOT be called an EXTRA time by _process_opcode's shadow stack.
        # With the fix, the property getter is called exactly once (by the
        # thread). Without the fix, _process_opcode also calls it via
        # getattr, so count would be > 1.
        assert result.num_explored >= 1
        assert call_count[0] == 1, (
            f"Property getter was called {call_count[0]} times, expected exactly 1. "
            f"_process_opcode is likely triggering the property getter via getattr()."
        )

    def test_load_method_uses_safe_getattr(self) -> None:
        """LOAD_METHOD handler (Python 3.10) should also use _safe_getattr."""
        import inspect

        from frontrun.dpor import _process_opcode

        source = inspect.getsource(_process_opcode)

        # Find the LOAD_METHOD section
        if "LOAD_METHOD" not in source:
            pytest.skip("LOAD_METHOD not in _process_opcode (Python 3.11+)")

        load_method_idx = source.index('"LOAD_METHOD"')
        next_section = source.index("elif op ==", load_method_idx + 1)
        load_method_section = source[load_method_idx:next_section]

        assert "_safe_getattr" in load_method_section, (
            "_process_opcode LOAD_METHOD handler does not use _safe_getattr. "
            "Bare getattr(obj, attr) triggers property descriptors, causing "
            "recursive lock deadlock when a property does DB access."
        )
