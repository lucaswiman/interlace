"""Tests for two bugs found via code review.

Bug 1: _trace_format._get_instruction uses id(code) as cache key, which can
       return stale results when a code object is GC'd and a new one reuses its
       address.  The same bug was already fixed in dpor.py (see _INSTR_CACHE
       comment on line 140-147).

Bug 2: async_trace_markers.py uses a processed_locations set that prevents
       markers from firing more than once per (filename, lineno).  The sync
       variant has no such deduplication, so markers in loops work in sync
       mode but silently skip in async mode.
"""

from __future__ import annotations

import dis
import gc
import types

from frontrun._trace_format import _get_instruction, _instr_cache
from frontrun.async_trace_markers import AsyncTraceExecutor
from frontrun.common import Schedule, Step


class TestStaleInstructionCacheId:
    """Bug 1: id(code) reuse after GC causes stale cache entries."""

    def test_id_reuse_returns_wrong_instruction(self) -> None:
        """Create a code object, cache its instructions, delete it, create
        another code object that reuses the same id, and show the cache
        returns stale (wrong) instructions for the new code object."""
        _instr_cache.clear()

        # Create first code object and cache it.
        # Use two different programs whose bytecode differs at a known offset.
        # "x = 1" has STORE_NAME('x') and "y = 2" has STORE_NAME('y') at offset 4.
        code1 = compile("x = 1", "<test1>", "exec")
        # Find an offset where the two programs differ (skip RESUME at offset 0)
        code2_ref = compile("y = 2", "<test2>", "exec")
        instr1_map = {i.offset: i for i in dis.get_instructions(code1)}
        instr2_map = {i.offset: i for i in dis.get_instructions(code2_ref)}
        differing_offset = None
        for off in sorted(instr1_map):
            if off in instr2_map and instr1_map[off].argval != instr2_map[off].argval:
                differing_offset = off
                break
        assert differing_offset is not None, "Test programs must differ at some offset"
        del code2_ref

        result1 = _get_instruction(code1, differing_offset)
        assert result1 is not None

        code1_id = id(code1)

        # Delete code1 so its address can be reused
        del code1
        gc.collect()

        # Create code objects until one reuses the same id
        attempts = 0
        code2 = None
        stash: list[types.CodeType] = []
        while attempts < 100_000:
            candidate = compile("y = 2", "<test2>", "exec")
            if id(candidate) == code1_id:
                code2 = candidate
                break
            stash.append(candidate)
            attempts += 1

        if code2 is None:
            # Can't reproduce id reuse -- skip rather than fail
            import pytest

            pytest.skip("Could not reproduce id(code) reuse after 100k attempts")

        # Now _get_instruction should return instructions for code2,
        # but with the id(code) bug it returns stale code1 instructions.
        result2 = _get_instruction(code2, differing_offset)
        assert result2 is not None

        # Build the correct mapping for code2 to compare.
        correct_mapping: dict[int, dis.Instruction] = {}
        for inst in dis.get_instructions(code2):
            correct_mapping[inst.offset] = inst
        correct = correct_mapping.get(differing_offset)

        # With the bug: result2 will be code1's instruction (stale).
        # After the fix: result2 should match code2's correct instruction.
        assert correct is not None
        assert result2.argval == correct.argval, (
            f"Stale cache returned argval={result2.argval!r} from old code object "
            f"instead of {correct.argval!r} from new code object"
        )


class TestAsyncMarkerInLoopSkipped:
    """Bug 2: async trace markers skip on second hit due to processed_locations set."""

    def test_marker_fires_on_every_loop_iteration(self) -> None:
        """A marker inside a loop should fire on every iteration, not just the first.

        The sync TraceExecutor correctly fires on every hit. The async variant
        has a processed_locations set that deduplicates by (filename, lineno),
        causing markers in loops to fire only once.
        """
        results: list[int] = []

        async def task1() -> None:
            for i in range(2):
                # frontrun: loop_marker
                results.append(i)  # noqa: PERF402

        async def task2() -> None:
            for i in range(2):
                # frontrun: loop_marker
                results.append(10 + i)  # noqa: PERF401

        # Schedule: task1 iter0, task2 iter0, task1 iter1, task2 iter1
        schedule = Schedule(
            [
                Step("task1", "loop_marker"),
                Step("task2", "loop_marker"),
                Step("task1", "loop_marker"),
                Step("task2", "loop_marker"),
            ]
        )

        executor = AsyncTraceExecutor(schedule, deadlock_timeout=5.0)
        executor.run(
            {
                "task1": task1,
                "task2": task2,
            },
            timeout=10.0,
        )

        # With the bug, only 2 of 4 markers fire (one per task), so the schedule
        # is incomplete and raises TimeoutError, or the order is wrong.
        # After the fix, all 4 markers fire and the execution order is:
        # task1 iter0 (append 0), task2 iter0 (append 10), task1 iter1 (append 1), task2 iter1 (append 11)
        assert results == [0, 10, 1, 11]
