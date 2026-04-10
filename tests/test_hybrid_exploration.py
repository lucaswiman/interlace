"""Tests for hybrid marker + bytecode exploration.

This is the TDD red-phase: all tests here target ``explore_hybrid_interleavings``,
which does NOT yet exist.  The import itself is expected to fail with
``ImportError`` until the feature is implemented.

Extension 3 from the testing-strategies roadmap:
    For each valid marker-level schedule, run bytecode-level (opcode-shuffling)
    exploration within each marker window.  This finds bugs that require precise
    bytecode timing *within* a marker window — races that pure marker exploration
    misses because it only controls thread ordering at the coarse marker
    granularity.
"""

import os
import threading

import pytest

# Skip under the frontrun CLI wrapper — hybrid tests create many concurrent
# threads and the LD_PRELOAD overhead causes OOM in constrained environments.
# These tests don't use DPOR or I/O detection, so the wrapper adds no value.
if os.environ.get("FRONTRUN_ACTIVE"):
    pytest.skip("hybrid tests are too thread-heavy for the frontrun wrapper", allow_module_level=True)

# ---------------------------------------------------------------------------
# 1. Import verification — this MUST fail until the feature is implemented.
# ---------------------------------------------------------------------------


def test_import_explore_hybrid_interleavings():
    """explore_hybrid_interleavings is importable from frontrun.trace_markers."""
    # This import should raise ImportError until the function is added.
    from frontrun.trace_markers import explore_hybrid_interleavings  # noqa: F401


# ---------------------------------------------------------------------------
# Shared test subjects
# ---------------------------------------------------------------------------


class SharedCounter:
    """Non-atomic counter — classic lost-update candidate."""

    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value  # frontrun: read
        self.value = temp + 1  # frontrun: write


class SafeCounter:
    """Lock-protected counter — should be race-free."""

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            temp = self.value  # frontrun: read
            self.value = temp + 1  # frontrun: write


class TwoPhaseSharedState:
    """State for a two-phase (prepare/commit) racy operation.

    The race only manifests when thread-2's LOAD occurs between
    thread-1's prepare_value assignment and its commit_value
    assignment, *and* thread-1's commit is interleaved *within*
    thread-2's bytecode sequence between LOAD and STORE.

    Pure marker exploration catches the coarse-level interleaving.
    Bytecode exploration within each window additionally exercises
    the fine-grained ordering inside each marker region.
    """

    def __init__(self):
        self.prepared = 0
        self.committed = 0

    def do_work(self, amount: int):
        # "prepare" window: two bytecode instructions that can be
        # split — LOAD_FAST and STORE_ATTR — giving bytecode-level
        # races within the marker window.
        self.prepared = self.prepared + amount  # frontrun: prepare
        self.committed = self.committed + amount  # frontrun: commit


class BytecodeRaceState:
    """State where a race only occurs at the sub-marker (bytecode) level.

    Both threads share a list.  The invariant is that the list length
    equals 2 after both threads append.  However, the list is built
    from a *read-then-write* at the Python level that can race between
    opcodes even when both threads are nominally in the same marker
    window.

    Pure marker exploration with a single "append" marker per thread
    sees no interleaving variation at the marker level (there is only
    one marker per thread and both threads complete their region).
    Bytecode exploration within that window can expose the race if
    the implementation uses a non-atomic read-modify-write.
    """

    def __init__(self):
        # Using a plain integer to simulate a non-atomic accumulator.
        self.count = 0

    def racy_append(self):
        # Simulate non-atomic += at the bytecode level (LOAD / ADD / STORE).
        # The single marker means marker exploration sees only one schedule;
        # bytecode exploration within that window can interleave the three ops.
        current = self.count  # frontrun: append
        self.count = current + 1


# ---------------------------------------------------------------------------
# 2. Basic race detection via hybrid exploration
# ---------------------------------------------------------------------------


class TestHybridBasicRaceDetection:
    """Hybrid exploration finds lost-update races."""

    def test_detects_lost_update_race(self):
        """Two threads doing non-atomic increment: hybrid finds the violation."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            bytecode_attempts=5,
        )

        assert not result.property_holds, (
            "Expected hybrid exploration to find the lost-update race; "
            f"got property_holds=True after {result.num_explored} explorations"
        )
        assert result.counterexample is not None

    def test_detects_race_with_multiple_threads(self):
        """Three-thread lost-update is caught by hybrid exploration."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
                "t3": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 3,
            stop_on_first=True,
            bytecode_attempts=5,
        )

        assert not result.property_holds


# ---------------------------------------------------------------------------
# 3. Safe code passes hybrid exploration
# ---------------------------------------------------------------------------


class TestHybridSafeCode:
    """When the invariant truly holds, hybrid exploration reports property_holds=True."""

    @pytest.mark.intentionally_leaves_dangling_threads
    def test_safe_counter_passes(self):
        """Lock-protected counter is safe under all hybrid explorations.

        Note: marked intentionally_leaves_dangling_threads because the
        TraceExecutor creates daemon threads that may not fully clean up when
        threading.Lock() causes scheduling-level stalls in some interleavings.
        """
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SafeCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            bytecode_attempts=3,
        )

        assert result.property_holds, (
            "Lock-protected counter should satisfy invariant under all "
            f"interleavings; counterexample: {result.counterexample}"
        )

    def test_trivially_safe_sequential_code(self):
        """Code with no concurrency bug is always safe."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        class SingleWriter:
            def __init__(self):
                self.value = 0

            def write(self, amount: int):
                self.value = amount  # frontrun: write  (atomic assignment)

        # Each thread writes its own distinct value; invariant checks any positive.
        result = explore_hybrid_interleavings(
            setup=SingleWriter,
            threads={
                "t1": (lambda s: s.write(1), ["write"]),
                "t2": (lambda s: s.write(1), ["write"]),
            },
            invariant=lambda s: s.value == 1,
            bytecode_attempts=5,
        )

        assert result.property_holds


# ---------------------------------------------------------------------------
# 4. num_explored reflects combined marker + bytecode exploration
# ---------------------------------------------------------------------------


class TestHybridNumExplored:
    """num_explored is >= the number of marker schedules explored."""

    def test_num_explored_at_least_marker_count(self):
        """With two markers per thread, there are 6 marker schedules (C(4,2)).
        Each gets bytecode_attempts runs, so num_explored >= 6."""
        from frontrun.trace_markers import all_marker_schedules, explore_hybrid_interleavings

        marker_decl = {"t1": ["read", "write"], "t2": ["read", "write"]}
        expected_marker_schedules = len(all_marker_schedules(marker_decl))

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=3,
        )

        assert result.num_explored >= expected_marker_schedules, (
            f"Expected num_explored >= {expected_marker_schedules} "
            f"(one per marker schedule), got {result.num_explored}"
        )

    def test_num_explored_scales_with_bytecode_attempts(self):
        """More bytecode_attempts → more total explorations."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result_low = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=1,
        )

        result_high = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=3,
        )

        assert result_high.num_explored > result_low.num_explored, (
            f"Higher bytecode_attempts should yield more explorations; "
            f"low={result_low.num_explored}, high={result_high.num_explored}"
        )


# ---------------------------------------------------------------------------
# 5. stop_on_first behaviour
# ---------------------------------------------------------------------------


class TestHybridStopOnFirst:
    """stop_on_first=True returns as soon as a violation is found."""

    def test_stop_on_first_returns_early(self):
        """When stop_on_first=True, num_explored is typically small."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            bytecode_attempts=3,
        )

        assert not result.property_holds
        # With stop_on_first the exploration should terminate early,
        # long before exhausting all marker schedules × bytecode_attempts.
        # C(4,2)=6 marker schedules × 100 attempts = 600 max without early exit.
        # We expect far fewer explorations before the first violation.
        assert result.num_explored < 100, (
            f"stop_on_first=True should stop early, got num_explored={result.num_explored}"
        )

    def test_stop_on_first_false_explores_all(self):
        """When stop_on_first=False, exploration continues past the first violation."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result_first = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            bytecode_attempts=3,
        )

        result_all = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=3,
        )

        assert result_all.num_explored >= result_first.num_explored, (
            "stop_on_first=False should explore at least as many as stop_on_first=True"
        )


# ---------------------------------------------------------------------------
# 6. bytecode_attempts parameter affects thoroughness
# ---------------------------------------------------------------------------


class TestBytecodeAttemptsParameter:
    """bytecode_attempts controls how many random opcode schedules are tried
    per marker window."""

    def test_zero_attempts_only_runs_marker_schedules(self):
        """bytecode_attempts=0: only pure marker-level schedules are run,
        no intra-window bytecode randomisation."""
        from frontrun.trace_markers import all_marker_schedules, explore_hybrid_interleavings

        marker_decl = {"t1": ["read", "write"], "t2": ["read", "write"]}
        num_markers = len(all_marker_schedules(marker_decl))

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=0,
        )

        # With bytecode_attempts=0, num_explored should equal the number of
        # distinct marker schedules (no extra bytecode runs).
        assert result.num_explored == num_markers, (
            f"bytecode_attempts=0 should run exactly {num_markers} schedules, "
            f"got {result.num_explored}"
        )

    def test_higher_attempts_runs_more_explorations(self):
        """More bytecode_attempts = more total explorations, increasing the
        chance of finding races under OS-level scheduling."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result_few = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=1,
        )

        result_many = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=3,
        )

        assert result_many.num_explored > result_few.num_explored, (
            f"More attempts should run more explorations: "
            f"few={result_few.num_explored}, many={result_many.num_explored}"
        )


# ---------------------------------------------------------------------------
# 7. Return type and counterexample structure
# ---------------------------------------------------------------------------


class TestHybridReturnType:
    """explore_hybrid_interleavings returns a well-formed InterleavingResult."""

    def test_returns_interleaving_result(self):
        """Return value is an InterleavingResult instance."""
        from frontrun.common import InterleavingResult
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            bytecode_attempts=3,
        )

        assert isinstance(result, InterleavingResult)

    def test_counterexample_is_not_none_when_violation_found(self):
        """When property_holds is False, counterexample is set."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            bytecode_attempts=5,
        )

        assert not result.property_holds
        assert result.counterexample is not None, (
            "counterexample should be set when property_holds=False"
        )

    @pytest.mark.intentionally_leaves_dangling_threads
    def test_counterexample_is_none_when_property_holds(self):
        """When property_holds is True, counterexample is None.

        Note: marked intentionally_leaves_dangling_threads because SafeCounter
        uses threading.Lock() which may leave TraceExecutor daemon threads.
        """
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SafeCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            bytecode_attempts=3,
        )

        assert result.property_holds
        assert result.counterexample is None

    def test_num_explored_is_positive(self):
        """num_explored is always > 0 when threads have markers."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            bytecode_attempts=3,
        )

        assert result.num_explored > 0


# ---------------------------------------------------------------------------
# 8. More thorough than markers alone (the key selling point)
# ---------------------------------------------------------------------------


class TestHybridMoreThoroughThanMarkersAlone:
    """Hybrid exploration uncovers races that pure marker exploration can miss.

    When the race requires a specific intra-window bytecode interleaving,
    pure marker exploration with coarse-grained synchronisation points will
    not trigger it.  Hybrid exploration adds random bytecode scheduling within
    each window to cover these cases.
    """

    def test_hybrid_explores_more_than_pure_markers(self):
        """Hybrid exploration runs more total explorations than pure markers.

        Pure marker exploration runs exactly ``len(all_marker_schedules(...))``
        executions.  Hybrid adds bytecode-level concurrent runs for each
        marker schedule, so it explores strictly more schedules and has a
        higher chance of exercising subtle thread-scheduling-dependent races.
        """
        from frontrun.trace_markers import all_marker_schedules, explore_hybrid_interleavings, explore_marker_interleavings

        marker_decl = {"t1": ["read", "write"], "t2": ["read", "write"]}
        pure_marker_count = len(all_marker_schedules(marker_decl))

        marker_result = explore_marker_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
        )

        hybrid_result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=5,
        )

        # Hybrid must have explored more than pure markers
        assert hybrid_result.num_explored > marker_result.num_explored, (
            f"Hybrid ({hybrid_result.num_explored}) should exceed pure marker "
            f"count ({marker_result.num_explored}) with bytecode_attempts=5"
        )
        # Hybrid explores marker_count * (1 + bytecode_attempts) total
        assert hybrid_result.num_explored == pure_marker_count * (1 + 5), (
            f"Expected {pure_marker_count * 6} explorations, got {hybrid_result.num_explored}"
        )

    def test_hybrid_num_explored_exceeds_pure_marker_count(self):
        """Hybrid explores strictly more schedules than pure marker exploration."""
        from frontrun.trace_markers import all_marker_schedules, explore_hybrid_interleavings

        marker_decl = {"t1": ["read", "write"], "t2": ["read", "write"]}
        pure_marker_count = len(all_marker_schedules(marker_decl))

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=False,
            bytecode_attempts=5,
        )

        assert result.num_explored > pure_marker_count, (
            f"Hybrid ({result.num_explored}) should exceed pure marker "
            f"count ({pure_marker_count}) since bytecode_attempts=5 adds "
            "extra runs per marker schedule"
        )


# ---------------------------------------------------------------------------
# 9. Timeout and deadlock_timeout parameters are accepted
# ---------------------------------------------------------------------------


class TestHybridTimeoutParameters:
    """The function accepts and respects timeout/deadlock_timeout kwargs."""

    def test_accepts_timeout_parameter(self):
        """timeout kwarg is accepted without error."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        # Should not raise even with a generous timeout.
        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            timeout=30.0,
            bytecode_attempts=3,
        )

        assert isinstance(result.num_explored, int)

    def test_accepts_deadlock_timeout_parameter(self):
        """deadlock_timeout kwarg is accepted without error."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            deadlock_timeout=2.0,
            bytecode_attempts=3,
        )

        assert isinstance(result.num_explored, int)

    def test_accepts_none_timeout(self):
        """timeout=None means no timeout."""
        from frontrun.trace_markers import explore_hybrid_interleavings

        result = explore_hybrid_interleavings(
            setup=SharedCounter,
            threads={
                "t1": (lambda c: c.increment(), ["read", "write"]),
                "t2": (lambda c: c.increment(), ["read", "write"]),
            },
            invariant=lambda c: c.value == 2,
            stop_on_first=True,
            timeout=None,
            bytecode_attempts=5,
        )

        assert isinstance(result.num_explored, int)
