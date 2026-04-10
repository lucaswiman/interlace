"""Tests for marker coverage tracking (Extension 8).

TDD red-phase: these tests verify the API contract for marker_coverage_report
and MarkerCoverageResult, which do not yet exist.  All tests are expected to
FAIL until the feature is implemented.
"""

import pytest

from frontrun.common import Schedule
from frontrun.trace_markers import all_marker_schedules

# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------


class Counter:
    """Shared state with a classic lost-update race."""

    def __init__(self):
        self.value = 0

    def increment(self) -> None:
        temp = self.value  # frontrun: read
        self.value = temp + 1  # frontrun: write


class IndependentWork:
    """Shared state with no race — each thread writes to its own list."""

    def __init__(self):
        self.log_a: list[str] = []
        self.log_b: list[str] = []

    def work_a(self) -> None:
        self.log_a.append("start")  # frontrun: step1
        self.log_a.append("end")  # frontrun: step2

    def work_b(self) -> None:
        self.log_b.append("start")  # frontrun: step1
        self.log_b.append("end")  # frontrun: step2


# ---------------------------------------------------------------------------
# Test 1 — import works
# ---------------------------------------------------------------------------


class TestImport:
    """marker_coverage_report and MarkerCoverageResult can be imported."""

    def test_import_marker_coverage_report(self):
        """marker_coverage_report is importable from frontrun.trace_markers."""
        from frontrun.trace_markers import marker_coverage_report  # noqa: F401

    def test_import_marker_coverage_result(self):
        """MarkerCoverageResult is importable from frontrun.trace_markers."""
        from frontrun.trace_markers import MarkerCoverageResult  # noqa: F401

    def test_imported_names_are_callable_and_class(self):
        """marker_coverage_report is callable; MarkerCoverageResult is a class."""
        from frontrun.trace_markers import MarkerCoverageResult, marker_coverage_report

        assert callable(marker_coverage_report)
        assert isinstance(MarkerCoverageResult, type)


# ---------------------------------------------------------------------------
# Test 2 — full coverage: stop_on_first=False, invariant always holds
# ---------------------------------------------------------------------------


class TestFullCoverage:
    """When invariant always holds and stop_on_first=False, all schedules run."""

    def test_coverage_ratio_is_1_0(self):
        """coverage_ratio == 1.0 when all schedules are executed."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: s.log_a == ["start", "end"] and s.log_b == ["start", "end"],
            stop_on_first=False,
        )
        assert result.coverage_ratio == 1.0

    def test_missed_schedules_is_empty(self):
        """missed_schedules is empty when all schedules are executed."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: s.log_a == ["start", "end"] and s.log_b == ["start", "end"],
            stop_on_first=False,
        )
        assert result.missed_schedules == []

    def test_executed_schedule_list_contains_all(self):
        """executed_schedule_list contains all 6 schedules for 2x2 markers."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: s.log_a == ["start", "end"] and s.log_b == ["start", "end"],
            stop_on_first=False,
        )
        assert len(result.executed_schedule_list) == 6

    def test_executed_schedules_field_equals_total(self):
        """executed_schedules count matches total_schedules when fully covered."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: s.log_a == ["start", "end"] and s.log_b == ["start", "end"],
            stop_on_first=False,
        )
        assert result.executed_schedules == result.total_schedules


# ---------------------------------------------------------------------------
# Test 3 — partial coverage: stop_on_first=True with a bug
# ---------------------------------------------------------------------------


class TestPartialCoverage:
    """When a bug is found early with stop_on_first=True, coverage is partial."""

    def test_coverage_ratio_less_than_1(self):
        """coverage_ratio < 1.0 when exploration stops early."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=Counter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
            stop_on_first=True,
        )
        assert result.coverage_ratio < 1.0

    def test_missed_schedules_is_non_empty(self):
        """missed_schedules is non-empty when exploration stopped early."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=Counter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
            stop_on_first=True,
        )
        assert len(result.missed_schedules) > 0

    def test_missed_schedules_are_schedule_objects(self):
        """Items in missed_schedules are Schedule instances."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=Counter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
            stop_on_first=True,
        )
        for sched in result.missed_schedules:
            assert isinstance(sched, Schedule), f"Expected Schedule, got {type(sched)}"

    def test_interleaving_result_property_holds_false(self):
        """interleaving_result.property_holds is False when bug is found."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=Counter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
            stop_on_first=True,
        )
        assert not result.interleaving_result.property_holds


# ---------------------------------------------------------------------------
# Test 4 — total_schedules matches all_marker_schedules
# ---------------------------------------------------------------------------


class TestTotalSchedulesMatchesAllMarkerSchedules:
    """total_schedules == len(all_marker_schedules(...)) for various thread configs."""

    def test_two_threads_two_markers(self):
        """total_schedules == 6 for 2 threads × 2 markers (C(4,2))."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        expected = len(all_marker_schedules({"t1": ["step1", "step2"], "t2": ["step1", "step2"]}))
        assert result.total_schedules == expected

    def test_total_schedules_is_6_for_two_threads_two_markers(self):
        """total_schedules is exactly 6 for the canonical 2×2 case."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        assert result.total_schedules == 6

    def test_single_thread_total_is_one(self):
        """total_schedules == 1 for a single thread (only one possible order)."""
        from frontrun.trace_markers import marker_coverage_report

        class SingleWorkerState:
            def __init__(self):
                self.items: list[str] = []

            def work(self) -> None:
                self.items.append("a")  # frontrun: step_a
                self.items.append("b")  # frontrun: step_b

        result = marker_coverage_report(
            setup=SingleWorkerState,
            threads={
                "t1": (lambda s: s.work(), ["step_a", "step_b"]),
            },
            invariant=lambda s: s.items == ["a", "b"],
            stop_on_first=False,
        )
        assert result.total_schedules == 1


# ---------------------------------------------------------------------------
# Test 5 — executed + missed == total
# ---------------------------------------------------------------------------


class TestExecutedPlusMissedEqualsTotal:
    """len(executed_schedule_list) + len(missed_schedules) == total_schedules always."""

    def test_full_coverage(self):
        """Partition holds when all schedules run."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        assert len(result.executed_schedule_list) + len(result.missed_schedules) == result.total_schedules

    def test_partial_coverage(self):
        """Partition holds when exploration stops early."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=Counter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
            stop_on_first=True,
        )
        assert len(result.executed_schedule_list) + len(result.missed_schedules) == result.total_schedules

    def test_no_overlap_between_executed_and_missed(self):
        """No schedule appears in both executed_schedule_list and missed_schedules."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=Counter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
            stop_on_first=True,
        )

        def _schedule_key(s: Schedule) -> tuple[tuple[str, str], ...]:
            return tuple((step.execution_name, step.marker_name) for step in s.steps)

        executed_keys = {_schedule_key(s) for s in result.executed_schedule_list}
        missed_keys = {_schedule_key(s) for s in result.missed_schedules}
        assert executed_keys.isdisjoint(missed_keys), "Executed and missed schedules must not overlap"


# ---------------------------------------------------------------------------
# Test 6 — interleaving_result is a valid InterleavingResult
# ---------------------------------------------------------------------------


class TestInterleavingResultField:
    """interleaving_result is a properly populated InterleavingResult."""

    def test_interleaving_result_type(self):
        """interleaving_result is an InterleavingResult instance."""
        from frontrun.common import InterleavingResult
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        assert isinstance(result.interleaving_result, InterleavingResult)

    def test_interleaving_result_property_holds_true_for_correct_code(self):
        """interleaving_result.property_holds is True when invariant always holds."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: s.log_a == ["start", "end"] and s.log_b == ["start", "end"],
            stop_on_first=False,
        )
        assert result.interleaving_result.property_holds

    def test_interleaving_result_num_explored_matches_executed_schedules(self):
        """interleaving_result.num_explored == result.executed_schedules."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        assert result.interleaving_result.num_explored == result.executed_schedules

    def test_interleaving_result_has_counterexample_when_bug_found(self):
        """interleaving_result.counterexample is not None when a bug is found."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=Counter,
            threads={
                "t1": (lambda s: s.increment(), ["read", "write"]),
                "t2": (lambda s: s.increment(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
            stop_on_first=True,
        )
        assert result.interleaving_result.counterexample is not None


# ---------------------------------------------------------------------------
# Test 7 — two-thread coverage: verify all 6 schedules are tracked
# ---------------------------------------------------------------------------


class TestTwoThreadCoverageTracking:
    """Detailed verification of schedule tracking for the canonical 2×2 case."""

    def test_six_schedules_total(self):
        """Exactly 6 schedules exist for 2 threads × 2 markers."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        assert result.total_schedules == 6

    def test_six_schedules_all_executed(self):
        """All 6 schedules are in executed_schedule_list when stop_on_first=False."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        assert len(result.executed_schedule_list) == 6

    def test_executed_schedules_match_all_marker_schedules(self):
        """The executed schedule set equals the full all_marker_schedules set."""
        from frontrun.trace_markers import marker_coverage_report

        thread_decl = {"t1": ["step1", "step2"], "t2": ["step1", "step2"]}

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )

        def _schedule_key(s: Schedule) -> tuple[tuple[str, str], ...]:
            return tuple((step.execution_name, step.marker_name) for step in s.steps)

        all_keys = {_schedule_key(s) for s in all_marker_schedules(thread_decl)}
        executed_keys = {_schedule_key(s) for s in result.executed_schedule_list}
        assert executed_keys == all_keys

    def test_executed_schedule_list_items_are_schedule_objects(self):
        """All items in executed_schedule_list are Schedule instances."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        for sched in result.executed_schedule_list:
            assert isinstance(sched, Schedule), f"Expected Schedule, got {type(sched)}"

    def test_each_executed_schedule_preserves_thread_order(self):
        """Each schedule in executed_schedule_list preserves per-thread marker order."""
        from frontrun.trace_markers import marker_coverage_report

        result = marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )
        for sched in result.executed_schedule_list:
            t1_markers = [s.marker_name for s in sched.steps if s.execution_name == "t1"]
            t2_markers = [s.marker_name for s in sched.steps if s.execution_name == "t2"]
            assert t1_markers == ["step1", "step2"], f"t1 order violated in schedule: {sched}"
            assert t2_markers == ["step1", "step2"], f"t2 order violated in schedule: {sched}"


# ---------------------------------------------------------------------------
# Test 8 — MarkerCoverageResult dataclass structure
# ---------------------------------------------------------------------------


class TestMarkerCoverageResultStructure:
    """MarkerCoverageResult has all required fields with correct types."""

    @pytest.fixture()
    def full_result(self):
        from frontrun.trace_markers import marker_coverage_report

        return marker_coverage_report(
            setup=IndependentWork,
            threads={
                "t1": (lambda s: s.work_a(), ["step1", "step2"]),
                "t2": (lambda s: s.work_b(), ["step1", "step2"]),
            },
            invariant=lambda s: True,
            stop_on_first=False,
        )

    def test_has_interleaving_result(self, full_result):
        """Result has interleaving_result attribute."""
        assert hasattr(full_result, "interleaving_result")

    def test_has_total_schedules(self, full_result):
        """Result has total_schedules attribute (int)."""
        assert hasattr(full_result, "total_schedules")
        assert isinstance(full_result.total_schedules, int)

    def test_has_executed_schedules(self, full_result):
        """Result has executed_schedules attribute (int)."""
        assert hasattr(full_result, "executed_schedules")
        assert isinstance(full_result.executed_schedules, int)

    def test_has_coverage_ratio(self, full_result):
        """Result has coverage_ratio attribute (float in [0.0, 1.0])."""
        assert hasattr(full_result, "coverage_ratio")
        assert isinstance(full_result.coverage_ratio, float)
        assert 0.0 <= full_result.coverage_ratio <= 1.0

    def test_has_missed_schedules(self, full_result):
        """Result has missed_schedules attribute (list)."""
        assert hasattr(full_result, "missed_schedules")
        assert isinstance(full_result.missed_schedules, list)

    def test_has_executed_schedule_list(self, full_result):
        """Result has executed_schedule_list attribute (list)."""
        assert hasattr(full_result, "executed_schedule_list")
        assert isinstance(full_result.executed_schedule_list, list)

    def test_coverage_ratio_computation(self, full_result):
        """coverage_ratio == executed_schedules / total_schedules."""
        if full_result.total_schedules == 0:
            assert full_result.coverage_ratio == 0.0
        else:
            expected = full_result.executed_schedules / full_result.total_schedules
            assert abs(full_result.coverage_ratio - expected) < 1e-9
