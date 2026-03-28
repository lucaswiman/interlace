"""Tests for the DPOR exploration HTML report."""

from __future__ import annotations

import json
import os
import tempfile

from frontrun._report import (
    ExecutionRecord,
    ExplorationReport,
    LockEvent,
    SwitchPoint,
    _safe_repr,
    generate_html_report,
)
from frontrun.dpor import _append_unique_lock_event, explore_dpor


def test_safe_repr_truncation():
    """_safe_repr truncates long strings."""
    short = _safe_repr(42)
    assert short == "42"

    long_str = "x" * 200
    result = _safe_repr(long_str)
    assert len(result) <= 80
    assert result.endswith("...")


def test_safe_repr_exception():
    """_safe_repr handles objects that raise on repr()."""

    class BadRepr:
        def __repr__(self):
            raise RuntimeError("boom")

    result = _safe_repr(BadRepr())
    assert "BadRepr" in result


def test_exploration_report_to_json():
    """ExplorationReport serializes to valid JSON."""
    report = ExplorationReport(
        num_threads=2,
        thread_names=["Thread 0", "Thread 1"],
        executions=[
            ExecutionRecord(
                index=0,
                schedule_trace=[0, 0, 1, 1, 0, 1],
                switch_points=[
                    SwitchPoint(
                        schedule_index=2,
                        from_thread=0,
                        to_thread=1,
                        filename="test.py",
                        lineno=10,
                        function_name="increment",
                        opcode="LOAD_ATTR",
                        source_line="temp = self.value",
                        shadow_stack_top5=["<Counter>", "42"],
                        access_type="read",
                        attr_name="value",
                        obj_type_name="Counter",
                    ),
                ],
                invariant_held=True,
                was_deadlock=False,
                race_info=None,
            ),
        ],
    )
    json_str = report.to_json()
    data = json.loads(json_str)
    assert data["version"] == 1
    assert data["num_threads"] == 2
    assert len(data["executions"]) == 1
    assert data["executions"][0]["schedule_trace"] == [0, 0, 1, 1, 0, 1]
    assert len(data["executions"][0]["switch_points"]) == 1
    sp = data["executions"][0]["switch_points"][0]
    assert sp["opcode"] == "LOAD_ATTR"
    assert sp["shadow_stack_top5"] == ["<Counter>", "42"]


def test_generate_html_report():
    """generate_html_report produces a valid HTML file with embedded JSON."""
    report = ExplorationReport(
        num_threads=2,
        thread_names=["Thread 0", "Thread 1"],
        executions=[
            ExecutionRecord(
                index=0,
                schedule_trace=[0, 1],
                switch_points=[],
                invariant_held=True,
                was_deadlock=False,
            ),
        ],
    )
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        generate_html_report(report, path)
        with open(path) as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "dpor-report" in html
        # The JSON should be embedded
        assert '"num_threads":2' in html
        assert "/* __DPOR_REPORT_DATA__ */" not in html
    finally:
        os.unlink(path)


def test_explore_dpor_with_report():
    """explore_dpor generates a report when _global_report_path is set."""
    import frontrun._report

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        frontrun._report._global_report_path = path

        class Counter:
            def __init__(self):
                self.value = 0

            def increment(self):
                temp = self.value
                self.value = temp + 1

        result = explore_dpor(
            setup=lambda: Counter(),
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value == 2,
            preemption_bound=2,
        )

        # Should have generated a report file
        assert os.path.exists(path)
        with open(path) as f:
            html = f.read()

        # Extract JSON from the HTML
        assert "dpor-report" in html
        assert '"num_threads":2' in html

        # Parse the embedded JSON to verify structure
        # Find the JSON between the script tags
        start_marker = '<script type="application/json" id="dpor-data">'
        end_marker = "</script>"
        start = html.index(start_marker) + len(start_marker)
        end = html.index(end_marker, start)
        json_str = html[start:end]
        data = json.loads(json_str)

        assert data["num_threads"] == 2
        assert len(data["executions"]) > 0
        # First execution should have a schedule trace
        assert len(data["executions"][0]["schedule_trace"]) > 0

        # Since there's a race condition, there should be multiple executions
        assert len(data["executions"]) >= 2, f"Expected multiple executions, got {len(data['executions'])}"

        # At least some executions should have switch points
        total_switches = sum(len(ex["switch_points"]) for ex in data["executions"])
        assert total_switches > 0, "Expected at least one switch point"

        # Check switch point structure
        for ex in data["executions"]:
            for sp in ex["switch_points"]:
                assert "schedule_index" in sp
                assert "from_thread" in sp
                assert "to_thread" in sp
                assert "opcode" in sp
                assert "source_line" in sp
                assert "shadow_stack_top5" in sp

    finally:
        frontrun._report._global_report_path = None
        if os.path.exists(path):
            os.unlink(path)


def test_append_unique_lock_event_deduplicates_adjacent_duplicates():
    events: list[LockEvent] = []

    first = LockEvent(schedule_index=13, thread_id=2, event_type="wait", lock_id=7)
    duplicate = LockEvent(schedule_index=13, thread_id=2, event_type="wait", lock_id=7)
    distinct = LockEvent(schedule_index=14, thread_id=2, event_type="wait", lock_id=7)

    _append_unique_lock_event(events, first)
    _append_unique_lock_event(events, duplicate)
    _append_unique_lock_event(events, distinct)

    assert events == [first, distinct]
