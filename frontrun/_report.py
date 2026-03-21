"""Interactive HTML report for DPOR exploration visualization.

Records per-execution data (schedule traces, thread switch points with
source/stack context, detected races) and generates a self-contained HTML
file with an SVG timeline viewer built on web components.
"""

from __future__ import annotations

import json
import linecache
import os
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Global sentinel set by the pytest plugin (--frontrun-report flag)
# ---------------------------------------------------------------------------
_global_report_path: str | None = None

# Maximum number of executions to record (avoid unbounded memory)
_MAX_RECORDED_EXECUTIONS = 1000


def _safe_repr(obj: Any, max_len: int = 80) -> str:
    """Return a truncated repr() of an object, safe for JSON embedding."""
    try:
        r = repr(obj)
    except Exception:
        r = f"<{type(obj).__name__}>"
    if len(r) > max_len:
        return r[: max_len - 3] + "..."
    return r


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StepEvent:
    """Lightweight per-step record: what happened at each schedule step."""

    thread_id: int
    filename: str
    lineno: int
    function_name: str
    opcode: str
    source_line: str
    access_type: str | None = None
    attr_name: str | None = None
    obj_type_name: str | None = None
    value_repr: str | None = None  # repr of the value loaded/stored


@dataclass(slots=True)
class SwitchPoint:
    """Data captured at a thread switch during one DPOR execution."""

    schedule_index: int
    from_thread: int | None  # None at the very start
    to_thread: int
    filename: str
    lineno: int
    function_name: str
    opcode: str
    source_line: str
    shadow_stack_top5: list[str]
    access_type: str | None = None
    attr_name: str | None = None
    obj_type_name: str | None = None


@dataclass(slots=True)
class ExecutionRecord:
    """Record of one DPOR execution."""

    index: int
    schedule_trace: list[int]
    switch_points: list[SwitchPoint]
    invariant_held: bool
    was_deadlock: bool
    race_info: list[dict[str, Any]] | None = None
    step_events: dict[int, StepEvent] = field(default_factory=dict)


@dataclass
class ExplorationReport:
    """Full DPOR exploration data for visualization."""

    num_threads: int
    thread_names: list[str]
    executions: list[ExecutionRecord] = field(default_factory=list)
    source_files: dict[str, list[str]] = field(default_factory=dict)

    def _collect_source_files(self) -> None:
        """Populate source_files from filenames referenced in switch points and step events."""
        seen: set[str] = set()
        for ex in self.executions:
            for sp in ex.switch_points:
                if sp.filename and sp.filename not in seen and not sp.filename.startswith("<"):
                    seen.add(sp.filename)
            for se in ex.step_events.values():
                if se.filename and se.filename not in seen and not se.filename.startswith("<"):
                    seen.add(se.filename)
        for filename in sorted(seen):
            try:
                lines = linecache.getlines(filename)
                self.source_files[filename] = [line.rstrip("\n") for line in lines]
            except Exception:
                pass

    def to_json(self) -> str:
        """Serialize to JSON string for embedding in HTML."""
        self._collect_source_files()
        exec_dicts = []
        for ex in self.executions:
            d = asdict(ex)
            # Convert step_events from {int: StepEvent} to {str: dict} for JSON
            d["step_events"] = {str(k): v for k, v in d["step_events"].items()}
            exec_dicts.append(d)
        data = {
            "version": 1,
            "num_threads": self.num_threads,
            "thread_names": self.thread_names,
            "executions": exec_dicts,
            "source_files": self.source_files,
        }
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "_report_template.html")
_JSON_PLACEHOLDER = "/* __DPOR_REPORT_DATA__ */"


def generate_html_report(report: ExplorationReport, output_path: str) -> None:
    """Generate a self-contained HTML report file."""
    with open(_TEMPLATE_PATH) as f:
        template = f.read()
    json_data = report.to_json()
    # Escape </script> in JSON to prevent premature tag closing
    json_data = json_data.replace("</", "<\\/")
    html = template.replace(_JSON_PLACEHOLDER, json_data)
    with open(output_path, "w") as f:
        f.write(html)
