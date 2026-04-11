"""Public sync DPOR API and compatibility re-exports."""

from __future__ import annotations

from frontrun._dpor_runtime._shared import (
    _USE_SYS_MONITORING,
    PyDporEngine,
    PyExecution,
    ShadowStack,
    StableObjectIds,
    _append_unique_lock_event,
    _dpor_tls,
    _make_object_key,
    _process_opcode,
)
from frontrun._dpor_runtime.explore import explore_dpor
from frontrun._dpor_runtime.preload_bridge import _PreloadBridge
from frontrun._dpor_runtime.replay import _reproduce_dpor_counterexample, _run_dpor_schedule
from frontrun._dpor_runtime.runner import DporBytecodeRunner
from frontrun._dpor_runtime.scheduler import (
    DporScheduler,
    _IOAnchoredReplayScheduler,
    _ReplayDporScheduler,
    _ReplayEngine,
    _ReplayExecution,
)

__all__ = [
    "DporBytecodeRunner",
    "DporScheduler",
    "PyDporEngine",
    "PyExecution",
    "ShadowStack",
    "StableObjectIds",
    "_IOAnchoredReplayScheduler",
    "_PreloadBridge",
    "_ReplayDporScheduler",
    "_ReplayEngine",
    "_ReplayExecution",
    "_USE_SYS_MONITORING",
    "_append_unique_lock_event",
    "_dpor_tls",
    "_make_object_key",
    "_process_opcode",
    "_reproduce_dpor_counterexample",
    "_run_dpor_schedule",
    "explore_dpor",
]
