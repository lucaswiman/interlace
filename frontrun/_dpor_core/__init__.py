"""Pure helpers shared by sync and async DPOR pipelines.

This package holds the bits of the DPOR drivers that don't depend on
threading, asyncio, ``sys.settrace``/``sys.monitoring``, ``ContextVars``,
or any other execution-mode-specific machinery.  Everything here is a
pure function (or close to it) that operates on simple data passed in
by the caller.

The two callers today are:

* ``frontrun/_dpor_runtime/explore.py`` (sync DPOR driver)
* ``frontrun/async_dpor.py``           (async DPOR driver)

Keeping these helpers in one place locks in the shared type vocabulary
without forcing the two drivers through a common backend abstraction yet.
"""

from __future__ import annotations

from frontrun._dpor_core.engine import make_dpor_engine
from frontrun._dpor_core.invariants import (
    compute_serializable_baseline_async,
    compute_serializable_baseline_sync,
    format_race_failure_explanation,
)

__all__ = [
    "compute_serializable_baseline_async",
    "compute_serializable_baseline_sync",
    "format_race_failure_explanation",
    "make_dpor_engine",
]
