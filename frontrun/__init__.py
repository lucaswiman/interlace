"""
Frontrun: Deterministic concurrency testing for Python.

Trace markers (sync)::

    from frontrun.common import Schedule, Step
    from frontrun import TraceExecutor

Async trace markers (pass a dict of task names to coroutines)::

    from frontrun import TraceExecutor
    from frontrun.common import Schedule, Step
    executor = TraceExecutor(schedule)
    executor.run({"task1": coro_factory1, "task2": coro_factory2})

Unified exploration entry point (recommended)::

    from frontrun import explore

    # Sync DPOR (default)
    result = explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    result.assert_holds()

    # Async — detected automatically
    async def worker(state): ...
    result = await explore(setup=make_state, workers=worker, count=2, invariant=...)

    # Strategy selection
    result = explore(..., strategy="dpor")    # default
    result = explore(..., strategy="random")  # formerly explore_interleavings

DPOR (Dynamic Partial Order Reduction) systematic exploration (deprecated)::

    from frontrun.dpor import explore_dpor  # deprecated — use frontrun.explore instead

Async DPOR systematic exploration (deprecated)::

    from frontrun.async_dpor import explore_async_dpor  # deprecated — use frontrun.explore instead

Bytecode exploration (deprecated)::

    from frontrun import explore_random  # canonical new name
    from frontrun import explore_interleavings  # deprecated alias

Async shuffler exploration (deprecated)::

    from frontrun import explore_async_random  # canonical new name
    from frontrun import explore_async_interleavings  # deprecated alias

Contrib helpers (use threads= for sync, tasks= for async)::

    from frontrun.contrib.django import django_dpor
    from frontrun.contrib.sqlalchemy import sqlalchemy_dpor, get_connection, get_async_connection
"""

import importlib
import warnings
from importlib.metadata import version as _metadata_version
from typing import TYPE_CHECKING, Any

from frontrun.common import DEPRECATION_MESSAGES, NondeterministicSQLError
from frontrun.explore import explore
from frontrun.trace_markers import TraceExecutor

if TYPE_CHECKING:
    from frontrun.async_shuffler import explore_async_random as explore_async_random
    from frontrun.bytecode import explore_random as explore_random

try:
    __version__: str = _metadata_version("frontrun")
except Exception:
    __version__ = "0.0.0"


# ---------------------------------------------------------------------------
# Deprecated aliases — resolved lazily via module-level __getattr__ (PEP 562).
# Each access warns; each call also warns via the underlying ``deprecate``-wrapped
# shim.  Python's default warnings filter deduplicates by (message, category,
# module, lineno), so a given call site only produces one DeprecationWarning
# per process — users aren't spammed.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "explore_random": ("frontrun.bytecode", "explore_random"),
    "explore_async_random": ("frontrun.async_shuffler", "explore_async_random"),
}

_DEPRECATED_ALIASES: dict[str, tuple[str, str, str]] = {
    "explore_dpor": ("frontrun._dpor_runtime.explore", "explore_dpor", DEPRECATION_MESSAGES["explore_dpor"]),
    "explore_async_dpor": (
        "frontrun.async_dpor",
        "explore_async_dpor",
        DEPRECATION_MESSAGES["explore_async_dpor"],
    ),
    "explore_async_interleavings": (
        "frontrun.async_shuffler",
        "explore_interleavings",
        DEPRECATION_MESSAGES["explore_async_interleavings"],
    ),
}


def _deprecated_explore_interleavings(*args: Any, **kwargs: Any) -> Any:
    """Deprecated sync/async dispatch shim for the old ``explore_interleavings`` API.

    The pre-0.5 function accepted either ``threads=`` (sync) or ``tasks=``
    (async) and dispatched internally. Preserved here through the 0.5 and 0.6
    series; removed in 0.6.
    """
    warnings.warn(DEPRECATION_MESSAGES["explore_interleavings"], DeprecationWarning, stacklevel=2)
    has_threads = "threads" in kwargs
    has_tasks = "tasks" in kwargs
    if has_threads == has_tasks:
        raise TypeError("explore_interleavings() requires exactly one of 'threads' or 'tasks'")
    if has_tasks:
        from frontrun.async_shuffler import explore_async_random as _async_impl

        return _async_impl(*args, **kwargs)
    from frontrun.bytecode import explore_random as _sync_impl

    return _sync_impl(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_path), attr)
    if name == "explore_interleavings":
        warnings.warn(DEPRECATION_MESSAGES["explore_interleavings"], DeprecationWarning, stacklevel=2)
        return _deprecated_explore_interleavings
    if name in _DEPRECATED_ALIASES:
        module_path, attr, msg = _DEPRECATED_ALIASES[name]
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module 'frontrun' has no attribute {name!r}")


__all__ = [
    "NondeterministicSQLError",
    "TraceExecutor",
    "__version__",
    "explore",
    "explore_random",
    "explore_async_random",
    # Deprecated aliases — not in __all__ so static analysis won't suggest them,
    # but still accessible via __getattr__ for backward compat.
]
