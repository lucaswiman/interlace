"""Thin pure-Python wrappers around :class:`frontrun._dpor.PyDporEngine`.

These helpers exist only to give sync and async DPOR a single place to
construct the Rust engine with a stable keyword-only signature.  No
threading, asyncio, or scheduler state lives here.
"""

from __future__ import annotations

from typing import Any

try:
    from frontrun._dpor import PyDporEngine  # type: ignore[reportAttributeAccessIssue]
except ModuleNotFoundError as _err:
    raise ModuleNotFoundError(
        "frontrun._dpor_core requires the frontrun._dpor Rust extension.\n"
        "Build it with:  make build-dpor-3.14   (or build-dpor-3.10 / build-dpor-3.14t)\n"
        "Or install from source:  pip install -e ."
    ) from _err


def make_dpor_engine(
    *,
    num_threads: int,
    preemption_bound: int | None,
    max_branches: int,
    max_executions: int | None,
    search: str | None = None,
) -> Any:
    """Construct a :class:`PyDporEngine` with a stable keyword-only signature.

    The Rust engine accepts the same constructor arguments for both sync
    and async exploration; this wrapper only exists so the two callers
    share an import site and a normalised set of defaults.
    """
    return PyDporEngine(
        num_threads=num_threads,
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        max_executions=max_executions,
        search=search,
    )
