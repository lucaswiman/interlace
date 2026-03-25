"""
Bytecode-tracing DPOR (Dynamic Partial Order Reduction) for frontrun.

This module implements systematic interleaving exploration using DPOR,
completely separate from the existing bytecode.py random exploration.

The approach:
1. A Rust DPOR engine (frontrun._dpor) manages the exploration tree,
   vector clocks, and wakeup tree exploration.
2. Python drives execution: runs threads under sys.settrace opcode
   tracing, uses a shadow stack to detect shared-memory accesses,
   and feeds access/sync events to the Rust engine.
3. Cooperative threading primitives (lock, event, etc.) are monkey-patched
   to yield control back to the DPOR scheduler and report synchronization
   events for happens-before tracking.

Usage::

    from frontrun.dpor import explore_dpor

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
    )
    assert result.property_holds, result.explanation  # fails — lost update!
"""

from __future__ import annotations

import dis
import linecache
import sys
import threading
import time
import types
from collections.abc import Callable
from typing import Any, TypeVar

from frontrun._cooperative import (
    clear_context,
    patch_locks,
    real_condition,
    real_lock,
    set_context,
    set_sync_reporter,
    unpatch_locks,
)
from frontrun._deadlock import DeadlockError, SchedulerAbort, install_wait_for_graph, uninstall_wait_for_graph
from frontrun._io_detection import (
    patch_io,
    set_io_reporter,
    unpatch_io,
)
from frontrun._redis_client import (
    is_redis_tid_suppressed,
    patch_redis,
    unpatch_redis,
)
from frontrun._sql_anomaly import classify_sql_anomaly
from frontrun._sql_cursor import (
    clear_sql_metadata,
    get_active_sql_io_context,
    is_tid_suppressed,
    patch_sql,
    unpatch_sql,
)
from frontrun._sql_insert_tracker import check_uncaptured_inserts, clear_insert_tracker
from frontrun._trace_format import TraceRecorder, build_call_chain, format_trace
from frontrun._tracing import TraceFilter as _TraceFilter
from frontrun._tracing import is_cmdline_user_code as _is_cmdline_user_code
from frontrun._tracing import is_dynamic_code as _is_dynamic_code
from frontrun._tracing import set_active_trace_filter as _set_active_trace_filter
from frontrun._tracing import should_trace_file as _should_trace_file
from frontrun.cli import require_active as _require_frontrun_env
from frontrun.common import InterleavingResult

try:
    from frontrun._dpor import PyDporEngine, PyExecution  # type: ignore[reportAttributeAccessIssue]
except ModuleNotFoundError as _err:
    raise ModuleNotFoundError(
        "explore_dpor requires the frontrun._dpor Rust extension.\n"
        "Build it with:  make build-dpor-3.14t   (or build-dpor-3.10 / build-dpor-3.14)\n"
        "Or install from source:  pip install -e ."
    ) from _err

T = TypeVar("T")

_PY_VERSION = sys.version_info[:2]
# sys.monitoring (PEP 669) is available since 3.12 and is required for
# free-threaded builds (3.13t/3.14t) where sys.settrace + f_trace_opcodes
# has a known crash bug (CPython #118415).
_USE_SYS_MONITORING = _PY_VERSION >= (3, 12)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shadow Stack for shared-access detection
# ---------------------------------------------------------------------------


class ShadowStack:
    """Mirrors CPython's evaluation stack to track object identity.

    When LOAD_ATTR/STORE_ATTR execute, we peek at our shadow stack
    to identify which object is being accessed.
    """

    __slots__ = ("stack",)

    def __init__(self) -> None:
        self.stack: list[Any] = []

    def push(self, val: Any) -> None:
        self.stack.append(val)

    def pop(self) -> Any:
        return self.stack.pop() if self.stack else None

    def peek(self, n: int = 0) -> Any:
        idx = -(n + 1)
        return self.stack[idx] if abs(idx) <= len(self.stack) else None

    def clear(self) -> None:
        self.stack.clear()


# Pre-analyzed instruction cache: code object -> {offset -> instruction}.
#
# Keyed by the code object itself (not ``id(code)``).  Using the code
# object as the dict key keeps a strong reference, which prevents the
# object from being garbage-collected while cached.  This eliminates the
# stale-cache bug where a GC'd code object's id was reused by a new one
# within a single DPOR execution.
_INSTR_CACHE: dict[Any, dict[int, dis.Instruction]] = {}
# Must use real_lock() — not threading.Lock() — to avoid cooperative re-entry
# when the pytest plugin patches threading.Lock globally.
_INSTR_CACHE_LOCK = real_lock()


def _get_instructions(code: Any) -> dict[int, dis.Instruction]:
    """Get a mapping from byte offset to Instruction for a code object."""
    # Fast path: already cached (safe to read without lock on GIL builds;
    # on free-threaded builds dict reads are internally locked)
    cached = _INSTR_CACHE.get(code)
    if cached is not None:
        return cached
    with _INSTR_CACHE_LOCK:
        # Double-check after acquiring lock
        if code in _INSTR_CACHE:
            return _INSTR_CACHE[code]
        mapping = {}
        # show_caches parameter was added in Python 3.11
        if _PY_VERSION >= (3, 11):
            instructions = dis.get_instructions(code, show_caches=False)
        else:
            instructions = dis.get_instructions(code)
        for instr in instructions:
            mapping[instr.offset] = instr
        _INSTR_CACHE[code] = mapping
        return mapping


# ---------------------------------------------------------------------------
# Thread-local state for the DPOR scheduler
# ---------------------------------------------------------------------------

_dpor_tls = threading.local()


# ---------------------------------------------------------------------------
# LD_PRELOAD I/O event bridge
# ---------------------------------------------------------------------------


class _PreloadBridge:
    """Routes I/O events from the LD_PRELOAD library to DPOR threads.

    The LD_PRELOAD library intercepts C-level ``send()``/``recv()`` calls
    (e.g. from psycopg2's libpq) and writes events to a pipe.  An
    :class:`~frontrun._preload_io.IOEventDispatcher` reads the pipe in a
    background thread and invokes :meth:`listener` for each event.

    This bridge maps OS thread IDs to DPOR logical thread IDs and buffers
    events in per-thread lists.  The DPOR scheduler drains these buffers
    at each scheduling point via :meth:`drain`.
    """

    def __init__(self, dispatcher: Any = None) -> None:
        self._lock = real_lock()
        self._tid_to_dpor: dict[int, int] = {}
        self._pending: dict[int, list[tuple[int, str, str, str | None, list[str] | None]]] = {}
        self._active = False
        self._dispatcher = dispatcher  # IOEventDispatcher (for poll())

    def register_thread(self, os_tid: int, dpor_id: int) -> None:
        """Map an OS thread ID to a DPOR logical thread ID."""
        with self._lock:
            self._tid_to_dpor[os_tid] = dpor_id
            self._pending.setdefault(dpor_id, [])
            self._active = True

    def unregister_thread(self, os_tid: int) -> None:
        """Remove an OS thread ID mapping."""
        with self._lock:
            self._tid_to_dpor.pop(os_tid, None)
            if not self._tid_to_dpor:
                self._active = False

    def clear(self) -> None:
        """Clear all mappings and pending events (between executions)."""
        with self._lock:
            self._tid_to_dpor.clear()
            self._pending.clear()
            self._active = False

    def listener(self, event: Any) -> None:
        """IOEventDispatcher callback — buffer the event for the right thread."""
        if not self._active:
            return
        # Skip close events — closing a file descriptor doesn't mutate the
        # external resource and creates many spurious conflict points that
        # force DPOR to explore uninteresting interleavings first.
        if event.kind == "close":
            return
        # Skip if this thread's cursor.execute() or Redis client already reported at a higher level
        if is_tid_suppressed(event.tid) or is_redis_tid_suppressed(event.tid):
            return
        with self._lock:
            dpor_id = self._tid_to_dpor.get(event.tid)
            if dpor_id is None:
                return
            # Map libc I/O operations to DPOR access kinds.  Using the
            # actual send/recv distinction (write/read) is critical: the
            # DPOR engine's ObjectState tracks per-thread latest-read and
            # latest-write separately.  If we treated all socket I/O as
            # "write", only the LAST write per thread would be tracked,
            # and early access positions (e.g. a SELECT recv) would be
            # overwritten by later ones (e.g. a COMMIT recv).  With
            # read/write distinction, DPOR iteratively explores wakeup tree
            # branches through the send/recv pairs to reach the critical interleaving.
            kind = "write" if event.kind == "write" else "read"
            obj_key = _make_object_key(hash(event.resource_id), event.resource_id)
            detail, call_chain = get_active_sql_io_context(event.tid)
            self._pending.setdefault(dpor_id, []).append((obj_key, kind, event.resource_id, detail, call_chain))

    def drain(self, dpor_id: int) -> list[tuple[int, str, str, str | None, list[str] | None]]:
        """Return and clear buffered events for a DPOR thread.

        Each item is ``(object_key, kind, resource_id, detail, call_chain)``.

        On free-threaded Python the background reader may not have
        processed pipe data yet, so we poll the dispatcher first to
        flush any pending bytes into listener callbacks.
        """
        if self._dispatcher is not None:
            self._dispatcher.poll()
        with self._lock:
            events = self._pending.get(dpor_id)
            if events:
                self._pending[dpor_id] = []
                return events
            return []


# ---------------------------------------------------------------------------
# DPOR Opcode Scheduler
# ---------------------------------------------------------------------------


_SENTINEL = object()


class DporScheduler:
    """Controls thread execution at opcode granularity, driven by the DPOR engine.

    Unlike the random OpcodeScheduler in bytecode.py, this scheduler gets
    its scheduling decisions from the Rust DPOR engine.

    Deadlock detection uses a fallback timeout plus instant lock-ordering
    cycle detection via the :class:`~frontrun._deadlock.WaitForGraph`.
    """

    def __init__(
        self,
        engine: PyDporEngine,
        execution: PyExecution,
        num_threads: int,
        engine_lock: threading.Lock | None = None,
        deadlock_timeout: float = 5.0,
        trace_recorder: TraceRecorder | None = None,
        preload_bridge: _PreloadBridge | None = None,
        detect_io: bool = False,
        stable_ids: StableObjectIds | None = None,
        switch_point_collector: list[Any] | None = None,
        track_dunder_dict_accesses: bool = False,
    ) -> None:
        self.engine = engine
        self.execution = execution
        self.num_threads = num_threads
        self.deadlock_timeout = deadlock_timeout
        self.trace_recorder = trace_recorder
        self._preload_bridge = preload_bridge
        self._detect_io = detect_io
        self._stable_ids = stable_ids if stable_ids is not None else StableObjectIds()
        self._switch_point_collector = switch_point_collector
        self._track_dunder_dict_accesses = track_dunder_dict_accesses
        self._step_event_collector: dict[int, Any] | None = {} if switch_point_collector is not None else None
        self._lock_event_collector: list[Any] | None = [] if switch_point_collector is not None else None
        # Captured at the moment the first error fires (schedule_trace length then).
        # Steps at/after this index are teardown artifacts and should not be rendered.
        self._deadlock_at: int | None = None
        # On free-threaded Python, PyO3 &mut self borrows are non-blocking
        # (try-or-panic).  A single engine_lock serialises ALL calls to the
        # engine and execution objects across worker threads, the sync
        # reporter, and the main explore_dpor loop.
        self._engine_lock: threading.Lock = engine_lock if engine_lock is not None else real_lock()
        self._lock = real_lock()
        self._condition = real_condition(self._lock)
        self._finished = False
        self._error: Exception | None = None
        self._threads_done: set[int] = set()
        self._waiting_count: int = 0
        self._current_thread: int | None = None

        # Shadow stacks are per-thread (each thread only accesses its own),
        # stored in thread-local storage. This avoids cross-thread access
        # entirely, which is critical for free-threaded builds.
        # Format: _dpor_tls._shadow_stacks = {frame_id: ShadowStack}

        # Tracks which threads are waiting for which locks (lock_id → {thread_ids}).
        # Used to block threads in the DPOR execution when they're spinning
        # on a cooperative lock, and unblock them when the lock is released.
        self._lock_waiters: dict[int, set[int]] = {}
        # Per-thread deferred I/O buffers. Lists are shared with thread-local
        # storage so other threads can flush deferred I/O when they reach a
        # real competing I/O boundary.
        self._pending_io_by_thread: dict[int, list[tuple[int, str]]] = {}
        # Per-thread lock nesting depth mirrored from TLS for cross-thread
        # deferred-I/O flush decisions.
        self._lock_depth_by_thread: dict[int, int] = {}

        # Maps iterator id → original container object. When GET_ITER creates
        # an iterator from a mutable container, we record the mapping so that
        # FOR_ITER can report reads on the underlying container.
        self._iter_to_container: dict[int, Any] = {}

        # Row-lock registry: resource_id → thread_id holding the lock.
        # SELECT FOR UPDATE is exclusive — only one thread can hold it at a time.
        self._active_row_locks: dict[str, int] = {}
        # Reverse index: thread_id → set of resource_ids held by that thread.
        # Avoids O(n) scan in _release_row_locks_unlocked.
        self._thread_row_locks: dict[int, set[str]] = {}

        # Stable integer IDs for row-lock resources (for WaitForGraph nodes).
        # String resource IDs are assigned monotonically increasing integers so
        # row-lock nodes ("row_lock", int) are disjoint from cooperative-lock
        # nodes ("lock", id(obj)) in the WaitForGraph.
        self._row_lock_ids: dict[str, int] = {}
        self._row_lock_next_id: int = 0

        # Threads currently blocked waiting for a DPOR row lock.
        # Maps blocked_thread_id → holder_thread_id.  Used by
        # _schedule_next() to skip blocked threads and schedule their
        # holders instead, preventing the scheduler from cycling
        # indefinitely between a blocked thread and its holder.
        self._row_lock_blocked: dict[int, int] = {}

        # Last path_id snapshot from _schedule_next, used to attribute
        # lock events to the correct scheduling step on free-threaded Python.
        self._last_scheduled_path_id: int | None = None

        # Request the first scheduling decision
        self._current_thread = self._schedule_next()

    def _schedule_next(self) -> int | None:
        """Ask the DPOR engine which thread to run next.

        If the engine selects a thread that is blocked on a DPOR row lock,
        override the decision and schedule the lock holder instead.  This
        prevents the scheduler from cycling between a blocked thread and
        its holder (defect #6).

        Also snapshots ``engine.path_position`` under the engine lock so
        that ``report_and_wait`` can attribute subsequent lock events to
        the correct scheduling step (see ``_last_scheduled_path_id``).
        """
        with self._engine_lock:
            runnable = self.execution.runnable_threads()
            if not runnable:
                self._last_scheduled_path_id = None
                return None

            scheduled = self.engine.schedule(self.execution)
            # Snapshot path position under engine_lock. On free-threaded
            # Python, another thread may call schedule() concurrently
            # after we release the lock, advancing path.pos.  The saved
            # position ensures _sync_reporter attributes lock events to
            # the correct step.
            _pp = getattr(self.engine, "path_position", None)
            self._last_scheduled_path_id = _pp - 1 if _pp is not None else None
            if scheduled is not None and scheduled in self._row_lock_blocked:
                holder = self._row_lock_blocked[scheduled]
                if holder not in self._threads_done:
                    return holder
                # Holder is done — lock should have been released via
                # mark_done → _release_row_locks_unlocked.  Clean up the
                # stale blocked entry so scheduled can proceed.
                self._row_lock_blocked.pop(scheduled, None)
            return scheduled

    def wait_for_turn(self, thread_id: int) -> bool:
        """Block until it's this thread's turn. Returns False when done."""
        return self._report_and_wait(None, thread_id)

    def _all_other_live_threads_blocked_by_current(self, thread_id: int) -> bool:
        from frontrun._deadlock import get_wait_for_graph

        graph = get_wait_for_graph()
        if graph is None:
            return False
        live_other_threads = {
            tid for tid in range(self.num_threads) if tid != thread_id and tid not in self._threads_done
        }
        if not live_other_threads:
            return True
        blocked_threads = graph.reverse_reachable_threads_from(thread_id)
        return live_other_threads.issubset(blocked_threads)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        """Report accesses for an opcode and wait for this thread's turn.

        Combines ``_process_opcode`` and the wait-for-turn logic under a
        single lock acquisition so that ``engine.report_access()`` and
        ``engine.schedule()`` can never be called concurrently.  This is
        critical on free-threaded Python (3.13t/3.14t) where there is no
        GIL to serialise PyO3 ``&mut self`` borrows.
        """
        return self._report_and_wait(frame, thread_id)

    def _report_and_wait(self, frame: Any | None, thread_id: int) -> bool:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            # Set reentrancy guard so GC-triggered __del__ (e.g.,
            # redis.Redis.__del__) won't re-enter the scheduler.
            # See defect #7.
            _scheduler_tls._in_dpor_machinery = True
            try:

                def _flush_pending_io_for(flush_thread_id: int, *, allow_inside_lock: bool = False) -> None:
                    pending_io = self._pending_io_by_thread.get(flush_thread_id)
                    if not pending_io:
                        return
                    if not allow_inside_lock and self._lock_depth_by_thread.get(flush_thread_id, 0) > 0:
                        return
                    engine = self.engine
                    execution = self.execution
                    for obj_key, io_kind in pending_io:
                        with self._engine_lock:
                            engine.report_io_access(execution, flush_thread_id, obj_key, io_kind)
                    pending_io.clear()

                def _flush_other_pending_io_for_current_io() -> None:
                    current_pending = self._pending_io_by_thread.get(thread_id)
                    if not current_pending:
                        return
                    for other_thread_id, pending_io in list(self._pending_io_by_thread.items()):
                        if other_thread_id == thread_id or not pending_io:
                            continue
                        # Another thread reached a real I/O boundary, so any
                        # deferred I/O from this thread is now part of a real
                        # race window and must become visible to the engine.
                        _flush_pending_io_for(other_thread_id, allow_inside_lock=True)

                # Merge LD_PRELOAD I/O events (C-level send/recv from e.g.
                # psycopg2) into the thread's pending_io list.  The preload
                # bridge buffers events from the pipe reader thread, keyed by
                # DPOR thread ID.
                _pending_io: list[tuple[int, str]] | None = getattr(_dpor_tls, "pending_io", None)
                if self._preload_bridge is not None:
                    _preload_events = self._preload_bridge.drain(thread_id)
                    if _preload_events:
                        # Record into trace for human-readable output.  These events
                        # come from C extensions (e.g. libpq) with no Python frame.
                        _recorder = self.trace_recorder
                        if _recorder is not None:
                            for _, _kind, _resource_id, _detail, _call_chain in _preload_events:
                                _recorder.record_io(
                                    thread_id,
                                    _resource_id,
                                    _kind,
                                    call_chain=_call_chain,
                                    detail=_detail,
                                )
                        # Convert 3-tuples to 2-tuples for the pending list
                        _io_pairs = [(_key, _kind) for _key, _kind, _, _, _ in _preload_events]
                        if _pending_io is not None:
                            _pending_io.extend(_io_pairs)
                        else:
                            _dpor_tls.pending_io = _io_pairs
                            _pending_io = _io_pairs
                # NOTE: We intentionally do NOT skip scheduling inside explicit
                # SQL transactions.  SQL atomicity is handled separately by
                # _tx_buffer in _sql_cursor.py (SQL events are buffered during
                # BEGIN...COMMIT and flushed atomically at COMMIT).  Non-SQL
                # shared state (Python objects) modified inside a transaction
                # body must still be interleaved by DPOR to find races.

                while True:
                    if self._finished or self._error:
                        return False
                    if self._current_thread == thread_id:
                        current_pending = self._pending_io_by_thread.get(thread_id)
                        if (
                            frame is None
                            and current_pending
                            and self._lock_depth_by_thread.get(thread_id, 0) > 0
                            and self._all_other_live_threads_blocked_by_current(thread_id)
                        ):
                            return True
                        _flush_other_pending_io_for_current_io()
                        # Flush deferred I/O only once this thread actually owns
                        # the current DPOR step. On free-threaded Python a thread
                        # can reach report_and_wait while another thread still owns
                        # the step; flushing earlier stamps the access onto the
                        # wrong path_id and can hide the wakeup tree insertion point.
                        _flush_pending_io_for(thread_id)
                        # Process opcode accesses only when it's our turn.
                        # Deferring this until the thread is scheduled ensures
                        # that accesses are recorded at the correct path_id
                        # (after any intervening operations by other threads).
                        # Without this, a preempted thread's accesses land at the
                        # preemption branch where the other thread is Active,
                        # making wakeup tree insertions at that position impossible.
                        # Save frame info before _process_opcode clears it,
                        # in case we need to record a switch/step point.
                        _switch_frame = frame
                        # Snapshot shadow stack before _process_opcode pops values.
                        # For STORE_ATTR: stack is [..., value, obj] — we want value (TOS1).
                        # For LOAD_ATTR: stack is [..., obj] — value will be on TOS after.
                        _pre_opcode_stack = None
                        if self._step_event_collector is not None and frame is not None:
                            _pre_stacks = getattr(_dpor_tls, "_shadow_stacks", None)
                            if _pre_stacks:
                                _pre_shadow = _pre_stacks.get(id(frame))
                                if _pre_shadow and _pre_shadow.stack:
                                    _pre_opcode_stack = list(_pre_shadow.stack[-3:])  # last 3 elements
                        if frame is not None:
                            _process_opcode(frame, self, thread_id)
                            frame = None  # only process once
                        # Record step event for the report
                        if self._switch_point_collector is not None and _switch_frame is not None:
                            self._capture_step_event(_switch_frame, thread_id, _pre_opcode_stack)
                        # It's our turn. After executing one opcode, schedule next.
                        next_thread = self._schedule_next()
                        # _schedule_next saves the path position in
                        # self._last_scheduled_path_id (under engine_lock).
                        # Copy it to TLS so _sync_reporter can attribute lock
                        # events to this thread's scheduling step, not a later
                        # step advanced by another thread on free-threaded Python.
                        _pp = self._last_scheduled_path_id
                        if _pp is not None:
                            _dpor_tls._last_path_id = _pp
                        # Record switch point if thread changes and collector is active
                        if (
                            self._switch_point_collector is not None
                            and next_thread is not None
                            and next_thread != thread_id
                            and _switch_frame is not None
                        ):
                            self._capture_switch_point(_switch_frame, thread_id, next_thread)
                        self._current_thread = next_thread
                        if next_thread is None:
                            self._finished = True
                        self._condition.notify_all()
                        return True

                    # Wait for our turn (fallback timeout for C-blocked threads)
                    alive = self.num_threads - len(self._threads_done)
                    self._waiting_count += 1
                    try:
                        all_waiting = self._waiting_count >= alive and alive > 0
                        timeout = 0.1 if all_waiting else self.deadlock_timeout
                        if not self._condition.wait(timeout=timeout):
                            if self._current_thread in self._threads_done:
                                next_thread = self._schedule_next()
                                self._current_thread = next_thread
                                if next_thread is None:
                                    self._finished = True
                                self._condition.notify_all()
                                continue
                            self._error = TimeoutError(
                                f"DPOR deadlock: waiting for thread {thread_id}, current is {self._current_thread}"
                            )
                            self._condition.notify_all()
                            return False
                    finally:
                        self._waiting_count -= 1
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def mark_done(self, thread_id: int) -> None:
        with self._condition:
            self._threads_done.add(thread_id)
            with self._engine_lock:
                self.execution.finish_thread(thread_id)
            # Release any row locks the thread may still hold (safety net).
            # _release_row_locks_unlocked avoids re-acquiring self._condition.
            self._release_row_locks_unlocked(thread_id)
            # Clean up stale row-lock-blocked entry (safety net).
            self._row_lock_blocked.pop(thread_id, None)
            # If the done thread was the current one, schedule next
            if self._current_thread == thread_id:
                next_thread = self._schedule_next()
                self._current_thread = next_thread
                if next_thread is None and len(self._threads_done) >= self.num_threads:
                    self._finished = True
            self._condition.notify_all()

    def report_error(self, error: Exception) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def _capture_switch_point(self, frame: Any, from_thread: int, to_thread: int) -> None:
        """Capture a SwitchPoint when the scheduler switches threads."""
        from frontrun._report import SwitchPoint, _safe_repr

        code = frame.f_code
        lineno = frame.f_lineno
        instr = _get_instructions(code).get(frame.f_lasti)
        opcode = instr.opname if instr else ""
        source_line = linecache.getline(code.co_filename, lineno).strip()

        # Snapshot shadow stack top 5
        shadow_top5: list[str] = []
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks:
            shadow = stacks.get(id(frame))
            if shadow and shadow.stack:
                shadow_top5.extend(_safe_repr(shadow.stack[-(i + 1)]) for i in range(min(5, len(shadow.stack))))

        # Get access info from the most recent trace event
        access_type: str | None = None
        attr_name: str | None = None
        obj_type_name: str | None = None
        if self.trace_recorder and self.trace_recorder.events:
            last_ev = self.trace_recorder.events[-1]
            if last_ev.thread_id == from_thread:
                access_type = last_ev.access_type
                attr_name = last_ev.attr_name
                obj_type_name = last_ev.obj_type_name

        # schedule_trace length gives current position (after schedule() appended)
        with self._engine_lock:
            schedule_len = len(self.execution.schedule_trace)
        schedule_index = schedule_len - 1  # index of the just-scheduled step

        sp = SwitchPoint(
            schedule_index=schedule_index,
            from_thread=from_thread,
            to_thread=to_thread,
            filename=code.co_filename,
            lineno=lineno,
            function_name=code.co_name,
            opcode=opcode,
            source_line=source_line,
            shadow_stack_top5=shadow_top5,
            access_type=access_type,
            attr_name=attr_name,
            obj_type_name=obj_type_name,
        )
        self._switch_point_collector.append(sp)  # type: ignore[union-attr]

    def _capture_step_event(self, frame: Any, thread_id: int, pre_opcode_stack: list[Any] | None = None) -> None:
        """Capture a StepEvent keyed by schedule index (path_id)."""
        from frontrun._report import StepEvent, _safe_repr

        if self._step_event_collector is None:
            return
        code = frame.f_code
        lineno = frame.f_lineno
        instr = _get_instructions(code).get(frame.f_lasti)
        opcode = instr.opname if instr else ""
        source_line = linecache.getline(code.co_filename, lineno).strip()

        # Get access info from the most recent trace event
        access_type: str | None = None
        attr_name: str | None = None
        obj_type_name: str | None = None
        if self.trace_recorder and self.trace_recorder.events:
            last_ev = self.trace_recorder.events[-1]
            if last_ev.thread_id == thread_id:
                access_type = last_ev.access_type
                attr_name = last_ev.attr_name
                obj_type_name = last_ev.obj_type_name

        # Capture the value involved in the access.
        # The trace callback fires *before* the instruction executes, so for
        # LOAD_ATTR we read the attribute that's about to be loaded, and for
        # STORE_ATTR we show the current (pre-store) value.
        # We read from the actual Python objects rather than the shadow stack,
        # which only tracks object identity (its elements are often None).
        value_repr: str | None = None
        try:
            if attr_name and obj_type_name and opcode in ("LOAD_ATTR", "STORE_ATTR") and instr:
                attr = instr.argval
                if attr:
                    # Find the object in frame locals — look for instances of the right type
                    for local_val in frame.f_locals.values():
                        if type(local_val).__name__ == obj_type_name:
                            val = getattr(local_val, attr, _SENTINEL)
                            if val is not _SENTINEL and not callable(val):
                                value_repr = _safe_repr(val)
                                break
            elif opcode in ("LOAD_GLOBAL", "STORE_GLOBAL") and instr:
                name = instr.argval
                if name and name in frame.f_globals:
                    val = frame.f_globals[name]
                    if not callable(val):
                        value_repr = _safe_repr(val)
        except Exception:
            pass

        # Key by schedule index (= len(schedule_trace) - 1, the most recently
        # scheduled step). This aligns with path_id used in race detection.
        with self._engine_lock:
            schedule_idx = len(self.execution.schedule_trace) - 1

        self._step_event_collector[schedule_idx] = StepEvent(
            thread_id=thread_id,
            filename=code.co_filename,
            lineno=lineno,
            function_name=code.co_name,
            opcode=opcode,
            source_line=source_line,
            access_type=access_type,
            attr_name=attr_name,
            obj_type_name=obj_type_name,
            value_repr=value_repr,
        )

    @staticmethod
    def get_shadow_stack(frame_id: int) -> ShadowStack:
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks is None:
            stacks = {}
            _dpor_tls._shadow_stacks = stacks
        if frame_id not in stacks:
            stacks[frame_id] = ShadowStack()
        return stacks[frame_id]

    @staticmethod
    def remove_shadow_stack(frame_id: int) -> None:
        stacks = getattr(_dpor_tls, "_shadow_stacks", None)
        if stacks is not None:
            stacks.pop(frame_id, None)

    def _row_lock_int_id(self, res_id: str) -> int:
        """Return a stable monotonic integer ID for *res_id* (allocated on first call)."""
        lid = self._row_lock_ids.get(res_id)
        if lid is None:
            lid = self._row_lock_next_id
            self._row_lock_next_id += 1
            self._row_lock_ids[res_id] = lid
        return lid

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> None:
        """Block until all *resource_ids* can be held by *thread_id*.

        If another thread holds a conflicting lock, waits on the condition
        variable.  On timeout (the holder is likely blocked in C too), lets
        the C call proceed — the ``lock_timeout`` PostgreSQL safety net
        will handle it as a fast error rather than an indefinite hang.

        When a WaitForGraph is installed, registers waiting/holding edges for
        instant cycle-based deadlock detection.
        """
        from frontrun._deadlock import DeadlockError, SchedulerAbort, format_cycle, get_wait_for_graph

        graph = get_wait_for_graph()
        with self._condition:
            for res_id in resource_ids:
                lock_int_id = self._row_lock_int_id(res_id)
                while True:
                    holder = self._active_row_locks.get(res_id)
                    if holder is None or holder == thread_id:
                        break
                    # Another thread holds this row lock — check for cycle first
                    if graph is not None:
                        cycle = graph.add_waiting(thread_id, lock_int_id, kind="row_lock")
                        if cycle is not None:
                            graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                            desc = format_cycle(cycle, {v: k for k, v in self._row_lock_ids.items()})
                            err = DeadlockError(f"Row-lock deadlock detected: {desc}", desc)
                            if self._error is None:
                                self._error = err
                            self._condition.notify_all()
                            raise SchedulerAbort(str(err))
                    # Register this thread as row-lock-blocked so that
                    # _schedule_next() skips it and schedules the holder
                    # instead of cycling (defect #6).
                    self._row_lock_blocked[thread_id] = holder
                    # Yield scheduling to the holder so it can run and
                    # either release the lock or block on one of ours
                    # (triggering WaitForGraph cycle detection).
                    if self._current_thread == thread_id:
                        self._current_thread = holder
                        self._condition.notify_all()
                    # Wait for the holder to release
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        self._row_lock_blocked.pop(thread_id, None)
                        if graph is not None:
                            graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                        if self._finished or self._error:
                            return
                        # Timeout — the holder is probably blocked in C too.
                        # Let the C call proceed; lock_timeout safety net will handle it.
                        return
                    self._row_lock_blocked.pop(thread_id, None)
                    if graph is not None:
                        graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                    if self._finished or self._error:
                        return
                self._active_row_locks[res_id] = thread_id
                self._thread_row_locks.setdefault(thread_id, set()).add(res_id)
                if graph is not None:
                    graph.add_holding(thread_id, lock_int_id, kind="row_lock")

    def _release_row_locks_unlocked(self, thread_id: int) -> bool:
        """Remove row locks for *thread_id*. Caller must hold ``self._condition``."""
        from frontrun._deadlock import get_wait_for_graph

        held = self._thread_row_locks.pop(thread_id, None)
        if not held:
            return False
        graph = get_wait_for_graph()
        for r in held:
            self._active_row_locks.pop(r, None)
            if graph is not None:
                lid = self._row_lock_ids.get(r)
                if lid is not None:
                    graph.remove_holding(thread_id, lid, kind="row_lock")
        return True

    def release_row_locks(self, thread_id: int) -> None:
        """Release all row locks held by *thread_id* (called on COMMIT/ROLLBACK)."""
        with self._condition:
            if self._release_row_locks_unlocked(thread_id):
                self._condition.notify_all()


class _ReplayEngine:
    """No-op engine used when replaying a fixed DPOR schedule."""

    def report_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_first_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_io_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_sync(
        self, execution: Any, thread_id: int, event_type: str, sync_id: int, path_id: int | None = None
    ) -> None:
        return None


class _ReplayExecution:
    """Minimal execution object for DporBytecodeRunner TLS plumbing."""

    def finish_thread(self, thread_id: int) -> None:
        return None

    def block_thread(self, thread_id: int) -> None:
        return None

    def unblock_thread(self, thread_id: int) -> None:
        return None


class _ReplayDporScheduler(DporScheduler):
    """Replay a fixed DPOR schedule using the DPOR runner and SQL row-lock logic."""

    def __init__(
        self,
        schedule: list[int],
        num_threads: int,
        *,
        deadlock_timeout: float = 5.0,
        trace_recorder: TraceRecorder | None = None,
        detect_io: bool = False,
    ) -> None:
        self._replay_schedule = list(schedule)
        self._replay_index = 0
        self._replay_max_ops = len(self._replay_schedule) * 10 + 10_000
        super().__init__(
            _ReplayEngine(),  # type: ignore[arg-type]
            _ReplayExecution(),  # type: ignore[arg-type]
            num_threads,
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
        )

    def _extend_schedule(self) -> bool:
        if self._replay_index >= self._replay_max_ops:
            return False
        active = [t for t in range(self.num_threads) if t not in self._threads_done]
        if not active:
            return False
        self._replay_schedule.extend(active)
        return True

    def _schedule_next(self) -> int | None:
        while True:
            if self._replay_index >= len(self._replay_schedule):
                if not self._extend_schedule():
                    return None
            scheduled = self._replay_schedule[self._replay_index]
            self._replay_index += 1
            if scheduled in self._threads_done:
                continue
            return scheduled

    def wait_for_turn(self, thread_id: int) -> bool:
        return self._wait_for_turn(thread_id)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        return self._wait_for_turn(thread_id)

    def _wait_for_turn(self, thread_id: int) -> bool:
        with self._condition:
            while True:
                if self._finished or self._error:
                    return False

                if self._current_thread in self._threads_done:
                    self._current_thread = self._schedule_next()
                    if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                        self._condition.notify_all()
                        return False

                if self._current_thread == thread_id:
                    self._current_thread = self._schedule_next()
                    if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                    self._condition.notify_all()
                    return True

                alive = self.num_threads - len(self._threads_done)
                self._waiting_count += 1
                try:
                    all_waiting = self._waiting_count >= alive and alive > 0
                    timeout = 0.1 if all_waiting else self.deadlock_timeout
                    if not self._condition.wait(timeout=timeout):
                        if self._current_thread in self._threads_done:
                            self._current_thread = self._schedule_next()
                            if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                                self._finished = True
                            self._condition.notify_all()
                            continue
                        self._error = TimeoutError(
                            f"DPOR replay deadlock: waiting for thread {thread_id}, current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        return False
                finally:
                    self._waiting_count -= 1


# ---------------------------------------------------------------------------
# Opcode trace callback with shadow stack access detection
# ---------------------------------------------------------------------------


# C-level method access classification.
#
# Python bytecode tracing can't see inside C method calls — list.append(),
# set.add(), dict.update(), etc. all execute opaquely.  We detect these in the
# CALL handler by checking if the callable is a builtin_function_or_method
# (i.e. bound to a C-implemented object) and classifying the call as a read or
# write based on the method name.
#
# Design: immutable types are excluded entirely (calling str.upper() can't
# cause a data race).  For mutable types, known read-only methods report READ;
# everything else defaults to WRITE.
_BUILTIN_METHOD_TYPE = type(len)  # builtin_function_or_method
_WRAPPER_DESCRIPTOR_TYPE = type(object.__setattr__)  # wrapper_descriptor
_METHOD_WRAPPER_TYPE = type("".__str__)  # method-wrapper

_IMMUTABLE_TYPES = (str, bytes, int, float, bool, complex, tuple, frozenset, type(None), types.ModuleType)

# C-level methods that are read-only (don't mutate the object).
_C_METHOD_READ_ONLY = frozenset(
    {
        # Lookup / iteration (common to multiple container types)
        "__contains__",
        "__getitem__",
        "__getattribute__",
        "__len__",
        "__iter__",
        "__reversed__",
        # list / tuple
        "count",
        "index",
        # dict
        "get",
        "keys",
        "values",
        "items",
        # set
        "issubset",
        "issuperset",
        "isdisjoint",
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        # copy
        "copy",
        "__copy__",
    }
)

# C-level methods on immutable types that iterate their FIRST ARGUMENT.
# The method's __self__ is immutable (e.g. str for str.join), so the standard
# C-method handler skips them.  We detect these by name and report a READ on
# the first argument instead.
_IMMUTABLE_SELF_ARG_READERS = frozenset({"join"})

# Type constructors that iterate their first argument (read it).
# These are `type` objects (list, dict, bytes, etc.) called as constructors.
# The CALL handler needs to report a READ on the first argument when one of
# these types is called.
_CONTAINER_CONSTRUCTORS: frozenset[type] = frozenset(
    {list, dict, set, frozenset, tuple, bytes, bytearray, enumerate, zip, map, filter, reversed}
)


# ---------------------------------------------------------------------------
# Passthrough builtins: functions that operate on their ARGUMENTS rather than
# __self__.  Keyed by id(function) for O(1) lookup.
# Format: {id(fn): (access_kind, obj_arg_index, name_arg_index_or_None)}
# ---------------------------------------------------------------------------
import builtins as _builtins_mod
import operator as _operator_mod

_PASSTHROUGH_BUILTINS: dict[int, tuple[str, int, int | None]] = {}


def _register_passthrough(fn: Any, kind: str, obj_idx: int, name_idx: int | None) -> None:
    _PASSTHROUGH_BUILTINS[id(fn)] = (kind, obj_idx, name_idx)


# Attribute writers: setattr(obj, name, val), delattr(obj, name)
_register_passthrough(_builtins_mod.setattr, "write", 0, 1)
_register_passthrough(_builtins_mod.getattr, "read", 0, 1)
_register_passthrough(_builtins_mod.delattr, "write", 0, 1)
_register_passthrough(_builtins_mod.hasattr, "read", 0, 1)
# operator module item access: operator.setitem(d, k, v), etc.
_register_passthrough(_operator_mod.setitem, "write", 0, 1)
_register_passthrough(_operator_mod.getitem, "read", 0, 1)
_register_passthrough(_operator_mod.delitem, "write", 0, 1)
# len() reads the container (needed to detect check-then-act races)
_register_passthrough(_builtins_mod.len, "read", 0, None)
# Container-iterating builtins: these read their first argument by iterating it.
# Without explicit registration, DPOR doesn't see the read because __self__ is a
# module (builtins / _functools) which is immutable.
_register_passthrough(_builtins_mod.sorted, "read", 0, None)
_register_passthrough(_builtins_mod.min, "read", 0, None)
_register_passthrough(_builtins_mod.max, "read", 0, None)
_register_passthrough(_builtins_mod.sum, "read", 0, None)
_register_passthrough(_builtins_mod.any, "read", 0, None)
_register_passthrough(_builtins_mod.all, "read", 0, None)
_register_passthrough(_builtins_mod.next, "read", 0, None)
import functools as _functools_mod

_register_passthrough(_functools_mod.reduce, "read", 1, None)


def _safe_getattr(obj: Any, attr: str) -> Any:
    """Get an attribute without triggering property descriptors.

    Uses the instance ``__dict__`` to bypass the descriptor protocol,
    preventing property getters from firing.  This is critical because
    ``_process_opcode`` runs inside ``DporScheduler._report_and_wait``'s
    locked section — a property getter that accesses the DB would call
    back into ``report_and_wait``, causing a recursive lock deadlock.

    For class-level attributes (methods, class variables), falls back to
    ``getattr`` only if the attribute is NOT a data descriptor (property,
    etc.).
    """
    # 1. Try instance __dict__ first — bypasses all descriptors
    try:
        inst_dict = object.__getattribute__(obj, "__dict__")
        if attr in inst_dict:
            return inst_dict[attr]
    except (AttributeError, TypeError):
        pass

    # 2. For class-level attributes, check if it's a data descriptor
    #    (property, cached_property, etc.) and skip to avoid side effects.
    for cls in type(obj).__mro__:
        cls_dict = cls.__dict__
        if attr in cls_dict:
            candidate = cls_dict[attr]
            # Data descriptors have __set__ or __delete__ in addition to __get__.
            # These include property, cached_property, and ORM descriptors.
            # Skip them to avoid triggering side-effectful getters.
            if hasattr(candidate, "__set__") or hasattr(candidate, "__delete__"):
                return None
            # Non-data descriptors (regular methods, staticmethod, classmethod)
            # are safe to invoke via getattr.
            return getattr(obj, attr)

    # 3. Fallback for dynamic attributes (__getattr__ etc.)
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _make_object_key(obj_id: int, name: Any) -> int:
    """Create a non-negative u64 object key for the Rust engine."""
    return hash((obj_id, name)) & 0xFFFFFFFFFFFFFFFF


# Module-level reverse map: object_key -> human-readable description.
# Set to a dict when report collection is active, None otherwise.
_object_key_reverse_map: dict[int, str] | None = None


class StableObjectIds:
    """Assign monotonically increasing stable IDs to Python objects.

    Replaces ``id(obj)`` in object key generation.  Since ``explore_dpor``
    creates fresh ``state = setup()`` each execution, ``id(obj)`` changes
    between executions for the same logical object.  This class assigns a
    counter-based ID on first access, producing the same ID across executions
    as long as objects are accessed in the same deterministic order during
    replay.

    The mapping is maintained per ``explore_dpor`` call and reset at the start
    of each execution via ``reset_for_execution()``.
    """

    __slots__ = ("_map", "_next_id")

    def __init__(self) -> None:
        self._map: dict[int, int] = {}
        self._next_id = 0

    def get(self, obj: object) -> int:
        """Return the stable ID for *obj*, assigning one on first access."""
        py_id = id(obj)
        stable_id = self._map.get(py_id)
        if stable_id is None:
            stable_id = self._next_id
            self._map[py_id] = stable_id
            self._next_id += 1
        return stable_id

    def reset_for_execution(self) -> None:
        """Clear the mapping at the start of each execution.

        Since ``explore_dpor`` creates fresh state objects each execution,
        old ``id(obj)`` values are stale.  The mapping is rebuilt during
        replay, where the same objects are accessed in the same
        deterministic order, producing the same stable IDs.
        """
        self._map.clear()
        self._next_id = 0


def _register_object_key(key: int, obj: Any, name: Any) -> None:
    """Register a human-readable description for an object key in the reverse map."""
    rmap = _object_key_reverse_map
    if rmap is not None and key not in rmap:
        type_name = type(obj).__name__
        name_str = str(name) if name is not None else ""
        rmap[key] = f"{type_name}.{name_str}" if name_str else type_name


def _report_read(
    engine: PyDporEngine,
    execution: PyExecution,
    thread_id: int,
    obj: Any,
    name: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
) -> None:
    if obj is not None:
        key = _make_object_key(stable_ids.get(obj), name)
        _register_object_key(key, obj, name)
        with lock:
            engine.report_access(execution, thread_id, key, "read")


def _report_first_read(
    engine: PyDporEngine,
    execution: PyExecution,
    thread_id: int,
    obj: Any,
    name: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
) -> None:
    """Like ``_report_read`` but preserves the **earliest** read position.

    Used for LOAD_GLOBAL reads so that ``global += 1`` (which also does
    LOAD_GLOBAL) doesn't overwrite the position of an earlier read like
    ``tmp = global_var``.  Preserving the earliest read is critical for
    detecting TOCTOU patterns.
    """
    if obj is not None:
        key = _make_object_key(stable_ids.get(obj), name)
        _register_object_key(key, obj, name)
        with lock:
            engine.report_first_access(execution, thread_id, key, "read")


def _report_write(
    engine: PyDporEngine,
    execution: PyExecution,
    thread_id: int,
    obj: Any,
    name: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
) -> None:
    if obj is not None:
        key = _make_object_key(stable_ids.get(obj), name)
        _register_object_key(key, obj, name)
        with lock:
            engine.report_access(execution, thread_id, key, "write")


def _report_weak_read(
    engine: PyDporEngine,
    execution: PyExecution,
    thread_id: int,
    obj: Any,
    name: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
) -> None:
    """Like ``_report_read`` but uses ``weak_read`` access kind.

    A weak read conflicts with writes but NOT with weak writes or other
    weak reads.  Used for LOAD_ATTR on mutable values so that loading a
    container to subscript it doesn't create a spurious conflict with
    ``STORE_SUBSCR``'s weak write on disjoint keys, while still
    conflicting with C-method writes (append, clear, etc.).
    """
    if obj is not None:
        key = _make_object_key(stable_ids.get(obj), name)
        _register_object_key(key, obj, name)
        with lock:
            engine.report_access(execution, thread_id, key, "weak_read")


def _report_weak_write(
    engine: PyDporEngine,
    execution: PyExecution,
    thread_id: int,
    obj: Any,
    name: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
) -> None:
    """Like ``_report_write`` but uses ``weak_write`` access kind.

    A weak write conflicts with reads and writes but NOT with other weak
    writes.  Used for container-level subscript tracking so that two
    ``STORE_SUBSCR`` on disjoint keys don't create a spurious conflict,
    while still conflicting with C-method reads (iteration, ``len()``, etc.).
    """
    if obj is not None:
        key = _make_object_key(stable_ids.get(obj), name)
        _register_object_key(key, obj, name)
        with lock:
            engine.report_access(execution, thread_id, key, "weak_write")


def _subscript_key_name(key: Any) -> Any:
    """Normalize a subscript key for object key computation.

    For string keys, return the string directly so it matches LOAD_ATTR/STORE_ATTR
    argval (e.g. 'value' instead of "'value'").  For non-string keys, use repr()
    as a fallback to distinguish types (e.g. int 0 vs string '0').
    """
    if isinstance(key, str):
        return key
    return repr(key)


def _expand_slice_reads(
    engine: PyDporEngine,
    execution: PyExecution,
    thread_id: int,
    container: Any,
    key: Any,
    lock: threading.Lock,
    stable_ids: StableObjectIds,
) -> None:
    """For slice subscript reads, report individual element reads.

    When a slice like ``buf[0:4]`` is read, DPOR only sees a single read on
    key ``'slice(0, 4, None)'``.  Individual element writes like ``buf[0] = x``
    use key ``'0'``.  These keys don't match, so DPOR doesn't see the conflict.

    This function expands a slice read into per-element reads so that each
    write position gets its own conflict point, enabling DPOR to explore
    interleavings where the slice read happens between individual writes.
    """
    if not isinstance(key, slice):
        return
    try:
        length = len(container)
    except (TypeError, AttributeError):
        return
    indices = range(*key.indices(length))
    for idx in indices:
        _report_read(engine, execution, thread_id, container, repr(idx), lock, stable_ids)


# ---------------------------------------------------------------------------
# Scheduling coarsening: only yield at shared-access opcodes
# ---------------------------------------------------------------------------
#
# Most opcodes (LOAD_FAST, STORE_FAST, BINARY_OP +, COPY, SWAP, etc.) only
# manipulate thread-local state (the evaluation stack and f_locals).  They
# never call _report_read/_report_write, so the DPOR engine doesn't need to
# consider them as scheduling points.  By skipping the scheduler yield for
# these opcodes, we dramatically reduce the DPOR search tree.
#
# Opcodes in _SHARED_ACCESS_OPCODES *may* report shared memory accesses and
# must go through the full scheduler yield path.  Everything else just updates
# the shadow stack without yielding.

_SHARED_ACCESS_OPCODES = frozenset(
    {
        # Attribute access on (potentially shared) objects
        "LOAD_ATTR",
        "STORE_ATTR",
        "DELETE_ATTR",
        "LOAD_METHOD",  # 3.10 only
        "LOAD_SPECIAL",  # 3.14+ context manager __enter__/__exit__
        # Global / name access
        "LOAD_GLOBAL",
        "STORE_GLOBAL",
        "LOAD_NAME",
        # Closure / cell variable access
        "LOAD_DEREF",
        "STORE_DEREF",
        # Subscript / slice access
        "BINARY_SUBSCR",
        "STORE_SUBSCR",
        "DELETE_SUBSCR",
        "BINARY_SLICE",
        "STORE_SLICE",
        # BINARY_OP may be subscript on 3.14 (checked at call site)
        "BINARY_OP",
        # Function / method calls (may invoke C-level methods)
        "CALL",
        "CALL_FUNCTION",
        "CALL_METHOD",
        "CALL_KW",
        "CALL_FUNCTION_KW",
        "CALL_FUNCTION_EX",
        # Iterator operations on (potentially mutable) containers
        "GET_ITER",
        "FOR_ITER",
    }
)


_CALL_OPCODES = frozenset({"CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_KW", "CALL_FUNCTION_KW", "CALL_FUNCTION_EX"})


def _call_might_report_access(shadow: ShadowStack, argc: int) -> bool:
    """Quick pre-check: does this CALL likely involve a C-method that reports access?

    Mirrors the detection strategies in ``_process_opcode``'s CALL handler.
    Returns True if any detectable C-method pattern is found on the shadow stack.
    False negatives would miss accesses; false positives are harmless (extra scheduling).
    """
    scan_depth = min(argc + 3, len(shadow.stack))
    for i in range(scan_depth):
        item = shadow.stack[-(i + 1)]
        if item is None:
            continue
        item_type = type(item)

        # Strategy 1: Passthrough builtins (setattr, getattr, len, etc.)
        if id(item) in _PASSTHROUGH_BUILTINS:
            return True

        # Strategy 2: Bound C methods on mutable __self__
        if item_type is _BUILTIN_METHOD_TYPE or item_type is _METHOD_WRAPPER_TYPE:
            self_obj = getattr(item, "__self__", None)
            if self_obj is not None:
                if not isinstance(self_obj, _IMMUTABLE_TYPES):
                    return True  # C method on mutable object
                # Check for immutable-self methods that read arguments (e.g. str.join)
                method_name = getattr(item, "__name__", None)
                if method_name in _IMMUTABLE_SELF_ARG_READERS:
                    return True

        # Strategy 2b: Container constructors (list, dict, etc.)
        if item_type is type and item in _CONTAINER_CONSTRUCTORS:
            return True

        # Strategy 3: Wrapper descriptors on mutable types
        if item_type is _WRAPPER_DESCRIPTOR_TYPE:
            objclass = getattr(item, "__objclass__", None)
            if objclass is not None and not issubclass(objclass, _IMMUTABLE_TYPES):
                return True

    return False


def _is_shared_opcode(code: Any, instruction_offset: int) -> bool:
    """Check whether an opcode at the given offset might access shared state.

    Returns True for opcodes in _SHARED_ACCESS_OPCODES, with a special case
    for BINARY_OP: only subscript variants (``[]``, ``NB_SUBSCR``) are shared;
    arithmetic variants (+, -, *, etc.) are not.

    CALL opcodes are NOT checked here — they require shadow stack inspection
    via ``_call_might_report_access`` and are handled separately in the callbacks.
    """
    instrs = _get_instructions(code)
    instr = instrs.get(instruction_offset)
    if instr is None:
        return False
    op = instr.opname
    # CALL opcodes are handled separately (need shadow stack inspection)
    if op in _CALL_OPCODES:
        return False
    if op not in _SHARED_ACCESS_OPCODES:
        return False
    # BINARY_OP is only shared when it's a subscript operation (3.14+)
    if op == "BINARY_OP":
        argrepr = instr.argrepr
        if not argrepr or ("[" not in argrepr and "NB_SUBSCR" not in argrepr.upper()):
            return False
    return True


def _process_opcode(
    frame: Any,
    scheduler: DporScheduler,
    thread_id: int,
) -> None:
    """Process a single opcode, updating the shadow stack and reporting accesses.

    Handles opcodes across Python 3.10-3.14, including:
    - 3.13: LOAD_FAST_LOAD_FAST, STORE_FAST_STORE_FAST
    - 3.14: LOAD_FAST_BORROW, LOAD_FAST_BORROW_LOAD_FAST_BORROW,
            STORE_FAST_LOAD_FAST, STORE_FAST_MAYBE_NULL, LOAD_FAST_AND_CLEAR,
            LOAD_SMALL_INT, BINARY_SUBSCR removal
    """
    code = frame.f_code
    instrs = _get_instructions(code)
    instr = instrs.get(frame.f_lasti)
    if instr is None:
        return

    shadow = scheduler.get_shadow_stack(id(frame))
    op = instr.opname
    engine = scheduler.engine
    execution = scheduler.execution
    elock = scheduler._engine_lock
    recorder = scheduler.trace_recorder
    sids = scheduler._stable_ids

    # === LOAD instructions: push values onto the shadow stack ===

    if op in ("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"):
        # LOAD_FAST_BORROW is new in 3.14: same semantics as LOAD_FAST
        # but uses a borrowed reference internally.
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    elif op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
        # New in 3.13: pushes two locals in one instruction.
        # LOAD_FAST_BORROW_LOAD_FAST_BORROW is the 3.14 variant using
        # borrowed references internally (same observable semantics).
        # argval is a tuple of two variable names.
        argval = instr.argval
        if isinstance(argval, tuple) and len(argval) == 2:
            shadow.push(frame.f_locals.get(argval[0]))
            shadow.push(frame.f_locals.get(argval[1]))
        else:
            shadow.push(None)
            shadow.push(None)

    elif op == "LOAD_GLOBAL":
        val = frame.f_globals.get(instr.argval)
        if val is None:
            # Fall back to builtins (setattr, getattr, type, dict, object, etc.)
            _fb = getattr(frame, "f_builtins", None)
            if isinstance(_fb, dict):
                val = _fb.get(instr.argval)
        # On 3.11+, LOAD_GLOBAL with NULL flag (bit 0 of arg) pushes an
        # extra NULL slot.  The order differs by version:
        #   3.11-3.13: [NULL, value]  (NULL below, value on TOS)
        #   3.14+:     [value, NULL]  (value below, NULL on TOS)
        if _PY_VERSION >= (3, 11) and instr.arg is not None and instr.arg & 1:
            if _PY_VERSION >= (3, 14):
                shadow.push(val)
                shadow.push(None)
            else:
                shadow.push(None)
                shadow.push(val)
        else:
            shadow.push(val)
        # Report a READ on the module's globals dict for this variable name.
        # Without this, LOAD_GLOBAL/STORE_GLOBAL races are invisible to DPOR.
        # Uses first-access semantics so ``global += 1`` (LOAD_GLOBAL + STORE_GLOBAL)
        # doesn't overwrite the position of an earlier read, enabling DPOR to
        # insert into the wakeup tree between the read and a subsequent write.
        _report_first_read(engine, execution, thread_id, frame.f_globals, instr.argval, elock, sids)

    elif op == "LOAD_NAME":
        # Used in exec/eval code (module-level scope).  Like LOAD_GLOBAL
        # but checks locals first, then globals, then builtins.
        val = frame.f_locals.get(instr.argval)
        if val is None:
            val = frame.f_globals.get(instr.argval)
        if val is None:
            _fb = getattr(frame, "f_builtins", None)
            if isinstance(_fb, dict):
                val = _fb.get(instr.argval)
        shadow.push(val)

    elif op == "LOAD_DEREF":
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)
        # Report a READ on closure cell/free variables so DPOR sees
        # cross-thread conflicts.  Using code as the identity works because
        # threads sharing a closure function share the same code object.
        varname = instr.argval
        if varname in code.co_freevars or varname in code.co_cellvars:
            _report_first_read(engine, execution, thread_id, code, varname, elock, sids)

    elif op in ("LOAD_CONST", "LOAD_CONST_IMMORTAL", "LOAD_CONST_MORTAL"):
        shadow.push(instr.argval)

    elif op == "LOAD_SMALL_INT":
        # New in 3.14: pushes a small integer (the oparg itself).
        shadow.push(instr.arg)

    # === Stack manipulation ===

    elif op == "COPY":
        n = instr.arg
        if n is not None and len(shadow.stack) >= n:
            shadow.push(shadow.stack[-n])
        else:
            shadow.push(None)

    elif op == "SWAP":
        n = instr.arg
        if n is not None and len(shadow.stack) >= n:
            shadow.stack[-1], shadow.stack[-n] = shadow.stack[-n], shadow.stack[-1]

    # --- Python 3.10 stack manipulation (replaced by COPY/SWAP in 3.11) ---

    elif op == "DUP_TOP":
        shadow.push(shadow.peek())

    elif op == "DUP_TOP_TWO":
        b = shadow.peek(0)
        a = shadow.peek(1)
        shadow.push(a)
        shadow.push(b)

    elif op == "ROT_TWO":
        if len(shadow.stack) >= 2:
            shadow.stack[-1], shadow.stack[-2] = shadow.stack[-2], shadow.stack[-1]

    elif op == "ROT_THREE":
        if len(shadow.stack) >= 3:
            shadow.stack[-1], shadow.stack[-2], shadow.stack[-3] = (
                shadow.stack[-2],
                shadow.stack[-3],
                shadow.stack[-1],
            )

    elif op == "ROT_FOUR":
        if len(shadow.stack) >= 4:
            shadow.stack[-1], shadow.stack[-2], shadow.stack[-3], shadow.stack[-4] = (
                shadow.stack[-2],
                shadow.stack[-3],
                shadow.stack[-4],
                shadow.stack[-1],
            )

    # === Attribute access: the instructions we care about most ===

    elif op == "LOAD_ATTR":
        obj = shadow.pop()
        attr = instr.argval
        _report_read(engine, execution, thread_id, obj, attr, elock, sids)
        # Also report on obj.__dict__ so LOAD_ATTR conflicts with
        # STORE_SUBSCR on the same __dict__ (cross-path detection).
        # Off by default: doubles wakeup tree insertions for rare benefit.
        if getattr(scheduler, "_track_dunder_dict_accesses", False) and obj is not None:
            try:
                _obj_dict = object.__getattribute__(obj, "__dict__")
                _report_read(engine, execution, thread_id, _obj_dict, attr, elock, sids)
            except AttributeError:
                pass
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=attr, obj=obj)
        if obj is not None:
            try:
                val = _safe_getattr(obj, attr)
                shadow.push(val)
                # When the loaded value is a mutable object (but NOT a bound
                # method), report a WEAK READ on the object itself.  This
                # detects cases where a container is read indirectly —
                # e.g. passed to len() or iterated — creating a conflict
                # with C-level method WRITEs (append, add, etc.) reported
                # by the CALL handler below.
                #
                # We use weak_read (not read) so that loading a dict just
                # to subscript it doesn't conflict with STORE_SUBSCR's
                # weak_write on disjoint keys.
                #
                # We skip bound methods (loading .append is not a container
                # read) and immutable types (no mutation possible).
                if val is not None and type(val) is not _BUILTIN_METHOD_TYPE and isinstance(val, (list, dict, set)):
                    _report_weak_read(engine, execution, thread_id, val, "__cmethods__", elock, sids)
            except Exception:
                shadow.push(None)
        else:
            shadow.push(None)
        # On 3.12+, LOAD_ATTR with method flag (bit 0 of arg) pushes an
        # extra self/NULL slot after the callable, matching LOAD_METHOD's
        # stack layout.  On 3.11, LOAD_ATTR with the method flag has
        # stack_effect=0 (no extra push), so we skip it there.
        if _PY_VERSION >= (3, 12) and instr.arg is not None and instr.arg & 1:
            shadow.push(None)

    elif op == "STORE_ATTR":
        obj = shadow.pop()  # TOS = object
        _val = shadow.pop()  # TOS1 = value
        _report_write(engine, execution, thread_id, obj, instr.argval, elock, sids)
        # Also report on obj.__dict__ so STORE_ATTR conflicts with
        # STORE_SUBSCR on the same __dict__ (cross-path detection).
        # Off by default: doubles wakeup tree insertions for rare benefit.
        if getattr(scheduler, "_track_dunder_dict_accesses", False) and obj is not None:
            try:
                _obj_dict = object.__getattribute__(obj, "__dict__")
                _report_write(engine, execution, thread_id, _obj_dict, instr.argval, elock, sids)
            except AttributeError:
                pass
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=instr.argval, obj=obj)

    elif op == "LOAD_METHOD":
        # Python 3.10 only (replaced by LOAD_ATTR with method flag in 3.11+).
        # Pops owner, pushes (method, self/NULL) — net stack effect +1.
        obj = shadow.pop()
        attr = instr.argval
        _report_read(engine, execution, thread_id, obj, attr, elock, sids)
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=attr, obj=obj)
        if obj is not None:
            try:
                shadow.push(_safe_getattr(obj, attr))
            except Exception:
                shadow.push(None)
        else:
            shadow.push(None)
        # Extra push for the self/NULL slot (LOAD_METHOD pushes 2 values).
        shadow.push(None)

    elif op == "DELETE_ATTR":
        obj = shadow.pop()
        _report_write(engine, execution, thread_id, obj, instr.argval, elock, sids)
        if recorder is not None and obj is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=instr.argval, obj=obj)

    elif op == "LOAD_SPECIAL":
        # New in 3.14: replaces LOAD_ATTR for ``__enter__`` / ``__exit__``
        # in ``with`` statements.  Pops owner, pushes (attr, self_or_null).
        # Stack effect = +1 (−1 pop + 2 push).
        _special_names = {0: "__enter__", 1: "__exit__"}
        _arg = instr.arg if instr.arg is not None else -1
        attr = _special_names.get(_arg, f"__special_{_arg}__")
        obj = shadow.pop()
        _report_read(engine, execution, thread_id, obj, attr, elock, sids)
        if obj is not None:
            try:
                val = _safe_getattr(obj, attr)
                shadow.push(val)
            except Exception:
                shadow.push(None)
        else:
            shadow.push(None)
        shadow.push(None)  # self_or_null slot

    # === Subscript access (dict/list operations) ===

    elif op == "BINARY_SUBSCR":
        # Present on 3.10-3.13. Removed in 3.14 (replaced by BINARY_OP
        # with subscript oparg).
        key = shadow.pop()
        container = shadow.pop()
        _kname = _subscript_key_name(key)
        _report_read(engine, execution, thread_id, container, _kname, elock, sids)
        # Container-level read for conflict with C-methods and different subscript keys.
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_read(engine, execution, thread_id, container, "__cmethods__", elock, sids)
            # For slice accesses, also report reads on individual element keys
            # so DPOR sees per-element conflicts with STORE_SUBSCR writes.
            _expand_slice_reads(engine, execution, thread_id, container, key, elock, sids)
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=_kname, obj=container)
        shadow.push(None)

    elif op == "STORE_SUBSCR":
        key = shadow.pop()
        container = shadow.pop()
        _val = shadow.pop()
        _kname = _subscript_key_name(key)
        _report_write(engine, execution, thread_id, container, _kname, elock, sids)
        # Report a container-level weak-write so subscript writes conflict
        # with C-method reads (e.g. len(), iteration) but two subscript
        # writes on disjoint keys do NOT conflict with each other.
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_weak_write(engine, execution, thread_id, container, "__cmethods__", elock, sids)
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=_kname, obj=container)

    elif op == "BINARY_SLICE":
        # New in 3.12 (replaces BINARY_SUBSCR for slice operations).
        # Stack: [container, start, stop] → [result]
        _stop = shadow.pop()
        _start = shadow.pop()
        container = shadow.pop()
        _report_read(engine, execution, thread_id, container, "__slice__", elock, sids)
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_read(engine, execution, thread_id, container, "__cmethods__", elock, sids)
            _expand_slice_reads(engine, execution, thread_id, container, slice(_start, _stop), elock, sids)
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name="__slice__", obj=container)
        shadow.push(None)

    elif op == "STORE_SLICE":
        # New in 3.12 (replaces STORE_SUBSCR for slice operations).
        # Stack: [value, container, start, stop] → []
        _stop = shadow.pop()
        _start = shadow.pop()
        container = shadow.pop()
        _val = shadow.pop()
        _report_write(engine, execution, thread_id, container, "__slice__", elock, sids)
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_weak_write(engine, execution, thread_id, container, "__cmethods__", elock, sids)
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name="__slice__", obj=container)

    elif op == "DELETE_SUBSCR":
        key = shadow.pop()
        container = shadow.pop()
        _kname = _subscript_key_name(key)
        _report_write(engine, execution, thread_id, container, _kname, elock, sids)
        # Container-level write for delete too (regular last-access semantics).
        if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
            _report_write(engine, execution, thread_id, container, "__cmethods__", elock, sids)
        if recorder is not None and container is not None:
            recorder.record(thread_id, frame, opcode=op, access_type="write", attr_name=_kname, obj=container)

    # === Arithmetic and binary operations ===

    elif op == "BINARY_OP":
        # On 3.14, BINARY_OP also handles subscript operations (replacing
        # the removed BINARY_SUBSCR). Check the argrepr for "[]" / subscript.
        argrepr = instr.argrepr
        if argrepr and ("[" in argrepr or "NB_SUBSCR" in argrepr.upper()):
            key = shadow.pop()
            container = shadow.pop()
            _kname = _subscript_key_name(key)
            _report_read(engine, execution, thread_id, container, _kname, elock, sids)
            # Container-level read for subscript access (same as BINARY_SUBSCR).
            if container is not None and not isinstance(container, _IMMUTABLE_TYPES):
                _report_read(engine, execution, thread_id, container, "__cmethods__", elock, sids)
                # For slice accesses, also report reads on individual element keys
                _expand_slice_reads(engine, execution, thread_id, container, key, elock, sids)
            if recorder is not None and container is not None:
                recorder.record(thread_id, frame, opcode=op, access_type="read", attr_name=_kname, obj=container)
            shadow.push(None)
        else:
            shadow.pop()
            shadow.pop()
            shadow.push(None)

    elif op.startswith(("INPLACE_", "BINARY_")):
        # Python 3.10 INPLACE_ADD, INPLACE_SUBTRACT, etc. and
        # BINARY_ADD, BINARY_MULTIPLY, etc. All pop 2, push 1.
        # (BINARY_OP and BINARY_SUBSCR are already handled above.)
        shadow.pop()
        shadow.pop()
        shadow.push(None)

    # === Store instructions ===

    elif op == "STORE_GLOBAL":
        shadow.pop()
        # Report a WRITE on the module's globals dict for this variable name.
        _report_write(engine, execution, thread_id, frame.f_globals, instr.argval, elock, sids)

    elif op == "STORE_DEREF":
        shadow.pop()
        # Report a WRITE on closure cell/free variables.
        varname = instr.argval
        if varname in code.co_freevars or varname in code.co_cellvars:
            _report_write(engine, execution, thread_id, code, varname, elock, sids)

    elif op == "STORE_FAST":
        shadow.pop()

    elif op == "STORE_FAST_STORE_FAST":
        # New in 3.13: pops two values.
        shadow.pop()
        shadow.pop()

    elif op == "STORE_FAST_LOAD_FAST":
        # New in 3.14: stores TOS into one local, then loads another local.
        # Net stack effect is 0 (pop one, push one) but the fallback handler
        # sees effect=0 and does nothing, losing the pushed value.
        argval = instr.argval
        shadow.pop()
        if isinstance(argval, tuple) and len(argval) == 2:
            shadow.push(frame.f_locals.get(argval[1]))
        else:
            shadow.push(None)

    elif op == "STORE_FAST_MAYBE_NULL":
        # New in 3.14: like STORE_FAST but tolerates NULL on TOS.
        shadow.pop()

    elif op == "LOAD_FAST_AND_CLEAR":
        # Used in comprehensions: loads a local then clears it.
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    # === Return/pop ===

    elif op in ("RETURN_VALUE", "RETURN_CONST"):
        shadow.pop()

    elif op == "POP_TOP":
        shadow.pop()

    # === Build instructions (slice, list, etc.) ===

    elif op == "BUILD_SLICE":
        # BUILD_SLICE pops 2 or 3 items (start, stop, [step]) and pushes a slice object.
        # The fallback stack_effect handler adjusts the size correctly, but we need
        # the ACTUAL slice object on the shadow stack so that BINARY_SUBSCR can
        # detect slice accesses and expand them into per-element reads.
        argc = instr.arg or 2
        items: list[Any] = [shadow.pop() for _ in range(argc)]
        items.reverse()
        try:
            if argc == 2:
                shadow.push(slice(items[0], items[1]))
            else:
                shadow.push(slice(items[0], items[1], items[2]))
        except (TypeError, ValueError):
            shadow.push(None)

    # === Iterator operations ===
    # GET_ITER creates an iterator from a container. We record the mapping
    # so that FOR_ITER can report reads on the original container.

    elif op == "GET_ITER":
        # stack: [iterable] → [iterator]
        # GET_ITER pops the iterable and pushes an iterator (stack effect = 0).
        # We record the iterable→iterator mapping via a mutable marker on the
        # shadow stack so that FOR_ITER can report per-element reads.
        iterable = shadow.peek()
        if iterable is not None and not isinstance(iterable, _IMMUTABLE_TYPES):
            shadow.pop()
            # Mutable list marker: [tag, container, iteration_counter]
            # The counter tracks which element index FOR_ITER is reading,
            # enabling per-element conflict detection with STORE_SUBSCR writes.
            shadow.push(["__iter_source__", iterable, 0])
            # Report a container-level read now.  On Python 3.10, list
            # comprehensions are compiled as nested functions, so FOR_ITER
            # runs in a child frame with a fresh shadow stack that won't
            # see the __iter_source__ marker.  The read here ensures the
            # conflict with STORE_SUBSCR weak-writes is recorded in the
            # current frame before the iterator crosses frame boundaries.
            _report_first_read(engine, execution, thread_id, iterable, "__cmethods__", elock, sids)
        else:
            shadow.pop()
            shadow.push(None)

    elif op == "FOR_ITER":
        # stack: [iterator] → [iterator, next_value] or [−iterator] (exhausted)
        # FOR_ITER calls __next__ on the iterator. If the iterator was created
        # from a mutable container (tracked via GET_ITER), report reads on it.
        # stack effect = +1 (pushes the yielded value; TOS is the iterator).
        # We peek at the iterator marker to find the underlying container.
        top = shadow.peek()
        if isinstance(top, list) and len(top) == 3 and top[0] == "__iter_source__":
            _iter_container = top[1]
            _iter_counter = top[2]
            if _iter_container is not None and not isinstance(_iter_container, _IMMUTABLE_TYPES):
                # Per-element read using the iteration counter as the key.
                # For lists, counter 0, 1, 2... matches STORE_SUBSCR keys "0", "1", "2"...
                # This creates per-element conflicts enabling fine-grained interleaving.
                _report_first_read(engine, execution, thread_id, _iter_container, repr(_iter_counter), elock, sids)
                # Coarse-grained read for conflict with C-method writes (append,
                # insert, etc.) and other container-level operations.  Uses first-access
                # semantics: all FOR_ITER reads point back to the first iteration's
                # scheduling point, so wakeup insertions target that earliest position
                # instead of cascading backward through each iteration.
                _report_first_read(engine, execution, thread_id, _iter_container, "__cmethods__", elock, sids)
            # Increment counter for next iteration (mutable list, in-place update).
            top[2] = _iter_counter + 1
        shadow.push(None)  # push the yielded value

    elif op == "END_FOR":
        # End of for loop — pops the exhausted iterator value.
        shadow.pop()

    elif op == "POP_ITER":
        # Python 3.14: pops the iterator itself at end of for loop.
        shadow.pop()

    # === Function/method calls ===

    elif op == "PRECALL":
        # Python 3.11 only.  PRECALL is a no-op for the evaluation stack
        # (it's a cache/optimization hint for the interpreter).  However,
        # dis.stack_effect reports a negative effect equal to -argc, which
        # would corrupt the shadow stack if handled by the fallback.
        pass

    elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_KW", "CALL_FUNCTION_KW", "CALL_FUNCTION_EX"):
        # Detect C-level method calls and classify as read or write.
        #
        # Three detection strategies, tried in order:
        # 1. Passthrough builtins (setattr, getattr, operator.setitem, etc.)
        #    — operate on their arguments rather than __self__
        # 2. Bound C methods (list.append, dict.update, etc.)
        #    — __self__ is the mutable target object
        # 3. Wrapper descriptors (object.__setattr__, dict.__setitem__, etc.)
        #    — unbound C type methods, first argument is the target
        argc = instr.arg or 0
        scan_depth = min(argc + 3, len(shadow.stack))
        _call_handled = False
        # When a container constructor (enumerate, zip, etc.) wraps a mutable
        # iterable, we save the source container so that GET_ITER → FOR_ITER
        # can report per-element reads on the underlying container.
        _constructor_source: Any = None

        for i in range(scan_depth):
            item = shadow.stack[-(i + 1)]
            if item is None:
                continue
            item_type = type(item)

            # --- Strategy 1: Passthrough builtins ---
            # These are builtins whose __self__ is a module (builtins, operator)
            # but that access their ARGUMENTS.  Identified by id(function).
            _pt = _PASSTHROUGH_BUILTINS.get(id(item))
            if _pt is not None:
                _pt_kind, _pt_obj_idx, _pt_name_idx = _pt
                # Arguments are always in the top `argc` positions on the stack
                # regardless of Python version (3.10: [func, args], 3.11-3.13:
                # [NULL, func, args], 3.14: [func, NULL, args]).
                _slen = len(shadow.stack)
                _obj_depth = argc - _pt_obj_idx
                if argc >= _pt_obj_idx + 1 and 0 < _obj_depth <= _slen:
                    _pt_target = shadow.stack[-_obj_depth]
                    _pt_attr: Any = "__cmethods__"
                    if _pt_name_idx is not None and argc >= _pt_name_idx + 1:
                        _name_depth = argc - _pt_name_idx
                        if 0 < _name_depth <= _slen:
                            _raw = shadow.stack[-_name_depth]
                            if isinstance(_raw, str):
                                _pt_attr = _raw
                    if _pt_target is not None and not isinstance(_pt_target, _IMMUTABLE_TYPES):
                        if _pt_kind == "read":
                            _report_read(engine, execution, thread_id, _pt_target, _pt_attr, elock, sids)
                        else:
                            _report_write(engine, execution, thread_id, _pt_target, _pt_attr, elock, sids)
                _call_handled = True
                break

            # --- Strategy 2: Bound C methods (existing behavior) ---
            if item_type is _BUILTIN_METHOD_TYPE or item_type is _METHOD_WRAPPER_TYPE:
                self_obj = getattr(item, "__self__", None)
                if self_obj is not None and not isinstance(self_obj, _IMMUTABLE_TYPES):
                    method_name = getattr(item, "__name__", None)
                    if method_name in _C_METHOD_READ_ONLY:
                        _report_read(engine, execution, thread_id, self_obj, "__cmethods__", elock, sids)
                    else:
                        _report_write(engine, execution, thread_id, self_obj, "__cmethods__", elock, sids)
                    _call_handled = True
                    break
                # __self__ is immutable (e.g. str, module) — check if the method
                # iterates its first argument (e.g. str.join reads the iterable).
                if self_obj is not None:
                    method_name = getattr(item, "__name__", None)
                    if method_name in _IMMUTABLE_SELF_ARG_READERS and argc >= 1 and argc <= len(shadow.stack):
                        _arg_target = shadow.stack[-argc]
                        if _arg_target is not None and not isinstance(_arg_target, _IMMUTABLE_TYPES):
                            _report_read(engine, execution, thread_id, _arg_target, "__cmethods__", elock, sids)
                        _call_handled = True
                        break
                    # Otherwise fall through to continue scan

            # --- Strategy 2b: Type constructors that iterate arguments ---
            # list(iterable), dict(iterable), bytes(iterable), enumerate(iterable),
            # zip(iter1, iter2), map(func, iterable), filter(func, iterable), etc.
            if item_type is type and item in _CONTAINER_CONSTRUCTORS:
                # Report a READ on each mutable argument (they get iterated).
                # Also save the first mutable arg as the "source container" so that
                # if this constructor result is iterated via FOR_ITER, the reads
                # are attributed to the underlying container (not the wrapper).
                for _ci in range(argc):
                    _c_depth = argc - _ci
                    if _c_depth < 1 or _c_depth > len(shadow.stack):
                        continue
                    _c_arg = shadow.stack[-_c_depth]
                    if _c_arg is not None and not isinstance(_c_arg, _IMMUTABLE_TYPES):
                        _report_read(engine, execution, thread_id, _c_arg, "__cmethods__", elock, sids)
                        if _constructor_source is None:
                            _constructor_source = _c_arg
                _call_handled = True
                break

            # --- Strategy 3: Wrapper descriptors (unbound C type methods) ---
            if item_type is _WRAPPER_DESCRIPTOR_TYPE:
                objclass = getattr(item, "__objclass__", None)
                if objclass is not None and not issubclass(objclass, _IMMUTABLE_TYPES):
                    # First argument (self) is always at the bottom of the argc args
                    if argc >= 1 and argc <= len(shadow.stack):
                        _wd_target = shadow.stack[-argc]
                        if _wd_target is not None and not isinstance(_wd_target, _IMMUTABLE_TYPES):
                            method_name = getattr(item, "__name__", None)
                            if method_name in _C_METHOD_READ_ONLY:
                                _report_read(engine, execution, thread_id, _wd_target, "__cmethods__", elock, sids)
                            else:
                                _report_write(engine, execution, thread_id, _wd_target, "__cmethods__", elock, sids)
                _call_handled = True
                break

        # Standard stack effect handling.
        try:
            effect = dis.stack_effect(instr.opcode, instr.arg or 0)
            # On Python 3.11, PRECALL reported a stack effect of -argc
            # but we handle it as a no-op (it doesn't touch the real
            # stack).  Compensate by adding the missing pops to CALL.
            if _PY_VERSION[:2] == (3, 11) and op == "CALL" and argc > 0:
                effect -= argc
            for _ in range(max(0, -effect)):
                shadow.pop()
            for _ in range(max(0, effect)):
                shadow.push(None)
        except (ValueError, TypeError):
            shadow.clear()

        # CALL replaces the callable/args with the return value. Plain
        # stack_effect accounting leaves the bottom-most operand in place,
        # which can be the callable itself on 3.10+ and pollute subsequent
        # attribute/subscript tracking.
        if shadow.stack:
            shadow.stack[-1] = None
        else:
            shadow.push(None)

        # Fixup: when a container constructor (enumerate, zip, map, etc.) wraps
        # a mutable iterable, replace the None result on TOS with the source
        # container.  This way GET_ITER picks it up and FOR_ITER can report
        # per-element reads on the underlying container during iteration.
        if _constructor_source is not None and shadow.stack:
            shadow.stack[-1] = _constructor_source

    else:
        # Fallback: use dis.stack_effect for unknown opcodes.
        # This handles PUSH_NULL, RESUME, and any version-specific
        # opcodes we don't explicitly handle.
        try:
            effect = dis.stack_effect(instr.opcode, instr.arg or 0)
            for _ in range(max(0, -effect)):
                shadow.pop()
            for _ in range(max(0, effect)):
                shadow.push(None)
        except (ValueError, TypeError):
            shadow.clear()


# ---------------------------------------------------------------------------
# DPOR Bytecode Runner
# ---------------------------------------------------------------------------


class DporBytecodeRunner:
    """Runs threads under DPOR-controlled bytecode-level interleaving.

    Uses sys.monitoring (PEP 669) on Python 3.12+ for thread-safe opcode
    instrumentation. Falls back to sys.settrace on 3.10-3.11.
    """

    # sys.monitoring tool ID for DPOR (use PROFILER to avoid conflict with debuggers)
    _TOOL_ID: int | None = None

    def __init__(
        self,
        scheduler: DporScheduler,
        detect_io: bool = True,
        preload_bridge: _PreloadBridge | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.detect_io = detect_io
        self._preload_bridge = preload_bridge
        self.threads: list[threading.Thread] = []
        self.errors: dict[int, Exception] = {}
        self._lock_patched = False
        self._io_patched = False
        self._monitoring_active = False

    def _patch_locks(self) -> None:
        install_wait_for_graph()
        patch_locks()
        self._lock_patched = True

    def _unpatch_locks(self) -> None:
        if self._lock_patched:
            unpatch_locks()
            uninstall_wait_for_graph()
            self._lock_patched = False

    def _patch_io(self) -> None:
        if not self.detect_io:
            return
        patch_io()
        patch_sql()
        patch_redis()
        self._io_patched = True

    def _unpatch_io(self) -> None:
        if self._io_patched:
            unpatch_redis()
            unpatch_sql()
            unpatch_io()
            self._io_patched = False

    # --- sys.settrace backend (3.10-3.11) ---

    def _make_trace(self, thread_id: int) -> Callable[..., Any]:
        scheduler = self.scheduler
        _detect_io = scheduler._detect_io

        def trace(frame: Any, event: str, arg: Any) -> Any:
            if scheduler._finished or scheduler._error:
                return None

            if event == "call":
                filename = frame.f_code.co_filename
                if _should_trace_file(filename):
                    # Skip dynamically generated code (<string>, etc.)
                    # unless its caller is user code.  In I/O mode, skip
                    # all dynamic code unconditionally.  Exception: in
                    # python -c mode, functions defined in the -c string
                    # have filename "<string>" but ARE user code.
                    if _is_dynamic_code(filename) and not _is_cmdline_user_code(filename, frame.f_globals):
                        if _detect_io:
                            return None
                        caller = frame.f_back
                        if caller is None or not _should_trace_file(caller.f_code.co_filename):
                            return None
                    frame.f_trace_opcodes = True
                    return trace
                return None

            if event == "opcode":
                # --- Scheduling coarsening ---
                # Non-shared opcodes only manipulate thread-local state.
                # Process the shadow stack directly without a scheduler yield.
                # CALL opcodes need shadow stack inspection to decide.
                # When detect_io is enabled, skip coarsening for CALL opcodes
                # because I/O events arrive from C-level LD_PRELOAD interception
                # and need scheduling points around open/read/write calls.
                if not _is_shared_opcode(frame.f_code, frame.f_lasti):
                    instrs = _get_instructions(frame.f_code)
                    instr = instrs.get(frame.f_lasti)
                    if instr is not None and instr.opname in _CALL_OPCODES:
                        if _detect_io:
                            scheduler.report_and_wait(frame, thread_id)
                            frame.f_locals  # noqa: B018  — refresh f_locals before LocalsToFast
                            return trace
                        shadow = scheduler.get_shadow_stack(id(frame))
                        argc = instr.arg or 0
                        if _call_might_report_access(shadow, argc):
                            scheduler.report_and_wait(frame, thread_id)
                            frame.f_locals  # noqa: B018  — refresh f_locals before LocalsToFast
                            return trace
                    _process_opcode(frame, scheduler, thread_id)
                    return trace
                scheduler.report_and_wait(frame, thread_id)
                # CPython 3.10-3.11 bug workaround: after our trace callback
                # returns, CPython calls PyFrame_LocalsToFast(frame, 1) which
                # copies f_locals dict values back to cell/free variable cells
                # (see CPython ceval.c call_trace).  If this thread waited in
                # report_and_wait while another thread modified a shared cell,
                # the stale f_locals snapshot would overwrite the new value.
                # Re-accessing frame.f_locals triggers PyFrame_FastToLocals
                # (see CPython frameobject.c frame_getlocals), refreshing the
                # snapshot so LocalsToFast writes back the current value.
                # This is not needed on 3.12+ where PEP 667 replaced f_locals
                # with a proxy and removed LocalsToFast from the trace path.
                frame.f_locals  # noqa: B018  — refresh f_locals before LocalsToFast
                return trace

            if event == "return":
                scheduler.remove_shadow_stack(id(frame))
                return trace

            return trace

        return trace

    # --- sys.monitoring backend (3.12+) ---

    def _setup_monitoring(self) -> None:
        """Set up sys.monitoring INSTRUCTION events for all code objects."""
        if not _USE_SYS_MONITORING:
            return

        mon = sys.monitoring
        tool_id = mon.PROFILER_ID  # type: ignore[attr-defined]
        DporBytecodeRunner._TOOL_ID = tool_id

        # Defensively free the tool ID if it's still claimed from a previous
        # run that was interrupted (e.g. by pytest-timeout) before
        # _teardown_monitoring could call free_tool_id.  Without this,
        # use_tool_id raises "tool N is already in use" and every subsequent
        # test in the suite fails.
        try:
            mon.use_tool_id(tool_id, "frontrun._dpor")  # type: ignore[attr-defined]
        except ValueError:
            # Tool ID still held from a previous interrupted run — force cleanup.
            mon.set_events(tool_id, 0)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_START, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_RETURN, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.INSTRUCTION, None)  # type: ignore[attr-defined]
            mon.free_tool_id(tool_id)  # type: ignore[attr-defined]
            mon.use_tool_id(tool_id, "frontrun._dpor")  # type: ignore[attr-defined]

        mon.set_events(tool_id, mon.events.PY_START | mon.events.PY_RETURN | mon.events.INSTRUCTION)  # type: ignore[attr-defined]

        scheduler = self.scheduler

        _detect_io = scheduler._detect_io

        def handle_py_start(code: Any, instruction_offset: int) -> Any:
            # Only use mon.DISABLE for code that should *never* be traced
            # (stdlib, site-packages, frontrun internals).  Do NOT disable
            # for transient conditions like scheduler._finished — DISABLE
            # permanently removes INSTRUCTION events from the code object,
            # corrupting monitoring state for subsequent DPOR iterations
            # and tests that share the same tool ID.
            if not _should_trace_file(code.co_filename):
                return mon.DISABLE  # type: ignore[attr-defined]
            # In I/O-detection mode, skip dynamically generated code
            # (e.g. dataclass __init__ from exec/compile in libraries).
            # These create thousands of extra scheduling points that
            # drown out I/O-based wakeup tree entries.  Safe to DISABLE
            # because each exec() creates a fresh code object.
            if _detect_io and code.co_filename.startswith("<"):
                return mon.DISABLE  # type: ignore[attr-defined]
            return None

        def handle_py_return(code: Any, instruction_offset: int, retval: Any) -> Any:
            if not _should_trace_file(code.co_filename):
                return None
            thread_id = getattr(_dpor_tls, "thread_id", None)
            if thread_id is not None and getattr(_dpor_tls, "scheduler", None) is scheduler:
                frame = sys._getframe(1)
                scheduler.remove_shadow_stack(id(frame))
            return None

        def handle_instruction(code: Any, instruction_offset: int) -> Any:
            if scheduler._finished or scheduler._error:
                return None
            if not _should_trace_file(code.co_filename):
                return None
            # Skip dynamically generated code (<string>, etc.) unless its
            # caller is user code.  Libraries use exec/compile internally
            # (dataclass __init__, SQLAlchemy methods) creating thousands
            # of scheduling points in non-user code.  In I/O mode, skip
            # all dynamic code unconditionally.  Exception: in python -c
            # mode, functions defined in the -c string ARE user code.
            if _is_dynamic_code(code.co_filename):
                frame = sys._getframe(1)
                if not _is_cmdline_user_code(code.co_filename, frame.f_globals):
                    if _detect_io:
                        return None
                    caller = frame.f_back
                    if caller is None or not _should_trace_file(caller.f_code.co_filename):
                        return None

            thread_id = getattr(_dpor_tls, "thread_id", None)
            if thread_id is None:
                return None

            # Guard against zombie threads from a previous DporBytecodeRunner
            # whose monitoring was torn down and replaced by ours.  The zombie
            # still has TLS from the old scheduler, but this closure captures
            # the *new* scheduler.  Letting it through would call engine
            # methods on the wrong execution, causing PyO3 borrow conflicts.
            if getattr(_dpor_tls, "scheduler", None) is not scheduler:
                return None

            # --- Scheduling coarsening ---
            # Non-shared opcodes (LOAD_FAST, STORE_FAST, BINARY_OP +, COPY,
            # SWAP, etc.) only manipulate thread-local state.  Process the
            # shadow stack directly without entering the scheduler, avoiding
            # a yield that would create an unnecessary DPOR scheduling point.
            #
            # CALL opcodes need special handling: _is_shared_opcode returns
            # False for them so we check the shadow stack to see if the
            # callable is a C-method that would report shared access.
            if not _is_shared_opcode(code, instruction_offset):
                # Check if this is a CALL opcode that needs shadow stack inspection.
                # When detect_io is enabled, all CALL opcodes must yield because
                # I/O events arrive from C-level LD_PRELOAD interception.
                instrs = _get_instructions(code)
                instr = instrs.get(instruction_offset)
                frame = sys._getframe(1)
                if instr is not None and instr.opname in _CALL_OPCODES:
                    if _detect_io:
                        scheduler.report_and_wait(frame, thread_id)
                        return None
                    shadow = scheduler.get_shadow_stack(id(frame))
                    argc = instr.arg or 0
                    if _call_might_report_access(shadow, argc):
                        scheduler.report_and_wait(frame, thread_id)
                        return None
                    # No C-method detected; process shadow stack without yield
                    _process_opcode(frame, scheduler, thread_id)
                    return None
                _process_opcode(frame, scheduler, thread_id)
                return None

            # Use sys._getframe() to get the actual frame for _process_opcode.
            # report_and_wait runs _process_opcode and wait_for_turn under a
            # single lock so that engine.report_access() and engine.schedule()
            # cannot overlap on free-threaded builds.
            frame = sys._getframe(1)
            scheduler.report_and_wait(frame, thread_id)
            return None

        mon.register_callback(tool_id, mon.events.PY_START, handle_py_start)  # type: ignore[attr-defined]
        mon.register_callback(tool_id, mon.events.PY_RETURN, handle_py_return)  # type: ignore[attr-defined]
        mon.register_callback(tool_id, mon.events.INSTRUCTION, handle_instruction)  # type: ignore[attr-defined]
        self._monitoring_active = True

    def _teardown_monitoring(self) -> None:
        if not self._monitoring_active:
            return
        mon = sys.monitoring
        tool_id = DporBytecodeRunner._TOOL_ID
        if tool_id is not None:
            mon.set_events(tool_id, 0)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_START, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_RETURN, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.INSTRUCTION, None)  # type: ignore[attr-defined]
            mon.free_tool_id(tool_id)  # type: ignore[attr-defined]
        self._monitoring_active = False

    # --- Thread entry points ---

    def _setup_dpor_tls(self, thread_id: int) -> None:
        """Set up both shared cooperative TLS and DPOR-specific TLS."""
        scheduler = self.scheduler
        engine = scheduler.engine
        execution = scheduler.execution
        engine_lock = scheduler._engine_lock
        # Shared context for cooperative primitives
        set_context(self.scheduler, thread_id)

        # Sync reporter so cooperative Lock/RLock report to the DPOR engine.
        # Must hold the engine_lock to serialise PyO3 &mut self borrows.
        # Also handles block/unblock for cooperative lock spinning:
        #   "lock_wait"    → execution.block_thread  (DPOR skips this thread)
        #   "lock_acquire" → execution.unblock_thread (DPOR can schedule again)
        #   "lock_release" → unblock all waiters for this lock
        # XOR masks matching Rust DporEngine constants for virtual lock objects.
        lock_object_xor = 0x4C4F_434B_4C4F_434B  # "LOCKLOCK"
        lock_release_xor = 0x524C_5345_524C_5345  # "RLSERLSE"

        def _register_lock_in_reverse_map(stable_id: int, lock_obj: object) -> None:
            """Register virtual lock objects in the reverse map for human-readable race info."""
            rmap = _object_key_reverse_map
            if rmap is None:
                return
            type_name = type(lock_obj).__name__
            acq_key = stable_id ^ lock_object_xor
            rel_key = stable_id ^ lock_release_xor
            if acq_key not in rmap:
                rmap[acq_key] = f"{type_name}(id={stable_id}).acquire"
            if rel_key not in rmap:
                rmap[rel_key] = f"{type_name}(id={stable_id}).release"

        def _sync_reporter(event: str, obj_id: int, lock_obj: object) -> None:
            from frontrun._cooperative import _scheduler_tls

            # Use stable IDs so that lock identifiers are consistent across
            # DPOR executions (each execution creates fresh state via setup(),
            # so raw id() values change).  This is the same mechanism used for
            # shared-object access tracking.
            stable_lock_id = scheduler._stable_ids.get(lock_obj)
            _register_lock_in_reverse_map(stable_lock_id, lock_obj)

            # Set reentrancy guard so that GC-triggered __del__ chains
            # (e.g., redis.Redis.__del__) that acquire cooperative locks
            # will fall back to real blocking instead of re-entering the
            # scheduler.  See defect #7.
            _scheduler_tls._in_dpor_machinery = True
            try:
                if event == "lock_wait":
                    with engine_lock:
                        scheduler._lock_waiters.setdefault(stable_lock_id, set()).add(thread_id)
                        execution.block_thread(thread_id)
                        _trace_len_wait = (
                            len(execution.schedule_trace) if scheduler._lock_event_collector is not None else 0
                        )
                        _trace_snap_wait = (
                            list(execution.schedule_trace) if scheduler._lock_event_collector is not None else None
                        )
                    if scheduler._error is not None:
                        # Capture teardown boundary once, then suppress all further events.
                        if scheduler._deadlock_at is None:
                            scheduler._deadlock_at = _trace_len_wait
                        return
                    if _trace_snap_wait is not None:
                        from frontrun._report import LockEvent as _LockEvent

                        _wait_idx = next(
                            (i for i in range(len(_trace_snap_wait) - 1, -1, -1) if _trace_snap_wait[i] == thread_id),
                            max(0, len(_trace_snap_wait) - 1),
                        )
                        scheduler._lock_event_collector.append(  # type: ignore[union-attr]
                            _LockEvent(
                                schedule_index=_wait_idx, thread_id=thread_id, event_type="wait", lock_id=stable_lock_id
                            )
                        )
                    return
                if event == "lock_acquire":
                    with engine_lock:
                        waiter_set = scheduler._lock_waiters.get(stable_lock_id)
                        if waiter_set is not None and thread_id in waiter_set:
                            waiter_set.discard(thread_id)
                            execution.unblock_thread(thread_id)
                        _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                        engine.report_sync(execution, thread_id, "lock_acquire", stable_lock_id, _saved_path_id)
                        _trace_len_acq = (
                            len(execution.schedule_trace) if scheduler._lock_event_collector is not None else 0
                        )
                        _trace_snap_acq = (
                            list(execution.schedule_trace) if scheduler._lock_event_collector is not None else None
                        )
                    new_depth = getattr(_dpor_tls, "lock_depth", 0) + 1
                    _dpor_tls.lock_depth = new_depth
                    scheduler._lock_depth_by_thread[thread_id] = new_depth
                    if scheduler._error is not None:
                        if scheduler._deadlock_at is None:
                            scheduler._deadlock_at = _trace_len_acq
                        return
                    if _trace_snap_acq is not None:
                        from frontrun._report import LockEvent as _LockEvent

                        _acq_idx = next(
                            (i for i in range(len(_trace_snap_acq) - 1, -1, -1) if _trace_snap_acq[i] == thread_id),
                            max(0, len(_trace_snap_acq) - 1),
                        )
                        scheduler._lock_event_collector.append(  # type: ignore[union-attr]
                            _LockEvent(
                                schedule_index=_acq_idx,
                                thread_id=thread_id,
                                event_type="acquire",
                                lock_id=stable_lock_id,
                            )
                        )
                    return
                if event == "lock_release":
                    with engine_lock:
                        waiters = scheduler._lock_waiters.pop(stable_lock_id, set())
                        for waiter in waiters:
                            execution.unblock_thread(waiter)
                        _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                        engine.report_sync(execution, thread_id, "lock_release", stable_lock_id, _saved_path_id)
                        _trace_len_rel = (
                            len(execution.schedule_trace) if scheduler._lock_event_collector is not None else 0
                        )
                        _trace_snap_rel = (
                            list(execution.schedule_trace) if scheduler._lock_event_collector is not None else None
                        )
                    new_depth = max(0, getattr(_dpor_tls, "lock_depth", 1) - 1)
                    _dpor_tls.lock_depth = new_depth
                    scheduler._lock_depth_by_thread[thread_id] = new_depth
                    if scheduler._error is not None:
                        if scheduler._deadlock_at is None:
                            scheduler._deadlock_at = _trace_len_rel
                        return
                    if _trace_snap_rel is not None:
                        from frontrun._report import LockEvent as _LockEvent

                        _rel_idx = next(
                            (i for i in range(len(_trace_snap_rel) - 1, -1, -1) if _trace_snap_rel[i] == thread_id),
                            max(0, len(_trace_snap_rel) - 1),
                        )
                        scheduler._lock_event_collector.append(  # type: ignore[union-attr]
                            _LockEvent(
                                schedule_index=_rel_idx,
                                thread_id=thread_id,
                                event_type="release",
                                lock_id=stable_lock_id,
                            )
                        )
                    # Wake threads that may now be schedulable
                    with scheduler._condition:
                        scheduler._condition.notify_all()
                    return
                with engine_lock:
                    _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                    engine.report_sync(execution, thread_id, event, stable_lock_id, _saved_path_id)
            finally:
                _scheduler_tls._in_dpor_machinery = False

        set_sync_reporter(_sync_reporter)
        # DPOR-specific TLS for _process_opcode (shadow stacks, etc.)
        _dpor_tls.scheduler = self.scheduler
        _dpor_tls.thread_id = thread_id
        _dpor_tls.engine = engine
        _dpor_tls.execution = execution
        _dpor_tls._last_path_id = None  # set by report_and_wait after schedule()

        # IO detection: defer I/O reports while this thread is inside locks.
        # If another thread later reaches a real I/O boundary, it will flush
        # these deferred accesses before reporting its own I/O so the race
        # window is preserved without adding spurious paths for threads that
        # only block on the same lock.
        _dpor_tls.lock_depth = 0
        self.scheduler._lock_depth_by_thread[thread_id] = 0
        pending_io: list[tuple[int, str]] = []
        _dpor_tls.pending_io = pending_io
        self.scheduler._pending_io_by_thread[thread_id] = pending_io

        # Register OS TID → DPOR thread ID mapping for LD_PRELOAD events.
        if self._preload_bridge is not None:
            self._preload_bridge.register_thread(threading.get_native_id(), thread_id)

        from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id

        set_dpor_scheduler(self.scheduler)
        set_dpor_thread_id(thread_id)

        if self.detect_io:
            _recorder = scheduler.trace_recorder

            def _io_reporter(resource_id: str, kind: str) -> None:
                object_key = _make_object_key(hash(resource_id), resource_id)
                pending: list[tuple[int, str]] = _dpor_tls.pending_io
                pending.append((object_key, kind))
                # Record I/O event in the trace for human-readable output
                if _recorder is not None:
                    _frame = sys._getframe(1)
                    # Walk up to find user code (skip frontrun internals)
                    while _frame is not None and not _should_trace_file(_frame.f_code.co_filename):
                        _frame = _frame.f_back
                    if _frame is not None:
                        chain = build_call_chain(_frame, filter_fn=_should_trace_file)
                        _recorder.record(
                            thread_id=thread_id,
                            frame=_frame,
                            opcode="IO",
                            access_type=kind,
                            attr_name=resource_id,
                            call_chain=chain,
                        )

            set_io_reporter(_io_reporter)

    def _teardown_dpor_tls(self) -> None:
        """Clean up both shared and DPOR-specific TLS."""
        _tid = getattr(_dpor_tls, "thread_id", None)
        if self._preload_bridge is not None:
            self._preload_bridge.unregister_thread(threading.get_native_id())

        # Flush any orphaned pending_io before tearing down TLS.
        # _capture_insert_id adds the indexical alias WRITE to pending_io
        # AFTER the INSERT's report_and_wait scheduling point.  If the INSERT
        # is the last operation before the thread exits, the alias WRITE
        # would be silently discarded.  Flushing here ensures the DPOR engine
        # sees all I/O events, even those reported after the last scheduling
        # point.
        _pending = getattr(_dpor_tls, "pending_io", None)
        if _pending:
            _tid = getattr(_dpor_tls, "thread_id", None)
            _engine = getattr(_dpor_tls, "engine", None)
            _execution = getattr(_dpor_tls, "execution", None)
            if _tid is not None and _engine is not None and _execution is not None:
                _elock = self.scheduler._engine_lock
                for _obj_key, _io_kind in _pending:
                    with _elock:
                        _engine.report_io_access(_execution, _tid, _obj_key, _io_kind)
                _pending.clear()

        if self.detect_io:
            set_io_reporter(None)
        from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id

        set_dpor_scheduler(None)
        set_dpor_thread_id(None)
        clear_context()
        set_sync_reporter(None)
        _dpor_tls.scheduler = None
        _dpor_tls.thread_id = None
        _dpor_tls.engine = None
        _dpor_tls.execution = None
        _dpor_tls.lock_depth = 0
        _dpor_tls.pending_io = []
        if _tid is not None:
            self.scheduler._lock_depth_by_thread.pop(_tid, None)
            self.scheduler._pending_io_by_thread.pop(_tid, None)

    def _run_thread_settrace(
        self,
        thread_id: int,
        func: Callable[..., None],
        args: tuple[Any, ...],
    ) -> None:
        """Thread entry using sys.settrace (3.10-3.11)."""
        try:
            self._setup_dpor_tls(thread_id)

            trace_fn = self._make_trace(thread_id)
            sys.settrace(trace_fn)
            func(*args)
        except SchedulerAbort:
            pass  # scheduler already has the error; just exit cleanly
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            sys.settrace(None)
            self._teardown_dpor_tls()
            self.scheduler.mark_done(thread_id)

    def _run_thread_monitoring(
        self,
        thread_id: int,
        func: Callable[..., None],
        args: tuple[Any, ...],
    ) -> None:
        """Thread entry using sys.monitoring (3.12+)."""
        try:
            self._setup_dpor_tls(thread_id)

            func(*args)
        except SchedulerAbort:
            pass  # scheduler already has the error; just exit cleanly
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)
        finally:
            self._teardown_dpor_tls()
            self.scheduler.mark_done(thread_id)

    def run(
        self,
        funcs: list[Callable[..., None]],
        args: list[tuple[Any, ...]] | None = None,
        timeout: float = 10.0,
    ) -> None:
        if args is None:
            args = [() for _ in funcs]

        use_monitoring = _USE_SYS_MONITORING
        if use_monitoring:
            self._setup_monitoring()
            run_thread = self._run_thread_monitoring
        else:
            run_thread = self._run_thread_settrace

        try:
            for i, (func, a) in enumerate(zip(funcs, args)):
                t = threading.Thread(
                    target=run_thread,
                    args=(i, func, a),
                    name=f"dpor-{i}",
                    daemon=True,
                )
                self.threads.append(t)

            for t in self.threads:
                t.start()

            deadline = time.monotonic() + timeout
            for t in self.threads:
                remaining = max(0, deadline - time.monotonic())
                t.join(timeout=remaining)

            # Signal scheduler to abort if any threads are still alive.
            # On free-threaded Python, condition notifications can be lost
            # and threads may still be blocked in wait_for_turn.
            alive = [t for t in self.threads if t.is_alive()]
            if alive:
                self.scheduler._error = TimeoutError(f"Timed out waiting for {len(alive)} thread(s) to complete")
                with self.scheduler._condition:
                    self.scheduler._condition.notify_all()
                # Give threads a brief grace period to notice the error
                for t in alive:
                    t.join(timeout=0.5)
        finally:
            if use_monitoring:
                self._teardown_monitoring()

        if self.errors:
            first_error = next(iter(self.errors.values()))
            if not isinstance(first_error, TimeoutError):
                raise first_error


# ---------------------------------------------------------------------------
# Fixed-schedule replay
# ---------------------------------------------------------------------------


def _run_dpor_schedule(
    schedule: list[int],
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    timeout: float = 5.0,
    detect_io: bool = False,
    deadlock_timeout: float = 5.0,
    trace_recorder: TraceRecorder | None = None,
) -> T:
    """Replay a DPOR schedule using the DPOR runner rather than OpcodeScheduler."""
    scheduler = _ReplayDporScheduler(
        schedule,
        len(threads),
        deadlock_timeout=deadlock_timeout,
        trace_recorder=trace_recorder,
        detect_io=detect_io,
    )
    runner = DporBytecodeRunner(scheduler, detect_io=detect_io)

    runner._patch_locks()
    runner._patch_io()
    try:
        state = setup()

        def make_thread_func(thread_func: Callable[[T], None], thread_state: T) -> Callable[[], None]:
            def thread_wrapper() -> None:
                thread_func(thread_state)

            return thread_wrapper

        funcs: list[Callable[[], None]] = [make_thread_func(t, state) for t in threads]
        try:
            runner.run(funcs, timeout=timeout)
        except TimeoutError:
            pass
        if isinstance(scheduler._error, DeadlockError):
            raise scheduler._error
    finally:
        runner._unpatch_io()
        runner._unpatch_locks()
    return state


def _reproduce_dpor_counterexample(
    *,
    schedule_list: list[int],
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    timeout_per_run: float,
    deadlock_timeout: float,
    reproduce_on_failure: int,
    lock_timeout: int | None,
    invariant: Callable[[T], bool] | None = None,
    detect_io: bool = True,
) -> tuple[int, int]:
    """Measure how often a DPOR counterexample reproduces under the DPOR runner.

    Reproduction runs with the same IO interception (SQL, Redis) as
    exploration so that the replay scheduler can enforce interleavings at
    IO boundaries, not just bytecode boundaries.
    """
    from frontrun._preload_io import _set_preload_pipe_fd
    from frontrun._redis_client import patch_redis, set_redis_replay_mode, unpatch_redis
    from frontrun._sql_cursor import get_lock_timeout, patch_sql, set_lock_timeout, unpatch_sql

    _set_preload_pipe_fd(-1)
    if reproduce_on_failure <= 0:
        return reproduce_on_failure, 0

    _prev_lt = get_lock_timeout()
    _replay_lock_timeout = lock_timeout if lock_timeout is not None else 5000
    set_lock_timeout(_replay_lock_timeout)
    patch_sql()
    patch_redis()
    set_redis_replay_mode(True)
    successes = 0
    try:
        for _ in range(reproduce_on_failure):
            try:
                replay_state = _run_dpor_schedule(
                    schedule_list,
                    setup,
                    threads,
                    timeout=timeout_per_run,
                    detect_io=detect_io,
                    deadlock_timeout=deadlock_timeout,
                )
                if invariant is not None and not invariant(replay_state):
                    successes += 1
            except DeadlockError:
                if invariant is None:
                    successes += 1
            except Exception:
                pass  # timeout / crash during replay — not a reproduction
    finally:
        set_redis_replay_mode(False)
        unpatch_redis()
        unpatch_sql()
        set_lock_timeout(_prev_lt)
    return reproduce_on_failure, successes


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def explore_dpor(
    setup: Callable[[], T],
    threads: list[Callable[[T], None]],
    invariant: Callable[[T], bool],
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    timeout_per_run: float = 5.0,
    stop_on_first: bool = True,
    detect_io: bool = True,
    deadlock_timeout: float = 5.0,
    reproduce_on_failure: int = 10,
    total_timeout: float | None = None,
    warn_nondeterministic_sql: bool = True,
    lock_timeout: int | None = None,
    trace_packages: list[str] | None = None,
    track_dunder_dict_accesses: bool = False,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> InterleavingResult:
    """Systematically explore interleavings using DPOR.

    This is the DPOR replacement for ``explore_interleavings()``. Instead of
    random sampling, it uses the DPOR algorithm to explore only distinct
    interleavings (modulo independent operation reordering).

    Args:
        setup: Creates fresh shared state for each execution.
        threads: List of callables, each receiving the shared state.
        invariant: Predicate over shared state; must be True after all
            threads complete.
        max_executions: Safety limit on total executions (None = unlimited).
        preemption_bound: Limit on preemptions per execution. 2 catches most
            bugs. None = unbounded (full DPOR).
        max_branches: Maximum scheduling points per execution.
        timeout_per_run: Timeout for each individual run.
        stop_on_first: If True (default), stop exploring as soon as the
            first invariant violation is found.  Set to False to collect
            all failing interleavings.
        detect_io: Automatically detect socket/file I/O operations and
            report them as resource accesses (default True).  Two threads
            accessing the same endpoint or file will be treated as
            conflicting, enabling DPOR to explore their orderings.
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).
        reproduce_on_failure: When a counterexample is found, replay the
            same schedule this many times to measure reproducibility
            (default 10).  Set to 0 to skip reproduction testing.
        total_timeout: Maximum total time in seconds for the entire
            exploration (default None = unlimited).  When exceeded, returns
            results gathered so far.
        warn_nondeterministic_sql: If True (default), raise
            :class:`~frontrun.common.NondeterministicSQLError` when SQL
            INSERT statements are detected but ``lastrowid`` capture
            failed (e.g. psycopg2 without RETURNING).  Set to False to
            suppress.  When capture succeeds, INSERTs use stable
            indexical resource IDs automatically.
        lock_timeout: If set, automatically execute
            ``SET lock_timeout = '<N>ms'`` on every new PostgreSQL
            connection created through the patched ``psycopg2.connect``
            (or ``psycopg.connect``).  This prevents the cooperative
            scheduler from deadlocking when two threads contend on the
            same PostgreSQL row lock (defect #6).  Value is in
            milliseconds; 2000 (2 seconds) is a good default.
        trace_packages: List of package name patterns (fnmatch syntax) to
            trace in addition to user code.  By default, code in
            site-packages is skipped.  Use this to include specific
            installed packages, e.g. ``["django_*", "mylib.*"]``.
        track_dunder_dict_accesses: If True, report accesses on ``obj.__dict__``
            in addition to direct attribute accesses.  This catches the
            rare case where one thread uses ``self.x = v`` and another
            uses ``self.__dict__['x'] = v``, but doubles wakeup tree
            insertions and can cause combinatorial explosion.  Default
            False.

    Returns:
        InterleavingResult with exploration statistics and any counterexample found.

    .. note::

       When running under **pytest**, this function requires the
       ``frontrun`` CLI wrapper (``frontrun pytest ...``) or the
       ``--frontrun-patch-locks`` flag.  Without it, the test is
       automatically skipped.
    """
    _require_frontrun_env("explore_dpor")
    if trace_packages is not None:
        _set_active_trace_filter(_TraceFilter(trace_packages))
    num_threads = len(threads)
    pb = None if preemption_bound is None else preemption_bound
    me = None if max_executions is None else max_executions
    engine = PyDporEngine(
        num_threads=num_threads,
        preemption_bound=pb,
        max_branches=max_branches,
        max_executions=me,
    )

    result = InterleavingResult(property_holds=True)
    stable_ids = StableObjectIds()
    # Shared lock serialising ALL PyO3 calls to engine/execution objects.
    # On free-threaded Python, PyO3 &mut self borrows panic rather than
    # block when contested, so we need a Python-level lock shared across
    # worker threads, the sync reporter, and the main loop.
    engine_lock = real_lock()
    total_deadline = time.monotonic() + total_timeout if total_timeout is not None else None

    # Set up the LD_PRELOAD → DPOR bridge for C-level I/O detection.
    # When code under test uses C extensions that call libc send()/recv()
    # directly (e.g. psycopg2/libpq), the Python-level monkey-patches in
    # _io_detection can't see those calls.  The LD_PRELOAD library
    # intercepts them at the C level and writes events to a pipe.  The
    # IOEventDispatcher reads the pipe in a background thread and the
    # _PreloadBridge routes events to the correct DPOR thread for
    # conflict analysis.
    preload_dispatcher = None
    preload_bridge: _PreloadBridge | None = None
    if detect_io:
        from frontrun._preload_io import IOEventDispatcher

        preload_dispatcher = IOEventDispatcher()
        preload_bridge = _PreloadBridge(dispatcher=preload_dispatcher)
        preload_dispatcher.add_listener(preload_bridge.listener)
        preload_dispatcher.start()

    clear_sql_metadata()

    # Warm SQL parsers (sqlglot) BEFORE the first _patch_locks() call.
    # sqlglot creates a module-level _import_lock = threading.RLock() on
    # first import.  If that import happens after _patch_locks() replaces
    # threading.RLock with CooperativeRLock, the lock becomes cooperative.
    # If a worker thread is then killed while holding it (e.g. timeout),
    # the underlying real lock stays locked forever, causing deadlocks in
    # later phases (_reproduce_dpor_counterexample).  Warming here ensures
    # the lock is a real RLock.
    if detect_io:
        from frontrun._sql_cursor import _warm_sql_parsers

        _warm_sql_parsers()

    # Inject SET lock_timeout on new PG connections (defect #6 workaround).
    from frontrun._sql_cursor import get_lock_timeout, set_lock_timeout

    prev_lock_timeout = get_lock_timeout()
    if lock_timeout is not None:
        set_lock_timeout(lock_timeout)

    # Set up report collection if --frontrun-report is active
    from frontrun._report import (
        _MAX_RECORDED_EXECUTIONS,
        ExecutionRecord,
        ExplorationReport,
        _global_report_path,
        generate_html_report,
    )

    def _build_race_info(raw_races: list[tuple[int, int, int, int | None]]) -> list[dict[str, Any]] | None:
        if not raw_races:
            return None
        rmap = _object_key_reverse_map or {}
        result = []
        # Track (prev_step, current_step, thread_id) to deduplicate dict.__dict__
        # shadows of real attribute accesses
        seen_steps: set[tuple[int, int, int]] = set()
        for r in raw_races:
            obj_name = rmap.get(r[3], f"object {r[3]}") if r[3] is not None else None
            key = (r[0], r[1], r[2])
            # Skip dict.X entries when we already have a Type.X entry for the same steps
            if obj_name and obj_name.startswith("dict."):
                if key in seen_steps:
                    continue
            seen_steps.add(key)
            result.append(
                {
                    "prev_step": r[0],
                    "current_step": r[1],
                    "thread_id": r[2],
                    "object": obj_name,
                }
            )
        return result or None

    global _object_key_reverse_map

    report_path = _global_report_path
    report: ExplorationReport | None = None
    if report_path is not None:
        report = ExplorationReport(
            num_threads=num_threads,
            thread_names=[f"Thread {i}" for i in range(num_threads)],
        )
        _object_key_reverse_map = {}

    try:
        while True:
            if total_deadline is not None and time.monotonic() > total_deadline:
                break
            clear_insert_tracker()
            stable_ids.reset_for_execution()
            with engine_lock:
                execution = engine.begin_execution()
            recorder = TraceRecorder()
            # Clear bridge state for this new execution.
            if preload_bridge is not None:
                preload_bridge.clear()
            # Set up switch point collection for the report
            _collecting_report = report is not None and len(report.executions) < _MAX_RECORDED_EXECUTIONS
            switch_points: list[Any] = [] if _collecting_report else []
            scheduler = DporScheduler(
                engine,
                execution,
                num_threads,
                engine_lock=engine_lock,
                deadlock_timeout=deadlock_timeout,
                trace_recorder=recorder,
                preload_bridge=preload_bridge,
                detect_io=detect_io,
                stable_ids=stable_ids,
                switch_point_collector=switch_points if _collecting_report else None,
                track_dunder_dict_accesses=track_dunder_dict_accesses,
            )
            runner = DporBytecodeRunner(scheduler, detect_io=detect_io, preload_bridge=preload_bridge)

            runner._patch_locks()
            runner._patch_io()
            try:
                state = setup()

                def make_thread_func(thread_func: Callable[[T], None], s: T) -> Callable[[], None]:
                    def wrapper() -> None:
                        thread_func(s)

                    return wrapper

                funcs = [make_thread_func(t, state) for t in threads]
                try:
                    runner.run(funcs, timeout=timeout_per_run)
                except TimeoutError:
                    pass
            finally:
                runner._unpatch_io()
                runner._unpatch_locks()

            result.num_explored += 1
            if on_progress is not None:
                on_progress(result.num_explored, None)

            # Check for deadlock before running the invariant — a deadlock
            # means the program never completed, so the invariant can never be
            # satisfied.  Report it as a property violation with a clear message.
            if isinstance(scheduler._error, DeadlockError):
                result.property_holds = False
                with engine_lock:
                    schedule = execution.schedule_trace
                schedule_list = list(schedule)
                result.failures.append((result.num_explored, schedule_list))
                if result.counterexample is None:
                    result.counterexample = schedule_list
                if result.explanation is None:
                    result.explanation = (
                        f"Deadlock detected after {result.num_explored} interleaving(s).\n\n"
                        f"{scheduler._error.cycle_description}"
                    )

                # Replay the counterexample to measure reproducibility
                if reproduce_on_failure > 0 and result.reproduction_attempts == 0:
                    attempts, successes = _reproduce_dpor_counterexample(
                        schedule_list=schedule_list,
                        setup=setup,
                        threads=threads,
                        timeout_per_run=timeout_per_run,
                        deadlock_timeout=deadlock_timeout,
                        reproduce_on_failure=reproduce_on_failure,
                        lock_timeout=lock_timeout,
                        invariant=None,
                        detect_io=detect_io,
                    )
                    result.reproduction_attempts = attempts
                    result.reproduction_successes = successes

                    from frontrun._preload_io import _set_preload_pipe_fd

                    if preload_dispatcher is not None and preload_dispatcher._write_fd is not None:
                        _set_preload_pipe_fd(preload_dispatcher._write_fd)

                if stop_on_first:
                    with _INSTR_CACHE_LOCK:
                        _INSTR_CACHE.clear()
                    if report is not None and report_path is not None:
                        # Record the failing execution before generating report
                        if _collecting_report:
                            with engine_lock:
                                schedule_trace_r = list(execution.schedule_trace)
                                raw_races = engine.pending_races()
                            race_info = _build_race_info(raw_races)
                            report.executions.append(
                                ExecutionRecord(
                                    index=len(report.executions),
                                    schedule_trace=schedule_trace_r,
                                    switch_points=switch_points,
                                    invariant_held=False,
                                    was_deadlock=True,
                                    race_info=race_info,
                                    step_events=scheduler._step_event_collector or {},
                                    lock_events=scheduler._lock_event_collector or [],
                                    deadlock_at=scheduler._deadlock_at,
                                    deadlock_cycle_description=getattr(scheduler._error, "cycle_description", None),
                                )
                            )
                        generate_html_report(report, report_path)
                    return result

            if warn_nondeterministic_sql:
                check_uncaptured_inserts()

            if not invariant(state):
                result.property_holds = False
                with engine_lock:
                    schedule = execution.schedule_trace
                schedule_list = list(schedule)
                result.failures.append((result.num_explored, schedule_list))
                if result.counterexample is None:
                    result.counterexample = schedule_list

                # Replay the counterexample to measure reproducibility
                if reproduce_on_failure > 0 and result.reproduction_attempts == 0:
                    attempts, successes = _reproduce_dpor_counterexample(
                        schedule_list=schedule_list,
                        setup=setup,
                        threads=threads,
                        timeout_per_run=timeout_per_run,
                        deadlock_timeout=deadlock_timeout,
                        reproduce_on_failure=reproduce_on_failure,
                        lock_timeout=lock_timeout,
                        invariant=invariant,
                        detect_io=detect_io,
                    )
                    result.reproduction_attempts = attempts
                    result.reproduction_successes = successes

                    # Re-enable pipe writes for subsequent DPOR executions.
                    from frontrun._preload_io import _set_preload_pipe_fd

                    if preload_dispatcher is not None and preload_dispatcher._write_fd is not None:
                        _set_preload_pipe_fd(preload_dispatcher._write_fd)

                if result.explanation is None:
                    result.explanation = format_trace(
                        recorder.events,
                        num_threads=num_threads,
                        num_explored=result.num_explored,
                        reproduction_attempts=result.reproduction_attempts,
                        reproduction_successes=result.reproduction_successes,
                    )
                if result.sql_anomaly is None:
                    result.sql_anomaly = classify_sql_anomaly(recorder.events)
                if stop_on_first:
                    # Clear cache before returning
                    with _INSTR_CACHE_LOCK:
                        _INSTR_CACHE.clear()
                    if report is not None and report_path is not None:
                        if _collecting_report:
                            with engine_lock:
                                schedule_trace_r2 = list(execution.schedule_trace)
                                raw_races2 = engine.pending_races()
                            race_info2 = _build_race_info(raw_races2)
                            report.executions.append(
                                ExecutionRecord(
                                    index=len(report.executions),
                                    schedule_trace=schedule_trace_r2,
                                    switch_points=switch_points,
                                    invariant_held=False,
                                    was_deadlock=False,
                                    race_info=race_info2,
                                    step_events=scheduler._step_event_collector or {},
                                    lock_events=scheduler._lock_event_collector or [],
                                    deadlock_at=scheduler._deadlock_at,
                                )
                            )
                        generate_html_report(report, report_path)
                    return result

            # Clear instruction cache between executions to avoid stale code ids
            with _INSTR_CACHE_LOCK:
                _INSTR_CACHE.clear()

            # Collect report data before next_execution() consumes pending races
            if _collecting_report and report is not None:
                with engine_lock:
                    schedule_trace = list(execution.schedule_trace)
                    raw_races = engine.pending_races()
                race_info = _build_race_info(raw_races)
                was_deadlock = isinstance(scheduler._error, DeadlockError)
                # Check if this specific execution failed: it was appended to failures
                # with the current num_explored as its execution number
                this_exec_failed = any(n == result.num_explored for n, _ in result.failures)
                invariant_held = not was_deadlock and not this_exec_failed
                report.executions.append(
                    ExecutionRecord(
                        index=len(report.executions),
                        schedule_trace=schedule_trace,
                        switch_points=switch_points,
                        invariant_held=invariant_held,
                        was_deadlock=was_deadlock,
                        race_info=race_info,
                        step_events=scheduler._step_event_collector or {},
                        lock_events=scheduler._lock_event_collector or [],
                        deadlock_at=scheduler._deadlock_at,
                        deadlock_cycle_description=getattr(scheduler._error, "cycle_description", None)
                        if was_deadlock
                        else None,
                    )
                )

            with engine_lock:
                if not engine.next_execution():
                    break
    finally:
        _set_active_trace_filter(None)
        set_lock_timeout(prev_lock_timeout)
        if preload_dispatcher is not None:
            preload_dispatcher.stop()

    # Generate HTML report if requested
    if report is not None and report_path is not None:
        generate_html_report(report, report_path)
    _object_key_reverse_map = None

    return result
