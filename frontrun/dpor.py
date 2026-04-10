"""
Bytecode-tracing DPOR (Dynamic Partial Order Reduction) for frontrun.

This module implements systematic interleaving exploration using DPOR.
Shares cooperative threading primitives and shadow-stack infrastructure
with bytecode.py (random exploration).

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

import linecache
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from frontrun._cooperative import (
    clear_context,
    patch_locks,
    patch_sleep,
    real_condition,
    real_lock,
    set_context,
    set_sync_reporter,
    unpatch_locks,
    unpatch_sleep,
)
from frontrun._opcode_observer import (
    ShadowStack,
    StableObjectIds,
    _CALL_OPCODES,
    _call_might_report_access,
    _get_instructions,
    _is_shared_opcode,
    _make_object_key,
    _process_opcode,
    _register_object_key,
    clear_instr_cache,
    get_object_key_reverse_map,
    set_object_key_reverse_map,
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
    is_sql_endpoint_suppressed,
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
from frontrun.common import InterleavingResult, check_serializability_violation

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
# Thread-local state for the DPOR scheduler
# ---------------------------------------------------------------------------

_dpor_tls = threading.local()


def _append_unique_lock_event(lock_events: list[Any], event: Any) -> None:
    """Append a lock event unless it duplicates the immediately previous one."""
    if lock_events:
        last = lock_events[-1]
        if (
            last.schedule_index == event.schedule_index
            and last.thread_id == event.thread_id
            and last.event_type == event.event_type
            and last.lock_id == event.lock_id
        ):
            return
    lock_events.append(event)


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
        # Skip events to known SQL/Redis socket endpoints — the cursor/client
        # layer already reports at a higher granularity (table/row/key level).
        # Also skip socket events from permanently-suppressed SQL threads (covers
        # the race window where connect() events arrive before the endpoint is
        # registered).  Non-socket events (file I/O) always pass through even
        # for SQL threads, so DPOR can detect non-SQL conflicts.
        is_socket = event.resource_id.startswith("socket:")
        if is_sql_endpoint_suppressed(event.resource_id):
            return
        if is_socket and is_tid_suppressed(event.tid):
            return
        if is_redis_tid_suppressed(event.tid):
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
        self._pending_io_by_thread: dict[int, list[tuple[int, str, bool]]] = {}
        # Per-thread lock nesting depth mirrored from TLS for cross-thread
        # deferred-I/O flush decisions.
        self._lock_depth_by_thread: dict[int, int] = {}

        # IO trace: records (thread_id, resource_id) in IO execution order.
        # Populated by after_io() under the condition lock.  See defect #16.
        self._io_trace: list[tuple[int, str]] = []
        # Explicit Python-level I/O boundary currently in progress. While set,
        # the owning thread keeps the scheduler turn until after_io() runs.
        self._active_io_thread: int | None = None
        self._next_thread_after_io: int | None = None
        # Like the I/O path, cooperative lock/semaphore retries need to keep
        # the scheduler turn until the real non-blocking probe completes.
        # Otherwise free-threaded builds turn the probe into an OS-level race
        # between multiple awakened waiters.
        self._active_sync_thread: int | None = None
        self._next_thread_after_sync: int | None = None

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

    def before_sync_retry(self, thread_id: int) -> bool:
        """Wait for a turn and keep it through one external sync probe."""
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                while True:
                    if self._finished or self._error:
                        return False

                    if self._active_sync_thread is not None and self._active_sync_thread != thread_id:
                        pass
                    elif self._current_thread == thread_id:
                        next_thread = self._schedule_next()
                        _pp = self._last_scheduled_path_id
                        if _pp is not None:
                            _dpor_tls._last_path_id = _pp
                        self._active_sync_thread = thread_id
                        self._next_thread_after_sync = next_thread
                        self._current_thread = thread_id
                        self._condition.notify_all()
                        return True

                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._current_thread in self._threads_done:
                            next_thread = self._schedule_next()
                            self._current_thread = next_thread
                            if next_thread is None:
                                self._finished = True
                            self._condition.notify_all()
                            continue
                        self._error = TimeoutError(
                            f"DPOR sync deadlock: waiting for thread {thread_id}, current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        return False
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def after_sync_retry(self, thread_id: int) -> None:
        with self._condition:
            if self._active_sync_thread == thread_id:
                self._active_sync_thread = None
                self._current_thread = self._next_thread_after_sync
                self._next_thread_after_sync = None
                if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                    self._finished = True
                self._condition.notify_all()

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

    def _flush_pending_io_for_unlocked(self, flush_thread_id: int, *, allow_inside_lock: bool = False) -> None:
        pending_io = self._pending_io_by_thread.get(flush_thread_id)
        if not pending_io:
            return
        inside_lock = self._lock_depth_by_thread.get(flush_thread_id, 0) > 0
        if not allow_inside_lock and inside_lock:
            # Even though we defer most events, synced events
            # (Python-level Redis/SQL) can be flushed now — they use
            # dpor_vv which respects lock HB, so recording them at
            # the in-lock path_id is correct and avoids position
            # misattribution from deferred flushing.
            engine = self.engine
            execution = self.execution
            remaining: list[tuple[int, str, bool]] = []
            for obj_key, io_kind, synced in pending_io:
                if synced:
                    with self._engine_lock:
                        engine.report_synced_io_access(execution, flush_thread_id, obj_key, io_kind)
                else:
                    remaining.append((obj_key, io_kind, synced))
            pending_io.clear()
            pending_io.extend(remaining)
            return
        engine = self.engine
        execution = self.execution
        for obj_key, io_kind, synced in pending_io:
            with self._engine_lock:
                if synced:
                    engine.report_synced_io_access(execution, flush_thread_id, obj_key, io_kind)
                else:
                    engine.report_io_access(execution, flush_thread_id, obj_key, io_kind)
        pending_io.clear()

    def _flush_other_pending_io_for_current_io_unlocked(self, thread_id: int) -> None:
        current_pending = self._pending_io_by_thread.get(thread_id)
        if not current_pending:
            return
        for other_thread_id, pending_io in list(self._pending_io_by_thread.items()):
            if other_thread_id == thread_id or not pending_io:
                continue
            # Another thread reached a real I/O boundary, so any
            # deferred I/O from this thread is now part of a real
            # race window and must become visible to the engine.
            self._flush_pending_io_for_unlocked(other_thread_id, allow_inside_lock=True)

    def _report_and_wait(self, frame: Any | None, thread_id: int) -> bool:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            # Set reentrancy guard so GC-triggered __del__ (e.g.,
            # redis.Redis.__del__) won't re-enter the scheduler.
            # See defect #7.
            _scheduler_tls._in_dpor_machinery = True
            try:
                # Merge LD_PRELOAD I/O events (C-level send/recv from e.g.
                # psycopg2) into the thread's pending_io list.  The preload
                # bridge buffers events from the pipe reader thread, keyed by
                # DPOR thread ID.
                _pending_io: list[tuple[int, str, bool]] | None = getattr(_dpor_tls, "pending_io", None)
                if self._preload_bridge is not None:
                    _preload_events = self._preload_bridge.drain(thread_id)
                    # Drop events for known SQL/Redis endpoints that raced past
                    # the listener() check due to async pipe delivery.
                    # Also drop socket events from permanently-suppressed SQL
                    # threads (belt-and-suspenders for connect-time race).
                    _is_sql_tid = is_tid_suppressed(threading.get_native_id())
                    if _preload_events:
                        _preload_events = [
                            ev
                            for ev in _preload_events
                            if not is_sql_endpoint_suppressed(ev[2])
                            and not (_is_sql_tid and ev[2].startswith("socket:"))
                        ]
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
                        # LD_PRELOAD events inside a Python lock should respect lock HB
                        # (synced=True uses dpor_vv).  Events outside locks use io_vv
                        # for TOCTOU detection.
                        _inside_lock = self._lock_depth_by_thread.get(thread_id, 0) > 0
                        _io_pairs = [(_key, _kind, _inside_lock) for _key, _kind, _, _, _ in _preload_events]
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
                            # Flush synced IO events even when skipping the
                            # scheduling point — they use dpor_vv and should
                            # be recorded at the correct in-lock position.
                            self._flush_pending_io_for_unlocked(thread_id)
                            return True
                        self._flush_other_pending_io_for_current_io_unlocked(thread_id)
                        # Flush deferred I/O only once this thread actually owns
                        # the current DPOR step. On free-threaded Python a thread
                        # can reach report_and_wait while another thread still owns
                        # the step; flushing earlier stamps the access onto the
                        # wrong path_id and can hide the wakeup tree insertion point.
                        self._flush_pending_io_for_unlocked(thread_id)
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
                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._current_thread in self._threads_done:
                            # Current thread is done, try scheduling again
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
                _scheduler_tls._in_dpor_machinery = False

    def before_io(self, thread_id: int, resource_id: str) -> None:
        """Enter an explicit Python-level I/O boundary.

        Unlike wait_for_turn(), this does not release the scheduler turn
        immediately. The current thread keeps running until after_io()
        records completion and hands off to the precomputed next thread.
        """
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                while True:
                    if self._finished or self._error:
                        return

                    if self._active_io_thread is not None and self._active_io_thread != thread_id:
                        pass
                    elif self._current_thread == thread_id:
                        current_pending = self._pending_io_by_thread.get(thread_id)
                        if (
                            current_pending
                            and self._lock_depth_by_thread.get(thread_id, 0) > 0
                            and self._all_other_live_threads_blocked_by_current(thread_id)
                        ):
                            # Keep ownership of the turn and avoid forcing a
                            # preemption inside a lock-held deadlock-avoidance path.
                            self._flush_pending_io_for_unlocked(thread_id)
                            self._active_io_thread = thread_id
                            self._next_thread_after_io = thread_id
                            self._condition.notify_all()
                            return

                        self._flush_other_pending_io_for_current_io_unlocked(thread_id)
                        self._flush_pending_io_for_unlocked(thread_id)
                        next_thread = self._schedule_next()
                        _pp = self._last_scheduled_path_id
                        if _pp is not None:
                            _dpor_tls._last_path_id = _pp
                        self._active_io_thread = thread_id
                        self._next_thread_after_io = next_thread
                        self._current_thread = thread_id
                        self._condition.notify_all()
                        return

                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._current_thread in self._threads_done:
                            next_thread = self._schedule_next()
                            self._current_thread = next_thread
                            if next_thread is None:
                                self._finished = True
                            self._condition.notify_all()
                            continue
                        self._error = TimeoutError(
                            f"DPOR I/O deadlock before {resource_id}: waiting for thread {thread_id}, "
                            f"current is {self._current_thread}"
                        )
                        self._condition.notify_all()
                        return
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def after_io(self, thread_id: int, resource_id: str) -> None:
        """Called immediately after an IO command completes.

        During exploration, records the IO event and releases the turn to
        the next thread chosen at before_io().
        """
        with self._condition:
            self._io_trace.append((thread_id, resource_id))
            if self._active_io_thread == thread_id:
                self._active_io_thread = None
                self._current_thread = self._next_thread_after_io
                self._next_thread_after_io = None
                if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                    self._finished = True
                self._condition.notify_all()

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
                # Report row-lock acquire to the DPOR engine so vector clocks
                # reflect the serialization from database row locking.
                _elock = getattr(self, "_engine_lock", None)
                if _elock is not None:
                    _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                    with _elock:
                        self.engine.report_sync(self.execution, thread_id, "lock_acquire", lock_int_id, _saved_path_id)

    def _release_row_locks_unlocked(self, thread_id: int) -> bool:
        """Remove row locks for *thread_id*. Caller must hold ``self._condition``."""
        from frontrun._deadlock import get_wait_for_graph

        held = self._thread_row_locks.pop(thread_id, None)
        if not held:
            return False
        graph = get_wait_for_graph()
        for r in held:
            self._active_row_locks.pop(r, None)
            lid = self._row_lock_ids.get(r)
            if graph is not None and lid is not None:
                graph.remove_holding(thread_id, lid, kind="row_lock")
            # Report row-lock release to the DPOR engine so vector clocks
            # reflect the serialization from database row locking.
            _elock = getattr(self, "_engine_lock", None)
            if lid is not None and _elock is not None:
                _saved_path_id = getattr(_dpor_tls, "_last_path_id", None)
                with _elock:
                    self.engine.report_sync(self.execution, thread_id, "lock_release", lid, _saved_path_id)
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

    def report_synced_io_access(self, execution: Any, thread_id: int, object_id: int, kind: str) -> None:
        return None

    def report_sync(
        self, execution: Any, thread_id: int, event_type: str, sync_id: int, path_id: int | None = None
    ) -> None:
        return None

    def register_resource_group(self, object_id: int, group_id: int) -> None:
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
        # Process the opcode to keep the shadow stack in sync with
        # exploration.  Without this, _call_might_report_access sees an
        # empty shadow stack during replay and skips CALL scheduling
        # points that existed during exploration, desynchronising the
        # schedule and preventing reproduction.
        if frame is not None:
            _process_opcode(frame, self, thread_id)
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

                if not self._condition.wait(timeout=self.deadlock_timeout):
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


class _IOAnchoredReplayScheduler(DporScheduler):
    """Replay using only IO boundaries as schedule anchors (defect #16).

    When detect_io=True, every CALL opcode is a scheduling point. If the code
    under test has state-dependent paths (e.g., early returns that skip Redis
    operations), the number of opcode-level scheduling points can change
    between exploration and replay, desynchronising the schedule.

    This scheduler uses a two-phase IO protocol:

    1. ``before_io(tid, resource_id)`` checks the next recorded anchor and
       blocks if it's not this thread's turn.

    2. ``after_io(tid, resource_id)`` (post-IO hook): called from the Redis
       interception layer *after* the command completes, inside the
       scheduler's condition lock.  Atomically records the IO event and
       switches ``current_thread`` to the next IO schedule entry.

    Between IO boundaries, one thread runs exclusively (enforced by opcode-
    level ``report_and_wait(frame, tid)`` blocking on ``current_thread``).
    """

    def __init__(
        self,
        io_schedule: list[tuple[int, str]],
        num_threads: int,
        *,
        deadlock_timeout: float = 5.0,
        trace_recorder: TraceRecorder | None = None,
        detect_io: bool = False,
    ) -> None:
        self._io_schedule = list(io_schedule)
        self._io_index = 0
        super().__init__(
            _ReplayEngine(),  # type: ignore[arg-type]
            _ReplayExecution(),  # type: ignore[arg-type]
            num_threads,
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
        )
        # _schedule_next() in super().__init__ consumed an entry.
        # Reset so the first IO scheduling point matches entry 0.
        self._io_index = 0
        # Set initial current_thread to the first IO schedule entry.
        self._current_thread = io_schedule[0][0] if io_schedule else 0

    def _schedule_next(self) -> int | None:
        """Override to use IO schedule instead of DPOR engine."""
        if self._io_index >= len(self._io_schedule):
            active = [t for t in range(self.num_threads) if t not in self._threads_done]
            return active[0] if active else None
        return self._io_schedule[self._io_index][0]

    def wait_for_turn(self, thread_id: int) -> bool:
        return self._wait_for_turn(thread_id)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        if frame is not None:
            _process_opcode(frame, self, thread_id)
        return self._wait_for_turn(thread_id)

    def before_io(self, thread_id: int, resource_id: str) -> None:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                while True:
                    if self._finished or self._error:
                        return

                    if self._current_thread in self._threads_done:
                        self._current_thread = self._schedule_next()
                        if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                            self._finished = True
                            self._condition.notify_all()
                            return

                    if self._active_io_thread is not None and self._active_io_thread != thread_id:
                        pass
                    elif self._current_thread == thread_id:
                        if self._io_index >= len(self._io_schedule):
                            self._error = RuntimeError(
                                "DPOR IO-anchored replay desynchronised: "
                                f"unexpected extra I/O anchor {(thread_id, resource_id)!r}"
                            )
                            self._condition.notify_all()
                            return

                        expected_tid, expected_resource_id = self._io_schedule[self._io_index]
                        if expected_tid != thread_id or expected_resource_id != resource_id:
                            self._error = RuntimeError(
                                "DPOR IO-anchored replay desynchronised: "
                                f"expected {(expected_tid, expected_resource_id)!r}, "
                                f"got {(thread_id, resource_id)!r}"
                            )
                            self._condition.notify_all()
                            return

                        self._io_index += 1
                        self._active_io_thread = thread_id
                        self._next_thread_after_io = self._schedule_next()
                        self._current_thread = thread_id
                        self._condition.notify_all()
                        return

                    if not self._condition.wait(timeout=self.deadlock_timeout):
                        if self._current_thread in self._threads_done:
                            self._current_thread = self._schedule_next()
                            if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                                self._finished = True
                            self._condition.notify_all()
                            continue
                        self._error = TimeoutError(
                            f"DPOR IO-anchored replay deadlock before {resource_id}: "
                            f"waiting for thread {thread_id}, current is {self._current_thread}, "
                            f"io_index={self._io_index}"
                        )
                        self._condition.notify_all()
                        return
            finally:
                _scheduler_tls._in_dpor_machinery = False

    def after_io(self, thread_id: int, resource_id: str) -> None:
        from frontrun._cooperative import _scheduler_tls

        with self._condition:
            _scheduler_tls._in_dpor_machinery = True
            try:
                if self._finished or self._error:
                    return
                if self._active_io_thread == thread_id:
                    self._active_io_thread = None
                    self._current_thread = self._next_thread_after_io
                    self._next_thread_after_io = None
                    if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                        self._finished = True
                self._condition.notify_all()
            finally:
                _scheduler_tls._in_dpor_machinery = False

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
                    self._condition.notify_all()
                    continue

                if self._active_io_thread == thread_id:
                    return True

                if self._current_thread == thread_id:
                    return True

                if not self._condition.wait(timeout=self.deadlock_timeout):
                    if self._current_thread in self._threads_done:
                        self._current_thread = self._schedule_next()
                        if self._current_thread is None and len(self._threads_done) >= self.num_threads:
                            self._finished = True
                        self._condition.notify_all()
                        continue
                    self._error = TimeoutError(
                        f"DPOR IO-anchored replay deadlock: waiting for thread {thread_id}, "
                        f"current is {self._current_thread}, io_index={self._io_index}"
                    )
                    self._condition.notify_all()
                    return False


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
        self._sleep_patched = False
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

    def _patch_sleep(self) -> None:
        patch_sleep()
        self._sleep_patched = True

    def _unpatch_sleep(self) -> None:
        if self._sleep_patched:
            unpatch_sleep()
            self._sleep_patched = False

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
            rmap = get_object_key_reverse_map()
            if rmap is None:
                return
            type_name = type(lock_obj).__name__
            acq_key = stable_id ^ lock_object_xor
            rel_key = stable_id ^ lock_release_xor
            if acq_key not in rmap:
                rmap[acq_key] = f"{type_name}(id={stable_id}).acquire"
            if rel_key not in rmap:
                rmap[rel_key] = f"{type_name}(id={stable_id}).release"

        def _append_lock_event(schedule_index: int, event_type: str, lock_id: int) -> None:
            if scheduler._lock_event_collector is None:
                return
            from frontrun._report import LockEvent as _LockEvent

            event = _LockEvent(
                schedule_index=schedule_index,
                thread_id=thread_id,
                event_type=event_type,
                lock_id=lock_id,
            )
            _append_unique_lock_event(scheduler._lock_event_collector, event)

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
                        _wait_idx = next(
                            (i for i in range(len(_trace_snap_wait) - 1, -1, -1) if _trace_snap_wait[i] == thread_id),
                            max(0, len(_trace_snap_wait) - 1),
                        )
                        _append_lock_event(_wait_idx, "wait", stable_lock_id)
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
                        _acq_idx = next(
                            (i for i in range(len(_trace_snap_acq) - 1, -1, -1) if _trace_snap_acq[i] == thread_id),
                            max(0, len(_trace_snap_acq) - 1),
                        )
                        _append_lock_event(_acq_idx, "acquire", stable_lock_id)
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
                        _rel_idx = next(
                            (i for i in range(len(_trace_snap_rel) - 1, -1, -1) if _trace_snap_rel[i] == thread_id),
                            max(0, len(_trace_snap_rel) - 1),
                        )
                        _append_lock_event(_rel_idx, "release", stable_lock_id)
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
        pending_io: list[tuple[int, str, bool]] = []
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
            _registered_groups: set[int] = set()

            def _io_reporter(resource_id: str, kind: str) -> None:
                object_key = _make_object_key(hash(resource_id), resource_id)
                pending: list[tuple[int, str, bool]] = _dpor_tls.pending_io
                pending.append((object_key, kind, True))  # synced=True: Python-level I/O respects locks
                # Register resource group for SQL resources so the DPOR engine
                # can skip inline wakeup insertion and rely on deferred notdep
                # processing. This prevents backtrack explosion from intermediate
                # operations on unrelated tables (Defect #15).
                if resource_id.startswith("sql:"):
                    parts = resource_id.split(":")
                    table_group = f"sql:{parts[1]}" if len(parts) >= 2 else resource_id
                    group_key = hash(table_group) & 0xFFFFFFFFFFFFFFFF
                    if object_key not in _registered_groups:
                        with engine_lock:
                            engine.register_resource_group(object_key, group_key)
                        _registered_groups.add(object_key)
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
                for _obj_key, _io_kind, _synced in _pending:
                    with _elock:
                        if _synced:
                            _engine.report_synced_io_access(_execution, _tid, _obj_key, _io_kind)
                        else:
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
    io_schedule: list[tuple[int, str]] | None = None,
    patch_sleep: bool = True,
) -> T:
    """Replay a DPOR schedule using the DPOR runner rather than OpcodeScheduler.

    When *io_schedule* is provided and *detect_io* is True, uses the
    IO-anchored replay scheduler (defect #16) which only enforces the
    schedule at IO boundaries, tolerating state-dependent changes in
    opcode-level scheduling points.
    """
    if io_schedule is not None and detect_io:
        scheduler: DporScheduler = _IOAnchoredReplayScheduler(
            io_schedule,
            len(threads),
            deadlock_timeout=deadlock_timeout,
            trace_recorder=trace_recorder,
            detect_io=detect_io,
        )
    else:
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
    if patch_sleep:
        runner._patch_sleep()
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
        runner._unpatch_sleep()
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
    io_schedule: list[tuple[int, str]] | None = None,
    patch_sleep: bool = True,
) -> tuple[int, int]:
    """Measure how often a DPOR counterexample reproduces under the DPOR runner.

    Reproduction runs with the same IO interception (SQL, Redis) as
    exploration so that the replay scheduler can enforce interleavings at
    IO boundaries, not just bytecode boundaries.

    When *io_schedule* is provided, replay is anchored to explicit I/O
    boundaries (defect #16) so state-dependent opcode paths do not
    desynchronise the schedule.
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
                    io_schedule=io_schedule,
                    patch_sleep=patch_sleep,
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
    search: str | None = None,
    patch_sleep: bool = True,
    serializable_invariant: Callable[[T], Any] | bool = False,
    error_on_any_race: bool = False,
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
        search: Controls the order in which wakeup tree branches are
            explored.  All strategies visit the same set of Mazurkiewicz
            trace equivalence classes; only the order differs.  Accepted
            values:

            - ``None`` or ``"dfs"`` — classic DFS, lowest thread ID first
              (default, matches the paper's Algorithm 2).  **Best for
              exhaustive exploration** (``stop_on_first=False``): produces
              the optimal (minimum) number of executions.
            - ``"bit-reversal"`` or ``"bit-reversal:<seed>"`` — visit
              children in bit-reversal permutation order for maximal
              spread across distinct conflict points early.
            - ``"round-robin"`` or ``"round-robin:<seed>"`` — cycle
              through available threads in rotating order.
            - ``"stride"`` or ``"stride:<seed>"`` — visit every s-th
              sibling (s coprime to branching factor, derived from seed).
            - ``"conflict-first"`` — reverse of DFS (highest thread ID
              first), preferring threads added by race reversals.

            **Use a non-DFS strategy when the trace space is large and
            you have a limited execution budget** (``stop_on_first=True``
            or a low ``max_executions``).  DFS explores traces in a fixed
            order and may spend many executions on similar interleavings
            before reaching a bug.  The alternative strategies spread
            exploration across different conflict points earlier, finding
            bugs faster on average.
        patch_sleep: If True (default), replace ``time.sleep`` with a
            no-op that yields to the scheduler.  This prevents threads
            from actually sleeping during exploration (which would be
            extremely slow) while preserving sleep calls as scheduling
            points.  Set to False if your code depends on real delays.

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

    # Compute serializable baseline if requested.
    serial_valid_states: set[Any] | None = None
    serial_hash_fn: Callable[[Any], Any] = repr
    if serializable_invariant is not False:
        try:
            from frontrun.common import compute_serializable_states, resolve_serializable_hash_fn

            serial_hash_fn = resolve_serializable_hash_fn(serializable_invariant) or repr
            serial_valid_states = compute_serializable_states(setup, threads, state_hash=serial_hash_fn)
        except BaseException:
            _set_active_trace_filter(None)
            raise

    num_threads = len(threads)
    engine = PyDporEngine(
        num_threads=num_threads,
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        max_executions=max_executions,
        search=search,
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
        rmap = get_object_key_reverse_map() or {}
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

    report_path = _global_report_path
    report: ExplorationReport | None = None
    if report_path is not None:
        report = ExplorationReport(
            num_threads=num_threads,
            thread_names=[f"Thread {i}" for i in range(num_threads)],
        )
        set_object_key_reverse_map({})

    def _record_and_emit_report(*, was_deadlock: bool = False) -> None:
        """Record the current execution to the report and write the HTML file."""
        if report is None or report_path is None:
            return
        if not _collecting_report:
            generate_html_report(report, report_path)
            return
        with engine_lock:
            sched = list(execution.schedule_trace)
            races = engine.pending_races()
        report.executions.append(
            ExecutionRecord(
                index=len(report.executions),
                schedule_trace=sched,
                switch_points=switch_points,
                invariant_held=False,
                was_deadlock=was_deadlock,
                race_info=_build_race_info(races),
                step_events=scheduler._step_event_collector or {},
                lock_events=scheduler._lock_event_collector or [],
                deadlock_at=scheduler._deadlock_at,
                deadlock_cycle_description=getattr(scheduler._error, "cycle_description", None)
                if was_deadlock
                else None,
            )
        )
        generate_html_report(report, report_path)

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
            # Clear persistent SQL suppression flags from previous execution.
            from frontrun._sql_cursor import clear_permanent_suppressions

            clear_permanent_suppressions()
            # Set up switch point collection for the report
            _collecting_report = report is not None and len(report.executions) < _MAX_RECORDED_EXECUTIONS
            switch_points: list[Any] = []
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
            if patch_sleep:
                runner._patch_sleep()
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
                runner._unpatch_sleep()
                runner._unpatch_io()
                runner._unpatch_locks()

            result.num_explored += 1

            # Check for deadlock before running the invariant — a deadlock
            # means the program never completed, so the invariant can never be
            # satisfied.  Report it as a property violation with a clear message.
            _deadlock_err = scheduler._error if isinstance(scheduler._error, DeadlockError) else None
            is_deadlock = _deadlock_err is not None
            if _deadlock_err is not None:
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
                        f"{_deadlock_err.cycle_description}"
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
                        io_schedule=list(scheduler._io_trace) if detect_io and scheduler._io_trace else None,
                        patch_sleep=patch_sleep,
                    )
                    result.reproduction_attempts = attempts
                    result.reproduction_successes = successes

                    from frontrun._preload_io import _set_preload_pipe_fd

                    if preload_dispatcher is not None and preload_dispatcher._write_fd is not None:
                        _set_preload_pipe_fd(preload_dispatcher._write_fd)

                if stop_on_first:
                    clear_instr_cache()
                    _record_and_emit_report(was_deadlock=True)
                    return result

            if warn_nondeterministic_sql:
                check_uncaptured_inserts()

            # --- error_on_any_race: treat unsynchronized races as failures ---
            if error_on_any_race and not is_deadlock:
                with engine_lock:
                    raw_races_check = engine.attribute_races()
                if raw_races_check:
                    result.property_holds = False
                    result.races_detected = True
                    with engine_lock:
                        schedule = execution.schedule_trace
                    schedule_list = list(schedule)
                    result.failures.append((result.num_explored, schedule_list))
                    if result.counterexample is None:
                        result.counterexample = schedule_list
                    if result.explanation is None:
                        result.explanation = (
                            f"Unsynchronized race detected in execution {result.num_explored}.\n"
                            f"{len(raw_races_check)} race(s) found between threads on shared objects."
                        )
                    if stop_on_first:
                        clear_instr_cache()
                        _record_and_emit_report()
                        return result

            # --- serializable_invariant: check against sequential baselines ---
            if serial_valid_states is not None and not is_deadlock:
                ser_explanation = check_serializability_violation(
                    state, serial_valid_states, serial_hash_fn, result.num_explored
                )
                if ser_explanation is not None:
                    result.property_holds = False
                    with engine_lock:
                        schedule = execution.schedule_trace
                    schedule_list = list(schedule)
                    result.failures.append((result.num_explored, schedule_list))
                    if result.counterexample is None:
                        result.counterexample = schedule_list
                    if result.explanation is None:
                        result.explanation = ser_explanation
                    if stop_on_first:
                        clear_instr_cache()
                        _record_and_emit_report()
                        return result

            if not is_deadlock and not invariant(state):
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
                        io_schedule=list(scheduler._io_trace) if detect_io and scheduler._io_trace else None,
                        patch_sleep=patch_sleep,
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
                    clear_instr_cache()
                    _record_and_emit_report()
                    return result

            # Clear instruction cache between executions to avoid stale code ids
            clear_instr_cache()

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
        if trace_packages is not None:
            _set_active_trace_filter(None)
        set_lock_timeout(prev_lock_timeout)
        if preload_dispatcher is not None:
            preload_dispatcher.stop()
        set_object_key_reverse_map(None)

    # Generate HTML report if requested
    if report is not None and report_path is not None:
        generate_html_report(report, report_path)

    return result
