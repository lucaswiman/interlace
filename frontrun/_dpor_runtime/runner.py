# ruff: noqa: F403, F405

from __future__ import annotations

from contextlib import contextmanager

from frontrun._threaded_runner import PatchScope, notify_scheduler_timeout, run_thread_group

from ._shared import *
from ._shared import (
    _USE_SYS_MONITORING,
    _append_unique_lock_event,
    _dpor_tls,
    _make_object_key,
    _should_trace_file,
)
from .preload_bridge import _PreloadBridge
from .scheduler import DporScheduler


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

    def patch_scope(self, *, patch_sleep: bool = True) -> PatchScope:
        scope = PatchScope()
        scope.add(self._patch_locks, self._unpatch_locks)
        scope.add(self._patch_io, self._unpatch_io)
        scope.add(self._patch_sleep, self._unpatch_sleep, enabled=patch_sleep)
        return scope

    # --- sys.settrace backend (3.10-3.11) ---

    def _make_trace(self, thread_id: int) -> Callable[..., Any]:
        scheduler = self.scheduler
        _detect_io = scheduler._detect_io

        def _get_tid() -> int | None:
            return thread_id

        def _on_opcode(code: Any, offset: int, frame: Any, tid: int) -> bool:
            return process_opcode_with_coarsening(code, offset, frame, scheduler, tid, _detect_io)

        return make_settrace_callback(
            get_thread_id=_get_tid,
            on_opcode=_on_opcode,
            remove_shadow_stack=scheduler.remove_shadow_stack,
            detect_io=_detect_io,
            is_active=lambda: not (scheduler._finished or scheduler._error),
        )

    # --- sys.monitoring backend (3.12+) ---

    def _setup_monitoring(self) -> None:
        """Set up sys.monitoring INSTRUCTION events for all code objects."""
        if not _USE_SYS_MONITORING:
            return

        scheduler = self.scheduler
        _detect_io = scheduler._detect_io

        def _get_tid() -> int | None:
            tid = getattr(_dpor_tls, "thread_id", None)
            if tid is not None and getattr(_dpor_tls, "scheduler", None) is scheduler:
                return tid  # type: ignore[no-any-return]
            return None

        def _on_opcode(code: Any, offset: int, frame: Any, tid: int) -> bool:
            return process_opcode_with_coarsening(code, offset, frame, scheduler, tid, _detect_io)

        handle_py_start, handle_py_return, handle_instruction = make_monitoring_callbacks(
            get_thread_id=_get_tid,
            on_opcode=_on_opcode,
            remove_shadow_stack=scheduler.remove_shadow_stack,
            detect_io=_detect_io,
            is_active=lambda: not (scheduler._finished or scheduler._error),
        )

        DporBytecodeRunner._TOOL_ID = setup_opcode_monitoring(
            tool_name="frontrun._dpor",
            handle_py_start=handle_py_start,
            handle_py_return=handle_py_return,
            handle_instruction=handle_instruction,
        )
        self._monitoring_active = True

    def _teardown_monitoring(self) -> None:
        if not self._monitoring_active:
            return
        teardown_opcode_monitoring(DporBytecodeRunner._TOOL_ID)
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

    @contextmanager
    def _thread_runtime(
        self,
        thread_id: int,
        *,
        trace_fn: Callable[..., Any] | None = None,
    ):
        self._setup_dpor_tls(thread_id)
        if trace_fn is not None:
            sys.settrace(trace_fn)
        try:
            yield
        finally:
            if trace_fn is not None:
                sys.settrace(None)
            self._teardown_dpor_tls()
            self.scheduler.mark_done(thread_id)

    def _run_thread(
        self,
        thread_id: int,
        func: Callable[..., None],
        args: tuple[Any, ...],
        *,
        trace_fn: Callable[..., Any] | None = None,
    ) -> None:
        try:
            with self._thread_runtime(thread_id, trace_fn=trace_fn):
                func(*args)
        except SchedulerAbort:
            pass  # scheduler already has the error; just exit cleanly
        except Exception as e:
            self.errors[thread_id] = e
            self.scheduler.report_error(e)

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
        run_thread = self._run_thread

        def make_thread_target(
            thread_id: int,
            func: Callable[..., None],
            thread_args: tuple[Any, ...],
        ) -> Callable[[], None]:
            def target() -> None:
                run_thread(
                    thread_id,
                    func,
                    thread_args,
                    trace_fn=None if use_monitoring else self._make_trace(thread_id),
                )

            return target

        def on_timeout(alive: list[threading.Thread]) -> None:
            notify_scheduler_timeout(self.scheduler, alive)

        run_thread_group(
            funcs=funcs,
            args=args,
            make_thread_target=make_thread_target,
            name_prefix="dpor",
            timeout=timeout,
            thread_store=self.threads,
            teardown=self._teardown_monitoring if use_monitoring else None,
            on_timeout=on_timeout,
        )

        if self.errors:
            first_error = next(iter(self.errors.values()))
            if not isinstance(first_error, TimeoutError):
                raise first_error


# ---------------------------------------------------------------------------
# Fixed-schedule replay
# ---------------------------------------------------------------------------
