"""Async DPOR (Dynamic Partial Order Reduction) for frontrun.

Combines the DPOR engine's systematic interleaving exploration with the
async scheduler's await-point-level control.  Instead of the random
schedule sampling used by ``async_shuffler.py``, this module uses the
Rust DPOR engine to explore every meaningfully distinct interleaving
exactly once.

The approach:
1. A Rust DPOR engine (frontrun._dpor) manages the exploration tree,
   vector clocks, and wakeup tree exploration.
2. Python drives execution: each task's coroutine is wrapped with
   ``_AutoPauseCoroutine`` which intercepts every ``await`` yield and
   inserts a DPOR scheduling decision.  No user code changes needed.
3. At each yield, the scheduler reports the access to the DPOR engine
   and asks it which task to run next.
4. ``asyncio.Lock`` is monkey-patched to a cooperative version with
   deadlock detection (WaitForGraph) and explicit scheduling points.
5. SQL queries are intercepted via async cursor patching and reported
   as I/O resource accesses to the DPOR engine.

Usage::

    import asyncio
    from frontrun.async_dpor import explore_async_dpor

    class Counter:
        def __init__(self):
            self.value = 0

        async def increment(self):
            temp = self.value
            await asyncio.sleep(0)  # any natural await works
            self.value = temp + 1

    result = await explore_async_dpor(
        setup=lambda: Counter(),
        tasks=[lambda c: c.increment(), lambda c: c.increment()],
        invariant=lambda c: c.value == 2,
    )
    assert result.property_holds, result.explanation  # fails — lost update!
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
import time
from collections.abc import Awaitable, Callable, Coroutine, Generator
from typing import Any, TypeVar

from frontrun._deadlock import DeadlockError, WaitForGraph, format_cycle
from frontrun._sql_cursor import clear_sql_metadata, get_lock_timeout, set_lock_timeout
from frontrun._sql_insert_tracker import check_uncaptured_inserts, clear_insert_tracker
from frontrun._tracing import TraceFilter as _TraceFilter
from frontrun._tracing import is_cmdline_user_code as _is_cmdline_user_code
from frontrun._tracing import is_dynamic_code as _is_dynamic_code
from frontrun._tracing import set_active_trace_filter as _set_active_trace_filter
from frontrun._tracing import should_trace_file as _should_trace_file
from frontrun.async_scheduler import InterleavedLoop
from frontrun.common import InterleavingResult
from frontrun.dpor import _USE_SYS_MONITORING, ShadowStack, StableObjectIds, _process_opcode

try:
    from frontrun._dpor import PyDporEngine, PyExecution  # type: ignore[reportAttributeAccessIssue]
except ModuleNotFoundError as _err:
    raise ModuleNotFoundError(
        "explore_async_dpor requires the frontrun._dpor Rust extension.\n"
        "Build it with:  make build-dpor-3.14   (or build-dpor-3.10 / build-dpor-3.14t)\n"
        "Or install from source:  pip install -e ."
    ) from _err

# Lazy import for async SQL patching (avoid hard dependency)
_sql_async_available = False
try:
    from frontrun._sql_cursor_async import patch_sql_async, unpatch_sql_async

    _sql_async_available = True
except ImportError:

    def patch_sql_async() -> None:  # type: ignore[misc]
        pass

    def unpatch_sql_async() -> None:  # type: ignore[misc]
        pass


# Lazy import for async Redis patching (avoid hard dependency)
_redis_async_available = False
try:
    from frontrun._redis_client_async import patch_redis_async, unpatch_redis_async

    _redis_async_available = True
except ImportError:

    def patch_redis_async() -> None:  # type: ignore[misc]
        pass

    def unpatch_redis_async() -> None:  # type: ignore[misc]
        pass


T = TypeVar("T")


class _NoOpLock:
    """Context-manager-shaped lock for single-threaded async DPOR engine calls."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


# Context variables to track the active scheduler and task ID
_scheduler_var: contextvars.ContextVar[AsyncDporScheduler | None] = contextvars.ContextVar(
    "_async_dpor_scheduler", default=None
)
_task_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("_async_dpor_task_id", default=None)

# When True, the coroutine wrapper handles scheduling automatically.
# _CooperativeAsyncLock checks this to avoid double-scheduling.
_auto_pause_active: contextvars.ContextVar[bool] = contextvars.ContextVar("_auto_pause_active", default=False)

# When >0, the _AutoPauseIterator should NOT insert scheduling points.
# Incremented by scheduler.pause() so the wrapper doesn't double-schedule
# on the pause's own yields (sleep(0), condition.wait()).
_in_scheduler_pause: contextvars.ContextVar[int] = contextvars.ContextVar("_in_scheduler_pause", default=0)

# Guards against re-entering async opcode tracing while _process_opcode()
# is already running for the current task.
_in_trace_processing: contextvars.ContextVar[bool] = contextvars.ContextVar("_in_trace_processing", default=False)


# ---------------------------------------------------------------------------
# Object key helper (shared with sync dpor.py)
# ---------------------------------------------------------------------------


def _make_object_key(obj_id: int, name: Any) -> int:
    """Create a non-negative u64 object key for the Rust engine."""
    return hash((obj_id, name)) & 0xFFFFFFFFFFFFFFFF


# Synchronization acquisition points must still be explored even when the
# individual lock resources differ, because future blocking can make those
# orderings observably distinct (deadlock, starvation, etc.).
_SHARED_SYNC_ACQUIRE_KEY = _make_object_key(0, "__async_dpor_sync_acquire__")


# ---------------------------------------------------------------------------
# Cooperative asyncio.Lock with deadlock detection
# ---------------------------------------------------------------------------

_real_asyncio_lock = asyncio.Lock
_async_lock_patched = False

# Per-lock wait-for graph for async DPOR deadlock detection.
_async_wait_graph: WaitForGraph | None = None
# Map from lock id(obj) → owning task_id
_async_lock_owners: dict[int, int] = {}
# Reverse map: task_id → set of lock objects held by that task.
# Used to force-release locks when a task finishes without calling release().
_async_task_held_locks: dict[int, set[Any]] = {}


class _CooperativeAsyncLock:
    """Drop-in asyncio.Lock replacement with wait-for graph deadlock detection.

    When a task tries to acquire a held lock, registers a waiting edge
    in the global WaitForGraph.  If adding the edge creates a cycle,
    raises DeadlockError immediately instead of blocking forever.

    Every acquire() is also a DPOR scheduling point (via await_point()).
    This is necessary because asyncio.Lock.acquire() on a free lock
    completes synchronously without yielding, which would prevent DPOR
    from interleaving lock acquisitions across tasks.
    """

    def __init__(self) -> None:
        self._lock = _real_asyncio_lock()
        self._owner: int | None = None

    def locked(self) -> bool:
        return self._lock.locked()

    async def acquire(self) -> bool:
        task_id = _task_id_var.get()
        graph = _async_wait_graph

        # Make lock acquisition a DPOR scheduling point so the engine
        # can interleave different tasks' lock acquisitions.  Without
        # this, asyncio.Lock.acquire() on a free lock completes
        # synchronously and DPOR never sees the interleaving where
        # two tasks hold conflicting locks simultaneously.
        #
        # The scheduler.pause() call sets _in_scheduler_pause so the
        # coroutine wrapper won't insert a redundant scheduling point
        # for the pause's own yields.
        scheduler = _scheduler_var.get()
        if scheduler is not None and task_id is not None and _in_scheduler_pause.get() == 0:
            await scheduler.pause(task_id, ("lock_acquire", id(self)))

        if task_id is not None and graph is not None and self._lock.locked():
            lock_id = id(self)
            # Register: this task is waiting for this lock
            cycle = graph.add_waiting(task_id, lock_id)
            if cycle is not None:
                graph.remove_waiting(task_id, lock_id)
                desc = format_cycle(cycle)
                raise DeadlockError(f"Async lock deadlock detected: {desc}", desc)
            # Mark this task as blocked in the DPOR execution so the engine
            # won't schedule it while it's waiting for the lock.  Also track
            # the lock holder so _schedule_next can redirect to the holder
            # if needed.
            if scheduler is not None:
                scheduler.execution.block_thread(task_id)
                if self._owner is not None:
                    scheduler._lock_blocked[task_id] = self._owner
            # Set _in_scheduler_pause so the AutoPauseCoroutine passes
            # the lock's internal yields through to the event loop without
            # inserting scheduling points.  Without this, the DPOR scheduler
            # would try to schedule at every yield of the lock's acquire,
            # creating a deadlock (the blocked task can't proceed but the
            # scheduler keeps picking it).
            depth = _in_scheduler_pause.get()
            _in_scheduler_pause.set(depth + 1)
            try:
                result = await self._lock.acquire()
            finally:
                _in_scheduler_pause.set(depth)
                graph.remove_waiting(task_id, lock_id)
                if scheduler is not None:
                    scheduler.execution.unblock_thread(task_id)
                    scheduler._lock_blocked.pop(task_id, None)
        else:
            result = await self._lock.acquire()

        # Record ownership
        if task_id is not None and graph is not None:
            lock_id = id(self)
            self._owner = task_id
            _async_lock_owners[lock_id] = task_id
            _async_task_held_locks.setdefault(task_id, set()).add(self)
            graph.add_holding(task_id, lock_id)
            scheduler = _scheduler_var.get()
            if scheduler is not None:
                scheduler.engine.report_sync(scheduler.execution, task_id, "lock_acquire", lock_id)

        return result

    def release(self) -> None:
        graph = _async_wait_graph
        if self._owner is not None:
            task_id = self._owner
            lock_id = id(self)
            if graph is not None:
                graph.remove_holding(task_id, lock_id)
            _async_lock_owners.pop(lock_id, None)
            held = _async_task_held_locks.get(task_id)
            if held is not None:
                held.discard(self)
            scheduler = _scheduler_var.get()
            if scheduler is not None:
                scheduler.engine.report_sync(scheduler.execution, task_id, "lock_release", lock_id)
            self._owner = None
        self._lock.release()

    async def __aenter__(self) -> bool:
        return await self.acquire()

    async def __aexit__(self, *args: Any) -> None:
        self.release()


def _release_task_async_locks(task_id: int) -> None:
    """Force-release all asyncio.Lock objects held by *task_id*.

    Called when a task finishes (normally or via exception) without
    explicitly releasing its locks.  Cleans up both the WaitForGraph
    holding edges and the underlying real asyncio.Lock objects.
    """
    held = _async_task_held_locks.pop(task_id, None)
    if not held:
        return
    graph = _async_wait_graph
    for lock_obj in list(held):
        lock_id = id(lock_obj)
        if graph is not None:
            graph.remove_holding(task_id, lock_id)
        _async_lock_owners.pop(lock_id, None)
        lock_obj._owner = None
        if lock_obj._lock.locked():
            lock_obj._lock.release()


def _patch_asyncio_lock() -> None:
    """Replace asyncio.Lock with cooperative deadlock-detecting version."""
    global _async_lock_patched, _async_wait_graph  # noqa: PLW0603
    if _async_lock_patched:
        return
    _async_wait_graph = WaitForGraph()
    _async_lock_owners.clear()
    asyncio.Lock = _CooperativeAsyncLock  # type: ignore[assignment,misc]
    _async_lock_patched = True


def _unpatch_asyncio_lock() -> None:
    """Restore original asyncio.Lock."""
    global _async_lock_patched, _async_wait_graph  # noqa: PLW0603
    if not _async_lock_patched:
        return
    asyncio.Lock = _real_asyncio_lock  # type: ignore[assignment,misc]
    if _async_wait_graph is not None:
        _async_wait_graph.clear()
    _async_wait_graph = None
    _async_lock_owners.clear()
    _async_task_held_locks.clear()
    _async_lock_patched = False


# ---------------------------------------------------------------------------
# Await point
# ---------------------------------------------------------------------------


async def await_point() -> None:
    """Yield to the DPOR scheduler at an await point.

    When auto-pause is active (the default in ``explore_async_dpor``),
    this is equivalent to ``await asyncio.sleep(0)`` — the coroutine
    wrapper handles scheduling automatically.  Kept for backward
    compatibility with code that uses explicit scheduling markers.

    If no scheduler is active, this function returns immediately.
    """
    if _auto_pause_active.get():
        # The coroutine wrapper intercepts all yields, so a simple
        # sleep(0) is sufficient to create a scheduling point.
        await asyncio.sleep(0)
        return
    scheduler = _scheduler_var.get()
    if scheduler is not None:
        task_id = _task_id_var.get()
        if task_id is not None:
            await scheduler.pause(task_id)


# ---------------------------------------------------------------------------
# Auto-pause coroutine wrapper
# ---------------------------------------------------------------------------


class _AutoPauseIterator:
    """Wraps a coroutine to insert DPOR scheduling points at every yield.

    Intercepts the coroutine's send/throw protocol.  Before each step of the
    inner coroutine (i.e. before forwarding a ``send()`` call), the wrapper
    drives a ``scheduler.pause(task_id)`` coroutine.  This gives the DPOR
    engine a scheduling decision point at every natural ``await`` expression
    in user code, without requiring explicit ``await await_point()`` calls.

    The wrapper alternates between two phases:
    - **Pause phase**: driving the pause coroutine, yielding its futures to
      the event loop.
    - **Inner phase**: forwarding the buffered value to the inner coroutine,
      yielding the inner's future to the event loop.

    There is no recursion risk because the pause coroutine's yields go
    directly to the event loop (via the wrapper's own ``send`` return),
    not back through the inner coroutine.
    """

    __slots__ = ("_inner", "_task_id", "_scheduler", "_pause_iter", "_buffered_value")

    def __init__(self, inner_coro: Any, task_id: int, scheduler: AsyncDporScheduler) -> None:
        self._inner = inner_coro
        self._task_id = task_id
        self._scheduler = scheduler
        self._pause_iter: Any | None = None
        self._buffered_value: Any = None

    def __next__(self) -> Any:
        return self.send(None)

    def send(self, value: Any) -> Any:
        # Continue an active pause coroutine
        if self._pause_iter is not None:
            try:
                return self._pause_iter.send(value)
            except StopIteration:
                self._pause_iter = None
                # Pause completed — forward the buffered value to inner
                return self._inner.send(self._buffered_value)

        # If we're inside a scheduler.pause() call (e.g. from
        # _CooperativeAsyncLock), don't insert another pause — just
        # forward to the inner coroutine.
        if _in_scheduler_pause.get() > 0:
            return self._inner.send(value)

        # Buffer the incoming value and start a new pause
        self._buffered_value = value
        pause_coro = self._scheduler.pause(self._task_id)
        self._pause_iter = pause_coro.__await__()
        try:
            return next(self._pause_iter)
        except StopIteration:
            # Pause completed immediately (no scheduler active)
            self._pause_iter = None
            return self._inner.send(self._buffered_value)

    def throw(self, typ: Any, val: Any = None, tb: Any = None) -> Any:
        if self._pause_iter is not None:
            self._pause_iter.close()
            self._pause_iter = None
        if val is None and tb is None:
            return self._inner.throw(typ)
        return self._inner.throw(typ, val, tb)

    def close(self) -> None:
        if self._pause_iter is not None:
            self._pause_iter.close()
            self._pause_iter = None
        self._inner.close()


class _AutoPauseCoroutine:
    """Awaitable wrapper that makes a coroutine auto-schedule via DPOR."""

    __slots__ = ("_iter",)

    def __init__(self, coro: Any, task_id: int, scheduler: AsyncDporScheduler) -> None:
        self._iter = _AutoPauseIterator(coro, task_id, scheduler)

    def __await__(self) -> Generator[Any, Any, None]:
        return self._iter  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Async DPOR Scheduler
# ---------------------------------------------------------------------------


class AsyncDporScheduler(InterleavedLoop):
    """Controls async task execution at await-point granularity using DPOR.

    Instead of following a fixed schedule, uses the Rust DPOR engine to
    decide which task runs next.  Shared-memory accesses inside each
    await-delimited block are traced at opcode/instruction granularity
    and reported to the engine before the next scheduling decision.
    """

    _TOOL_ID: int | None = None

    def __init__(
        self,
        engine: PyDporEngine,
        execution: PyExecution,
        num_tasks: int,
        *,
        deadlock_timeout: float = 5.0,
        detect_sql: bool = False,
        detect_redis: bool = False,
        stable_ids: StableObjectIds | None = None,
    ) -> None:
        super().__init__(deadlock_timeout=deadlock_timeout)
        self.engine = engine
        self.execution = execution
        self._num_engine_tasks = num_tasks
        self._current_task: int | None = None
        self._detect_sql = detect_sql
        self._detect_redis = detect_redis
        self._engine_lock = _NoOpLock()
        self.trace_recorder = None
        self._iter_to_container: dict[int, Any] = {}
        self._shadow_stacks: dict[int, ShadowStack] = {}
        self._monitoring_active = False
        self._stable_ids = stable_ids if stable_ids is not None else StableObjectIds()
        # Pending I/O accesses per task (from SQL interception)
        self._pending_io: dict[int, list[tuple[int, str, bool]]] = {i: [] for i in range(num_tasks)}

        # Track tasks blocked on asyncio.Lock: task_id → lock-holder task_id.
        # When DPOR schedules a blocked task, override to run the holder.
        self._lock_blocked: dict[int, int] = {}

        # Row lock tracking: resource_id → holding task_id
        self._active_row_locks: dict[str, int] = {}
        # task_id → set of held resource_ids
        self._task_row_locks: dict[int, set[str]] = {}
        # resource_id → integer ID for WaitForGraph
        self._row_lock_ids: dict[str, int] = {}
        self._row_lock_next_id: int = 0

        # Request the first scheduling decision
        self._current_task = self._schedule_next()

    def _schedule_next(self) -> int | None:
        """Ask the DPOR engine which task to run next.

        If the engine picks a task blocked on an asyncio.Lock, override
        the decision to schedule the lock holder instead.  This prevents
        the scheduler from cycling between a blocked task and the event
        loop, causing false deadlock timeouts.
        """
        runnable = self.execution.runnable_threads()
        if not runnable:
            return None
        scheduled = self.engine.schedule(self.execution)
        if scheduled is not None and scheduled in self._lock_blocked:
            holder = self._lock_blocked[scheduled]
            if holder not in self._tasks_done:
                return holder
            # Holder is done — lock should be released. Clean up stale entry.
            self._lock_blocked.pop(scheduled, None)
        return scheduled

    # -- InterleavedLoop policy -----------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        if self._current_task is None:
            self._finished = True
            return True
        if self._current_task == task_id:
            return True
        # If the currently-scheduled task is blocked on a lock held by
        # task_id, let task_id proceed so it can release the lock.
        if self._current_task in self._lock_blocked:
            holder = self._lock_blocked[self._current_task]
            if holder == task_id:
                return True
        return False

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        # Flush any pending I/O accesses before advancing
        self._flush_pending_io(task_id)
        if isinstance(marker, tuple) and marker and marker[0] == "lock_acquire":
            self.engine.report_access(
                self.execution,
                task_id,
                _SHARED_SYNC_ACQUIRE_KEY,
                "write",
            )

        # Schedule next task.  If the engine returns None (no runnable
        # threads), keep _current_task as the current task_id so the
        # current task can continue.  Only set _finished when ALL tasks
        # are done.
        next_task = self._schedule_next()
        if next_task is not None:
            self._current_task = next_task
        # If next_task is None but there are alive tasks, keep current
        # task so it can continue running to completion.

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        self._error = TimeoutError(
            f"Deadlock: DPOR async scheduler wants task {self._current_task} "
            f"but task {task_id} is waiting at marker {marker!r}"
        )
        self._condition.notify_all()

    def _setup_task_context(self, task_id: Any) -> None:
        _scheduler_var.set(self)
        _task_id_var.set(task_id)
        # Set up IO reporter context so SQL interception can report to us
        from frontrun._io_detection import _io_tls, set_dpor_scheduler, set_dpor_thread_id, set_io_reporter

        set_dpor_scheduler(self)
        set_dpor_thread_id(task_id)

        if self._detect_sql or self._detect_redis:

            def _io_reporter(resource_id: str, kind: str) -> None:
                # Dynamically read the current task ID so that when multiple
                # async tasks share the same thread-local reporter, each
                # I/O event is attributed to the task that actually runs the
                # Redis/SQL command, not whichever task was set up last.
                current_task = _task_id_var.get()
                if current_task is None:
                    current_task = task_id
                object_key = _make_object_key(hash(resource_id), resource_id)
                self._pending_io.setdefault(current_task, []).append((object_key, kind, True))

            set_io_reporter(_io_reporter)

        # Reset transaction state for this task
        _io_tls._in_transaction = False
        _io_tls._is_autobegin = False
        _io_tls._tx_buffer = []
        _io_tls._tx_savepoints = {}

    async def run_all(
        self,
        task_funcs: dict[Any, Callable[..., Awaitable[None]]] | list[Callable[..., Awaitable[None]]],
        timeout: float = 10.0,
    ) -> None:
        """Run tasks with DPOR-controlled interleaving.

        Each task's coroutine is wrapped with ``_AutoPauseCoroutine``,
        which automatically inserts a DPOR scheduling point before every
        step of the inner coroutine (i.e. at every natural ``await``).
        No explicit ``await await_point()`` calls are needed.
        """
        if isinstance(task_funcs, list):
            task_funcs = dict(enumerate(task_funcs))

        # Wrap each user function so every await becomes a DPOR scheduling point
        wrapped: dict[Any, Callable[..., Awaitable[None]]] = {}
        for tid, func in task_funcs.items():

            async def _wrapped(f: Callable[..., Awaitable[None]] = func, t: Any = tid) -> None:
                _auto_pause_active.set(True)
                await _AutoPauseCoroutine(f(), t, self)

            wrapped[tid] = _wrapped

        if _USE_SYS_MONITORING:
            self._setup_monitoring()
            try:
                await super().run_all(wrapped, timeout=timeout)
            finally:
                self._teardown_monitoring()
                self._shadow_stacks.clear()
        else:
            self._setup_settrace()
            try:
                await super().run_all(wrapped, timeout=timeout)
            finally:
                sys.settrace(None)
                self._shadow_stacks.clear()

    async def pause(self, task_id: Any, marker: Any = None) -> None:
        """DPOR-aware pause that ensures fair task wakeup.

        After proceeding from a pause, yields to the event loop so other
        tasks that were notified can process their condition waits.
        Without this, a single task can reacquire the condition lock
        before other notified tasks, causing false deadlock detection.

        Sets ``_in_scheduler_pause`` so the coroutine wrapper knows not
        to insert a redundant scheduling point for this pause's yields.
        """
        depth = _in_scheduler_pause.get()
        _in_scheduler_pause.set(depth + 1)
        try:
            # Yield to let any previously-notified tasks process their wakeups
            await asyncio.sleep(0)
            await super().pause(task_id, marker)
        finally:
            _in_scheduler_pause.set(depth)

    async def _mark_done(self, task_id: Any) -> None:
        """Mark a task as finished in both InterleavedLoop and the DPOR engine."""
        self.execution.finish_thread(task_id)
        # If this was the current task, schedule the next one
        async with self._condition:
            self._tasks_done.add(task_id)
            if self._current_task == task_id:
                next_task = self._schedule_next()
                self._current_task = next_task
                if next_task is None and len(self._tasks_done) >= self._num_tasks:
                    self._finished = True
            self._condition.notify_all()

    def _cleanup_task_context(self, task_id: Any) -> None:
        # Flush any remaining pending I/O before cleanup
        self._flush_pending_io(task_id)

        # Release any row locks still held (task finished without COMMIT)
        self.release_row_locks(task_id)

        # Release any asyncio.Lock objects still held (task crashed without release())
        _release_task_async_locks(task_id)

        _scheduler_var.set(None)
        _task_id_var.set(None)
        from frontrun._io_detection import set_dpor_scheduler, set_dpor_thread_id, set_io_reporter

        # Only clear thread-local DPOR/IO state when ALL tasks are done.
        # In async mode, all tasks share the same thread-local storage.
        # Clearing the reporter when one task finishes would break I/O
        # detection for remaining tasks on the same thread.
        # Note: _cleanup_task_context runs BEFORE _mark_done, so the
        # current task_id is not yet in _tasks_done; +1 accounts for it.
        if len(self._tasks_done) + 1 >= self._num_engine_tasks:
            set_dpor_scheduler(None)
            set_dpor_thread_id(None)
            if self._detect_sql or self._detect_redis:
                set_io_reporter(None)

    def get_shadow_stack(self, frame_id: int) -> ShadowStack:
        stack = self._shadow_stacks.get(frame_id)
        if stack is None:
            stack = ShadowStack()
            self._shadow_stacks[frame_id] = stack
        return stack

    def remove_shadow_stack(self, frame_id: int) -> None:
        self._shadow_stacks.pop(frame_id, None)

    def _trace_user_opcode(self, frame: Any) -> None:
        task_id = _task_id_var.get()
        if task_id is None or _scheduler_var.get() is not self or _in_trace_processing.get():
            return
        # When I/O detection (Redis/SQL) is active, skip opcode-level access
        # reporting entirely.  The I/O-level reporters already capture the
        # real key/table conflicts.  Running _process_opcode on user frames
        # would still pick up shared Python state (module globals, class
        # objects, connection pool internals) creating false DPOR wakeup tree
        # entries and excess path exploration for independent I/O operations.
        if self._detect_sql or self._detect_redis:
            return
        token = _in_trace_processing.set(True)
        try:
            with self._engine_lock:
                _process_opcode(frame, self, task_id)  # type: ignore[arg-type]
        finally:
            _in_trace_processing.reset(token)

    def _setup_settrace(self) -> None:
        def trace(frame: Any, event: str, arg: Any) -> Any:
            if event == "call":
                filename = frame.f_code.co_filename
                if _should_trace_file(filename):
                    if _is_dynamic_code(filename) and not _is_cmdline_user_code(filename, frame.f_globals):
                        caller = frame.f_back
                        if caller is None or not _should_trace_file(caller.f_code.co_filename):
                            return None
                    frame.f_trace_opcodes = True
                    return trace
                return None

            if event == "opcode":
                self._trace_user_opcode(frame)
                return trace

            if event == "return":
                self.remove_shadow_stack(id(frame))
                return trace

            return trace

        sys.settrace(trace)

    def _setup_monitoring(self) -> None:
        if not _USE_SYS_MONITORING:
            return

        mon = sys.monitoring
        tool_id = mon.PROFILER_ID  # type: ignore[attr-defined]
        AsyncDporScheduler._TOOL_ID = tool_id

        mon.use_tool_id(tool_id, "frontrun.async_dpor")  # type: ignore[attr-defined]
        mon.set_events(tool_id, mon.events.PY_START | mon.events.PY_RETURN | mon.events.INSTRUCTION)  # type: ignore[attr-defined]

        def handle_py_start(code: Any, instruction_offset: int) -> Any:
            if not _should_trace_file(code.co_filename):
                return mon.DISABLE  # type: ignore[attr-defined]
            return None

        def handle_py_return(code: Any, instruction_offset: int, retval: Any) -> Any:
            if not _should_trace_file(code.co_filename):
                return None
            if _task_id_var.get() is None or _scheduler_var.get() is not self:
                return None
            frame = sys._getframe(1)
            self.remove_shadow_stack(id(frame))
            return None

        def handle_instruction(code: Any, instruction_offset: int) -> Any:
            if not _should_trace_file(code.co_filename):
                return None
            if _task_id_var.get() is None or _scheduler_var.get() is not self:
                return None

            frame = sys._getframe(1)
            if _is_dynamic_code(code.co_filename) and not _is_cmdline_user_code(code.co_filename, frame.f_globals):
                caller = frame.f_back
                if caller is None or not _should_trace_file(caller.f_code.co_filename):
                    return None

            self._trace_user_opcode(frame)
            return None

        mon.register_callback(tool_id, mon.events.PY_START, handle_py_start)  # type: ignore[attr-defined]
        mon.register_callback(tool_id, mon.events.PY_RETURN, handle_py_return)  # type: ignore[attr-defined]
        mon.register_callback(tool_id, mon.events.INSTRUCTION, handle_instruction)  # type: ignore[attr-defined]
        self._monitoring_active = True

    def _teardown_monitoring(self) -> None:
        if not self._monitoring_active:
            return
        mon = sys.monitoring
        tool_id = AsyncDporScheduler._TOOL_ID
        if tool_id is not None:
            mon.set_events(tool_id, 0)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_START, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.PY_RETURN, None)  # type: ignore[attr-defined]
            mon.register_callback(tool_id, mon.events.INSTRUCTION, None)  # type: ignore[attr-defined]
            mon.free_tool_id(tool_id)  # type: ignore[attr-defined]
        self._monitoring_active = False

    def _row_lock_int_id(self, res_id: str) -> int:
        """Return a stable integer ID for *res_id* (allocated on first call)."""
        lid = self._row_lock_ids.get(res_id)
        if lid is None:
            lid = self._row_lock_next_id
            self._row_lock_next_id += 1
            self._row_lock_ids[res_id] = lid
        return lid

    def acquire_row_locks(self, thread_id: int, resource_ids: list[str]) -> None:
        """Track SQL row locks in the async WaitForGraph for cross-resource deadlock detection.

        In the async single-threaded context we cannot block waiting for a
        holder to release.  Instead we:
        - Record the holding edge when the lock is free (or already ours).
        - Detect cycles instantly via WaitForGraph when another task holds
          the lock, raising DeadlockError so explore_async_dpor reports it.
        """
        graph = _async_wait_graph
        self.engine.report_access(
            self.execution,
            thread_id,
            _SHARED_SYNC_ACQUIRE_KEY,
            "write",
        )
        for res_id in resource_ids:
            lock_int_id = self._row_lock_int_id(res_id)
            holder = self._active_row_locks.get(res_id)
            if holder is not None and holder != thread_id:
                # Another task holds this — check for deadlock cycle
                if graph is not None:
                    cycle = graph.add_waiting(thread_id, lock_int_id, kind="row_lock")
                    if cycle is not None:
                        graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                        desc = format_cycle(cycle, {v: k for k, v in self._row_lock_ids.items()})
                        raise DeadlockError(f"Row-lock deadlock detected: {desc}", desc)
                    # No cycle but contention — remove waiting edge (we can't
                    # actually block in async), let the SQL proceed.  The DB
                    # will handle the actual blocking and lock_timeout safety
                    # net prevents indefinite hangs.
                    graph.remove_waiting(thread_id, lock_int_id, kind="row_lock")
                # Even without a graph, record ownership optimistically so
                # that the DPOR engine can explore alternative interleavings.
            self._active_row_locks[res_id] = thread_id
            self._task_row_locks.setdefault(thread_id, set()).add(res_id)
            if graph is not None:
                graph.add_holding(thread_id, lock_int_id, kind="row_lock")

    def release_row_locks(self, thread_id: int) -> None:
        """Release all row locks held by *thread_id* (called on COMMIT/ROLLBACK)."""
        graph = _async_wait_graph
        held = self._task_row_locks.pop(thread_id, None)
        if not held:
            return
        for res_id in held:
            self._active_row_locks.pop(res_id, None)
            if graph is not None:
                lid = self._row_lock_ids.get(res_id)
                if lid is not None:
                    graph.remove_holding(thread_id, lid, kind="row_lock")

    def _flush_pending_io(self, task_id: int) -> None:
        """Flush pending I/O accesses to the DPOR engine."""
        pending = self._pending_io.get(task_id)
        if pending:
            for obj_key, kind, synced in pending:
                if synced:
                    self.engine.report_synced_io_access(self.execution, task_id, obj_key, kind)
                else:
                    self.engine.report_io_access(self.execution, task_id, obj_key, kind)
            pending.clear()

    def report_and_wait_sync(self, task_id: int) -> None:
        """Synchronous report-and-wait for use from SQL cursor interception.

        SQL cursor patching calls ``_get_dpor_context()`` which returns
        ``(scheduler, thread_id)``.  The sync DPOR scheduler has a
        ``report_and_wait(frame, thread_id)`` method.  For async DPOR,
        SQL interception needs a way to force a scheduling point
        synchronously (the cursor patch is called from inside an await).
        We just flush pending I/O here; the actual scheduling happens
        at the next ``await_point()``.
        """
        self._flush_pending_io(task_id)

    def report_and_wait(self, frame: Any, thread_id: int) -> bool:
        """Compatibility method for SQL cursor interception.

        The sync ``_sql_cursor.py`` and ``_sql_cursor_async.py`` call
        ``dpor_ctx[0].report_and_wait(None, thread_id)`` to force a
        scheduling point after SQL operations.  For async DPOR, we
        flush pending I/O but the actual scheduling happens at await
        points.  Returns True to indicate the task should continue.
        """
        self._flush_pending_io(thread_id)
        return True

    def finish_task(self, task_id: int) -> None:
        """Mark a task as finished in the DPOR engine."""
        self.execution.finish_thread(task_id)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


async def run_with_schedule_dpor(
    engine: PyDporEngine,
    execution: PyExecution,
    setup: Callable[[], Any],
    tasks: list[Callable[[Any], Coroutine[Any, Any, None]]],
    timeout: float = 5.0,
    deadlock_timeout: float = 5.0,
    detect_sql: bool = False,
    detect_redis: bool = False,
) -> Any:
    """Run one async DPOR execution and return the state object.

    Args:
        engine: The DPOR engine instance.
        execution: The current execution instance from the engine.
        setup: Returns fresh shared state.
        tasks: Async callables that each receive the state as their argument.
        timeout: Max seconds.
        deadlock_timeout: Seconds to wait before declaring a deadlock.
        detect_sql: If True, patch async DBAPI drivers for SQL tracking.
        detect_redis: If True, patch async Redis clients for key-level
            conflict detection.

    Returns:
        The state object after execution.
    """
    num_tasks = len(tasks)
    scheduler = AsyncDporScheduler(
        engine,
        execution,
        num_tasks,
        deadlock_timeout=deadlock_timeout,
        detect_sql=detect_sql,
        detect_redis=detect_redis,
    )

    state = setup()

    task_funcs: dict[int, Callable[..., Coroutine[Any, Any, None]]] = {
        i: (lambda s=state, t=t: t(s))  # type: ignore[assignment]
        for i, t in enumerate(tasks)
    }

    try:
        await scheduler.run_all(task_funcs, timeout=timeout)  # type: ignore[arg-type]
    except TimeoutError:
        pass

    # Mark any unfinished tasks as done in the DPOR engine
    for i in range(num_tasks):
        if i not in scheduler._tasks_done:
            scheduler.finish_task(i)

    return state


def _format_async_trace(schedule: list[int], num_tasks: int) -> str:
    """Generate a human-readable explanation of an async DPOR interleaving.

    Converts the raw schedule (list of task IDs at each scheduling point)
    into a readable description of task switches, making it easier to
    understand the interleaving that caused an invariant violation.
    """
    if not schedule:
        return "Invariant violation detected (empty schedule)."

    lines: list[str] = []
    lines.append("Invariant violation found after exploring interleaving schedule.")
    lines.append(f"Tasks: {num_tasks}, Schedule steps: {len(schedule)}")
    lines.append("")
    lines.append("Task interleaving (task ID at each scheduling point):")

    # Group consecutive runs by the same task
    runs: list[tuple[int, int]] = []  # (task_id, count)
    for tid in schedule:
        if runs and runs[-1][0] == tid:
            runs[-1] = (tid, runs[-1][1] + 1)
        else:
            runs.append((tid, 1))

    for i, (tid, count) in enumerate(runs):
        step_label = "step" if count == 1 else "steps"
        lines.append(f"  [{i + 1}] Task {tid}: {count} {step_label}")

    return "\n".join(lines)


class _ReplayAsyncScheduler(InterleavedLoop):
    """Replay a fixed schedule for async counterexample reproduction."""

    def __init__(self, schedule: list[int], num_tasks: int, *, deadlock_timeout: float = 5.0) -> None:
        super().__init__(deadlock_timeout=deadlock_timeout)
        self._replay_schedule = list(schedule)
        self._replay_index = 0
        self._replay_max_ops = len(self._replay_schedule) * 10 + 10_000
        self._current_task: int | None = schedule[0] if schedule else None
        self._num_replay_tasks = num_tasks

    def _extend_schedule(self) -> bool:
        if self._replay_index >= self._replay_max_ops:
            return False
        active = [t for t in range(self._num_replay_tasks) if t not in self._tasks_done]
        if not active:
            return False
        self._replay_schedule.extend(active)
        return True

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        if self._current_task is None:
            self._finished = True
            return True
        return self._current_task == task_id

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        while True:
            if self._replay_index >= len(self._replay_schedule):
                if not self._extend_schedule():
                    self._current_task = None
                    self._finished = True
                    return
            scheduled = self._replay_schedule[self._replay_index]
            self._replay_index += 1
            if scheduled not in self._tasks_done:
                self._current_task = scheduled
                return

    def finish_task(self, task_id: int) -> None:
        self._tasks_done.add(task_id)


async def _reproduce_async_counterexample(
    schedule_list: list[int],
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool] | None,
    num_tasks: int,
    reproduce_on_failure: int,
    timeout_per_run: float,
    deadlock_timeout: float,
) -> tuple[int, int]:
    """Measure how often an async DPOR counterexample reproduces."""
    successes = 0
    for _ in range(reproduce_on_failure):
        scheduler = _ReplayAsyncScheduler(schedule_list, num_tasks, deadlock_timeout=deadlock_timeout)
        state = setup()
        task_funcs: dict[int, Callable[..., Coroutine[Any, Any, None]]] = {
            i: (lambda s=state, t=t: t(s))  # type: ignore[assignment]
            for i, t in enumerate(tasks)
        }
        deadlocked = False
        try:
            await scheduler.run_all(task_funcs, timeout=timeout_per_run)  # type: ignore[arg-type]
        except DeadlockError:
            if invariant is None:
                successes += 1
            deadlocked = True
        except (TimeoutError, Exception):
            continue
        if not deadlocked:
            if invariant is not None and not invariant(state):
                successes += 1
    return reproduce_on_failure, successes


async def explore_async_dpor(
    setup: Callable[[], T],
    tasks: list[Callable[[T], Coroutine[Any, Any, None]]],
    invariant: Callable[[T], bool],
    max_executions: int | None = None,
    preemption_bound: int | None = 2,
    max_branches: int = 100_000,
    timeout_per_run: float = 5.0,
    stop_on_first: bool = True,
    deadlock_timeout: float = 5.0,
    detect_sql: bool = False,
    detect_redis: bool = False,
    trace_packages: list[str] | None = None,
    reproduce_on_failure: int = 10,
    total_timeout: float | None = None,
    warn_nondeterministic_sql: bool = True,
    lock_timeout: int | None = None,
) -> InterleavingResult:
    """Systematically explore async interleavings using DPOR.

    This is the async equivalent of ``explore_dpor()``.  Instead of threads,
    it runs async tasks with ``await_point()`` as the scheduling granularity.
    The Rust DPOR engine systematically explores every distinct interleaving,
    using vector clocks to prune redundant orderings.

    When ``detect_sql=True``, async database drivers (asyncpg, aiosqlite,
    psycopg AsyncCursor, aiomysql) are monkey-patched to report SQL
    table-level accesses to the DPOR engine.  This enables DPOR to detect
    SQL-level conflicts (e.g. two tasks writing the same table) and explore
    their orderings.

    When ``detect_redis=True``, async Redis clients (redis.asyncio, coredis)
    are monkey-patched to report key-level accesses to the DPOR engine.

    Args:
        setup: Creates fresh shared state for each execution.
        tasks: List of async callables, each receiving the shared state.
        invariant: Predicate over shared state; must be True after all
            tasks complete.
        max_executions: Safety limit on total executions (None = unlimited).
        preemption_bound: Limit on preemptions per execution. 2 catches most
            bugs. None = unbounded (full DPOR).
        max_branches: Maximum scheduling points per execution.
        timeout_per_run: Timeout for each individual run.
        stop_on_first: If True (default), stop on first invariant violation.
        deadlock_timeout: Seconds to wait before declaring a deadlock.
        detect_sql: If True, patch async DBAPI drivers for SQL tracking.
        detect_redis: If True, patch async Redis clients for key-level
            conflict detection.
        trace_packages: List of package name patterns (fnmatch syntax) to
            trace in addition to user code.  By default, code in
            site-packages is skipped.  Use this to include specific
            installed packages, e.g. ``["django_*", "mylib.*"]``.
        reproduce_on_failure: When a counterexample is found, replay the
            same schedule this many times to measure reproducibility.
            Set to 0 to skip.
        total_timeout: Maximum total time in seconds for the entire
            exploration.  None means no global deadline.
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
            scheduler from deadlocking when two tasks contend on the
            same PostgreSQL row lock.  Value is in milliseconds;
            2000 (2 seconds) is a good default.

    Returns:
        InterleavingResult with exploration statistics and any counterexample.
    """
    if trace_packages is not None:
        _set_active_trace_filter(_TraceFilter(trace_packages))
    num_tasks = len(tasks)
    pb = None if preemption_bound is None else preemption_bound
    me = None if max_executions is None else max_executions
    engine = PyDporEngine(
        num_threads=num_tasks,
        preemption_bound=pb,
        max_branches=max_branches,
        max_executions=me,
    )

    result = InterleavingResult(property_holds=True)
    stable_ids = StableObjectIds()
    total_deadline = time.monotonic() + total_timeout if total_timeout is not None else None

    if detect_sql and _sql_async_available:
        patch_sql_async()

    if detect_redis and _redis_async_available:
        patch_redis_async()

    clear_sql_metadata()

    # Inject SET lock_timeout on new PG connections (defect #6 workaround).
    prev_lock_timeout = get_lock_timeout()
    if lock_timeout is not None:
        set_lock_timeout(lock_timeout)

    _patch_asyncio_lock()

    try:
        while True:
            if total_deadline is not None and time.monotonic() > total_deadline:
                break
            clear_insert_tracker()
            stable_ids.reset_for_execution()
            execution = engine.begin_execution()

            # Clear wait-for graph and held-locks tracking between executions
            if _async_wait_graph is not None:
                _async_wait_graph.clear()
            _async_lock_owners.clear()
            _async_task_held_locks.clear()

            scheduler = AsyncDporScheduler(
                engine,
                execution,
                num_tasks,
                deadlock_timeout=deadlock_timeout,
                detect_sql=detect_sql,
                detect_redis=detect_redis,
                stable_ids=stable_ids,
            )

            state = setup()

            task_funcs: dict[int, Callable[..., Coroutine[Any, Any, None]]] = {
                i: (lambda s=state, t=t: t(s))  # type: ignore[assignment]
                for i, t in enumerate(tasks)
            }

            deadlock_error: DeadlockError | None = None
            task_error: Exception | None = None
            timed_out = False
            try:
                await scheduler.run_all(task_funcs, timeout=timeout_per_run)  # type: ignore[arg-type]
            except DeadlockError as e:
                deadlock_error = e
            except TimeoutError:
                timed_out = True
            except Exception as e:
                # Task raised an exception (not deadlock/timeout).
                # This is a valid exploration outcome — the cleanup already
                # happened in _run's finally block, so lock state is clean.
                # Record it and check the invariant below.
                task_error = e

            # Mark any unfinished tasks as done in the DPOR engine
            unfinished = [i for i in range(num_tasks) if i not in scheduler._tasks_done]
            for i in unfinished:
                scheduler.finish_task(i)

            result.num_explored += 1

            # Check for deadlock: explicit DeadlockError from wait-for
            # graph cycle detection, or timeout (tasks blocked on locks
            # get cancelled by run_all and appear "done" via _mark_done
            # in the finally block, so we can't rely on unfinished alone).
            is_deadlock = False
            deadlock_explanation = ""
            if deadlock_error is not None:
                is_deadlock = True
                deadlock_explanation = f"Deadlock detected: {deadlock_error.cycle_description}"
            elif timed_out:
                is_deadlock = True
                if unfinished:
                    stuck = ", ".join(str(t) for t in unfinished)
                    deadlock_explanation = (
                        f"Deadlock detected: tasks [{stuck}] did not complete. "
                        f"All tasks were blocked and could not make progress."
                    )
                else:
                    deadlock_explanation = (
                        "Deadlock detected: all tasks were blocked and timed out. "
                        "Tasks could not make progress (likely waiting on locks held by each other)."
                    )

            if is_deadlock:
                result.property_holds = False
                schedule_list = list(execution.schedule_trace)
                result.failures.append((result.num_explored, schedule_list))
                if result.counterexample is None:
                    result.counterexample = schedule_list
                    result.explanation = deadlock_explanation
                if reproduce_on_failure > 0 and result.reproduction_attempts == 0:
                    attempts, successes = await _reproduce_async_counterexample(
                        schedule_list=schedule_list,
                        setup=setup,
                        tasks=tasks,
                        invariant=None,
                        num_tasks=num_tasks,
                        reproduce_on_failure=reproduce_on_failure,
                        timeout_per_run=timeout_per_run,
                        deadlock_timeout=deadlock_timeout,
                    )
                    result.reproduction_attempts = attempts
                    result.reproduction_successes = successes
                if stop_on_first:
                    return result
            elif task_error is not None:
                result.property_holds = False
                schedule_list = list(execution.schedule_trace)
                result.failures.append((result.num_explored, schedule_list))
                if result.counterexample is None:
                    result.counterexample = schedule_list
                    exc_type = type(task_error).__name__
                    result.explanation = f"Task crash in execution {result.num_explored}: {exc_type}: {task_error}"
                if stop_on_first:
                    return result

            if warn_nondeterministic_sql:
                check_uncaptured_inserts()

            if not is_deadlock and task_error is None and not invariant(state):
                result.property_holds = False
                schedule_list = list(execution.schedule_trace)
                result.failures.append((result.num_explored, schedule_list))
                if result.counterexample is None:
                    result.counterexample = schedule_list
                    result.explanation = _format_async_trace(schedule_list, num_tasks)
                if reproduce_on_failure > 0 and result.reproduction_attempts == 0:
                    attempts, successes = await _reproduce_async_counterexample(
                        schedule_list=schedule_list,
                        setup=setup,
                        tasks=tasks,
                        invariant=invariant,
                        num_tasks=num_tasks,
                        reproduce_on_failure=reproduce_on_failure,
                        timeout_per_run=timeout_per_run,
                        deadlock_timeout=deadlock_timeout,
                    )
                    result.reproduction_attempts = attempts
                    result.reproduction_successes = successes
                if stop_on_first:
                    return result

            if not engine.next_execution():
                break
    finally:
        _set_active_trace_filter(None)
        set_lock_timeout(prev_lock_timeout)
        _unpatch_asyncio_lock()
        if detect_sql and _sql_async_available:
            unpatch_sql_async()
        if detect_redis and _redis_async_available:
            unpatch_redis_async()

    return result
