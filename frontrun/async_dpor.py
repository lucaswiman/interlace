"""Async DPOR (Dynamic Partial Order Reduction) for frontrun.

Combines the DPOR engine's systematic interleaving exploration with the
async scheduler's await-point-level control.  Instead of the random
schedule sampling used by ``async_bytecode.py``, this module uses the
Rust DPOR engine to explore every meaningfully distinct interleaving
exactly once.

The approach:
1. A Rust DPOR engine (frontrun._dpor) manages the exploration tree,
   vector clocks, and backtrack set computation.
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
from collections.abc import Awaitable, Callable, Coroutine, Generator
from typing import Any, TypeVar

from frontrun._deadlock import DeadlockError, WaitForGraph, format_cycle
from frontrun.async_scheduler import InterleavedLoop
from frontrun.common import InterleavingResult

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


T = TypeVar("T")

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


# ---------------------------------------------------------------------------
# Object key helper (shared with sync dpor.py)
# ---------------------------------------------------------------------------


def _make_object_key(obj_id: int, name: Any) -> int:
    """Create a non-negative u64 object key for the Rust engine."""
    return hash((obj_id, name)) & 0xFFFFFFFFFFFFFFFF


# A single shared resource key that all tasks write to at every await point.
# This creates write-write conflicts between all tasks, ensuring DPOR explores
# all distinct orderings.
_SHARED_AWAIT_KEY = _make_object_key(0, "__async_dpor_await_point__")


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
        if scheduler is not None and task_id is not None:
            await scheduler.pause(task_id)

        if task_id is not None and graph is not None and self._lock.locked():
            lock_id = id(self)
            # Register: this task is waiting for this lock
            cycle = graph.add_waiting(task_id, lock_id)
            if cycle is not None:
                graph.remove_waiting(task_id, lock_id)
                desc = format_cycle(cycle)
                raise DeadlockError(f"Async lock deadlock detected: {desc}", desc)
            try:
                result = await self._lock.acquire()
            finally:
                graph.remove_waiting(task_id, lock_id)
        else:
            result = await self._lock.acquire()

        # Record ownership
        if task_id is not None and graph is not None:
            lock_id = id(self)
            self._owner = task_id
            _async_lock_owners[lock_id] = task_id
            _async_task_held_locks.setdefault(task_id, set()).add(self)
            graph.add_holding(task_id, lock_id)

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
    decide which task runs next.  At each await point, reports the access
    to the engine and asks it for the next scheduling decision.
    """

    def __init__(
        self,
        engine: PyDporEngine,
        execution: PyExecution,
        num_tasks: int,
        *,
        deadlock_timeout: float = 5.0,
        detect_sql: bool = False,
    ) -> None:
        super().__init__(deadlock_timeout=deadlock_timeout)
        self.engine = engine
        self.execution = execution
        self._num_engine_tasks = num_tasks
        self._current_task: int | None = None
        self._detect_sql = detect_sql
        # Pending I/O accesses per task (from SQL interception)
        self._pending_io: dict[int, list[tuple[int, str]]] = {i: [] for i in range(num_tasks)}

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
        """Ask the DPOR engine which task to run next."""
        runnable = self.execution.runnable_threads()
        if not runnable:
            return None
        return self.engine.schedule(self.execution)

    # -- InterleavedLoop policy -----------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        if self._current_task is None:
            self._finished = True
            return True
        return self._current_task == task_id

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        # Flush any pending I/O accesses before advancing
        self._flush_pending_io(task_id)

        # Report the await point as a WRITE to a single shared resource.
        # All tasks write to the same "await_point" resource, creating
        # write-write conflicts that force DPOR to explore alternative
        # orderings at every await point.  This is the async equivalent
        # of opcode-level tracing in sync DPOR.
        self.engine.report_access(
            self.execution,
            task_id,
            _SHARED_AWAIT_KEY,
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

        if self._detect_sql:

            def _io_reporter(resource_id: str, kind: str) -> None:
                object_key = _make_object_key(hash(resource_id), resource_id)
                self._pending_io.setdefault(task_id, []).append((object_key, kind))

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

        await super().run_all(wrapped, timeout=timeout)

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

        set_dpor_scheduler(None)
        set_dpor_thread_id(None)
        if self._detect_sql:
            set_io_reporter(None)

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
            for obj_key, kind in pending:
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

    Returns:
        InterleavingResult with exploration statistics and any counterexample.
    """
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

    if detect_sql and _sql_async_available:
        patch_sql_async()

    _patch_asyncio_lock()

    try:
        while True:
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
            elif not invariant(state):
                result.property_holds = False
                schedule_list = list(execution.schedule_trace)
                result.failures.append((result.num_explored, schedule_list))
                if result.counterexample is None:
                    result.counterexample = schedule_list
                    result.explanation = _format_async_trace(schedule_list, num_tasks)
                if stop_on_first:
                    return result

            if not engine.next_execution():
                break
    finally:
        _unpatch_asyncio_lock()
        if detect_sql and _sql_async_available:
            unpatch_sql_async()

    return result
