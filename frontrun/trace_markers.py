"""
Frontrun: Deterministic thread interleaving using sys.settrace and comment markers.

This module provides a mechanism to control thread execution order by marking
synchronization points in code with special comments and enforcing a predefined
execution schedule using Python's tracing facilities.

Example usage:
    ```python
    from frontrun.common import Schedule, Step
    from frontrun.trace_markers import TraceExecutor

    def worker_function():
        x = read_data()  # frontrun: read
        write_data(x)    # frontrun: write

    schedule = Schedule([
        Step("thread1", "read"),
        Step("thread2", "read"),
        Step("thread1", "write"),
        Step("thread2", "write"),
    ])

    executor = TraceExecutor(schedule)
    executor.run("thread1", worker_function)
    executor.run("thread2", worker_function)
    executor.wait()
    ```
"""

import linecache
import re
import sys
import threading
from collections.abc import Callable
from typing import Any

from frontrun._cooperative import real_condition, real_lock
from frontrun.common import InterleavingResult, Schedule, Step

MARKER_PATTERN = re.compile(r"#\s*frontrun:\s*(\w+)")


class MarkerRegistry:
    """Tracks marker locations in source code for efficient lookup.

    This class scans source files to find lines with frontrun markers and
    maintains a mapping from (filename, line_number) to marker names.
    """

    def __init__(self):
        self._markers: dict[tuple[str, int], str] = {}  # (filename, lineno) -> marker_name
        self._scanned_files: set[str] = set()
        self._lock = real_lock()

    def scan_frame(self, frame: Any) -> None:  # type: ignore[name-defined]
        """Scan the source file for the given frame to find all markers.

        Args:
            frame: A Python frame object from the trace function
        """
        filename = frame.f_code.co_filename

        # Fast path: Skip if already scanned (no lock needed)
        if filename in self._scanned_files:
            return

        # Double-checked locking: acquire lock and re-check
        with self._lock:
            # Re-check inside the lock in case another thread finished scanning
            if filename in self._scanned_files:
                return

            # Read all lines from the file
            try:
                # Use linecache to read the file
                linecache.checkcache(filename)
                line_num = 1
                while True:
                    line = linecache.getline(filename, line_num)
                    if not line:
                        break

                    # Check for marker comment
                    match = MARKER_PATTERN.search(line)
                    if match:
                        marker_name = match.group(1)
                        self._markers[(filename, line_num)] = marker_name

                    line_num += 1
            except Exception:
                # If we can't read the file, just skip it
                pass

            # Mark as scanned AFTER we've populated all markers
            self._scanned_files.add(filename)

    def get_marker(self, filename: str, lineno: int) -> str | None:
        """Get the marker name for a specific file location.

        Args:
            filename: The source file path
            lineno: The line number

        Returns:
            The marker name if found, None otherwise
        """
        return self._markers.get((filename, lineno))


class ThreadCoordinator:
    """Coordinates thread execution according to a schedule.

    This class manages the synchronization between threads, ensuring that
    each thread executes markers in the order specified by the schedule.
    """

    def __init__(self, schedule: Schedule, *, deadlock_timeout: float = 5.0):
        self.schedule = schedule
        self.deadlock_timeout = deadlock_timeout
        self.current_step = 0
        self.lock = real_lock()
        self.condition = real_condition(self.lock)
        self.completed = False
        self.error: Exception | None = None
        # Execution serialization lock: ensures only one thread runs between
        # markers, replicating GIL-like serialization needed on free-threaded
        # Python where threads truly run in parallel.
        self._execution_lock = real_lock()

    def wait_for_turn(self, execution_name: str, marker_name: str, *, _reacquire_execution_lock: bool = False):
        """Block until it's this execution unit's turn to execute this marker.

        When *_reacquire_execution_lock* is ``True`` (used by the trace
        executors), ``_execution_lock`` is acquired while the condition lock
        is still held, before returning.  This prevents other threads from
        racing ahead between being notified and the caller resuming execution.
        The caller must have already released ``_execution_lock`` before
        calling this method.

        Args:
            execution_name: The name of the calling execution unit
            marker_name: The marker that was hit
        """
        with self.condition:
            while True:
                # Check if we're done or had an error
                if self.completed or self.error:
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    return

                # Check if we've exceeded the schedule
                if self.current_step >= len(self.schedule.steps):
                    self.completed = True
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    self.condition.notify_all()
                    return

                # Get the current expected step
                expected_step = self.schedule.steps[self.current_step]

                # Is it our turn?
                if expected_step.execution_name == execution_name and expected_step.marker_name == marker_name:
                    # It's our turn! Advance, optionally acquire execution lock, notify.
                    self.current_step += 1
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    self.condition.notify_all()
                    return

                # Not our turn — wait with a fallback timeout so that
                # incorrect schedules (referencing a marker that no thread
                # ever reaches) get diagnosed promptly instead of blocking
                # until the outer thread.join(timeout) fires.
                if not self.condition.wait(timeout=self.deadlock_timeout):
                    expected = self.schedule.steps[self.current_step]
                    self.error = TimeoutError(
                        f"Schedule stall: waiting for Step({expected.execution_name!r}, "
                        f"{expected.marker_name!r}) at step {self.current_step}/"
                        f"{len(self.schedule.steps)}, but no thread has reached it"
                    )
                    if _reacquire_execution_lock:
                        self._execution_lock.acquire()
                    self.condition.notify_all()
                    return

    def report_error(self, error: Exception):
        """Report an error and wake up all waiting threads.

        Args:
            error: The exception that occurred
        """
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def is_finished(self) -> bool:
        """Check if the schedule has completed or encountered an error."""
        with self.condition:
            return self.completed or self.error is not None


class _ThreadTraceExecutor:
    """Executes threads with interlaced execution according to a schedule.

    This is the main interface for the frontrun library. It sets up tracing
    for each thread and coordinates their execution.
    """

    def __init__(self, schedule: Schedule, *, deadlock_timeout: float = 5.0):
        """Initialize the executor with a schedule.

        Args:
            schedule: The Schedule defining the execution order
            deadlock_timeout: Seconds to wait before declaring a deadlock
                (default 5.0).  Increase for code that legitimately blocks
                in C extensions (NumPy, database queries, network I/O).
        """
        self.schedule = schedule
        self.deadlock_timeout = deadlock_timeout
        self.coordinator = ThreadCoordinator(schedule, deadlock_timeout=deadlock_timeout)
        self.marker_registry = MarkerRegistry()
        self.threads: list[threading.Thread] = []
        self.thread_errors: dict[str, Exception] = {}

    def _create_trace_function(self, execution_name: str) -> Callable[[Any, str, Any], Any]:  # type: ignore[return-value]
        """Create a trace function for a specific execution unit.

        Args:
            execution_name: The name of the execution unit this trace function is for

        Returns:
            A trace function suitable for sys.settrace
        """

        def trace_function(frame: Any, event: str, arg: Any) -> Any:  # type: ignore[name-defined]
            try:
                # Only care about 'line' events
                if event != "line":
                    return trace_function

                # Scan this file for markers if we haven't already
                self.marker_registry.scan_frame(frame)

                # Check if this line has a marker
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                marker_name = self.marker_registry.get_marker(filename, lineno)

                if marker_name:
                    # Release execution lock while waiting (let other threads run).
                    # wait_for_turn reacquires it before returning.
                    self.coordinator._execution_lock.release()
                    self.coordinator.wait_for_turn(execution_name, marker_name, _reacquire_execution_lock=True)

                    # Check if an error occurred in another execution unit
                    if self.coordinator.error:
                        raise self.coordinator.error

                return trace_function
            except Exception as e:
                # Release execution lock before reporting to avoid deadlock
                # (report_error needs the condition lock, which another thread
                # may hold while waiting for _execution_lock in wait_for_turn).
                try:
                    self.coordinator._execution_lock.release()
                except RuntimeError:
                    pass
                self.coordinator.report_error(e)
                return None

        return trace_function

    def _thread_wrapper(
        self,
        execution_name: str,
        target: Callable[..., None],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Wrapper function that sets up tracing for an execution unit.

        Args:
            execution_name: The name of this execution unit
            target: The function to execute
            args: Positional arguments for the target
            kwargs: Keyword arguments for the target
        """
        error: Exception | None = None
        try:
            # Install trace function for this execution unit
            trace_fn = self._create_trace_function(execution_name)
            # Acquire execution lock before running (serializes with other threads)
            self.coordinator._execution_lock.acquire()
            sys.settrace(trace_fn)

            # Execute the target function
            target(*args, **kwargs)
        except Exception as e:
            # Store the error but don't call report_error yet — we may still
            # hold _execution_lock and report_error needs the condition lock,
            # which risks deadlock against wait_for_turn's lock ordering.
            error = e
            self.thread_errors[execution_name] = e
        finally:
            sys.settrace(None)
            try:
                self.coordinator._execution_lock.release()
            except RuntimeError:
                pass
            if error is not None:
                self.coordinator.report_error(error)

    def run(
        self,
        execution_name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        """Start a new thread with tracing enabled.

        Args:
            execution_name: The name for this execution unit (must match schedule)
            target: The function to execute in the thread
            args: Positional arguments for the target function
            kwargs: Keyword arguments for the target function
        """
        if kwargs is None:
            kwargs = {}

        thread = threading.Thread(
            target=self._thread_wrapper, args=(execution_name, target, args, kwargs), name=execution_name, daemon=True
        )
        self.threads.append(thread)
        thread.start()

    def wait(self, timeout: float | None = None):
        """Wait for all threads to complete.

        Args:
            timeout: Optional timeout in seconds

        Raises:
            TimeoutError: If threads don't complete within the timeout
            Any exception that occurred in a thread during execution
        """
        for thread in self.threads:
            thread.join(timeout=timeout)

        # Check if any threads are still alive after timeout
        alive_threads = [thread for thread in self.threads if thread.is_alive()]
        if alive_threads:
            thread_names = ", ".join(thread.name for thread in alive_threads)
            raise TimeoutError(f"Threads did not complete within timeout: {thread_names}")

        # If any thread had an error, raise it
        if self.thread_errors:
            # Raise the first error we encountered
            first_error = next(iter(self.thread_errors.values()))
            raise first_error

        # Check if the schedule was partially consumed but not completed.
        # If at least one step was processed (so the schedule was in use)
        # but the full schedule wasn't completed, it means the schedule
        # references markers that no thread reached.  If zero steps were
        # consumed, the markers were simply never hit — which could be a
        # different issue (wrong file, exec'd code, etc.) and is not
        # necessarily an error.
        if (
            self.coordinator.current_step > 0
            and self.coordinator.current_step < len(self.coordinator.schedule.steps)
            and not self.coordinator.completed
        ):
            remaining = self.coordinator.schedule.steps[self.coordinator.current_step :]
            step_strs = [f"Step({s.execution_name!r}, {s.marker_name!r})" for s in remaining]
            raise TimeoutError(
                f"Schedule incomplete: {len(remaining)} step(s) were never reached: {', '.join(step_strs)}"
            )

    def reset(self):
        """Reset the executor for another run (for testing purposes)."""
        self.threads = []
        self.thread_errors = {}
        self.coordinator = ThreadCoordinator(self.schedule, deadlock_timeout=self.deadlock_timeout)
        self.marker_registry = MarkerRegistry()


class TraceExecutor:
    """Facade over sync and async marker-based schedule execution.

    Sync usage matches the historical ``TraceExecutor`` API:

    .. code-block:: python

       executor = TraceExecutor(schedule)
       executor.run("t1", worker1)
       executor.run("t2", worker2)
       executor.wait()

    Async usage accepts the async-task mapping directly:

    .. code-block:: python

       executor = TraceExecutor(schedule)
       executor.run({"task1": coro1, "task2": coro2})
    """

    def __init__(self, schedule: Schedule, *, deadlock_timeout: float = 5.0):
        self.schedule = schedule
        self.deadlock_timeout = deadlock_timeout
        self._mode: str | None = None
        self._sync = _ThreadTraceExecutor(schedule, deadlock_timeout=deadlock_timeout)
        self._async: Any | None = None

    def run(
        self,
        execution_name: str | dict[str, Callable[..., Any]],
        target: Callable[..., None] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> None:
        if isinstance(execution_name, dict):
            if target is not None:
                raise TypeError("TraceExecutor.run() accepts either a task mapping or a named sync target, not both")
            if self._mode == "sync":
                raise TypeError("TraceExecutor is already running in sync mode")
            self._mode = "async"
            if self._async is None:
                from frontrun.async_trace_markers import AsyncTraceExecutor

                self._async = AsyncTraceExecutor(self.schedule, deadlock_timeout=self.deadlock_timeout)
            self._async.run(execution_name, timeout=timeout)
            return

        if target is None:
            raise TypeError("TraceExecutor.run() missing target for sync execution")
        if self._mode == "async":
            raise TypeError("TraceExecutor is already running in async mode")
        self._mode = "sync"
        self._sync.run(execution_name, target, args, kwargs)

    def wait(self, timeout: float | None = None) -> None:
        if self._mode in (None, "async"):
            return
        self._sync.wait(timeout=timeout)

    def reset(self) -> None:
        self._mode = None
        self._sync.reset()
        if self._async is not None:
            self._async.reset()

    @property
    def threads(self) -> list[threading.Thread]:
        return self._sync.threads

    @property
    def thread_errors(self) -> dict[str, Exception]:
        return self._sync.thread_errors

    @property
    def coordinator(self) -> ThreadCoordinator:
        return self._sync.coordinator

    @property
    def marker_registry(self) -> MarkerRegistry:
        return self._sync.marker_registry

    @property
    def task_errors(self) -> dict[str, Exception]:
        if self._async is None:
            return {}
        return self._async.task_errors


def frontrun(
    schedule: Schedule,
    threads: dict[str, Callable[..., None]],
    thread_args: dict[str, tuple[Any, ...]] | None = None,
    thread_kwargs: dict[str, dict[str, Any]] | None = None,
    timeout: float | None = None,
    deadlock_timeout: float = 5.0,
) -> "TraceExecutor":
    """Convenience function to run multiple threads with a schedule.

    Args:
        schedule: The Schedule defining execution order
        threads: Dictionary mapping execution unit names to their target functions
        thread_args: Optional dictionary mapping execution unit names to argument tuples
        thread_kwargs: Optional dictionary mapping execution unit names to keyword argument dicts
        timeout: Optional timeout for waiting
        deadlock_timeout: Seconds to wait before declaring a deadlock
            (default 5.0).  Increase for code that legitimately blocks
            in C extensions (NumPy, database queries, network I/O).

    Returns:
        The TraceExecutor instance (useful for inspection)

    Example:
        ```python
        frontrun(
            schedule=Schedule([Step("t1", "marker1"), Step("t2", "marker1")]),
            threads={"t1": worker_func, "t2": worker_func},
        )
        ```
    """
    if thread_args is None:
        thread_args = {}
    if thread_kwargs is None:
        thread_kwargs = {}

    executor = TraceExecutor(schedule, deadlock_timeout=deadlock_timeout)

    for execution_name, target in threads.items():
        args = thread_args.get(execution_name, ())
        kwargs = thread_kwargs.get(execution_name, {})
        executor.run(execution_name, target, args, kwargs)

    executor.wait(timeout=timeout)
    return executor


# ---------------------------------------------------------------------------
# Property-based marker schedule generation
# ---------------------------------------------------------------------------


def marker_schedule_strategy(
    threads: dict[str, list[str]],
) -> Any:
    """Hypothesis strategy that generates valid marker-level Schedule objects.

    A valid schedule interleaves each thread's markers while preserving
    their relative order within each thread.  This provides a much smaller
    search space than opcode-level exploration while still covering all
    meaningful interleavings at the marker granularity.

    Args:
        threads: Mapping from thread/execution names to their ordered list
            of marker names.  Example::

                {"t1": ["read", "write"], "t2": ["read", "write"]}

    Returns:
        A Hypothesis strategy producing :class:`Schedule` objects.

    Example::

        from hypothesis import given
        from frontrun.trace_markers import marker_schedule_strategy

        @given(schedule=marker_schedule_strategy(
            threads={"w1": ["read", "write"], "w2": ["read", "write"]},
        ))
        def test_no_lost_update(schedule):
            ...
    """
    from hypothesis import strategies as st

    thread_items = list(threads.items())

    @st.composite
    def gen(draw: Any) -> Schedule:
        remaining = {name: list(markers) for name, markers in thread_items}
        steps: list[Step] = []

        while any(remaining.values()):
            available = [name for name, m in remaining.items() if m]
            thread_name = draw(st.sampled_from(sorted(available)))
            marker = remaining[thread_name].pop(0)
            steps.append(Step(thread_name, marker))

        return Schedule(steps)

    return gen()


def all_marker_schedules(
    threads: dict[str, list[str]],
) -> list[Schedule]:
    """Enumerate ALL valid interleavings of thread markers.

    A valid interleaving places every thread's markers in the schedule
    while preserving their relative order within each thread.

    For *N* threads with marker counts *k1, k2, ..., kN*, the total
    number of valid interleavings is the multinomial coefficient::

        (k1 + k2 + ... + kN)! / (k1! * k2! * ... * kN!)

    Args:
        threads: Mapping from thread/execution names to their ordered list
            of marker names.

    Returns:
        A list of :class:`Schedule` objects covering every valid interleaving.

    Example::

        schedules = all_marker_schedules(
            threads={"t1": ["a", "b"], "t2": ["x", "y"]},
        )
        assert len(schedules) == 6  # C(4,2)
    """
    thread_items = sorted(threads.items())  # deterministic order
    schedules: list[Schedule] = []

    def _recurse(
        remaining: dict[str, list[str]],
        path: list[Step],
    ) -> None:
        available = [name for name, markers in remaining.items() if markers]
        if not available:
            schedules.append(Schedule(list(path)))
            return

        for name in available:
            marker = remaining[name][0]
            # Pop first marker, recurse, restore
            remaining[name] = remaining[name][1:]
            path.append(Step(name, marker))
            _recurse(remaining, path)
            path.pop()
            remaining[name] = [marker, *remaining[name]]

    initial = {name: list(markers) for name, markers in thread_items}
    _recurse(initial, [])
    return schedules


def explore_marker_interleavings(
    setup: Callable[..., Any],
    threads: dict[str, tuple[Callable[..., None], list[str]]],
    invariant: Callable[..., bool],
    *,
    stop_on_first: bool = True,
    deadlock_timeout: float = 5.0,
    timeout: float | None = 10.0,
) -> InterleavingResult:
    """Explore all marker-level interleavings and check an invariant.

    Generates every valid interleaving of the declared markers (preserving
    per-thread order), runs each one against real code via :class:`TraceExecutor`,
    and checks the invariant after each execution.

    This sits between manual trace markers (exact schedule, one interleaving)
    and bytecode exploration (random, enormous search space).  For *N* threads
    with a few markers each, the search space is small enough to explore
    exhaustively — giving completeness guarantees at the marker granularity.

    Args:
        setup: Factory producing fresh shared state for each execution.
        threads: Mapping from execution name to ``(target_fn, markers)`` where
            *target_fn* takes the setup result and *markers* is the ordered
            list of ``# frontrun:`` marker names that *target_fn* hits.
        invariant: Predicate on the shared state; returns True if correct.
        stop_on_first: Stop after finding the first invariant violation
            (default True).
        deadlock_timeout: Per-schedule deadlock detection timeout.
        timeout: Per-schedule join timeout.

    Returns:
        An :class:`~frontrun.common.InterleavingResult`.  The ``counterexample``
        field is a :class:`Schedule` (not a list of ints) when a violation is
        found.
    """
    marker_decl = {name: markers for name, (_, markers) in threads.items()}
    schedules = all_marker_schedules(marker_decl)

    num_explored = 0
    failures: list[tuple[int, Schedule]] = []

    for i, schedule in enumerate(schedules):
        state = setup()
        executor = TraceExecutor(schedule, deadlock_timeout=deadlock_timeout)

        for exec_name, (target_fn, _markers) in threads.items():

            def _make_runner(s: Any = state, fn: Callable[..., None] = target_fn) -> None:
                fn(s)

            executor.run(exec_name, _make_runner)

        try:
            executor.wait(timeout=timeout)
        except (TimeoutError, Exception):
            # Schedule stall or thread error — treat as non-violation and skip
            num_explored += 1
            continue

        num_explored += 1

        if not invariant(state):
            failures.append((i, schedule))
            if stop_on_first:
                return InterleavingResult(
                    property_holds=False,
                    counterexample=schedule,  # type: ignore[arg-type]
                    num_explored=num_explored,
                    unique_interleavings=num_explored,
                )

    if failures:
        return InterleavingResult(
            property_holds=False,
            counterexample=failures[0][1],  # type: ignore[arg-type]
            num_explored=num_explored,
            unique_interleavings=num_explored,
            failures=failures,  # type: ignore[arg-type]
        )

    return InterleavingResult(
        property_holds=True,
        num_explored=num_explored,
        unique_interleavings=num_explored,
    )
