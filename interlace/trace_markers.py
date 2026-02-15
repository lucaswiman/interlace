"""
Interlace: Deterministic thread interleaving using sys.settrace and comment markers.

This module provides a mechanism to control thread execution order by marking
synchronization points in code with special comments and enforcing a predefined
execution schedule using Python's tracing facilities.

Example usage:
    ```python
    def worker_function():
        x = read_data()  # interlace: read
        write_data(x)    # interlace: write

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

import sys
import threading
import linecache
import re
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass


MARKER_PATTERN = re.compile(r'#\s*interlace:\s*(\w+)')


@dataclass
class Step:
    """Represents a single step in the execution schedule.

    Attributes:
        thread_name: The name of the thread that should execute this step
        marker_name: The marker name that identifies this synchronization point
    """
    thread_name: str
    marker_name: str

    def __repr__(self):
        return f"Step({self.thread_name!r}, {self.marker_name!r})"


class Schedule:
    """Defines the execution order for threads at synchronization points.

    A schedule is a linear sequence of steps that specify which thread should
    execute which marker in order.
    """

    def __init__(self, steps: List[Step]):
        """Initialize a schedule with a list of steps.

        Args:
            steps: Ordered list of Step objects defining the execution sequence
        """
        self.steps = steps
        self._validate()

    def _validate(self):
        """Validate that the schedule is well-formed."""
        if not self.steps:
            raise ValueError("Schedule must contain at least one step")

    def __repr__(self):
        return f"Schedule({self.steps!r})"


class MarkerRegistry:
    """Tracks marker locations in source code for efficient lookup.

    This class scans source files to find lines with interlace markers and
    maintains a mapping from (filename, line_number) to marker names.
    """

    def __init__(self):
        self._markers: Dict[tuple, str] = {}  # (filename, lineno) -> marker_name
        self._scanned_files: set = set()

    def scan_frame(self, frame):
        """Scan the source file for the given frame to find all markers.

        Args:
            frame: A Python frame object from the trace function
        """
        filename = frame.f_code.co_filename

        # Skip if already scanned
        if filename in self._scanned_files:
            return

        self._scanned_files.add(filename)

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

    def get_marker(self, filename: str, lineno: int) -> Optional[str]:
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

    def __init__(self, schedule: Schedule):
        self.schedule = schedule
        self.current_step = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.completed = False
        self.error: Optional[Exception] = None

    def wait_for_turn(self, thread_name: str, marker_name: str):
        """Block until it's this thread's turn to execute this marker.

        Args:
            thread_name: The name of the calling thread
            marker_name: The marker that was hit
        """
        with self.condition:
            while True:
                # Check if we're done or had an error
                if self.completed or self.error:
                    return

                # Check if we've exceeded the schedule
                if self.current_step >= len(self.schedule.steps):
                    self.completed = True
                    self.condition.notify_all()
                    return

                # Get the current expected step
                expected_step = self.schedule.steps[self.current_step]

                # Is it our turn?
                if (expected_step.thread_name == thread_name and
                    expected_step.marker_name == marker_name):
                    # It's our turn! Advance and notify others
                    self.current_step += 1
                    self.condition.notify_all()
                    return

                # Not our turn, wait
                self.condition.wait()

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


class TraceExecutor:
    """Executes threads with interlaced execution according to a schedule.

    This is the main interface for the interlace library. It sets up tracing
    for each thread and coordinates their execution.
    """

    def __init__(self, schedule: Schedule):
        """Initialize the executor with a schedule.

        Args:
            schedule: The Schedule defining the execution order
        """
        self.schedule = schedule
        self.coordinator = ThreadCoordinator(schedule)
        self.marker_registry = MarkerRegistry()
        self.threads: List[threading.Thread] = []
        self.thread_errors: Dict[str, Exception] = {}

    def _create_trace_function(self, thread_name: str):
        """Create a trace function for a specific thread.

        Args:
            thread_name: The name of the thread this trace function is for

        Returns:
            A trace function suitable for sys.settrace
        """
        def trace_function(frame, event, arg):
            try:
                # Only care about 'line' events
                if event != 'line':
                    return trace_function

                # Scan this file for markers if we haven't already
                self.marker_registry.scan_frame(frame)

                # Check if this line has a marker
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                marker_name = self.marker_registry.get_marker(filename, lineno)

                if marker_name:
                    # We hit a marker! Wait for our turn
                    self.coordinator.wait_for_turn(thread_name, marker_name)

                    # Check if an error occurred in another thread
                    if self.coordinator.error:
                        raise self.coordinator.error

                return trace_function
            except Exception as e:
                # Report error and stop tracing
                self.coordinator.report_error(e)
                return None

        return trace_function

    def _thread_wrapper(self, thread_name: str, target: Callable, args: tuple, kwargs: dict):
        """Wrapper function that sets up tracing for a thread.

        Args:
            thread_name: The name of this thread
            target: The function to execute
            args: Positional arguments for the target
            kwargs: Keyword arguments for the target
        """
        try:
            # Install trace function for this thread
            trace_fn = self._create_trace_function(thread_name)
            sys.settrace(trace_fn)

            # Execute the target function
            target(*args, **kwargs)
        except Exception as e:
            # Store the error
            self.thread_errors[thread_name] = e
            self.coordinator.report_error(e)
        finally:
            # Clean up trace function
            sys.settrace(None)

    def run(self, thread_name: str, target: Callable, args: tuple = (), kwargs: dict = None):
        """Start a new thread with tracing enabled.

        Args:
            thread_name: The name for this thread (must match schedule)
            target: The function to execute in the thread
            args: Positional arguments for the target function
            kwargs: Keyword arguments for the target function
        """
        if kwargs is None:
            kwargs = {}

        thread = threading.Thread(
            target=self._thread_wrapper,
            args=(thread_name, target, args, kwargs),
            name=thread_name
        )
        self.threads.append(thread)
        thread.start()

    def wait(self, timeout: Optional[float] = None):
        """Wait for all threads to complete.

        Args:
            timeout: Optional timeout in seconds

        Raises:
            Any exception that occurred in a thread during execution
        """
        for thread in self.threads:
            thread.join(timeout=timeout)

        # If any thread had an error, raise it
        if self.thread_errors:
            # Raise the first error we encountered
            first_error = next(iter(self.thread_errors.values()))
            raise first_error

    def reset(self):
        """Reset the executor for another run (for testing purposes)."""
        self.threads = []
        self.thread_errors = {}
        self.coordinator = ThreadCoordinator(self.schedule)
        self.marker_registry = MarkerRegistry()


def interlace(schedule: Schedule, threads: Dict[str, Callable],
              thread_args: Optional[Dict[str, tuple]] = None,
              thread_kwargs: Optional[Dict[str, dict]] = None,
              timeout: Optional[float] = None):
    """Convenience function to run multiple threads with a schedule.

    Args:
        schedule: The Schedule defining execution order
        threads: Dictionary mapping thread names to their target functions
        thread_args: Optional dictionary mapping thread names to argument tuples
        thread_kwargs: Optional dictionary mapping thread names to keyword argument dicts
        timeout: Optional timeout for waiting

    Returns:
        The TraceExecutor instance (useful for inspection)

    Example:
        ```python
        interlace(
            schedule=Schedule([Step("t1", "marker1"), Step("t2", "marker1")]),
            threads={"t1": worker_func, "t2": worker_func},
        )
        ```
    """
    if thread_args is None:
        thread_args = {}
    if thread_kwargs is None:
        thread_kwargs = {}

    executor = TraceExecutor(schedule)

    for thread_name, target in threads.items():
        args = thread_args.get(thread_name, ())
        kwargs = thread_kwargs.get(thread_name, {})
        executor.run(thread_name, target, args, kwargs)

    executor.wait(timeout=timeout)
    return executor
