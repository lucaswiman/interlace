"""
Interlace: Deterministic control of concurrent thread interleaving for testing.

This module provides a checkpoint-based API for testing race conditions by
controlling the exact ordering of operations across multiple threads. It uses
threading.Event to orchestrate execution and allows fine-grained control over
when threads proceed at specific points in their execution.

## Overview

Race conditions are notoriously difficult to test because they depend on
unpredictable thread scheduling. Interlace solves this by giving you precise
control over thread interleaving, making it possible to:

1. **Deterministically reproduce race conditions** - Force specific interleavings
   that expose bugs
2. **Test that races exist** - Prove that unprotected code has race conditions
3. **Verify fixes work** - Show that proper synchronization prevents races

## Basic Usage

The typical workflow is:

1. Add checkpoint() calls to your code at synchronization points
2. Define concurrent tasks using @task decorators
3. Specify the execution order using order()
4. Run the tasks with controlled interleaving using run()

## Example: Testing a Bank Account Race Condition

```python
class BankAccount:
    _interlace = None  # Set by tests to enable checkpoints

    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        current = self.balance  # READ
        if self._interlace:
            self._interlace.checkpoint('transfer', 'after_read')

        new_balance = current + amount  # COMPUTE

        if self._interlace:
            self._interlace.checkpoint('transfer', 'before_write')
        self.balance = new_balance  # WRITE
        return new_balance

# Test that forces a race condition
account = BankAccount(balance=100)

with Interlace() as il:
    BankAccount._interlace = il

    @il.task('thread1')
    def task1():
        account.transfer(50)

    @il.task('thread2')
    def task2():
        account.transfer(50)

    # Force both threads to read before either writes
    il.order([
        ('thread1', 'transfer', 'after_read'),   # T1 reads 100
        ('thread2', 'transfer', 'after_read'),   # T2 reads 100
        ('thread1', 'transfer', 'before_write'), # T1 writes 150
        ('thread2', 'transfer', 'before_write'), # T2 writes 150 (overwrites!)
    ])

    il.run()
    BankAccount._interlace = None

assert account.balance == 150  # One update was lost!
```

## Key Concepts

**Checkpoints**: Synchronization points where threads wait for permission to proceed.
Insert checkpoints in your code with `il.checkpoint(operation_name, phase_name)`.

**Tasks**: Concurrent operations defined with `@il.task('name')`. Each task runs
in its own thread.

**Order**: The sequence of checkpoints that defines the interleaving. Specified
as a list of (task_name, operation_name, phase_name) tuples.

**Phases**: Named stages within an operation (e.g., 'after_read', 'before_write').
You choose phase names that make sense for your code.

## API Reference

See the Interlace class for detailed API documentation.
"""

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock


class Interlace:
    """
    Context manager for controlling thread interleaving in tests.

    This class allows you to define concurrent tasks and specify the exact
    order in which their operations should execute, making it possible to
    deterministically reproduce race conditions.

    Attributes:
        tasks: Dictionary mapping task names to task functions
        threads: Dictionary mapping task names to Thread objects
        _events: Internal event coordination structure
        _patches: Active mock patches
        _current_thread_name: Thread-local storage for task names
    """

    def __init__(self):
        """Initialize a new Interlace coordinator."""
        self.tasks: Dict[str, Callable] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self._events: Dict[Tuple[str, str, str], threading.Event] = {}
        self._next_events: Dict[Tuple[str, str, str], threading.Event] = {}
        self._patches: List[mock._patch] = []
        self._current_thread_name = threading.local()
        self._sequence: List[Tuple[str, str, str]] = []
        self._intercepts: Dict[str, Any] = {}

    def task(self, name: str) -> Callable:
        """
        Decorator to register a function as a concurrent task.

        Args:
            name: Unique identifier for this task

        Returns:
            Decorator function that registers the task

        Example:
            >>> @il.task('writer1')
            ... def write_data():
            ...     data.append(1)
        """
        def decorator(func: Callable) -> Callable:
            self.tasks[name] = func
            return func
        return decorator

    def intercept(self, target: str, point: str = 'call') -> Callable:
        """
        Decorator to mark a method/function for interception and ordering.

        This is an alternative API that decorates the actual methods being tested
        rather than wrapping them via mock.patch. More explicit but requires
        modifying the class under test.

        Args:
            target: Name/identifier for this interception point
            point: 'call', 'entry', or 'exit' (default: 'call')

        Returns:
            Decorator that adds interception logic
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                thread_name = getattr(self._current_thread_name, 'name', None)
                if thread_name:
                    if point in ('call', 'entry'):
                        self._wait_for_turn(thread_name, target, 'entry')

                    if point == 'call':
                        result = func(*args, **kwargs)
                        self._signal_next(thread_name, target, 'exit')
                        return result
                    else:
                        result = func(*args, **kwargs)
                        if point == 'exit':
                            self._signal_next(thread_name, target, 'exit')
                        return result
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def order(self, sequence: List[Tuple[str, str, str]]):
        """
        Define the execution order of operations across tasks.

        Each tuple specifies (task_name, operation, phase) where:
        - task_name: The name given to a @task
        - operation: The method/function being intercepted (e.g., 'BankAccount.transfer')
        - phase: Either 'entry' (before execution) or 'exit' (after execution)

        Args:
            sequence: List of (task, operation, phase) tuples defining order

        Example:
            >>> il.order([
            ...     ('thread1', 'transfer', 'entry'),   # thread1 starts transfer
            ...     ('thread2', 'transfer', 'entry'),   # thread2 starts transfer
            ...     ('thread1', 'transfer', 'exit'),    # thread1 finishes transfer
            ...     ('thread2', 'transfer', 'exit'),    # thread2 finishes transfer
            ... ])
        """
        self._sequence = sequence

        # First pass: Create all events
        for i, step in enumerate(sequence):
            event = threading.Event()
            self._events[step] = event

            # First step starts ready
            if i == 0:
                event.set()

        # Second pass: Map each step to the next step's event
        for i, step in enumerate(sequence):
            if i < len(sequence) - 1:
                next_step = sequence[i + 1]
                self._next_events[step] = self._events[next_step]

    def patch(self, target: str, operation_name: Optional[str] = None):
        """
        Patch a method/function to intercept calls and enforce ordering.

        Args:
            target: Full path to the method (e.g., 'bank.BankAccount.transfer')
            operation_name: Name used in order() (defaults to class.method from target)

        Returns:
            Self for chaining

        Example:
            >>> il.patch('mymodule.BankAccount.transfer')
            >>> il.patch('mymodule.Account.withdraw', 'withdraw')
        """
        import importlib

        # Extract operation name from target if not provided
        if operation_name is None:
            parts = target.split('.')
            if len(parts) >= 2:
                operation_name = f"{parts[-2]}.{parts[-1]}"
            else:
                operation_name = parts[-1]

        # Store the target
        self._intercepts[operation_name] = target

        # Apply the patch immediately
        parts = target.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid target: {target}")

        module_path, attr_name = parts

        # Import the module and get the original function
        module_name, class_name = module_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        original = getattr(cls, attr_name)

        # Create wrapper for this specific operation
        wrapper = self._create_wrapper(original, operation_name)

        # Use mock.patch.object to replace the method on the class
        patcher = mock.patch.object(cls, attr_name, wrapper)
        patcher.start()

        self._patches.append(patcher)

        return self

    def _create_wrapper(self, original: Callable, operation_name: str) -> Callable:
        """
        Create a wrapper function that enforces ordering around the original.

        This wrapper enforces ordering at the CALL level - it controls when the
        entire function executes. For finer-grained control within a function,
        the function itself needs to call checkpoint() at specific points.

        Args:
            original: The original function being wrapped
            operation_name: Name used in the sequence

        Returns:
            Wrapped function with interleaving control
        """
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            thread_name = getattr(self._current_thread_name, 'name', None)

            # If not in a managed task, just call original
            if thread_name is None:
                return original(*args, **kwargs)

            # Wait for our turn to enter
            self._wait_for_turn(thread_name, operation_name, 'entry')
            # Signal that we've entered
            self._signal_next(thread_name, operation_name, 'entry')

            try:
                # Execute the original function
                result = original(*args, **kwargs)
            finally:
                # Wait for our turn to exit
                self._wait_for_turn(thread_name, operation_name, 'exit')
                # Signal that we've exited
                self._signal_next(thread_name, operation_name, 'exit')

            return result

        return wrapper

    def checkpoint(self, operation: str, phase: str = 'checkpoint'):
        """
        Insert a synchronization checkpoint within a function.

        This allows fine-grained control over execution order within
        a single function call, not just around the entire call.

        Args:
            operation: Name of the operation (matches what's in order())
            phase: Name of this checkpoint phase

        Example:
            >>> def my_function(self):
            ...     value = self.read()
            ...     il.checkpoint('my_function', 'after_read')
            ...     self.write(value + 1)
        """
        thread_name = getattr(self._current_thread_name, 'name', None)
        if thread_name:
            self._wait_for_turn(thread_name, operation, phase)
            self._signal_next(thread_name, operation, phase)

    def _wait_for_turn(self, task_name: str, operation: str, phase: str):
        """
        Block until it's this task's turn to execute the operation phase.

        Args:
            task_name: Name of the current task
            operation: Name of the operation being executed
            phase: 'entry' or 'exit'
        """
        key = (task_name, operation, phase)
        event = self._events.get(key)

        if event is not None:
            # Wait for our turn (with timeout to prevent hanging tests)
            if not event.wait(timeout=10.0):
                raise TimeoutError(
                    f"Timeout waiting for {key}. "
                    f"Check that all sequence steps are reachable."
                )

    def _signal_next(self, task_name: str, operation: str, phase: str):
        """
        Signal that this step is complete, allowing the next step to proceed.

        Args:
            task_name: Name of the current task
            operation: Name of the operation being executed
            phase: 'entry' or 'exit'
        """
        key = (task_name, operation, phase)
        next_event = self._next_events.get(key)

        if next_event is not None:
            next_event.set()

    def run(self):
        """
        Execute all registered tasks concurrently with controlled interleaving.

        This starts all tasks in separate threads and waits for them to complete.
        The order() specification controls how their operations interleave.

        Example:
            >>> with Interlace() as il:
            ...     @il.task('t1')
            ...     def task1(): ...
            ...     il.order([...])
            ...     il.run()  # Execute with controlled interleaving
        """
        # Create threads for each task
        for name, func in self.tasks.items():
            def make_runner(task_name: str, task_func: Callable) -> Callable:
                def runner():
                    # Set thread-local name so intercepted methods know who they are
                    self._current_thread_name.name = task_name
                    task_func()
                return runner

            thread = threading.Thread(
                target=make_runner(name, func),
                name=name
            )
            self.threads[name] = thread

        # Start all threads
        for thread in self.threads.values():
            thread.start()

        # Wait for all threads to complete
        for thread in self.threads.values():
            thread.join(timeout=15.0)
            if thread.is_alive():
                raise TimeoutError(
                    f"Thread {thread.name} did not complete. "
                    f"Check for deadlocks in your sequence."
                )

    def __enter__(self):
        """Enter the context manager."""
        # Patches are now applied in patch() method, not here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up patches."""
        # Stop all patches
        for patcher in self._patches:
            patcher.stop()
        self._patches.clear()

        return False


@contextmanager
def interlace():
    """
    Convenience context manager for creating an Interlace coordinator.

    Yields:
        Interlace: A new Interlace instance

    Example:
        >>> with interlace() as il:
        ...     @il.task('task1')
        ...     def task1(): ...
    """
    coordinator = Interlace()
    try:
        yield coordinator
    finally:
        pass


class InterleaveBuilder:
    """
    Fluent API builder for creating interleaving specifications.

    This provides an alternative, more readable way to define complex
    interleavings without manually writing out all tuples.

    Example:
        >>> builder = InterleaveBuilder()
        >>> builder.step('t1', 'method', 'entry') \\
        ...        .step('t2', 'method', 'entry') \\
        ...        .step('t1', 'method', 'exit') \\
        ...        .step('t2', 'method', 'exit')
        >>> il.order(builder.build())
    """

    def __init__(self):
        """Initialize an empty interleave builder."""
        self._steps: List[Tuple[str, str, str]] = []

    def step(self, task: str, operation: str, phase: str = 'call') -> 'InterleaveBuilder':
        """
        Add a step to the interleaving sequence.

        Args:
            task: Task name
            operation: Operation name
            phase: 'entry', 'exit', or 'call' (call adds both entry and exit)

        Returns:
            Self for chaining
        """
        if phase == 'call':
            self._steps.append((task, operation, 'entry'))
            self._steps.append((task, operation, 'exit'))
        else:
            self._steps.append((task, operation, phase))
        return self

    def build(self) -> List[Tuple[str, str, str]]:
        """
        Build and return the sequence.

        Returns:
            List of (task, operation, phase) tuples
        """
        return self._steps
