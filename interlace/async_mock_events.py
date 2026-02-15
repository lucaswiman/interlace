"""
AsyncInterlace: Deterministic control of async task interleaving for testing.

This module provides a checkpoint-based API for testing race conditions in async/await
code by controlling the exact ordering of operations across multiple async tasks.

Built on the shared InterleavedLoop abstraction, using checkpoint-based scheduling:
a list of (task_name, operation, phase) tuples defines the execution order.

## Key Insight: Async Race Conditions Only Happen at Await Points

Unlike threading where race conditions can occur anywhere due to preemptive scheduling,
async/await code is single-threaded and cooperative. The event loop only yields control
at explicit `await` statements. This means:

1. **Race conditions ONLY happen at await points** - Between two await statements,
   your code runs atomically without interruption.

2. **No need for sys.settrace** - We don't need to instrument every line of code.
   We only need to control the order of task resumption after await points.

3. **Simpler mental model** - You can see exactly where races can occur by looking
   for await statements in your code.

This makes async interlacing conceptually simpler than threading, though the API
remains similar for consistency with the threading-based mock_events.py.

## Overview

Race conditions in async code are easier to reason about than in threaded code, but
still need testing. AsyncInterlace solves this by giving you precise control over
task interleaving at await points, making it possible to:

1. **Deterministically reproduce race conditions** - Force specific interleavings
   that expose bugs in async code
2. **Test that races exist** - Prove that unprotected async operations have race conditions
3. **Verify fixes work** - Show that proper synchronization prevents races

## Basic Usage

The typical workflow is:

1. Add `await checkpoint()` calls to your async code at synchronization points
2. Define concurrent tasks using @task decorators (tasks must be async functions)
3. Specify the execution order using order()
4. Run the tasks with controlled interleaving using `await run()`

## Example: Testing an Async Bank Account Race Condition

```python
class BankAccount:
    _interlace = None  # Set by tests to enable checkpoints

    def __init__(self, balance=0):
        self.balance = balance

    async def transfer(self, amount):
        current = self.balance  # READ
        if self._interlace:
            await self._interlace.checkpoint('transfer', 'after_read')

        new_balance = current + amount  # COMPUTE

        if self._interlace:
            await self._interlace.checkpoint('transfer', 'before_write')
        self.balance = new_balance  # WRITE
        return new_balance

# Test that forces a race condition
account = BankAccount(balance=100)

async with AsyncInterlace() as il:
    BankAccount._interlace = il

    @il.task('task1')
    async def task1():
        await account.transfer(50)

    @il.task('task2')
    async def task2():
        await account.transfer(50)

    # Force both tasks to read before either writes
    il.order([
        ('task1', 'transfer', 'after_read'),   # T1 reads 100
        ('task2', 'transfer', 'after_read'),   # T2 reads 100
        ('task1', 'transfer', 'before_write'), # T1 writes 150
        ('task2', 'transfer', 'before_write'), # T2 writes 150 (overwrites!)
    ])

    await il.run()
    BankAccount._interlace = None

assert account.balance == 150  # One update was lost!
```

## Key Concepts

**Checkpoints**: Synchronization points (await points) where tasks wait for permission
to proceed. Insert checkpoints in your async code with
`await il.checkpoint(operation_name, phase_name)`.

**Tasks**: Concurrent async operations defined with `@il.task('name')`. Each task is
an async coroutine that runs in the same event loop.

**Order**: The sequence of checkpoints that defines the interleaving. Specified
as a list of (task_name, operation_name, phase_name) tuples.

**Phases**: Named stages within an operation (e.g., 'after_read', 'before_write').
You choose phase names that make sense for your code.

## Differences from Threading Version

1. **All tasks must be async** - Use `async def` and `await` syntax
2. **Checkpoints must be awaited** - Use `await il.checkpoint(...)` not `il.checkpoint(...)`
3. **run() must be awaited** - Use `await il.run()` not `il.run()`
4. **Single event loop** - All tasks run in the same event loop, no separate threads
5. **async context manager** - Use `async with AsyncInterlace()` not `with Interlace()`

## API Reference

See the AsyncInterlace class for detailed API documentation.
"""

import asyncio
import functools
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

from interlace.async_scheduler import InterleavedLoop


class _CheckpointLoop(InterleavedLoop):
    """Scheduling policy: linear sequence of (task, operation, phase) steps.

    This is the InterleavedLoop subclass used internally by AsyncInterlace.
    The marker passed to pause() is a (operation, phase) tuple.
    """

    def __init__(self, sequence: List[Tuple[str, str, str]]):
        super().__init__()
        self._sequence = sequence
        self._step = 0
        # task-name tracking: maps asyncio.Task id → task name
        self._task_locals: Dict[int, str] = {}

    # -- InterleavedLoop policy -----------------------------------------

    def should_proceed(self, task_id: Any, marker: Any = None) -> bool:
        # Skip past steps for done tasks
        while self._step < len(self._sequence):
            if self._sequence[self._step][0] in self._tasks_done:
                self._step += 1
                continue
            break

        if self._step >= len(self._sequence):
            self._finished = True
            return True

        expected = self._sequence[self._step]
        if marker is None:
            return False
        return expected == (task_id, marker[0], marker[1])

    def on_proceed(self, task_id: Any, marker: Any = None) -> None:
        if self._step < len(self._sequence):
            self._step += 1

    def _handle_timeout(self, task_id: Any, marker: Any = None) -> None:
        key = (task_id,) + tuple(marker) if marker else (task_id,)
        self._error = TimeoutError(
            f"Timeout waiting for {key}. "
            f"Check that all sequence steps are reachable."
        )
        self._condition.notify_all()

    def _setup_task_context(self, task_id: Any) -> None:
        task = asyncio.current_task()
        if task:
            self._task_locals[id(task)] = task_id

    def _cleanup_task_context(self, task_id: Any) -> None:
        task = asyncio.current_task()
        if task:
            self._task_locals.pop(id(task), None)

    # -- Helpers --------------------------------------------------------

    def get_current_task_name(self) -> Optional[str]:
        """Return the task name for the currently running asyncio Task."""
        task = asyncio.current_task()
        return self._task_locals.get(id(task)) if task else None


class AsyncInterlace:
    """
    Async context manager for controlling task interleaving in tests.

    This class allows you to define concurrent async tasks and specify the exact
    order in which their operations should execute, making it possible to
    deterministically reproduce race conditions in async code.

    Internally delegates to an InterleavedLoop subclass for coordination.
    """

    def __init__(self):
        """Initialize a new AsyncInterlace coordinator."""
        self.tasks: Dict[str, Callable] = {}
        self._loop: Optional[_CheckpointLoop] = None
        self._sequence: List[Tuple[str, str, str]] = []
        self._patches: List[mock._patch] = []

    def task(self, name: str) -> Callable:
        """
        Decorator to register an async function as a concurrent task.

        Args:
            name: Unique identifier for this task

        Returns:
            Decorator function that registers the task

        Example:
            >>> @il.task('writer1')
            ... async def write_data():
            ...     await data.append(1)
        """
        def decorator(func: Callable) -> Callable:
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"Task '{name}' must be an async function (use 'async def')")
            self.tasks[name] = func
            return func
        return decorator

    def intercept(self, target: str, point: str = 'call') -> Callable:
        """
        Decorator to mark an async method/function for interception and ordering.

        Args:
            target: Name/identifier for this interception point
            point: 'call', 'entry', or 'exit' (default: 'call')

        Returns:
            Decorator that adds interception logic
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                task_name = self._loop.get_current_task_name() if self._loop else None

                if task_name:
                    if point in ('call', 'entry'):
                        await self._loop.pause(task_name, (target, 'entry'))

                    result = await func(*args, **kwargs)

                    if point in ('call', 'exit'):
                        await self._loop.pause(task_name, (target, 'exit'))

                    return result
                else:
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def patch(self, target: str, operation_name: Optional[str] = None):
        """
        Patch an async method/function to intercept calls and enforce ordering.

        Args:
            target: Full path to the method (e.g., 'bank.BankAccount.transfer')
            operation_name: Name used in order() (defaults to class.method from target)

        Returns:
            Self for chaining
        """
        import importlib

        if operation_name is None:
            parts = target.split('.')
            if len(parts) >= 2:
                operation_name = f"{parts[-2]}.{parts[-1]}"
            else:
                operation_name = parts[-1]

        parts = target.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid target: {target}")

        module_path, attr_name = parts
        module_name, class_name = module_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        original = getattr(cls, attr_name)

        @functools.wraps(original)
        async def wrapper(*args, **kwargs):
            task_name = self._loop.get_current_task_name() if self._loop else None
            if task_name is None:
                return await original(*args, **kwargs)

            await self._loop.pause(task_name, (operation_name, 'entry'))
            try:
                result = await original(*args, **kwargs)
            finally:
                await self._loop.pause(task_name, (operation_name, 'exit'))
            return result

        patcher = mock.patch.object(cls, attr_name, wrapper)
        patcher.start()
        self._patches.append(patcher)
        return self

    def order(self, sequence: List[Tuple[str, str, str]]):
        """
        Define the execution order of operations across tasks.

        Each tuple specifies (task_name, operation, phase) where:
        - task_name: The name given to a @task
        - operation: The operation name used in checkpoint() calls
        - phase: The phase name used in checkpoint() calls

        Args:
            sequence: List of (task, operation, phase) tuples defining order

        Example:
            >>> il.order([
            ...     ('task1', 'transfer', 'after_read'),
            ...     ('task2', 'transfer', 'after_read'),
            ...     ('task1', 'transfer', 'before_write'),
            ...     ('task2', 'transfer', 'before_write'),
            ... ])
        """
        self._sequence = sequence
        self._loop = _CheckpointLoop(sequence)

    async def checkpoint(self, operation: str, phase: str = 'checkpoint'):
        """
        Insert a synchronization checkpoint within an async function.

        This allows fine-grained control over execution order within
        a single async function call. The checkpoint must be awaited.

        Args:
            operation: Name of the operation (matches what's in order())
            phase: Name of this checkpoint phase
        """
        if self._loop is None:
            return

        task_name = self._loop.get_current_task_name()
        if task_name:
            await self._loop.pause(task_name, (operation, phase))

    async def run(self):
        """
        Execute all registered tasks concurrently with controlled interleaving.

        Delegates to the InterleavedLoop for task lifecycle management.
        """
        if self._loop is None:
            raise RuntimeError("Must call order() before run()")

        await self._loop.run_all(self.tasks, timeout=15.0)

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager and clean up patches."""
        for patcher in self._patches:
            patcher.stop()
        self._patches.clear()
        return False


@asynccontextmanager
async def async_interlace():
    """
    Convenience async context manager for creating an AsyncInterlace coordinator.

    Yields:
        AsyncInterlace: A new AsyncInterlace instance

    Example:
        >>> async with async_interlace() as il:
        ...     @il.task('task1')
        ...     async def task1(): ...
    """
    coordinator = AsyncInterlace()
    try:
        yield coordinator
    finally:
        pass


class AsyncInterleaveBuilder:
    """
    Fluent API builder for creating interleaving specifications.

    This provides an alternative, more readable way to define complex
    interleavings without manually writing out all tuples.

    Example:
        >>> builder = AsyncInterleaveBuilder()
        >>> builder.step('t1', 'method', 'after_read') \\
        ...        .step('t2', 'method', 'after_read') \\
        ...        .step('t1', 'method', 'before_write') \\
        ...        .step('t2', 'method', 'before_write')
        >>> il.order(builder.build())
    """

    def __init__(self):
        """Initialize an empty interleave builder."""
        self._steps: List[Tuple[str, str, str]] = []

    def step(self, task: str, operation: str, phase: str = 'call') -> 'AsyncInterleaveBuilder':
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
