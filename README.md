# Interlace

A library for deterministic concurrency testing that helps you reliably reproduce and test race conditions.

## Overview

Interlace provides tools for controlling thread interleaving at a fine-grained level, allowing you to:

- **Deterministically reproduce race conditions** - Force specific interleavings to make race conditions happen reliably in tests
- **Test concurrent code exhaustively** - Explore different execution orders to find bugs
- **Verify synchronization correctness** - Ensure that proper locking prevents race conditions

Instead of relying on timing-based race detection (which is unreliable), Interlace lets you control exactly when threads execute, making concurrency testing deterministic and reproducible.

## Quick Start: Bank Account Race Condition

Here's a simple example showing how to use Interlace to detect a race condition:

```python
import threading
from interlace.mock_events import Interlace

class BankAccount:
    _interlace = None

    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        # This method has a race condition: read-modify-write without synchronization
        current = self.balance  # READ
        if self._interlace:
            self._interlace.checkpoint('transfer', 'after_read')

        new_balance = current + amount  # COMPUTE

        if self._interlace:
            self._interlace.checkpoint('transfer', 'before_write')
        self.balance = new_balance  # WRITE
        return new_balance

# Demonstrate the race condition
account = BankAccount(balance=100)

with Interlace() as il:
    BankAccount._interlace = il

    @il.task('thread1')
    def task1():
        account.transfer(50)

    @il.task('thread2')
    def task2():
        account.transfer(50)

    # Define an interleaving that triggers the race:
    # Both threads read before either writes
    il.order([
        ('thread1', 'transfer', 'after_read'),    # T1 reads 100
        ('thread2', 'transfer', 'after_read'),    # T2 reads 100 (both see same value!)
        ('thread1', 'transfer', 'before_write'),  # T1 writes 150
        ('thread2', 'transfer', 'before_write'),  # T2 writes 150 (overwrites T1's update!)
    ])

    il.run()
    BankAccount._interlace = None

# Result: balance is 150, not 200 - one update was lost!
assert account.balance == 150, "Race condition detected!"
```

## Usage Approaches

Interlace provides two different ways to control thread interleaving:

### 1. Trace Markers

Use comment-based markers for reproducing known race conditions or investigating whether a particular race condition is possible. Markers can be placed inline with code or on empty lines:

```python
from interlace.trace_markers import Schedule, Step, TraceExecutor

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        # Markers on empty lines (cleaner):
        # interlace: after_read
        temp = self.value
        temp += 1
        # interlace: before_write
        self.value = temp

        # Or use inline markers:
        # temp = self.value  # interlace: after_read
        # self.value = temp  # interlace: before_write

counter = Counter()

# Define execution order using markers
schedule = Schedule([
    Step("thread1", "after_read"),
    Step("thread2", "after_read"),
    Step("thread1", "before_write"),
    Step("thread2", "before_write"),
])

def worker1():
    counter.increment()

def worker2():
    counter.increment()

# Run with controlled interleaving
executor = TraceExecutor(schedule)
executor.run("thread1", worker1)
executor.run("thread2", worker2)
executor.wait(timeout=5.0)

assert counter.value == 1  # Race condition!
```

**Advantages:**
- Lightweight - just comments in code
- Two placement styles: empty lines (cleaner) or inline (compact)
- Automatic execution tracing
- Simple and readable
- No semantic code changes required
- Stable and well-tested

### 2. Bytecode Manipulation (Experimental)

> ⚠️ **Warning:** Bytecode instrumentation is **experimental**. It requires monkey-patching basic concurrency primitives during test execution and only works on python>=3.12 (which includes tracing opcodes rather than lines. Use with caution.

Automatically instrument functions using bytecode rewriting. No checkpoints needed!

```python
from interlace.bytecode import explore_interleavings

class Counter:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        temp = self.value
        self.value = temp + 1

# Use explore_interleavings to find race conditions
result = explore_interleavings(
    setup=lambda: Counter(value=0),
    threads=[
        lambda c: c.increment(),
        lambda c: c.increment(),
    ],
    invariant=lambda c: c.value == 2,  # This should hold if no races exist
    max_attempts=200,
    max_ops=200,
    seed=42,
)

if not result.property_holds:
    print("Race condition found!")
    print(f"After {result.num_explored} attempts")
    print(f"Expected value: 2, Got: {result.counterexample.value}")
```

Or use controlled interleaving with explicit schedules:

```python
from interlace.bytecode import controlled_interleaving, OpcodeScheduler, BytecodeInterlace

counter = Counter(value=0)

def thread1():
    counter.increment()

def thread2():
    counter.increment()

# Sequential schedule: thread 1 completes before thread 2 starts
schedule = [0] * 200 + [1] * 200
with controlled_interleaving(schedule, num_threads=2) as runner:
    runner.run([thread1, thread2])

print(f"Sequential result: {counter.value}")  # Correct: 2

# Interleaved schedule: rapidly alternate between threads
counter2 = Counter(value=0)
schedule = [0, 1] * 150  # Alternate rapidly
with controlled_interleaving(schedule, num_threads=2) as runner:
    runner.run([
        lambda: counter2.increment(),
        lambda: counter2.increment(),
    ])

print(f"Interleaved result: {counter2.value}")  # Might be wrong due to race
```

**Advantages (Experimental):**
- No manual checkpoint insertion needed
- Can test unmodified third-party code
- Property-based exploration finds edge cases
- Automatic bytecode instrumentation

**Limitations:**
- Results may vary across Python versions
- Some bytecode patterns may not be instrumented correctly
- Performance impact is higher than trace markers
- Less predictable behavior than explicit markers
- **API may change in future versions**

### When to Use Each Approach

**Use Trace Markers (recommended) if:**
- You want stable, predictable behavior
- You're comfortable adding comments to code
- You have specific synchronization points in mind
- You need deterministic test results
- You're testing code you can modify

**Use Bytecode Instrumentation (with caution) if:**
- You need to test unmodified third-party code
- You want automatic race condition discovery
- You're comfortable with experimental features
- You have time to handle potential API changes

## Async Support

Both approaches have async variants. Trace marker async support is stable, while bytecode instrumentation async support is experimental (see above).

### Async Trace Markers (Recommended)

```python
import asyncio
from interlace.async_trace_markers import AsyncSchedule, AsyncStep, AsyncTraceExecutor

class Counter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        # interlace: after_read
        temp = self.value
        await asyncio.sleep(0)  # Simulate async work
        # interlace: before_write
        self.value = temp + 1

async def test():
    counter = Counter()

    # Define execution order using markers
    schedule = AsyncSchedule([
        AsyncStep("task1", "after_read"),
        AsyncStep("task2", "after_read"),
        AsyncStep("task1", "before_write"),
        AsyncStep("task2", "before_write"),
    ])

    async def worker1():
        await counter.increment()

    async def worker2():
        await counter.increment()

    # Run with controlled interleaving
    executor = AsyncTraceExecutor(schedule)
    executor.run("task1", worker1)
    executor.run("task2", worker2)
    await executor.wait(timeout=5.0)

    assert counter.value == 1  # Race condition!

asyncio.run(test())
```

**Key insight:** In async code, race conditions only happen at `await` points since the event loop is single-threaded. Marker placement is crucial for proper synchronization.

## Development

### Running Tests

```bash
# From the interlace directory
make test

# Or from the root directory
make test-interlace
```

All tests pass successfully (40 tests total):

```
tests/test_bytecode.py ............... (11 tests)
tests/test_async_bytecode.py ......... (9 tests)
tests/test_trace_markers.py .......... (9 tests)
tests/test_async_trace_markers.py .... (9 tests)
tests/test_interlace.py .............. (2 tests)
```

### Project Structure

```
interlace/
├── Makefile                    # Project build targets
├── pyproject.toml              # Python packaging configuration
├── README.md                   # This file
├── docs/                       # Documentation (Sphinx/ReadTheDocs)
├── interlace/                  # Python package
│   ├── __init__.py
│   ├── trace_markers.py        # Trace marker approach (recommended)
│   ├── async_trace_markers.py  # Async trace markers (recommended)
│   ├── bytecode.py             # Bytecode instrumentation (experimental)
│   ├── async_bytecode.py       # Async bytecode instrumentation (experimental)
│   └── async_scheduler.py      # Async scheduling utilities
└── tests/                      # Test suite
    ├── test_trace_markers.py
    ├── test_async_trace_markers.py
    ├── test_bytecode.py        # Bytecode tests (experimental)
    └── test_async_bytecode.py  # Async bytecode tests (experimental)
```
