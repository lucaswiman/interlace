# Interlace

A library for deterministic concurrency testing that helps you reliably reproduce and test race conditions.

## Overview

Interlace provides tools for controlling thread interleaving at a fine-grained level, allowing you to:

- **Deterministically reproduce race conditions** - Force specific interleavings to make race conditions happen reliably in tests
- **Test concurrent code exhaustively** - Explore different execution orders to find bugs
- **Verify synchronization correctness** - Ensure that proper locking prevents race conditions

Instead of relying on timing-based race detection (which is unreliable), Interlace lets you control exactly when threads execute, making concurrency testing deterministic and reproducible.

## Quick Start: Bank Account Race Condition

Here's a simple example showing how to use Interlace to trigger a race condition using trace markers:

```python
from interlace.trace_markers import Schedule, Step, TraceExecutor

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        # This method has a race condition: read-modify-write without synchronization
        current = self.balance  # interlace: read_balance
        new_balance = current + amount
        self.balance = new_balance  # interlace: write_balance

# Demonstrate the race condition
account = BankAccount(balance=100)

# Define an interleaving that triggers the race:
# Both threads read before either writes
schedule = Schedule([
    Step("thread1", "read_balance"),    # T1 reads 100
    Step("thread2", "read_balance"),    # T2 reads 100 (both see same value!)
    Step("thread1", "write_balance"),   # T1 writes 150
    Step("thread2", "write_balance"),   # T2 writes 150 (overwrites T1's update!)
])

executor = TraceExecutor(schedule)
executor.run("thread1", lambda: account.transfer(50))
executor.run("thread2", lambda: account.transfer(50))
executor.wait(timeout=5.0)

# Result: balance is 150, not 200 - one update was lost!
assert account.balance == 150, "Race condition triggered!"
```

## Case Studies

See [detailed case studies](docs/CASE_STUDIES.rst) of searching for concurrency bugs in ten libraries: TPool, threadpoolctl, cachetools, PyDispatcher, pydis, pybreaker, urllib3, SQLAlchemy, amqtt, and pykka. Run the test suites with: `PYTHONPATH=interlace python interlace/docs/tests/run_external_tests.py`

## Usage Approaches

Interlace provides two different ways to control thread interleaving:

### 1. Trace Markers

Trace markers are special comments (`# interlace: <marker-name>`) which mark particular synchronization points in multithreaded or async code. These are intended to make it easier to reproduce race conditions in test cases and inspect whether some race conditions are possible.

The execution ordering is controlled with a "schedule" object that says what order the threads / markers should run in.

Each thread runs with a [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback that pauses at markers and waits for a schedule to grant the next execution turn. This gives deterministic control over execution order without modifying code semantics — markers are just comments. A marker **gates** the code that follows it: the thread pauses at the marker and only executes the gated code after the scheduler grants it a turn. Name markers after the operation they gate (e.g. `read_value`, `write_balance`) rather than with temporal prefixes like `before_` or `after_`.

Markers can be placed inline or on a separate line before the operation:

```python
from interlace.trace_markers import Schedule, Step, TraceExecutor

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value  # interlace: read_value
        temp += 1
        self.value = temp  # interlace: write_value

counter = Counter()

# Define execution order using markers
schedule = Schedule([
    Step("thread1", "read_value"),
    Step("thread2", "read_value"),
    Step("thread1", "write_value"),
    Step("thread2", "write_value"),
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

### 2. Bytecode Manipulation (Experimental)

> ⚠️ **Experimental:** Bytecode instrumentation is experimental and may change. It requires monkey-patching concurrency primitives and relies on `f_trace_opcodes` (Python 3.7+). Use with caution.

Automatically instrument functions using bytecode rewriting — no markers needed. Each thread fires a [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback at every bytecode instruction, pausing at each one to wait for its scheduler turn. This gives fine-grained control but requires monkey-patching standard threading primitives (`Lock`, `Semaphore`, `Event`, `Queue`, etc.) to prevent deadlocks.

For property-based exploration, `explore_interleavings()` generates random schedules and checks invariants, returning any counterexample schedule for deterministic reproduction.

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

### Choosing an Approach

**Trace markers** are stable and straightforward — add comments at synchronization points to control execution order. Use this for targeted testing of specific race conditions and when you can modify the code under test.

**Bytecode instrumentation** requires no markers and can test unmodified third-party code, but it's experimental and may change. The results depend on Python version and bytecode patterns, performance overhead is higher, and threading primitives must be monkey-patched. Use only if you need property-based exploration or must test unmodified library code.

## Async Support

Both approaches have async variants. Trace markers have stable async support, while bytecode instrumentation async support is experimental.

### Async Trace Markers

```python
from interlace.async_trace_markers import AsyncTraceExecutor
from interlace.common import Schedule, Step

class Counter:
    def __init__(self):
        self.value = 0

    async def get_value(self):
        return self.value

    async def set_value(self, new_value):
        self.value = new_value

    async def increment(self):
        # interlace: read_value
        temp = await self.get_value()
        # interlace: write_value
        await self.set_value(temp + 1)

counter = Counter()

# Define execution order using markers
schedule = Schedule([
    Step("task1", "read_value"),
    Step("task2", "read_value"),
    Step("task1", "write_value"),
    Step("task2", "write_value"),
])

# Run with controlled interleaving (synchronous method)
executor = AsyncTraceExecutor(schedule)
executor.run({
    "task1": counter.increment,
    "task2": counter.increment,
})

assert counter.value == 1  # Race condition!
```

**Key insight:** In async code, race conditions only happen at `await` points since the event loop is single-threaded.

## Development

### Running Tests

```bash
# From the interlace directory
make test

# Or from the root directory
make test-interlace
```

All tests pass successfully:

```
tests/test_trace_markers.py .............. (Sync trace markers)
tests/test_async_trace_markers.py ........ (Async trace markers)
tests/test_bytecode.py ................... (Bytecode instrumentation)
tests/test_async_bytecode.py ............. (Async bytecode instrumentation)
tests/test_threading_primitives.py ....... (Threading primitive wrappers)
tests/test_concurrency_bug_classes.py .... (Concurrency bug detection)
tests/test_interlace.py .................. (Integration tests)
```

### Project Structure

```
interlace/
├── Makefile                          # Project build targets
├── pyproject.toml                    # Python packaging configuration
├── README.md                         # This file
├── docs/                             # Documentation (Sphinx/ReadTheDocs)
├── interlace/                        # Python package
│   ├── __init__.py
│   ├── common.py                     # Shared types (Schedule, Step)
│   ├── trace_markers.py              # Trace marker approach
│   ├── async_trace_markers.py        # Async trace markers
│   ├── bytecode.py                   # Bytecode instrumentation (experimental)
│   ├── async_bytecode.py             # Async bytecode (experimental)
│   └── async_scheduler.py            # Async scheduling utilities
└── tests/                            # Test suite
    ├── conftest.py                   # Pytest configuration
    ├── buggy_programs.py             # Test programs with known bugs
    ├── test_trace_markers.py
    ├── test_async_trace_markers.py
    ├── test_bytecode.py
    ├── test_async_bytecode.py
    ├── test_threading_primitives.py
    ├── test_concurrency_bug_classes.py
    └── test_interlace.py             # Integration tests
```
