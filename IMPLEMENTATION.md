# Interlace Mock-and-Events Implementation

## Overview

This is a proof-of-concept implementation of the "interlace" library using a checkpoint-based approach with threading.Event objects for synchronization.

## Files Created

1. `/home/user/projects/interlace/interlace/mock_events.py` - The library implementation
2. `/home/user/projects/interlace/tests/test_mock_events.py` - Comprehensive tests
3. `/home/user/projects/interlace/demo.py` - Demo script showing usage

## Design

### Core Concept

The library allows you to control thread interleaving at a fine-grained level by inserting **checkpoints** in your code. At each checkpoint, a thread waits for permission to proceed based on a predefined **order**.

### Key Components

1. **Interlace Class**: The main coordinator that manages tasks, events, and execution order
2. **Checkpoints**: Synchronization points inserted via `il.checkpoint(operation, phase)`
3. **Tasks**: Concurrent functions decorated with `@il.task('name')`
4. **Order**: A sequence of (task, operation, phase) tuples defining execution order

### How It Works

1. **Setup Phase**:
   - Define concurrent tasks using `@il.task('name')` decorator
   - Specify the order of checkpoints using `il.order([...])`

2. **Execution Phase**:
   - `il.run()` starts all tasks in separate threads
   - Each thread has a thread-local name for identification

3. **Synchronization**:
   - When a thread hits a checkpoint, it waits on a threading.Event
   - Only the next checkpoint in the sequence is "ready" (Event is set)
   - After proceeding, the thread signals the next checkpoint in the sequence

### Example

```python
class BankAccount:
    _interlace = None

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

    # Both threads read before either writes -> race condition
    il.order([
        ('thread1', 'transfer', 'after_read'),
        ('thread2', 'transfer', 'after_read'),
        ('thread1', 'transfer', 'before_write'),
        ('thread2', 'transfer', 'before_write'),
    ])

    il.run()
    BankAccount._interlace = None

assert account.balance == 150  # One update was lost!
```

## API

### Interlace Methods

- `task(name)` - Decorator to register a concurrent task
- `checkpoint(operation, phase)` - Insert a synchronization point
- `order(sequence)` - Define the execution order
- `run()` - Execute all tasks with controlled interleaving
- `patch(target, operation_name)` - Legacy method for patching (kept for compatibility)

### InterleaveBuilder

Fluent API for building sequences:

```python
builder = InterleaveBuilder()
builder.step('t1', 'op', 'phase1') \
       .step('t2', 'op', 'phase1') \
       .step('t1', 'op', 'phase2')
il.order(builder.build())
```

## Trade-offs

### Advantages
- Simple, understandable implementation
- Fine-grained control over interleaving
- Only uses stdlib (threading, unittest.mock)
- Clean, readable test code

### Limitations
- Requires manual checkpoint insertion in code under test
- Uses a class variable (`_interlace`) for coordination
- Not suitable for testing third-party code without modification
- Checkpoints add overhead to production code (mitigated by if checks)

## Running Tests

```bash
cd /home/user/projects/interlace
python -m unittest tests.test_mock_events -v
```

## Running Demo

```bash
cd /home/user/projects/interlace
python demo.py
```

## Future Enhancements

1. **Automatic checkpoint injection** - Use AST rewriting or bytecode manipulation
2. **Context manager for enabling/disabling** - Better than class variables
3. **Timeout configuration** - Configurable wait timeouts
4. **Better error messages** - Show the full sequence when deadlocks occur
5. **Visualization** - Generate diagrams showing the interleaving
