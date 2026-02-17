"""
Tests for the interlace async_trace_markers module.

Demonstrates deterministic async task interleaving using comment-based markers.

These tests mirror the structure of test_trace_markers.py but adapted for async/await.
Each test uses asyncio.run() to execute the async test logic.
"""

import asyncio
import os
import sys

import pytest

# Add parent directory to path so we can import interlace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interlace.async_trace_markers import AsyncTraceExecutor, async_interlace
from interlace.common import Schedule, Step


class BankAccount:
    """A simple bank account class with a race condition vulnerability.

    This async version demonstrates how race conditions occur at await points.
    Uses # interlace: comments to mark synchronization points.
    """

    def __init__(self, balance=0):
        self.balance = balance

    async def get_balance(self):
        """Get current balance (simulates async I/O like database read)."""
        return self.balance

    async def set_balance(self, new_balance):
        """Set balance (simulates async I/O like database write)."""
        self.balance = new_balance

    async def transfer(self, amount):
        """Transfer money to this account (intentionally racy).

        This method has a race condition: it reads the balance, computes
        the new balance, then writes it back. Between the read and write,
        another task can interleave and cause a lost update.

        Args:
            amount: The amount to transfer
        """
        # interlace: read_balance
        current = await self.get_balance()
        # interlace: write_balance
        await self.set_balance(current + amount)


def test_race_condition_buggy_schedule():
    """Test that demonstrates the race condition bug with an unsafe schedule.

    Schedule: t1 reads, t2 reads, t1 writes, t2 writes
    This creates a lost update: both tasks read the same initial value,
    so the final balance is incorrect.
    """
    print("\n=== Test: Race Condition (Buggy Schedule) ===")

    account = BankAccount(balance=100)

    # Define the buggy schedule: both tasks read before either writes
    schedule = Schedule(
        [
            Step("task1", "read_balance"),
            Step("task2", "read_balance"),
            Step("task1", "write_balance"),
            Step("task2", "write_balance"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "task1": lambda: account.transfer(50),
            "task2": lambda: account.transfer(50),
        }
    )

    print("Initial balance: 100")
    print("Task 1 transfer: +50")
    print("Task 2 transfer: +50")
    print("Expected (buggy): 150")
    print(f"Actual balance: {account.balance}")

    # With the buggy schedule, we expect a lost update
    assert account.balance == 150, f"Expected 150 (lost update), got {account.balance}"
    print("✓ Race condition successfully reproduced!")


def test_race_condition_correct_schedule():
    """Test that demonstrates correct execution with a safe schedule.

    Schedule: t1 reads, t1 writes, t2 reads, t2 writes
    This ensures proper serialization: each task completes its transaction
    before the next one starts.
    """
    print("\n=== Test: Race Condition (Correct Schedule) ===")

    account = BankAccount(balance=100)

    # Define the correct schedule: each task completes before the next starts
    schedule = Schedule(
        [
            Step("task1", "read_balance"),
            Step("task1", "write_balance"),
            Step("task2", "read_balance"),
            Step("task2", "write_balance"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "task1": lambda: account.transfer(50),
            "task2": lambda: account.transfer(50),
        }
    )

    print("Initial balance: 100")
    print("Task 1 transfer: +50")
    print("Task 2 transfer: +50")
    print("Expected (correct): 200")
    print(f"Actual balance: {account.balance}")

    # With the correct schedule, we expect the right result
    assert account.balance == 200, f"Expected 200, got {account.balance}"
    print("✓ Correct execution verified!")


def test_multiple_markers_same_task():
    """Test a task hitting multiple markers in sequence."""
    print("\n=== Test: Multiple Markers Same Task ===")

    results = []

    async def worker_with_markers():
        # interlace: step1
        await asyncio.sleep(0)
        results.append("step1")
        # interlace: step2
        await asyncio.sleep(0)
        results.append("step2")
        # interlace: step3
        await asyncio.sleep(0)
        results.append("step3")

    schedule = Schedule(
        [
            Step("main", "step1"),
            Step("main", "step2"),
            Step("main", "step3"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "main": worker_with_markers,
        }
    )

    print(f"Results: {results}")
    assert results == ["step1", "step2", "step3"]
    print("✓ Multiple markers executed in order!")


def test_alternating_execution():
    """Test alternating execution between two tasks."""
    print("\n=== Test: Alternating Execution ===")

    results = []

    async def worker1():
        # interlace: marker_a
        await asyncio.sleep(0)
        results.append("t1_a")
        # interlace: marker_b
        await asyncio.sleep(0)
        results.append("t1_b")

    async def worker2():
        # interlace: marker_a
        await asyncio.sleep(0)
        results.append("t2_a")
        # interlace: marker_b
        await asyncio.sleep(0)
        results.append("t2_b")

    # Alternate between tasks at each marker
    schedule = Schedule(
        [
            Step("task1", "marker_a"),
            Step("task2", "marker_a"),
            Step("task1", "marker_b"),
            Step("task2", "marker_b"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "task1": worker1,
            "task2": worker2,
        }
    )

    print(f"Execution order: {results}")
    expected = ["t1_a", "t2_a", "t1_b", "t2_b"]
    assert results == expected, f"Expected {expected}, got {results}"
    print("✓ Alternating execution verified!")


def test_convenience_function():
    """Test the convenience async_interlace() function."""
    print("\n=== Test: Convenience Function ===")

    results = []

    async def worker1():
        # interlace: mark
        await asyncio.sleep(0)
        results.append("t1")

    async def worker2():
        # interlace: mark
        await asyncio.sleep(0)
        results.append("t2")

    schedule = Schedule(
        [
            Step("t1", "mark"),
            Step("t2", "mark"),
        ]
    )

    async_interlace(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)

    print(f"Results: {results}")
    assert results == ["t1", "t2"]
    print("✓ Convenience function works!")


def test_complex_race_scenario():
    """Test a more complex scenario with multiple shared resources.

    This test demonstrates how three tasks can all experience lost updates
    when they interleave their reads and writes in a maximally racy way.
    """
    print("\n=== Test: Complex Race Scenario ===")

    class SharedCounter:
        def __init__(self):
            self.value = 0

        async def get_value(self):
            """Get counter value (simulates async I/O)."""
            return self.value

        async def set_value(self, value):
            """Set counter value (simulates async I/O)."""
            self.value = value

        async def increment_racy(self):
            """Increment with race condition."""
            # interlace: read_counter
            temp = await self.get_value()
            # interlace: write_counter
            await self.set_value(temp + 1)

    counter = SharedCounter()

    # Three tasks, each incrementing once
    # We'll interleave them to maximize the race condition
    schedule = Schedule(
        [
            Step("t1", "read_counter"),
            Step("t2", "read_counter"),
            Step("t3", "read_counter"),
            Step("t1", "write_counter"),
            Step("t2", "write_counter"),
            Step("t3", "write_counter"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "t1": counter.increment_racy,
            "t2": counter.increment_racy,
            "t3": counter.increment_racy,
        }
    )

    print("Initial counter: 0")
    print("Three tasks each increment once")
    print("Expected (with race): 1")
    print(f"Actual counter: {counter.value}")

    # With this schedule, all three tasks read 0, then all write 1
    assert counter.value == 1, f"Expected 1 (lost updates), got {counter.value}"
    print("✓ Complex race condition reproduced!")


def test_synchronous_function_bodies():
    """Test that markers work even when awaited functions complete synchronously.

    This is a regression test for the bug where markers didn't work correctly
    when awaited functions had no suspension points in their body. Multiple
    markers would be detected in a single send() call, preventing the scheduler
    from interleaving execution.
    """
    print("\n=== Test: Synchronous Function Bodies ===")

    class Counter:
        def __init__(self):
            self.value = 0

        async def get_value(self):
            """No await in body - completes synchronously."""
            return self.value

        async def set_value(self, new_value):
            """No await in body - completes synchronously."""
            self.value = new_value

        async def increment(self):
            """Increment with markers, but awaited methods are synchronous."""
            # interlace: read_value
            current = await self.get_value()
            # interlace: write_value
            await self.set_value(current + 1)

    counter = Counter()

    # Buggy schedule: all reads before all writes (should cause lost update)
    schedule = Schedule(
        [
            Step("t1", "read_value"),
            Step("t2", "read_value"),
            Step("t3", "read_value"),
            Step("t1", "write_value"),
            Step("t2", "write_value"),
            Step("t3", "write_value"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "t1": counter.increment,
            "t2": counter.increment,
            "t3": counter.increment,
        }
    )

    print("Initial counter: 0")
    print("Three tasks each increment once")
    print("With buggy schedule (all reads before writes): expected 1")
    print(f"Actual counter: {counter.value}")

    # With this schedule, all three tasks read 0, then all write 1
    # This demonstrates the scheduler successfully interleaved execution
    assert counter.value == 1, f"Expected 1 (lost updates), got {counter.value}"
    print("✓ Markers worked correctly with synchronous function bodies!")


@pytest.mark.intentionally_leaves_dangling_threads
def test_timeout():
    """Test that timeout works correctly.

    This test intentionally leaves a sleeping thread running when the timeout
    occurs, so it needs the intentionally_leaves_dangling_threads marker.
    """
    print("\n=== Test: Timeout ===")

    # Create a schedule where t2 is waiting but will never get its turn
    schedule = Schedule(
        [
            Step("t1", "marker1"),
            Step("t2", "marker1"),
            # t1 needs to hit marker2, but it comes after t2's marker2 in the schedule
            Step("t2", "marker2"),
            Step("t1", "marker2"),
        ]
    )

    async def worker1():
        # interlace: marker1
        await asyncio.sleep(0)
        # This task will sleep and delay hitting marker2
        await asyncio.sleep(10)
        # interlace: marker2
        await asyncio.sleep(0)

    async def worker2():
        # interlace: marker1
        await asyncio.sleep(0)
        # interlace: marker2
        await asyncio.sleep(0)

    try:
        async_interlace(
            schedule=schedule,
            tasks={"t1": worker1, "t2": worker2},
            timeout=0.5,  # 500ms timeout
        )
        assert False, "Should have timed out"
    except TimeoutError:
        print("✓ Timeout correctly raised!")


def test_exception_propagation():
    """Test that exceptions in tasks are properly propagated."""
    print("\n=== Test: Exception Propagation ===")

    schedule = Schedule(
        [
            Step("t1", "marker1"),
            Step("t2", "marker1"),
        ]
    )

    async def worker1():
        # interlace: marker1
        await asyncio.sleep(0)
        raise ValueError("Intentional error in task1")

    async def worker2():
        # interlace: marker1
        await asyncio.sleep(0)

    try:
        async_interlace(schedule=schedule, tasks={"t1": worker1, "t2": worker2}, timeout=5.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Intentional error in task1"
        print("✓ Exception correctly propagated!")


def test_task_errors_tracked():
    """Test that task_errors is properly populated when tasks raise exceptions."""
    print("\n=== Test: Task Errors Tracked ===")

    schedule = Schedule(
        [
            Step("t1", "marker1"),
            Step("t2", "marker1"),
        ]
    )

    async def worker1():
        # interlace: marker1
        await asyncio.sleep(0)
        raise ValueError("Error in task1")

    async def worker2():
        # interlace: marker1
        await asyncio.sleep(0)

    executor = AsyncTraceExecutor(schedule)

    try:
        executor.run(
            {
                "t1": worker1,
                "t2": worker2,
            }
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Error in task1"
        assert "t1" in executor.task_errors
        assert isinstance(executor.task_errors["t1"], ValueError)
        print("✓ Task errors properly tracked!")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("INTERLACE ASYNC TRACE MARKERS - TEST SUITE")
    print("=" * 60)

    tests = [
        test_race_condition_buggy_schedule,
        test_race_condition_correct_schedule,
        test_multiple_markers_same_task,
        test_alternating_execution,
        test_convenience_function,
        test_complex_race_scenario,
        test_synchronous_function_bodies,
        test_timeout,
        test_exception_propagation,
        test_task_errors_tracked,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
