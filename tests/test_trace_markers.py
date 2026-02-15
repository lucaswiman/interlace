"""
Tests for the interlace trace_markers module.

Demonstrates deterministic thread interleaving using sys.settrace and comment markers.
"""

import sys
import os

# Add parent directory to path so we can import interlace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interlace.trace_markers import (
    Schedule, Step, TraceExecutor, interlace,
    MarkerRegistry, ThreadCoordinator
)
import threading
import time


class BankAccount:
    """A simple bank account class with a race condition vulnerability.

    The transfer method has interlace markers to control thread execution order.
    """

    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        """Transfer money to this account (intentionally racy).

        This method has a race condition: it reads the balance, then writes
        a new balance without locking. The interlace markers allow us to
        deterministically trigger the race.
        """
        current = self.balance  # interlace: read_balance
        # Simulate some work
        new_balance = current + amount
        self.balance = new_balance  # interlace: write_balance


def test_race_condition_buggy_schedule():
    """Test that demonstrates the race condition bug with an unsafe schedule.

    Schedule: t1 reads, t2 reads, t1 writes, t2 writes
    This creates a lost update: both threads read the same initial value,
    so the final balance is incorrect.
    """
    print("\n=== Test: Race Condition (Buggy Schedule) ===")

    account = BankAccount(balance=100)

    # Define the buggy schedule: both threads read before either writes
    schedule = Schedule([
        Step("thread1", "read_balance"),
        Step("thread2", "read_balance"),
        Step("thread1", "write_balance"),
        Step("thread2", "write_balance"),
    ])

    def worker1():
        account.transfer(50)

    def worker2():
        account.transfer(50)

    executor = TraceExecutor(schedule)
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    print(f"Initial balance: 100")
    print(f"Thread 1 transfer: +50")
    print(f"Thread 2 transfer: +50")
    print(f"Expected (buggy): 150")
    print(f"Actual balance: {account.balance}")

    # With the buggy schedule, we expect a lost update
    assert account.balance == 150, f"Expected 150 (lost update), got {account.balance}"
    print("✓ Race condition successfully reproduced!")


def test_race_condition_correct_schedule():
    """Test that demonstrates correct execution with a safe schedule.

    Schedule: t1 reads, t1 writes, t2 reads, t2 writes
    This ensures proper serialization: each thread completes its transaction
    before the next one starts.
    """
    print("\n=== Test: Race Condition (Correct Schedule) ===")

    account = BankAccount(balance=100)

    # Define the correct schedule: each thread completes before the next starts
    schedule = Schedule([
        Step("thread1", "read_balance"),
        Step("thread1", "write_balance"),
        Step("thread2", "read_balance"),
        Step("thread2", "write_balance"),
    ])

    def worker1():
        account.transfer(50)

    def worker2():
        account.transfer(50)

    executor = TraceExecutor(schedule)
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    print(f"Initial balance: 100")
    print(f"Thread 1 transfer: +50")
    print(f"Thread 2 transfer: +50")
    print(f"Expected (correct): 200")
    print(f"Actual balance: {account.balance}")

    # With the correct schedule, we expect the right result
    assert account.balance == 200, f"Expected 200, got {account.balance}"
    print("✓ Correct execution verified!")


def test_multiple_markers_same_thread():
    """Test a thread hitting multiple markers in sequence."""
    print("\n=== Test: Multiple Markers Same Thread ===")

    results = []

    def worker_with_markers():
        results.append("step1")  # interlace: step1
        results.append("step2")  # interlace: step2
        results.append("step3")  # interlace: step3

    schedule = Schedule([
        Step("main", "step1"),
        Step("main", "step2"),
        Step("main", "step3"),
    ])

    executor = TraceExecutor(schedule)
    executor.run("main", worker_with_markers)
    executor.wait(timeout=5.0)

    print(f"Results: {results}")
    assert results == ["step1", "step2", "step3"]
    print("✓ Multiple markers executed in order!")


def test_alternating_execution():
    """Test alternating execution between two threads."""
    print("\n=== Test: Alternating Execution ===")

    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    def worker1():
        x = 1  # interlace: marker_a
        append_safe("t1_a")
        y = 2  # interlace: marker_b
        append_safe("t1_b")

    def worker2():
        x = 1  # interlace: marker_a
        append_safe("t2_a")
        y = 2  # interlace: marker_b
        append_safe("t2_b")

    # Alternate between threads at each marker
    schedule = Schedule([
        Step("thread1", "marker_a"),
        Step("thread2", "marker_a"),
        Step("thread1", "marker_b"),
        Step("thread2", "marker_b"),
    ])

    executor = TraceExecutor(schedule)
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    print(f"Execution order: {results}")
    expected = ["t1_a", "t2_a", "t1_b", "t2_b"]
    assert results == expected, f"Expected {expected}, got {results}"
    print("✓ Alternating execution verified!")


def test_convenience_function():
    """Test the convenience interlace() function."""
    print("\n=== Test: Convenience Function ===")

    results = []
    lock = threading.Lock()

    def append_safe(value):
        with lock:
            results.append(value)

    def worker1():
        x = 1  # interlace: mark
        append_safe("t1")

    def worker2():
        x = 1  # interlace: mark
        append_safe("t2")

    schedule = Schedule([
        Step("t1", "mark"),
        Step("t2", "mark"),
    ])

    interlace(
        schedule=schedule,
        threads={"t1": worker1, "t2": worker2},
        timeout=5.0
    )

    print(f"Results: {results}")
    assert results == ["t1", "t2"]
    print("✓ Convenience function works!")


def test_marker_registry():
    """Test the MarkerRegistry class directly."""
    print("\n=== Test: MarkerRegistry ===")

    # Create a temporary function with markers to test scanning
    def test_function():
        x = 1  # interlace: marker1
        y = 2  # interlace: marker2
        return x + y

    # Get a frame from the function
    import inspect
    frame = None

    def get_frame():
        return inspect.currentframe()

    # We need to actually execute the function to get its frame via tracing
    registry = MarkerRegistry()
    found_markers = []

    def trace_func(frame, event, arg):
        if event == 'line':
            registry.scan_frame(frame)
            marker = registry.get_marker(frame.f_code.co_filename, frame.f_lineno)
            if marker:
                found_markers.append(marker)
        return trace_func

    sys.settrace(trace_func)
    try:
        test_function()
    finally:
        sys.settrace(None)

    print(f"Found markers: {found_markers}")
    assert "marker1" in found_markers
    assert "marker2" in found_markers
    print("✓ MarkerRegistry successfully scans and finds markers!")


def test_thread_coordinator():
    """Test the ThreadCoordinator synchronization logic."""
    print("\n=== Test: ThreadCoordinator ===")

    schedule = Schedule([
        Step("t1", "m1"),
        Step("t2", "m1"),
        Step("t1", "m2"),
    ])

    coordinator = ThreadCoordinator(schedule)
    results = []

    def thread1_work():
        results.append("t1_start")
        coordinator.wait_for_turn("t1", "m1")
        results.append("t1_m1")
        coordinator.wait_for_turn("t1", "m2")
        results.append("t1_m2")

    def thread2_work():
        results.append("t2_start")
        coordinator.wait_for_turn("t2", "m1")
        results.append("t2_m1")

    t1 = threading.Thread(target=thread1_work)
    t2 = threading.Thread(target=thread2_work)

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    print(f"Results: {results}")

    # The order should be: both start (unordered), then t1_m1, t2_m1, t1_m2
    assert "t1_m1" in results
    assert "t2_m1" in results
    assert "t1_m2" in results

    # Check the marker order is correct
    m1_index_t1 = results.index("t1_m1")
    m1_index_t2 = results.index("t2_m1")
    m2_index_t1 = results.index("t1_m2")

    assert m1_index_t1 < m1_index_t2, "t1 should hit m1 before t2"
    assert m1_index_t2 < m2_index_t1, "t2 should hit m1 before t1 hits m2"

    print("✓ ThreadCoordinator synchronizes correctly!")


def test_complex_race_scenario():
    """Test a more complex scenario with multiple shared resources."""
    print("\n=== Test: Complex Race Scenario ===")

    class SharedCounter:
        def __init__(self):
            self.value = 0

        def increment_racy(self):
            temp = self.value  # interlace: read_counter
            temp = temp + 1
            self.value = temp  # interlace: write_counter

    counter = SharedCounter()

    # Three threads, each incrementing twice
    # We'll interleave them to maximize the race condition
    schedule = Schedule([
        Step("t1", "read_counter"),
        Step("t2", "read_counter"),
        Step("t3", "read_counter"),
        Step("t1", "write_counter"),
        Step("t2", "write_counter"),
        Step("t3", "write_counter"),
    ])

    def worker():
        counter.increment_racy()

    executor = TraceExecutor(schedule)
    executor.run("t1", worker)
    executor.run("t2", worker)
    executor.run("t3", worker)
    executor.wait(timeout=5.0)

    print(f"Initial counter: 0")
    print(f"Three threads each increment once")
    print(f"Expected (with race): 1")
    print(f"Actual counter: {counter.value}")

    # With this schedule, all three threads read 0, then all write 1
    assert counter.value == 1, f"Expected 1 (lost updates), got {counter.value}"
    print("✓ Complex race condition reproduced!")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("INTERLACE TRACE MARKERS - TEST SUITE")
    print("=" * 60)

    tests = [
        test_race_condition_buggy_schedule,
        test_race_condition_correct_schedule,
        test_multiple_markers_same_thread,
        test_alternating_execution,
        test_convenience_function,
        test_marker_registry,
        test_thread_coordinator,
        test_complex_race_scenario,
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
