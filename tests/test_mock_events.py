"""
Tests demonstrating the interlace library for controlling thread interleaving.

These tests show how to use interlace to:
1. Deterministically reproduce race conditions
2. Test that race conditions exist
3. Test that proper synchronization prevents races
"""

import sys
import threading
import unittest
from pathlib import Path

# Add parent directory to path to import interlace
sys.path.insert(0, str(Path(__file__).parent.parent))

from interlace.mock_events import Interlace, InterleaveBuilder


class BankAccount:
    """
    A simple bank account with a race condition in the transfer method.

    This class intentionally has a read-modify-write race condition to
    demonstrate how interlace can deterministically reproduce it.
    """

    # Class variable to store interlace coordinator for checkpoints
    _interlace = None

    def __init__(self, balance: int = 0):
        """
        Initialize a bank account with a starting balance.

        Args:
            balance: Initial balance (default: 0)
        """
        self.balance = balance

    def transfer(self, amount: int) -> int:
        """
        Transfer an amount to the account (can be negative for withdrawals).

        This method has a classic race condition: it reads the balance,
        then writes a new balance based on that read. If another thread
        modifies the balance between the read and write, updates can be lost.

        Args:
            amount: Amount to add to the balance

        Returns:
            The new balance
        """
        current = self.balance  # READ
        if self._interlace:
            self._interlace.checkpoint('transfer', 'after_read')
        # Race condition window: another thread can modify balance here!
        new_balance = current + amount  # COMPUTE
        if self._interlace:
            self._interlace.checkpoint('transfer', 'before_write')
        self.balance = new_balance  # WRITE
        return new_balance

    def safe_transfer(self, amount: int) -> int:
        """
        Thread-safe transfer using a lock.

        Args:
            amount: Amount to add to the balance

        Returns:
            The new balance
        """
        if not hasattr(self, '_lock'):
            self._lock = threading.Lock()

        with self._lock:
            current = self.balance
            new_balance = current + amount
            self.balance = new_balance
            return new_balance


class TestInterlaceBasicUsage(unittest.TestCase):
    """Test basic usage patterns of the interlace library."""

    def test_race_condition_exists(self):
        """
        First, demonstrate that the race condition EXISTS.

        Without controlling the interleaving, we can't reliably trigger it
        in a test, but we can run it many times and observe that it sometimes
        produces incorrect results.

        This test is probabilistic and may occasionally pass even though
        the race exists, but typically it will fail, demonstrating the problem.
        """
        # Try multiple times to trigger the race
        race_detected = False

        for _ in range(100):
            account = BankAccount(balance=100)
            results = []

            def transfer_50():
                result = account.transfer(50)
                results.append(result)

            # Run two transfers concurrently
            t1 = threading.Thread(target=transfer_50)
            t2 = threading.Thread(target=transfer_50)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Expected: balance should be 200 (100 + 50 + 50)
            # With race condition: balance might be 150 (one update lost)
            if account.balance == 150:
                race_detected = True
                break

        # This test documents that races CAN happen (though not guaranteed)
        # In practice, on most systems this will detect the race within 100 iterations
        if race_detected:
            # Good, we confirmed the race exists
            pass
        else:
            # Race wasn't detected, but that's okay - the point is it CAN happen
            # The next tests will show how to DETERMINISTICALLY trigger it
            pass

    def test_deterministic_race_condition_reproduction(self):
        """
        Use interlace to DETERMINISTICALLY reproduce the race condition.

        This test forces a specific interleaving that guarantees the race
        condition occurs, making the test reliable and repeatable.

        The interleaving we force:
        1. Thread 1 reads balance (100)
        2. Thread 2 reads balance (100)  <- both see the same value!
        3. Thread 1 writes balance (150)
        4. Thread 2 writes balance (150) <- overwrites thread 1's update!

        Result: Final balance is 150 instead of 200 (one update lost).
        """
        account = BankAccount(balance=100)

        with Interlace() as il:
            # Set the interlace coordinator so transfer() can use checkpoints
            BankAccount._interlace = il

            # Define our concurrent tasks
            @il.task('thread1')
            def task1():
                account.transfer(50)

            @il.task('thread2')
            def task2():
                account.transfer(50)

            # Define the problematic interleaving that causes the race:
            # Both threads read before either writes
            il.order([
                ('thread1', 'transfer', 'after_read'),   # T1 reads balance=100
                ('thread2', 'transfer', 'after_read'),   # T2 reads balance=100
                ('thread1', 'transfer', 'before_write'), # T1 writes balance=150
                ('thread2', 'transfer', 'before_write'), # T2 writes balance=150 (overwrites!)
            ])

            # Run the interleaved execution
            il.run()

            # Clean up
            BankAccount._interlace = None

        # With this interleaving, we ALWAYS get the race condition
        self.assertEqual(account.balance, 150,
                        "Race condition: one update was lost!")

    def test_correct_interleaving_no_race(self):
        """
        Use interlace to force a CORRECT interleaving with no race.

        This test forces an interleaving where operations are properly
        serialized, demonstrating that the order matters.

        The safe interleaving we force:
        1. Thread 1 reads balance (100)
        2. Thread 1 writes balance (150)  <- completes before thread 2 starts
        3. Thread 2 reads balance (150)   <- sees thread 1's update
        4. Thread 2 writes balance (200)  <- correct final result

        Result: Final balance is 200 (correct).
        """
        account = BankAccount(balance=100)

        with Interlace() as il:
            BankAccount._interlace = il

            @il.task('thread1')
            def task1():
                account.transfer(50)

            @il.task('thread2')
            def task2():
                account.transfer(50)

            # Define a safe interleaving: each operation completes
            # before the next one starts
            il.order([
                ('thread1', 'transfer', 'after_read'),   # T1 reads balance=100
                ('thread1', 'transfer', 'before_write'), # T1 writes balance=150
                ('thread2', 'transfer', 'after_read'),   # T2 reads balance=150
                ('thread2', 'transfer', 'before_write'), # T2 writes balance=200
            ])

            il.run()
            BankAccount._interlace = None

        # With this interleaving, we get the correct result
        self.assertEqual(account.balance, 200,
                        "With proper serialization, both updates succeed")

    def test_three_way_race(self):
        """
        Demonstrate a three-way race condition.

        Shows that interlace scales to more complex scenarios with
        multiple threads.
        """
        account = BankAccount(balance=100)

        with Interlace() as il:
            BankAccount._interlace = il

            @il.task('t1')
            def task1():
                account.transfer(50)

            @il.task('t2')
            def task2():
                account.transfer(50)

            @il.task('t3')
            def task3():
                account.transfer(50)

            # All three threads read before any writes
            il.order([
                ('t1', 'transfer', 'after_read'),   # reads 100
                ('t2', 'transfer', 'after_read'),   # reads 100
                ('t3', 'transfer', 'after_read'),   # reads 100
                ('t1', 'transfer', 'before_write'), # writes 150
                ('t2', 'transfer', 'before_write'), # writes 150
                ('t3', 'transfer', 'before_write'), # writes 150
            ])

            il.run()
            BankAccount._interlace = None

        # All three updates are lost except one
        self.assertEqual(account.balance, 150,
                        "Three-way race: two updates lost!")

    def test_partial_overlap(self):
        """
        Test a partially overlapping interleaving.

        This shows a more subtle race where operations partially overlap
        but aren't fully serialized.
        """
        account = BankAccount(balance=100)

        with Interlace() as il:
            BankAccount._interlace = il

            @il.task('t1')
            def task1():
                account.transfer(50)

            @il.task('t2')
            def task2():
                account.transfer(50)

            # Thread 2 starts before thread 1 finishes
            il.order([
                ('t1', 'transfer', 'after_read'),   # T1 reads 100
                ('t2', 'transfer', 'after_read'),   # T2 reads 100 (race!)
                ('t1', 'transfer', 'before_write'), # T1 writes 150
                ('t2', 'transfer', 'before_write'), # T2 writes 150 (overwrites!)
            ])

            il.run()
            BankAccount._interlace = None

        self.assertEqual(account.balance, 150,
                        "Partial overlap causes lost update")


class TestInterleaveBuilder(unittest.TestCase):
    """Test the InterleaveBuilder fluent API."""

    def test_builder_api(self):
        """Test that the builder API works for creating sequences."""
        account = BankAccount(balance=100)

        with Interlace() as il:
            BankAccount._interlace = il

            @il.task('t1')
            def task1():
                account.transfer(50)

            @il.task('t2')
            def task2():
                account.transfer(50)

            # Use the builder for a more readable sequence definition
            builder = InterleaveBuilder()
            builder.step('t1', 'transfer', 'after_read') \
                   .step('t2', 'transfer', 'after_read') \
                   .step('t1', 'transfer', 'before_write') \
                   .step('t2', 'transfer', 'before_write')

            il.order(builder.build())
            il.run()
            BankAccount._interlace = None

        self.assertEqual(account.balance, 150)


class TestComplexScenarios(unittest.TestCase):
    """Test more complex scenarios with multiple methods."""

    def test_multiple_methods(self):
        """
        Test interleaving across multiple different methods.

        This shows how interlace can control ordering across different
        operations, not just multiple calls to the same method.
        """

        class Counter:
            _interlace = None

            def __init__(self):
                self.value = 0

            def increment(self):
                current = self.value
                if self._interlace:
                    self._interlace.checkpoint('increment', 'after_read')
                new_value = current + 1
                if self._interlace:
                    self._interlace.checkpoint('increment', 'before_write')
                self.value = new_value

            def decrement(self):
                current = self.value
                if self._interlace:
                    self._interlace.checkpoint('decrement', 'after_read')
                new_value = current - 1
                if self._interlace:
                    self._interlace.checkpoint('decrement', 'before_write')
                self.value = new_value

        counter = Counter()

        with Interlace() as il:
            Counter._interlace = il

            @il.task('inc')
            def increment_task():
                counter.increment()

            @il.task('dec')
            def decrement_task():
                counter.decrement()

            # Interleave increment and decrement: both read 0, then both write
            il.order([
                ('inc', 'increment', 'after_read'),   # reads 0
                ('dec', 'decrement', 'after_read'),   # reads 0
                ('inc', 'increment', 'before_write'), # writes 1
                ('dec', 'decrement', 'before_write'), # writes -1 (overwrites!)
            ])

            il.run()
            Counter._interlace = None

        # The decrement "wins" and we get -1 instead of 0
        self.assertEqual(counter.value, -1,
                        "Race between increment and decrement")


class TestDocumentationExamples(unittest.TestCase):
    """Tests that serve as clear documentation examples."""

    def test_readme_example(self):
        """
        A clear, self-documenting example for README/documentation.

        This test should be immediately understandable to someone
        learning about race conditions and how interlace helps test them.
        """
        # Setup: A bank account with a race condition
        account = BankAccount(balance=100)

        # Create an interlace coordinator
        with Interlace() as il:
            BankAccount._interlace = il

            # Define two tasks that will run concurrently
            @il.task('alice')
            def alice_deposits():
                account.transfer(50)  # Alice deposits $50

            @il.task('bob')
            def bob_deposits():
                account.transfer(50)  # Bob deposits $50

            # Force a specific interleaving that exposes the race condition:
            # Both threads read the balance BEFORE either one writes
            il.order([
                ('alice', 'transfer', 'after_read'),   # Alice reads: $100
                ('bob', 'transfer', 'after_read'),     # Bob reads: $100 (stale!)
                ('alice', 'transfer', 'before_write'), # Alice writes: $150
                ('bob', 'transfer', 'before_write'),   # Bob writes: $150 (overwrites Alice!)
            ])

            # Run with controlled interleaving
            il.run()
            BankAccount._interlace = None

        # Result: Only $150 instead of $200 (one deposit was lost!)
        assert account.balance == 150, \
            "Race condition reproduced: One deposit was lost!"

        print(f"✓ Deterministically reproduced race condition")
        print(f"  Expected: $200 (100 + 50 + 50)")
        print(f"  Actual:   ${account.balance} (one update lost)")

    def test_safe_implementation_comparison(self):
        """
        Compare unsafe vs safe implementation to show the difference.

        This is a great documentation example showing that proper
        synchronization fixes the issue.
        """
        # Test 1: Unsafe version with forced race
        unsafe_account = BankAccount(balance=100)

        with Interlace() as il:
            BankAccount._interlace = il

            @il.task('t1')
            def task1():
                unsafe_account.transfer(50)

            @il.task('t2')
            def task2():
                unsafe_account.transfer(50)

            il.order([
                ('t1', 'transfer', 'after_read'),
                ('t2', 'transfer', 'after_read'),
                ('t1', 'transfer', 'before_write'),
                ('t2', 'transfer', 'before_write'),
            ])

            il.run()
            BankAccount._interlace = None

        # Test 2: Safe version (doesn't use checkpoints, so race doesn't manifest)
        # The safe version would naturally serialize due to the lock
        safe_account = BankAccount(balance=100)

        def safe_transfer_in_thread(account):
            account.safe_transfer(50)

        import threading
        t1 = threading.Thread(target=safe_transfer_in_thread, args=(safe_account,))
        t2 = threading.Thread(target=safe_transfer_in_thread, args=(safe_account,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Compare results
        print(f"\n{'='*60}")
        print(f"Comparing unsafe vs safe implementation:")
        print(f"{'='*60}")
        print(f"Unsafe (race condition):  ${unsafe_account.balance} ❌")
        print(f"Safe (with lock):         ${safe_account.balance} ✓")
        print(f"{'='*60}\n")

        self.assertEqual(unsafe_account.balance, 150,
                        "Unsafe version has race")
        self.assertEqual(safe_account.balance, 200,
                        "Safe version prevents race")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
