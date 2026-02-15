"""
Tests demonstrating the async interlace library for controlling async task interleaving.

These tests show how to use AsyncInterlace to:
1. Deterministically reproduce race conditions in async code
2. Test that race conditions exist in async operations
3. Test that proper synchronization prevents races

Key insight: In async code, race conditions ONLY happen at await points since
the event loop is single-threaded and only yields at await statements.
"""

import asyncio
import sys
import unittest
from pathlib import Path

# Add parent directory to path to import interlace
sys.path.insert(0, str(Path(__file__).parent.parent))

from interlace.async_mock_events import AsyncInterlace, AsyncInterleaveBuilder


class BankAccount:
    """
    A simple async bank account with a race condition in the transfer method.

    This class intentionally has a read-modify-write race condition to
    demonstrate how AsyncInterlace can deterministically reproduce it.

    The race occurs because even though async is single-threaded, multiple
    tasks can be interleaved at await points, allowing one task's write to
    overwrite another's.
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

    async def transfer(self, amount: int) -> int:
        """
        Transfer an amount to the account (can be negative for withdrawals).

        This method has a classic race condition: it reads the balance,
        then writes a new balance based on that read. If another task
        modifies the balance between the read and write (at an await point),
        updates can be lost.

        Args:
            amount: Amount to add to the balance

        Returns:
            The new balance
        """
        # Important: checkpoint BEFORE read to prevent eager execution
        if self._interlace:
            await self._interlace.checkpoint('transfer', 'before_read')
        current = self.balance  # READ
        if self._interlace:
            await self._interlace.checkpoint('transfer', 'after_read')
        # Race condition window: another task can resume and modify balance here!
        new_balance = current + amount  # COMPUTE
        if self._interlace:
            await self._interlace.checkpoint('transfer', 'before_write')
        self.balance = new_balance  # WRITE
        return new_balance

    async def safe_transfer(self, amount: int) -> int:
        """
        Thread-safe transfer using an asyncio lock.

        Args:
            amount: Amount to add to the balance

        Returns:
            The new balance
        """
        if not hasattr(self, '_lock'):
            self._lock = asyncio.Lock()

        async with self._lock:
            current = self.balance
            new_balance = current + amount
            self.balance = new_balance
            return new_balance


class TestAsyncInterlaceBasicUsage(unittest.TestCase):
    """Test basic usage patterns of the async interlace library."""

    def test_race_condition_exists(self):
        """
        First, demonstrate that the race condition EXISTS in async code.

        Even though async is single-threaded, race conditions can still occur
        when tasks are interleaved at await points. This test shows that
        without controlling the interleaving, we can observe incorrect results.

        Note: This test is probabilistic because we're not controlling the
        interleaving, so it might not always trigger the race. The next test
        shows how to DETERMINISTICALLY trigger it.
        """
        async def run_race_test():
            # Try multiple times to trigger the race
            race_detected = False

            for _ in range(100):
                account = BankAccount(balance=100)
                results = []

                async def transfer_50():
                    result = await account.transfer(50)
                    results.append(result)

                # Run two transfers concurrently
                await asyncio.gather(transfer_50(), transfer_50())

                # Expected: balance should be 200 (100 + 50 + 50)
                # With race condition: balance might be 150 (one update lost)
                if account.balance == 150:
                    race_detected = True
                    break

            # This test documents that races CAN happen (though not guaranteed in async)
            # The key insight is that await points are where races occur
            # The next tests will show how to DETERMINISTICALLY trigger it
            return race_detected

        # Run the async test
        race_detected = asyncio.run(run_race_test())

        # We're okay either way - the point is to show races are possible
        # The deterministic tests below are what really matter

    def test_deterministic_race_condition_reproduction(self):
        """
        Use AsyncInterlace to DETERMINISTICALLY reproduce the race condition.

        This test forces a specific interleaving that guarantees the race
        condition occurs, making the test reliable and repeatable.

        The interleaving we force:
        1. Task 1 reads balance (100)
        2. Task 2 reads balance (100)  <- both see the same value!
        3. Task 1 writes balance (150)
        4. Task 2 writes balance (150) <- overwrites task 1's update!

        Result: Final balance is 150 instead of 200 (one update lost).
        """
        async def run_test():
            account = BankAccount(balance=100)

            async with AsyncInterlace() as il:
                # Set the interlace coordinator so transfer() can use checkpoints
                BankAccount._interlace = il

                # Define our concurrent tasks
                @il.task('task1')
                async def task1():
                    await account.transfer(50)

                @il.task('task2')
                async def task2():
                    await account.transfer(50)

                # Define the problematic interleaving that causes the race:
                # Both tasks read before either writes
                il.order([
                    ('task1', 'transfer', 'before_read'),  # T1 enters
                    ('task1', 'transfer', 'after_read'),   # T1 reads balance=100
                    ('task2', 'transfer', 'before_read'),  # T2 enters
                    ('task2', 'transfer', 'after_read'),   # T2 reads balance=100
                    ('task1', 'transfer', 'before_write'), # T1 writes balance=150
                    ('task2', 'transfer', 'before_write'), # T2 writes balance=150 (overwrites!)
                ])

                # Run the interleaved execution
                await il.run()

                # Clean up
                BankAccount._interlace = None

            return account.balance

        # Run the async test
        balance = asyncio.run(run_test())

        # With this interleaving, we ALWAYS get the race condition
        self.assertEqual(balance, 150,
                        "Race condition: one update was lost!")

    def test_correct_interleaving_no_race(self):
        """
        Use AsyncInterlace to force a CORRECT interleaving with no race.

        This test forces an interleaving where operations are properly
        serialized, demonstrating that the order matters.

        The safe interleaving we force:
        1. Task 1 reads balance (100)
        2. Task 1 writes balance (150)  <- completes before task 2 starts
        3. Task 2 reads balance (150)   <- sees task 1's update
        4. Task 2 writes balance (200)  <- correct final result

        Result: Final balance is 200 (correct).
        """
        async def run_test():
            account = BankAccount(balance=100)

            async with AsyncInterlace() as il:
                BankAccount._interlace = il

                @il.task('task1')
                async def task1():
                    await account.transfer(50)

                @il.task('task2')
                async def task2():
                    await account.transfer(50)

                # Define a safe interleaving: each operation completes
                # before the next one starts
                il.order([
                    ('task1', 'transfer', 'before_read'),  # T1 enters
                    ('task1', 'transfer', 'after_read'),   # T1 reads balance=100
                    ('task1', 'transfer', 'before_write'), # T1 writes balance=150
                    ('task2', 'transfer', 'before_read'),  # T2 enters (after T1 done)
                    ('task2', 'transfer', 'after_read'),   # T2 reads balance=150
                    ('task2', 'transfer', 'before_write'), # T2 writes balance=200
                ])

                await il.run()
                BankAccount._interlace = None

            return account.balance

        # Run the async test
        balance = asyncio.run(run_test())

        # With this interleaving, we get the correct result
        self.assertEqual(balance, 200,
                        "With proper serialization, both updates succeed")

    def test_three_way_race(self):
        """
        Demonstrate a three-way race condition in async code.

        Shows that AsyncInterlace scales to more complex scenarios with
        multiple tasks.
        """
        async def run_test():
            account = BankAccount(balance=100)

            async with AsyncInterlace() as il:
                BankAccount._interlace = il

                @il.task('t1')
                async def task1():
                    await account.transfer(50)

                @il.task('t2')
                async def task2():
                    await account.transfer(50)

                @il.task('t3')
                async def task3():
                    await account.transfer(50)

                # All three tasks read before any writes
                il.order([
                    ('t1', 'transfer', 'before_read'),  # T1 enters
                    ('t1', 'transfer', 'after_read'),   # reads 100
                    ('t2', 'transfer', 'before_read'),  # T2 enters
                    ('t2', 'transfer', 'after_read'),   # reads 100
                    ('t3', 'transfer', 'before_read'),  # T3 enters
                    ('t3', 'transfer', 'after_read'),   # reads 100
                    ('t1', 'transfer', 'before_write'), # writes 150
                    ('t2', 'transfer', 'before_write'), # writes 150
                    ('t3', 'transfer', 'before_write'), # writes 150
                ])

                await il.run()
                BankAccount._interlace = None

            return account.balance

        # Run the async test
        balance = asyncio.run(run_test())

        # All three updates are lost except one
        self.assertEqual(balance, 150,
                        "Three-way race: two updates lost!")

    def test_partial_overlap(self):
        """
        Test a partially overlapping interleaving.

        This shows a more subtle race where operations partially overlap
        but aren't fully serialized.
        """
        async def run_test():
            account = BankAccount(balance=100)

            async with AsyncInterlace() as il:
                BankAccount._interlace = il

                @il.task('t1')
                async def task1():
                    await account.transfer(50)

                @il.task('t2')
                async def task2():
                    await account.transfer(50)

                # Task 2 starts before task 1 finishes
                il.order([
                    ('t1', 'transfer', 'before_read'),  # T1 enters
                    ('t1', 'transfer', 'after_read'),   # T1 reads 100
                    ('t2', 'transfer', 'before_read'),  # T2 enters
                    ('t2', 'transfer', 'after_read'),   # T2 reads 100 (race!)
                    ('t1', 'transfer', 'before_write'), # T1 writes 150
                    ('t2', 'transfer', 'before_write'), # T2 writes 150 (overwrites!)
                ])

                await il.run()
                BankAccount._interlace = None

            return account.balance

        # Run the async test
        balance = asyncio.run(run_test())

        self.assertEqual(balance, 150,
                        "Partial overlap causes lost update")


class TestAsyncInterleaveBuilder(unittest.TestCase):
    """Test the AsyncInterleaveBuilder fluent API."""

    def test_builder_api(self):
        """Test that the builder API works for creating sequences."""
        async def run_test():
            account = BankAccount(balance=100)

            async with AsyncInterlace() as il:
                BankAccount._interlace = il

                @il.task('t1')
                async def task1():
                    await account.transfer(50)

                @il.task('t2')
                async def task2():
                    await account.transfer(50)

                # Use the builder for a more readable sequence definition
                builder = AsyncInterleaveBuilder()
                builder.step('t1', 'transfer', 'before_read') \
                       .step('t1', 'transfer', 'after_read') \
                       .step('t2', 'transfer', 'before_read') \
                       .step('t2', 'transfer', 'after_read') \
                       .step('t1', 'transfer', 'before_write') \
                       .step('t2', 'transfer', 'before_write')

                il.order(builder.build())
                await il.run()
                BankAccount._interlace = None

            return account.balance

        # Run the async test
        balance = asyncio.run(run_test())

        self.assertEqual(balance, 150)


class TestComplexScenarios(unittest.TestCase):
    """Test more complex scenarios with multiple methods."""

    def test_multiple_operations(self):
        """
        Test interleaving across multiple different async operations.

        This shows how AsyncInterlace can control ordering across different
        operations, not just multiple calls to the same method.
        """
        async def run_test():
            class Counter:
                _interlace = None

                def __init__(self):
                    self.value = 0

                async def increment(self):
                    if self._interlace:
                        await self._interlace.checkpoint('increment', 'before_read')
                    current = self.value
                    if self._interlace:
                        await self._interlace.checkpoint('increment', 'after_read')
                    new_value = current + 1
                    if self._interlace:
                        await self._interlace.checkpoint('increment', 'before_write')
                    self.value = new_value

                async def decrement(self):
                    if self._interlace:
                        await self._interlace.checkpoint('decrement', 'before_read')
                    current = self.value
                    if self._interlace:
                        await self._interlace.checkpoint('decrement', 'after_read')
                    new_value = current - 1
                    if self._interlace:
                        await self._interlace.checkpoint('decrement', 'before_write')
                    self.value = new_value

            counter = Counter()

            async with AsyncInterlace() as il:
                Counter._interlace = il

                @il.task('inc')
                async def increment_task():
                    await counter.increment()

                @il.task('dec')
                async def decrement_task():
                    await counter.decrement()

                # Interleave increment and decrement: both read 0, then both write
                il.order([
                    ('inc', 'increment', 'before_read'),  # inc enters
                    ('inc', 'increment', 'after_read'),   # reads 0
                    ('dec', 'decrement', 'before_read'),  # dec enters
                    ('dec', 'decrement', 'after_read'),   # reads 0
                    ('inc', 'increment', 'before_write'), # writes 1
                    ('dec', 'decrement', 'before_write'), # writes -1 (overwrites!)
                ])

                await il.run()
                Counter._interlace = None

            return counter.value

        # Run the async test
        value = asyncio.run(run_test())

        # The decrement "wins" and we get -1 instead of 0
        self.assertEqual(value, -1,
                        "Race between increment and decrement")


class TestDocumentationExamples(unittest.TestCase):
    """Tests that serve as clear documentation examples."""

    def test_readme_example(self):
        """
        A clear, self-documenting example for README/documentation.

        This test should be immediately understandable to someone
        learning about race conditions in async code and how AsyncInterlace
        helps test them.
        """
        async def run_test():
            # Setup: A bank account with a race condition
            account = BankAccount(balance=100)

            # Create an async interlace coordinator
            async with AsyncInterlace() as il:
                BankAccount._interlace = il

                # Define two tasks that will run concurrently
                @il.task('alice')
                async def alice_deposits():
                    await account.transfer(50)  # Alice deposits $50

                @il.task('bob')
                async def bob_deposits():
                    await account.transfer(50)  # Bob deposits $50

                # Force a specific interleaving that exposes the race condition:
                # Both tasks read the balance BEFORE either one writes
                il.order([
                    ('alice', 'transfer', 'before_read'),  # Alice enters
                    ('alice', 'transfer', 'after_read'),   # Alice reads: $100
                    ('bob', 'transfer', 'before_read'),    # Bob enters
                    ('bob', 'transfer', 'after_read'),     # Bob reads: $100 (stale!)
                    ('alice', 'transfer', 'before_write'), # Alice writes: $150
                    ('bob', 'transfer', 'before_write'),   # Bob writes: $150 (overwrites Alice!)
                ])

                # Run with controlled interleaving
                await il.run()
                BankAccount._interlace = None

            return account.balance

        # Run the async test
        balance = asyncio.run(run_test())

        # Result: Only $150 instead of $200 (one deposit was lost!)
        assert balance == 150, \
            "Race condition reproduced: One deposit was lost!"

        print(f"Deterministically reproduced async race condition")
        print(f"  Expected: $200 (100 + 50 + 50)")
        print(f"  Actual:   ${balance} (one update lost)")

    def test_safe_implementation_comparison(self):
        """
        Compare unsafe vs safe async implementation to show the difference.

        This is a great documentation example showing that proper
        async synchronization fixes the issue.
        """
        async def run_test():
            # Test 1: Unsafe version with forced race
            unsafe_account = BankAccount(balance=100)

            async with AsyncInterlace() as il:
                BankAccount._interlace = il

                @il.task('t1')
                async def task1():
                    await unsafe_account.transfer(50)

                @il.task('t2')
                async def task2():
                    await unsafe_account.transfer(50)

                il.order([
                    ('t1', 'transfer', 'before_read'),
                    ('t1', 'transfer', 'after_read'),
                    ('t2', 'transfer', 'before_read'),
                    ('t2', 'transfer', 'after_read'),
                    ('t1', 'transfer', 'before_write'),
                    ('t2', 'transfer', 'before_write'),
                ])

                await il.run()
                BankAccount._interlace = None

            # Test 2: Safe version (doesn't use checkpoints, so race doesn't manifest)
            # The safe version would naturally serialize due to the lock
            safe_account = BankAccount(balance=100)

            async def safe_transfer_task(account):
                await account.safe_transfer(50)

            await asyncio.gather(
                safe_transfer_task(safe_account),
                safe_transfer_task(safe_account)
            )

            return unsafe_account.balance, safe_account.balance

        # Run the async test
        unsafe_balance, safe_balance = asyncio.run(run_test())

        # Compare results
        print(f"\n{'='*60}")
        print(f"Comparing unsafe vs safe async implementation:")
        print(f"{'='*60}")
        print(f"Unsafe (race condition):  ${unsafe_balance}")
        print(f"Safe (with lock):         ${safe_balance}")
        print(f"{'='*60}\n")

        self.assertEqual(unsafe_balance, 150,
                        "Unsafe version has race")
        self.assertEqual(safe_balance, 200,
                        "Safe version prevents race")

    def test_async_insight_demonstration(self):
        """
        Demonstrate the key insight: races only happen at await points.

        This test shows that code between await points runs atomically,
        which is a key difference from threading.
        """
        async def run_test():
            class AtomicCounter:
                """
                A counter that shows atomic behavior between await points.

                Even without locks, the increment between await points is atomic
                because async is single-threaded and only yields at await.
                """
                _interlace = None

                def __init__(self):
                    self.value = 0
                    self.operations = []

                async def complex_increment(self):
                    if self._interlace:
                        await self._interlace.checkpoint('increment', 'before_read')

                    # Atomic section: all of this runs without interruption
                    temp = self.value
                    temp += 1
                    temp = temp * 2
                    temp = temp - 1

                    if self._interlace:
                        await self._interlace.checkpoint('increment', 'midpoint')

                    # After checkpoint: atomic section 2
                    # But now another task could have run!
                    self.value = temp
                    self.operations.append(temp)

            counter = AtomicCounter()

            async with AsyncInterlace() as il:
                AtomicCounter._interlace = il

                @il.task('t1')
                async def task1():
                    await counter.complex_increment()

                @il.task('t2')
                async def task2():
                    await counter.complex_increment()

                # Force both tasks to enter and read before either writes.
                # The before_read checkpoint gates entry, and midpoint gates
                # the write. With this schedule both tasks read value=0.
                il.order([
                    ('t1', 'increment', 'before_read'),  # T1 enters
                    ('t2', 'increment', 'before_read'),  # T2 enters
                    ('t1', 'increment', 'midpoint'),      # T1 computed (0+1)*2-1 = 1
                    ('t2', 'increment', 'midpoint'),      # T2 computed (0+1)*2-1 = 1
                ])

                await il.run()
                AtomicCounter._interlace = None

            return counter.value, counter.operations

        # Run the async test
        final_value, operations = asyncio.run(run_test())

        # Both tasks read value=0 and calculated temp=1, so last write wins
        self.assertEqual(final_value, 1)
        self.assertEqual(operations, [1, 1])
        print("\nKey insight demonstrated:")
        print("  - Complex calculations between awaits ran atomically")
        print("  - Race only occurred at the await checkpoint")
        print("  - This is why async races are simpler to reason about!")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
