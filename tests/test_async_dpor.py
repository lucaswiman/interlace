"""Tests for async DPOR (Dynamic Partial Order Reduction).

Verifies that explore_async_dpor systematically explores async
interleavings using the Rust DPOR engine, with await points as
scheduling granularity.
"""

from __future__ import annotations

import asyncio

from frontrun.cli import require_active


class TestAsyncDporBasic:
    """Basic async DPOR functionality tests."""

    def test_finds_lost_update(self) -> None:
        """DPOR should systematically find the lost-update race."""
        require_active("test_async_dpor_lost_update")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment],
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Async DPOR should find the lost update"
        assert result.num_explored >= 1

    def test_no_race_when_atomic(self) -> None:
        """DPOR should verify correctness when there's no race."""
        require_active("test_async_dpor_no_race")
        from frontrun.async_dpor import explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def safe_increment(counter: Counter) -> None:
            # Atomic: no await between read and write
            counter.value += 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[safe_increment, safe_increment],
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds, f"No race expected: {result.counterexample}"
        assert result.num_explored >= 1

    def test_three_tasks(self) -> None:
        """DPOR should handle three concurrent tasks."""
        require_active("test_async_dpor_three_tasks")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment, increment],
                invariant=lambda c: c.value == 3,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds, "Should find lost update with 3 tasks"

    def test_multiple_await_points(self) -> None:
        """DPOR should explore interleavings with multiple await points per task."""
        require_active("test_async_dpor_multiple_awaits")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.log: list[str] = []

        async def task_a(state: State) -> None:
            state.log.append("a1")
            await await_point()
            state.log.append("a2")
            await await_point()
            state.log.append("a3")

        async def task_b(state: State) -> None:
            state.log.append("b1")
            await await_point()
            state.log.append("b2")

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[task_a, task_b],
                invariant=lambda s: True,  # always passes
                deadlock_timeout=5.0,
            )
        )

        assert result.property_holds
        # DPOR should have explored multiple distinct interleavings
        assert result.num_explored >= 1

    def test_stop_on_first(self) -> None:
        """stop_on_first=True should stop after finding the first violation."""
        require_active("test_async_dpor_stop_on_first")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment],
                invariant=lambda c: c.value == 2,
                stop_on_first=True,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert len(result.failures) == 1


class TestAsyncDporDeadlock:
    """Deadlocks should surface as property_holds=False, not silently time out.

    Mirrors the sync TestDeadlockAsInvariantViolation tests in test_dpor.py,
    adapted for async coroutines with asyncio.Lock as the locking primitive.
    """

    def test_two_coroutine_row_lock_deadlock(self) -> None:
        """Classic lock-order inversion: C1 locks row1→row2, C2 locks row2→row1.

        C1 acquires row1, C2 acquires row2, then C1 tries row2 (blocked)
        and C2 tries row1 (blocked).  DPOR should find the deadlocking
        interleaving and report it — not actually deadlock.
        """
        require_active("test_async_dpor_two_coroutine_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.row1 = asyncio.Lock()
                self.row2 = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.row1.acquire()
            try:
                await await_point()
                await state.row2.acquire()
                state.row2.release()
            finally:
                state.row1.release()

        async def coroutine2(state: State) -> None:
            await state.row2.acquire()
            try:
                await await_point()
                await state.row1.acquire()
                state.row1.release()
            finally:
                state.row2.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,  # no data invariant — deadlock itself is the bug
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_three_coroutine_directed_cycle_deadlock(self) -> None:
        """Three-coroutine deadlock: C1→row1→row2, C2→row2→row3, C3→row3→row1.

        Forms the directed cycle C1→C2→C3→C1 when each holds its first
        lock and waits for its second.
        """
        require_active("test_async_dpor_three_coroutine_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.row1 = asyncio.Lock()
                self.row2 = asyncio.Lock()
                self.row3 = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.row1.acquire()
            try:
                await await_point()
                await state.row2.acquire()
                state.row2.release()
            finally:
                state.row1.release()

        async def coroutine2(state: State) -> None:
            await state.row2.acquire()
            try:
                await await_point()
                await state.row3.acquire()
                state.row3.release()
            finally:
                state.row2.release()

        async def coroutine3(state: State) -> None:
            await state.row3.acquire()
            try:
                await await_point()
                await state.row1.acquire()
                state.row1.release()
            finally:
                state.row3.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2, coroutine3],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "3-way deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_combined_asyncio_lock_and_sql_deadlock(self) -> None:
        """Mixed resource deadlock: one coroutine holds a real SQL row lock
        and waits for an asyncio.Lock, the other holds the asyncio.Lock and
        waits for the same SQL row lock.

        This cross-resource deadlock is invisible to both the DB backend
        (which only sees row locks) and asyncio (which only sees asyncio.Lock).
        Only the DPOR scheduler can detect it because it tracks both resource
        types in a unified WaitForGraph.

        Uses aiosqlite so no external database is needed.
        """
        require_active("test_async_dpor_combined_lock_deadlock")
        import os
        import tempfile

        import aiosqlite  # type: ignore[import-untyped]

        from frontrun.async_dpor import await_point, explore_async_dpor

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        class State:
            def __init__(self) -> None:
                self.app_lock = asyncio.Lock()
                self.db_path = db_path

        async def _setup_db() -> None:
            async with aiosqlite.connect(db_path) as db:
                await db.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)")
                await db.execute("INSERT OR REPLACE INTO t (id, v) VALUES (1, 0)")
                await db.commit()

        async def coroutine1(state: State) -> None:
            # Acquire a SQL row lock (UPDATE inside tx), then try the app lock
            async with aiosqlite.connect(state.db_path) as db:
                await db.execute("BEGIN")
                await db.execute("UPDATE t SET v = 1 WHERE id = 1")
                # Row lock on sql:t:(('id', 1)) is now held by this task
                await await_point()
                # Try to acquire the asyncio.Lock — may block if C2 holds it
                await state.app_lock.acquire()
                state.app_lock.release()
                await db.execute("COMMIT")

        async def coroutine2(state: State) -> None:
            # Acquire the app lock first, then try to get the SQL row lock
            await state.app_lock.acquire()
            try:
                await await_point()
                async with aiosqlite.connect(state.db_path) as db:
                    await db.execute("BEGIN")
                    # This tries to acquire the same row lock held by C1
                    await db.execute("UPDATE t SET v = 2 WHERE id = 1")
                    await db.execute("COMMIT")
            finally:
                state.app_lock.release()

        async def run() -> object:
            await _setup_db()
            return await explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,
                detect_sql=True,
                deadlock_timeout=2.0,
                timeout_per_run=5.0,
            )

        try:
            result = asyncio.run(run())
        finally:
            os.unlink(db_path)

        assert not result.property_holds, "Cross-resource deadlock should set property_holds=False"
        assert result.explanation is not None
        assert "deadlock" in result.explanation.lower()

    def test_partial_deadlock_with_completing_coroutine(self) -> None:
        """Three coroutines: two deadlock while a third completes normally.

        C1 and C2 have lock-order inversion; C3 does independent work.
        The partial deadlock should still be detected even though C3 finishes.
        """
        require_active("test_async_dpor_partial_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()
                self.c3_done = False

        async def coroutine1(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await state.lock_b.acquire()
            try:
                await await_point()
                await state.lock_a.acquire()
                state.lock_a.release()
            finally:
                state.lock_b.release()

        async def coroutine3(state: State) -> None:
            await await_point()
            state.c3_done = True

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2, coroutine3],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Partial deadlock should still be detected"

    def test_no_deadlock_consistent_lock_order(self) -> None:
        """Consistent lock ordering should not be reported as a deadlock.

        Both coroutines acquire locks in the same order (lock_a then lock_b),
        so no cycle is possible.
        """
        require_active("test_async_dpor_no_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert result.property_holds, "Consistent lock order should not be reported as deadlock"

    def test_self_deadlock_non_reentrant_lock(self) -> None:
        """Single coroutine tries to acquire the same non-reentrant asyncio.Lock twice.

        asyncio.Lock is not reentrant, so acquiring it while already held
        by the same coroutine is an instant deadlock.
        """
        require_active("test_async_dpor_self_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()

        async def coroutine1(state: State) -> None:
            await state.lock.acquire()
            await await_point()
            # Re-acquire the same non-reentrant lock — instant self-deadlock
            await state.lock.acquire()
            state.lock.release()
            state.lock.release()

        async def coroutine2(state: State) -> None:
            await await_point()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,
                max_executions=20,
                deadlock_timeout=1.0,
                timeout_per_run=2.0,
            )
        )

        assert not result.property_holds, "Self-deadlock should set property_holds=False"

    def test_asymmetric_await_points_before_deadlock(self) -> None:
        """Coroutines have different numbers of await points before the
        deadlocking acquire.  Verifies DPOR explores enough interleavings
        to reach the state where both hold one lock.
        """
        require_active("test_async_dpor_asymmetric_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()
                self.log: list[str] = []

        async def coroutine1(state: State) -> None:
            # Several await points of work before acquiring locks
            state.log.append("c1_step1")
            await await_point()
            state.log.append("c1_step2")
            await await_point()
            state.log.append("c1_step3")
            await await_point()
            # Now do the lock-order-inversion pattern
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            # Only one await point before locking
            state.log.append("c2_step1")
            await await_point()
            await state.lock_b.acquire()
            try:
                await await_point()
                await state.lock_a.acquire()
                state.lock_a.release()
            finally:
                state.lock_b.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,
                max_executions=50,
                deadlock_timeout=1.0,
                timeout_per_run=2.0,
            )
        )

        assert not result.property_holds, "Asymmetric deadlock should be found"

    def test_data_dependent_lock_order_deadlock(self) -> None:
        """Lock acquisition order depends on runtime state.

        C1 always acquires lock_a then lock_b.  C2 reads a shared flag
        (set by C1) that determines whether it acquires lock_b then lock_a
        (deadlock) or lock_a then lock_b (safe).  Only the interleaving
        where C1 sets the flag before C2 reads it triggers the deadlock.
        """
        require_active("test_async_dpor_data_dependent_deadlock")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock_a = asyncio.Lock()
                self.lock_b = asyncio.Lock()
                self.reverse_order = False

        async def coroutine1(state: State) -> None:
            state.reverse_order = True
            await await_point()
            await state.lock_a.acquire()
            try:
                await await_point()
                await state.lock_b.acquire()
                state.lock_b.release()
            finally:
                state.lock_a.release()

        async def coroutine2(state: State) -> None:
            await await_point()
            if state.reverse_order:
                # Opposite order from C1 → deadlock possible
                await state.lock_b.acquire()
                try:
                    await await_point()
                    await state.lock_a.acquire()
                    state.lock_a.release()
                finally:
                    state.lock_b.release()
            else:
                # Same order as C1 → safe
                await state.lock_a.acquire()
                try:
                    await await_point()
                    await state.lock_b.acquire()
                    state.lock_b.release()
                finally:
                    state.lock_a.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[coroutine1, coroutine2],
                invariant=lambda s: True,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        assert not result.property_holds, "Data-dependent deadlock should be found"

    def test_dining_philosophers_three(self) -> None:
        """Three dining philosophers: each acquires fork[i] then fork[(i+1)%3].

        Classic deadlock when all philosophers pick up their left fork
        simultaneously.  Uses 3 philosophers to keep the DPOR state space
        manageable (4 philosophers with preemption_bound=2 doesn't explore
        enough interleavings).
        """
        require_active("test_async_dpor_dining_philosophers")
        from frontrun.async_dpor import await_point, explore_async_dpor

        num_philosophers = 3

        class State:
            def __init__(self) -> None:
                self.forks = [asyncio.Lock() for _ in range(num_philosophers)]

        def make_philosopher(i: int):  # noqa: ANN202
            async def philosopher(state: State) -> None:
                left = i
                right = (i + 1) % num_philosophers
                await state.forks[left].acquire()
                try:
                    await await_point()
                    await state.forks[right].acquire()
                    state.forks[right].release()
                finally:
                    state.forks[left].release()

            return philosopher

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[make_philosopher(i) for i in range(num_philosophers)],
                invariant=lambda s: True,
                deadlock_timeout=1.0,
                timeout_per_run=2.0,
            )
        )

        assert not result.property_holds, "Dining philosophers deadlock should be found"


class TestAsyncDporCleanup:
    """Tests for resource cleanup when tasks finish without releasing locks."""

    def test_row_locks_released_on_task_finish(self) -> None:
        """Row locks should be released from the WaitForGraph when a task finishes.

        Bug: _cleanup_task_context doesn't call release_row_locks(task_id),
        so stale row lock entries remain in _active_row_locks and the
        WaitForGraph after a task completes without COMMIT.

        Scenario: Task 0 acquires a row lock and finishes without releasing.
        After cleanup, the scheduler's _active_row_locks should NOT still
        show task 0 as the holder.
        """
        require_active("test_async_dpor_row_lock_cleanup")
        from frontrun.async_dpor import AsyncDporScheduler, _patch_asyncio_lock, _unpatch_asyncio_lock, await_point

        async def run() -> None:
            from frontrun._dpor import PyDporEngine, PyExecution  # type: ignore[reportAttributeAccessIssue]

            _patch_asyncio_lock()
            try:
                engine = PyDporEngine(num_threads=2, preemption_bound=2)
                execution = engine.begin_execution()
                scheduler = AsyncDporScheduler(engine, execution, 2, deadlock_timeout=2.0)

                # Simulate task 0 acquiring row locks
                scheduler.acquire_row_locks(0, ["sql:t:(('id', 1))"])
                assert "sql:t:(('id', 1))" in scheduler._active_row_locks
                assert scheduler._active_row_locks["sql:t:(('id', 1))"] == 0

                # Simulate task 0 finishing (as _run's finally block does)
                scheduler._cleanup_task_context(0)

                # After cleanup, row locks should be released
                assert "sql:t:(('id', 1))" not in scheduler._active_row_locks, (
                    "Row lock should be released when task finishes, "
                    f"but _active_row_locks still contains: {scheduler._active_row_locks}"
                )
                assert 0 not in scheduler._task_row_locks, (
                    "Task's row lock set should be cleared on finish, "
                    f"but _task_row_locks still contains task 0: {scheduler._task_row_locks}"
                )
            finally:
                _unpatch_asyncio_lock()

        asyncio.run(run())

    def test_asyncio_lock_released_on_task_exception(self) -> None:
        """asyncio.Lock should be released when a task raises an exception.

        Bug: When a task finishes while holding a _CooperativeAsyncLock
        (e.g., exception without release()), the WaitForGraph holding edge
        remains AND the underlying real asyncio.Lock stays locked, blocking
        any other task that tries to acquire it.

        Scenario: Task 0 acquires a lock and crashes. Task 1 should be able
        to acquire the same lock without timing out.
        """
        require_active("test_async_dpor_lock_cleanup_on_exception")
        from frontrun.async_dpor import explore_async_dpor

        class State:
            def __init__(self) -> None:
                self.lock = asyncio.Lock()
                self.task1_got_lock = False

        async def task0_crashes(state: State) -> None:
            await state.lock.acquire()
            raise RuntimeError("intentional crash while holding lock")

        async def task1_acquires(state: State) -> None:
            # Task 1 should be able to acquire the lock after task 0 crashes
            await state.lock.acquire()
            state.task1_got_lock = True
            state.lock.release()

        result = asyncio.run(
            explore_async_dpor(
                setup=State,
                tasks=[task0_crashes, task1_acquires],
                invariant=lambda s: s.task1_got_lock,
                deadlock_timeout=2.0,
                timeout_per_run=3.0,
            )
        )

        # The invariant should hold: task 1 should always be able to get the lock
        # after task 0 crashes and its locks are cleaned up.
        assert result.property_holds, (
            "Task 1 should acquire the lock after task 0 crashes, "
            f"but got: property_holds={result.property_holds}, explanation={result.explanation}"
        )


class TestAsyncDporExplanation:
    """Tests for human-readable explanations of invariant violations."""

    def test_explanation_set_for_invariant_violation(self) -> None:
        """result.explanation should be non-None when an invariant violation is found.

        Bug: explore_async_dpor sets result.explanation for deadlocks but
        NOT for invariant violations. The explanation field stays None.
        """
        require_active("test_async_dpor_invariant_explanation")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment],
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert result.explanation is not None, (
            "result.explanation should be set for invariant violations, but it is None"
        )

    def test_explanation_contains_schedule_info(self) -> None:
        """Explanation should contain information about the interleaving schedule.

        For async DPOR, the explanation should describe which tasks ran
        at which points, making it possible to understand the race condition.
        """
        require_active("test_async_dpor_explanation_content")
        from frontrun.async_dpor import await_point, explore_async_dpor

        class Counter:
            def __init__(self) -> None:
                self.value = 0

        async def increment(counter: Counter) -> None:
            temp = counter.value
            await await_point()
            counter.value = temp + 1

        result = asyncio.run(
            explore_async_dpor(
                setup=Counter,
                tasks=[increment, increment],
                invariant=lambda c: c.value == 2,
                deadlock_timeout=5.0,
            )
        )

        assert not result.property_holds
        assert result.explanation is not None
        # Explanation should mention tasks/schedule
        explanation_lower = result.explanation.lower()
        assert "task" in explanation_lower or "schedule" in explanation_lower or "interleav" in explanation_lower, (
            f"Explanation should describe the interleaving, got: {result.explanation}"
        )
