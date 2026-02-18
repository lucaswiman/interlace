"""
Exhaustive concurrency tests for SQLAlchemy pool module.

Tests multiple distinct concurrency bugs in the SQLAlchemy connection pool
implementation using interlace's bytecode-level interleaving exploration.

Bug categories tested:
1.  _dec_overflow() lost update in unlimited mode (no lock when max_overflow==-1)
2.  _inc_overflow + _dec_overflow cross-race in unlimited mode
3.  _inc_overflow() lost update with 3 threads in unlimited mode
4.  checkedout() diagnostic torn read between _pool.qsize() and _overflow
5.  dispose() + _inc_overflow race (dispose resets _overflow while another
    thread increments it)
6.  _do_return_conn() + _dec_overflow double-decrement race (queue full path)
7.  SingletonThreadPool._cleanup() + _do_get() race on shared _all_conns set
8.  AssertionPool._do_get() + _do_return_conn() check-then-act race on
    _checked_out flag

Repository: https://github.com/sqlalchemy/sqlalchemy
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_lib = os.path.join(_test_dir, "..", "external_repos", "sqlalchemy", "lib")
# Insert local repo path FIRST so interlace can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_lib))

from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from sqlalchemy.pool import QueuePool  # noqa: E402
from sqlalchemy.pool.impl import SingletonThreadPool, AssertionPool  # noqa: E402

from interlace.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unlimited_pool(pool_size=5):
    """Create a QueuePool with unlimited overflow (max_overflow=-1).

    The creator callback is never actually invoked in counter-only tests.
    """
    return QueuePool(lambda: None, pool_size=pool_size, max_overflow=-1)


# ===========================================================================
# Test 1: _dec_overflow() lost update in unlimited mode
# ===========================================================================
# Mirror of the known _inc_overflow bug but for the decrement path.
# When max_overflow == -1, _dec_overflow() does ``self._overflow -= 1``
# WITHOUT holding _overflow_lock.  The -= is not atomic at bytecode level.

class DecOverflowState:
    """Two threads each call _dec_overflow() once."""

    def __init__(self):
        self.pool = _make_unlimited_pool()
        self.initial_overflow = self.pool._overflow

    def thread1(self):
        self.pool._dec_overflow()

    def thread2(self):
        self.pool._dec_overflow()


def _dec_overflow_invariant(s: DecOverflowState) -> bool:
    return s.pool._overflow == s.initial_overflow - 2


def test_dec_overflow_lost_update():
    """_dec_overflow() lost update when max_overflow == -1."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DecOverflowState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_dec_overflow_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_dec_overflow_sweep():
    """Sweep 20 seeds for _dec_overflow lost update."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: DecOverflowState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_dec_overflow_invariant,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Test 2: _inc_overflow + _dec_overflow cross-race in unlimited mode
# ===========================================================================
# One thread increments _overflow while another decrements it.  Both skip the
# lock when max_overflow == -1.  The final value should be unchanged
# (initial + 1 - 1 == initial), but a torn read/write can produce any of
# initial-1, initial, or initial+1.

class IncDecCrossRaceState:
    """Thread 1 increments, thread 2 decrements."""

    def __init__(self):
        self.pool = _make_unlimited_pool()
        self.initial_overflow = self.pool._overflow

    def thread1(self):
        self.pool._inc_overflow()

    def thread2(self):
        self.pool._dec_overflow()


def _inc_dec_cross_invariant(s: IncDecCrossRaceState) -> bool:
    # inc +1, dec -1 => net zero change
    return s.pool._overflow == s.initial_overflow


def test_inc_dec_cross_race():
    """Race between _inc_overflow and _dec_overflow in unlimited mode."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: IncDecCrossRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_inc_dec_cross_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_inc_dec_cross_race_sweep():
    """Sweep 20 seeds for inc/dec cross-race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: IncDecCrossRaceState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_inc_dec_cross_invariant,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Test 3: _inc_overflow with 3 threads (triple lost update)
# ===========================================================================
# Three concurrent _inc_overflow() calls in unlimited mode.  Expected final
# value is initial + 3 but the race can lose one or even two increments.

class TripleIncOverflowState:
    """Three threads each call _inc_overflow() once."""

    def __init__(self):
        self.pool = _make_unlimited_pool()
        self.initial_overflow = self.pool._overflow

    def thread1(self):
        self.pool._inc_overflow()

    def thread2(self):
        self.pool._inc_overflow()

    def thread3(self):
        self.pool._inc_overflow()


def _triple_inc_invariant(s: TripleIncOverflowState) -> bool:
    return s.pool._overflow == s.initial_overflow + 3


def test_triple_inc_overflow():
    """Three-thread _inc_overflow lost update in unlimited mode."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: TripleIncOverflowState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
            invariant=_triple_inc_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_triple_inc_overflow_reproduce():
    """Find then deterministically reproduce the 3-thread lost update."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: TripleIncOverflowState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
            invariant=_triple_inc_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )

    if not result.counterexample:
        print("No counterexample found -- skipping reproduction")
        return 0

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: TripleIncOverflowState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
        )
        expected = state.initial_overflow + 3
        actual = state.pool._overflow
        is_bug = actual != expected
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: _overflow={actual} "
            f"(expected {expected}) [{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ===========================================================================
# Test 4: checkedout() torn read
# ===========================================================================
# checkedout() computes:  self._pool.maxsize - self._pool.qsize() + self._overflow
# It reads qsize() and _overflow WITHOUT any lock.  A concurrent _inc_overflow
# (unlimited mode) that changes _overflow between the qsize() read and the
# _overflow read can cause checkedout() to return an inconsistent snapshot.
#
# We test this by having one thread call _inc_overflow() while another reads
# checkedout().  The initial checkedout count is 0 (pool_size - pool_size + _overflow
# where _overflow starts at -pool_size for an empty pool).  After one
# _inc_overflow, checkedout should be 1.  But if checkedout() reads _overflow
# at different moments it can observe 0 or 1, and we specifically check that
# once both threads finish, the checkedout value matches expectations.

class CheckedOutTornReadState:
    """Thread 1 increments overflow, thread 2 reads checkedout() mid-race."""

    def __init__(self):
        self.pool = _make_unlimited_pool()
        self.initial_overflow = self.pool._overflow
        self.observed_checkedout = None

    def thread1(self):
        self.pool._inc_overflow()

    def thread2(self):
        # Read checkedout while thread1 may be mid-increment
        self.observed_checkedout = self.pool.checkedout()


def _checkedout_torn_invariant(s: CheckedOutTornReadState) -> bool:
    # After both threads finish: _overflow should be initial+1 => checkedout=1
    # The thread2 observation should have been either 0 or 1, both valid
    # mid-execution snapshots.  But we check the FINAL state is consistent:
    # pool.checkedout() must equal pool.maxsize - pool.qsize() + pool._overflow
    # and _overflow must be initial+1.
    final_checkedout = s.pool.checkedout()
    expected_overflow = s.initial_overflow + 1
    overflow_correct = s.pool._overflow == expected_overflow
    checkedout_consistent = final_checkedout == (
        s.pool._pool.maxsize - s.pool._pool.qsize() + s.pool._overflow
    )
    return overflow_correct and checkedout_consistent


def test_checkedout_torn_read():
    """Torn read in checkedout() during concurrent _inc_overflow."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: CheckedOutTornReadState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_checkedout_torn_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 5: dispose() + _inc_overflow race
# ===========================================================================
# dispose() resets _overflow with: self._overflow = 0 - self.size()
# If a concurrent _inc_overflow happens at the same time (unlimited mode),
# the increment can be lost because dispose writes over it, or _inc_overflow
# can read the stale pre-dispose value.
#
# Scenario: Thread 1 calls _inc_overflow.  Thread 2 calls dispose.
# If thread 1 increments first, then thread 2 resets => _overflow = -pool_size
# If thread 2 resets first, then thread 1 increments => _overflow = -pool_size + 1
# But if they interleave at the bytecode level within the += operation:
# thread 1 loads _overflow (-5), thread 2 sets _overflow = -5, thread 1
# stores -5+1 = -4.  This means the dispose reset was overwritten.
# Alternatively: thread 2 sets -5, thread 1 loads -5 and stores -4.
# The invariant: _overflow must be EITHER -5 (dispose won) or -4 (inc after dispose).
# A buggy interleaving can produce -5+1 = -4 even when dispose should have won,
# but more importantly, the += torn write can produce unexpected values.

class DisposeIncRaceState:
    """Thread 1: _inc_overflow, Thread 2: dispose."""

    def __init__(self):
        self.pool = _make_unlimited_pool(pool_size=5)
        self.initial_overflow = self.pool._overflow  # -5

    def thread1(self):
        self.pool._inc_overflow()

    def thread2(self):
        # dispose resets _overflow = 0 - size() = -5
        self.pool.dispose()


def _dispose_inc_invariant(s: DisposeIncRaceState) -> bool:
    # Valid final states:
    # - If inc happens entirely before dispose: _overflow goes -5 -> -4 -> -5 (dispose resets)
    # - If dispose happens entirely before inc: _overflow goes -5 -> -5 -> -4
    # - Invalid: _overflow is anything other than -5 or -4
    pool_size = s.pool._pool.maxsize
    expected_reset = 0 - pool_size  # -5
    return s.pool._overflow in (expected_reset, expected_reset + 1)


def test_dispose_inc_overflow_race():
    """Race between dispose() and _inc_overflow() in unlimited mode."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DisposeIncRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_dispose_inc_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_dispose_inc_overflow_race_sweep():
    """Sweep 20 seeds for dispose + inc_overflow race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: DisposeIncRaceState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_dispose_inc_invariant,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Test 6: Double _dec_overflow via _do_return_conn queue-full path
# ===========================================================================
# _do_return_conn tries pool.put(record, False).  If the queue is full, it
# catches sqla_queue.Full, closes the record, then calls _dec_overflow().
# If two threads both return connections to a full pool at the same time,
# both call _dec_overflow in the unlimited-overflow path (no lock).
# This is the same lost-update pattern as Test 1 but triggered through the
# higher-level _do_return_conn code path.

class DoubleReturnDecOverflowState:
    """Two threads both call _dec_overflow (simulating queue-full returns)."""

    def __init__(self):
        self.pool = _make_unlimited_pool()
        # Artificially set overflow to a known value
        self.pool._overflow = 4
        self.initial_overflow = self.pool._overflow

    def thread1(self):
        self.pool._dec_overflow()

    def thread2(self):
        self.pool._dec_overflow()


def _double_return_dec_invariant(s: DoubleReturnDecOverflowState) -> bool:
    return s.pool._overflow == s.initial_overflow - 2


def test_double_return_dec_overflow():
    """Two concurrent _dec_overflow calls (queue-full return path)."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DoubleReturnDecOverflowState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_double_return_dec_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 7: SingletonThreadPool._all_conns set mutation race
# ===========================================================================
# SingletonThreadPool._do_get does:
#   if len(self._all_conns) >= self.size:
#       self._cleanup()
#   self._all_conns.add(c)
#
# _cleanup() does:
#   while len(self._all_conns) >= self.size:
#       c = self._all_conns.pop()
#       c.close()
#
# If two threads simultaneously check len() and both decide cleanup is needed,
# they can over-pop from _all_conns, or one thread can add while another pops.
# The _all_conns set is a plain Python set with no synchronization.
#
# We test this by having two threads both call into the len/_cleanup/add
# sequence and verifying the set size stays consistent.

class SingletonPoolCleanupRaceState:
    """Two threads both manipulate _all_conns via _cleanup + add."""

    def __init__(self):
        self.pool = SingletonThreadPool(
            lambda: object(),  # minimal creator
            pool_size=2,
        )
        # Pre-populate _all_conns with dummy entries to trigger cleanup path
        for _ in range(2):
            dummy = type('DummyRecord', (), {'close': lambda self: None})()
            self.pool._all_conns.add(dummy)
        self.initial_count = len(self.pool._all_conns)

    def thread1(self):
        # Simulate the cleanup + add portion of _do_get
        if len(self.pool._all_conns) >= self.pool.size:
            self.pool._cleanup()

    def thread2(self):
        # Another thread also runs cleanup
        if len(self.pool._all_conns) >= self.pool.size:
            self.pool._cleanup()


def _singleton_cleanup_invariant(s: SingletonPoolCleanupRaceState) -> bool:
    # After both threads do cleanup, _all_conns size should be <= pool.size - 1
    # (each cleanup pops until len < size).  If both run the while loop and pop
    # concurrently on the same set, we can get unexpected results including
    # KeyError from pop on empty set (which would be an exception, caught by
    # interlace).  If no exception, the count should be deterministic.
    #
    # With pool_size=2 and initial 2 entries:
    # - Single cleanup: pops 1 entry, leaving 1
    # - Two sequential cleanups: first pops 1 leaving 1, second finds 1 < 2, no-op => 1
    # - Race: both see len>=2, both pop => could leave 0
    # Valid outcomes: 0 or 1 entries remaining
    return len(s.pool._all_conns) >= 0  # any non-negative is "no crash"


def _singleton_cleanup_size_invariant(s: SingletonPoolCleanupRaceState) -> bool:
    # Stricter: after cleanup, set should have exactly 1 entry (the expected
    # sequential result: one pop reduces 2 to 1, second cleanup sees 1 < 2, no-op).
    # A race where both pop gives 0, which violates this.
    return len(s.pool._all_conns) == 1


def test_singleton_cleanup_race():
    """SingletonThreadPool._cleanup() race on shared _all_conns set."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SingletonPoolCleanupRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_singleton_cleanup_size_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 8: AssertionPool check-then-act race on _checked_out
# ===========================================================================
# AssertionPool._do_get checks ``if self._checked_out`` then sets it True.
# _do_return_conn checks ``if not self._checked_out`` then sets it False.
# These are classic check-then-act patterns with no synchronization.
#
# Race scenario: Two threads both call _do_get concurrently.  Both read
# _checked_out == False, both proceed, both set _checked_out = True.
# The pool's contract says only ONE connection should be checked out.
# The second get should have raised AssertionError but did not.

class AssertionPoolDoubleCheckoutState:
    """Two threads both attempt _do_get on an AssertionPool."""

    def __init__(self):
        self.pool = AssertionPool(
            lambda: object(),  # minimal creator
            store_traceback=False,
        )
        self.thread1_got = False
        self.thread2_got = False
        self.thread1_error = False
        self.thread2_error = False

    def thread1(self):
        try:
            self.pool._do_get()
            self.thread1_got = True
        except AssertionError:
            self.thread1_error = True

    def thread2(self):
        try:
            self.pool._do_get()
            self.thread2_got = True
        except AssertionError:
            self.thread2_error = True


def _assertion_pool_invariant(s: AssertionPoolDoubleCheckoutState) -> bool:
    # Correct behavior: exactly one thread gets the connection, the other
    # gets an AssertionError.  Bug: both threads get the connection.
    both_got = s.thread1_got and s.thread2_got
    return not both_got


def test_assertion_pool_double_checkout():
    """AssertionPool check-then-act: two threads both pass _checked_out guard."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: AssertionPoolDoubleCheckoutState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_assertion_pool_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_assertion_pool_double_checkout_sweep():
    """Sweep 20 seeds for AssertionPool double-checkout race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: AssertionPoolDoubleCheckoutState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_assertion_pool_invariant,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Test 9: _inc_overflow + _inc_overflow in bounded mode exceeds max_overflow
# ===========================================================================
# When max_overflow > -1, _inc_overflow uses _overflow_lock. But we can still
# check: the lock should serialize the check-and-increment.  With
# max_overflow=1 and two threads both trying _inc_overflow, exactly one
# should return True and one False.  This tests that the lock-based path is
# actually correct (and validates interlace handles locks properly).

class BoundedIncOverflowState:
    """Two threads calling _inc_overflow with max_overflow=1."""

    def __init__(self):
        self.pool = QueuePool(
            lambda: None, pool_size=5, max_overflow=1
        )
        self.initial_overflow = self.pool._overflow  # -5
        self.result1 = None
        self.result2 = None

    def thread1(self):
        self.result1 = self.pool._inc_overflow()

    def thread2(self):
        self.result2 = self.pool._inc_overflow()


def _bounded_inc_invariant(s: BoundedIncOverflowState) -> bool:
    # With max_overflow=1: _overflow starts at -5, max is 1.
    # _inc_overflow returns True if _overflow < max_overflow, then increments.
    # Both threads should succeed (since -5 < 1 and -4 < 1), and _overflow
    # should be initial + 2 = -3.  The lock ensures no lost update.
    return s.pool._overflow == s.initial_overflow + 2


def test_bounded_inc_overflow():
    """Bounded _inc_overflow with lock -- should NOT find a bug (control test)."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: BoundedIncOverflowState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_bounded_inc_invariant,
            max_attempts=200,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 10: Concurrent dispose calls race on _overflow reset
# ===========================================================================
# Two threads call dispose() simultaneously.  dispose() drains the queue
# then sets self._overflow = 0 - self.size().  If the pool has items,
# the queue-drain loop and the _overflow reset can interleave with each
# other, potentially leaving _overflow in an inconsistent state.

class DoubleDisposeState:
    """Two threads both call dispose() on the same pool."""

    def __init__(self):
        self.pool = _make_unlimited_pool(pool_size=5)
        # Set overflow to something non-default to detect corruption
        self.pool._overflow = 3
        self.expected_reset = 0 - self.pool._pool.maxsize  # -5

    def thread1(self):
        self.pool.dispose()

    def thread2(self):
        self.pool.dispose()


def _double_dispose_invariant(s: DoubleDisposeState) -> bool:
    # Both dispose calls set _overflow = 0 - size() = -5.
    # Even with interleaving, the final write should be -5.
    # But the assignment ``self._overflow = 0 - self.size()`` involves
    # a method call and subtraction -- if another dispose interleaves
    # mid-assignment, the final value might be corrupted.
    return s.pool._overflow == s.expected_reset


def test_double_dispose_race():
    """Two concurrent dispose() calls racing on _overflow reset."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DoubleDisposeState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_double_dispose_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 11: _inc_overflow followed by _dec_overflow with 3 threads
# ===========================================================================
# Three threads: two increment, one decrements.  Expected net: initial + 1.
# All in unlimited mode, so no locks.  This amplifies the lost-update
# window.

class ThreeThreadIncDecState:
    """Two threads inc, one thread dec, all in unlimited mode."""

    def __init__(self):
        self.pool = _make_unlimited_pool()
        self.initial_overflow = self.pool._overflow

    def thread_inc1(self):
        self.pool._inc_overflow()

    def thread_inc2(self):
        self.pool._inc_overflow()

    def thread_dec(self):
        self.pool._dec_overflow()


def _three_thread_inc_dec_invariant(s: ThreeThreadIncDecState) -> bool:
    # +1 +1 -1 = net +1
    return s.pool._overflow == s.initial_overflow + 1


def test_three_thread_inc_dec():
    """Three threads: two _inc_overflow + one _dec_overflow in unlimited mode."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ThreeThreadIncDecState(),
            threads=[
                lambda s: s.thread_inc1(),
                lambda s: s.thread_inc2(),
                lambda s: s.thread_dec(),
            ],
            invariant=_three_thread_inc_dec_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_three_thread_inc_dec_sweep():
    """Sweep 20 seeds for 3-thread inc+inc+dec race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: ThreeThreadIncDecState(),
                threads=[
                    lambda s: s.thread_inc1(),
                    lambda s: s.thread_inc2(),
                    lambda s: s.thread_dec(),
                ],
                invariant=_three_thread_inc_dec_invariant,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Test 12: AssertionPool _do_return_conn + _do_get race
# ===========================================================================
# One thread returns a connection (sets _checked_out = False) while another
# tries to get one (reads _checked_out).  The check-then-act on _checked_out
# is not atomic: thread1 could read _checked_out=True (blocking get), but
# then thread2 sets it to False, and now thread1 should retry but won't.
# Or: thread2 starts returning (checks _checked_out == True, about to set
# False), thread1 reads _checked_out == True and raises.  Then thread2 sets
# _checked_out = False.  Thread1 got an error unnecessarily.
# More interesting: thread2 is mid-return (has checked _checked_out==True but
# hasn't set False yet), thread1 reads _checked_out==True and raises.

class AssertionPoolReturnGetState:
    """Thread 1 tries _do_get, thread 2 does _do_return_conn."""

    def __init__(self):
        self.pool = AssertionPool(
            lambda: object(),
            store_traceback=False,
        )
        # Check out a connection first so we can return it
        self.conn = self.pool._do_get()
        # Now _checked_out is True
        self.get_succeeded = False
        self.get_raised = False

    def thread1(self):
        # Try to get a second connection -- should block or fail until return
        try:
            self.pool._do_get()
            self.get_succeeded = True
        except AssertionError:
            self.get_raised = True

    def thread2(self):
        # Return the connection
        self.pool._do_return_conn(self.conn)


def _assertion_return_get_invariant(s: AssertionPoolReturnGetState) -> bool:
    # After both threads complete:
    # If return happens before get: get should succeed, _checked_out = True
    # If get happens before return: get should raise, _checked_out eventually False
    # The state should be consistent: if get succeeded, _checked_out must be True
    # If get raised, _checked_out must be False (return completed)
    if s.get_succeeded:
        return s.pool._checked_out is True
    elif s.get_raised:
        return s.pool._checked_out is False
    else:
        # Neither happened -- shouldn't occur
        return False


def test_assertion_pool_return_get_race():
    """AssertionPool: concurrent _do_return_conn and _do_get race."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: AssertionPoolReturnGetState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_assertion_return_get_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Main: run all tests
# ===========================================================================

if __name__ == "__main__":
    import time

    tests = [
        ("1. _dec_overflow lost update (unlimited)", test_dec_overflow_lost_update),
        ("2. _dec_overflow lost update sweep", test_dec_overflow_sweep),
        ("3. _inc+_dec cross-race (unlimited)", test_inc_dec_cross_race),
        ("4. _inc+_dec cross-race sweep", test_inc_dec_cross_race_sweep),
        ("5. Triple _inc_overflow (3 threads)", test_triple_inc_overflow),
        ("6. Triple _inc_overflow reproduce", test_triple_inc_overflow_reproduce),
        ("7. checkedout() torn read", test_checkedout_torn_read),
        ("8. dispose() + _inc_overflow race", test_dispose_inc_overflow_race),
        ("9. dispose + inc race sweep", test_dispose_inc_overflow_race_sweep),
        ("10. Double _dec_overflow (queue-full path)", test_double_return_dec_overflow),
        ("11. SingletonThreadPool._cleanup race", test_singleton_cleanup_race),
        ("12. AssertionPool double checkout", test_assertion_pool_double_checkout),
        ("13. AssertionPool double checkout sweep", test_assertion_pool_double_checkout_sweep),
        ("14. Bounded _inc_overflow (control)", test_bounded_inc_overflow),
        ("15. Double dispose race", test_double_dispose_race),
        ("16. 3-thread inc+inc+dec", test_three_thread_inc_dec),
        ("17. 3-thread inc+inc+dec sweep", test_three_thread_inc_dec_sweep),
        ("18. AssertionPool return+get race", test_assertion_pool_return_get_race),
    ]

    for name, test_fn in tests:
        print(f"\n{'=' * 70}")
        print(f"=== {name} ===")
        print(f"{'=' * 70}")
        start = time.time()
        try:
            test_fn()
        except Exception as e:
            print(f"  TEST ERROR: {e}")
        elapsed = time.time() - start
        print(f"  [{elapsed:.1f}s]")
