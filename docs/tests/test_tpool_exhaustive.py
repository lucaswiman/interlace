"""
Exhaustive concurrency bug exploration for TPool (WildPool).

Tests every concurrency bug we can find in the WildPool class beyond the
single _should_keep_going() TOCTOU already covered in test_tpool_real.py.

Each test function targets a specific race condition, builds a minimal
state wrapper, and uses interlace's explore_interleavings to search for
a violating interleaving.

Bugs tested:
  1. _should_keep_going TOCTOU with concurrent stop() -- worker sees
     stale keep_going after stop() sets it to False.
  2. stop() vs join() race -- both pass their guards and proceed
     concurrently, double-sending sentinels and double-joining.
  3. Double stop() race -- two stop() calls both pass the _join_is_called
     check before either sets keep_going=False, leading to two sentinels
     and two worker.join() calls.
  4. join() reads self.worker without a lock -- concurrent start_worker()
     can replace self.worker between the None check and the .is_alive()/.join()
     call, causing join() to act on the wrong thread object.
  5. add_thread() after join() -- a task enqueued after the None sentinel
     means the worker exits without processing the task (task lost).
  6. Concurrent start_worker() -- two callers might both see the worker
     as dead and both try to spawn, with the second overwriting self.worker
     before the first's thread object is captured.
  7. _run_in_capsule pool tracking -- thread finishes before being added
     to self.pool, creating a zombie entry that _kick_dead_threads may
     or may not clean up depending on timing.
  8. Semaphore balance after None sentinel -- _jump_into_the_pool(None)
     releases the semaphore once, but if multiple sentinels are queued
     (from concurrent stop/join), the semaphore can exceed pool_size.
  9. stop() worker_alive TOCTOU -- stop() reads worker.is_alive() under
     worker_lock, releases the lock, then decides whether to join based
     on the stale value. The worker can terminate in between.
 10. _should_keep_going vs concurrent add_thread -- worker reads empty()
     as True under join_lock, but add_thread puts an item between empty()
     and the function returning False.
"""

import os
import sys
import threading

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "TPool", "src"))

from external_tests_helpers import print_exploration_result, print_seed_sweep_results
from TPool import WildPool

from interlace.bytecode import explore_interleavings, run_with_schedule


# ---------------------------------------------------------------------------
# Bug 1: _should_keep_going TOCTOU with concurrent stop()
#
# _should_keep_going() reads keep_going under worker_lock, then checks
# _join_is_called and bench.empty() under join_lock.  Between those two
# lock acquisitions, stop() can set keep_going=False and put a None
# sentinel on the bench.  The worker reads keep_going=True (stale), then
# sees _join_is_called=False and bench not empty, so it continues --
# consuming the sentinel and looping, potentially processing it as a
# real task or causing _jump_into_the_pool(None) to fire and release
# the semaphore unexpectedly.
# ---------------------------------------------------------------------------

class StopVsShouldKeepGoingState:
    """Worker calls _should_keep_going() while main calls stop()."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.pool.worker = threading.Thread(target=lambda: None)
        # Pretend the worker is alive by giving it a fake ident
        self.pool.worker._started = threading.Event()
        self.pool.worker._started.set()
        self.worker_saw_keep_going = None
        self.keep_going_after_stop = None

    def worker_thread(self):
        """Worker checks _should_keep_going."""
        result = self.pool._should_keep_going()
        self.worker_saw_keep_going = result

    def main_calls_stop(self):
        """Main thread calls stop()."""
        with self.pool.worker_lock:
            self.pool.keep_going = False
        self.pool.bench.put(None)
        self.keep_going_after_stop = self.pool.keep_going


def test_stop_vs_should_keep_going():
    """Find the race where worker sees keep_going=True after stop() sets it False."""

    result = explore_interleavings(
        setup=lambda: StopVsShouldKeepGoingState(),
        threads=[
            lambda s: s.worker_thread(),
            lambda s: s.main_calls_stop(),
        ],
        invariant=lambda s: (
            # If stop set keep_going=False, the worker should NOT see True
            s.keep_going_after_stop is None  # stop hasn't run yet
            or not s.worker_saw_keep_going  # worker correctly saw False
            or s.worker_saw_keep_going is None  # worker hasn't run yet
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 2: stop() vs join() race
#
# stop() checks _join_is_called under join_lock (line 233-235), and if
# False, proceeds.  join() checks _join_is_called under join_lock too
# (line 213-216).  If both are called concurrently, one passes and the
# other should bail.  But stop() doesn't SET _join_is_called -- it only
# reads it.  So the sequence can be:
#   1) stop() acquires join_lock, sees _join_is_called=False, releases
#   2) join() acquires join_lock, sees _join_is_called=False, sets it True
#   3) Both proceed: stop() puts None + does worker.join(),
#      join() puts None + does worker.join()
# This leads to two sentinels and two join attempts on the worker.
# ---------------------------------------------------------------------------

class StopVsJoinState:
    """Concurrent stop() and join() both try to shut down the pool."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.sentinels_added = 0
        self.stop_proceeded = False
        self.join_proceeded = False

    def call_stop(self):
        """Simulate stop() checking the guard and proceeding."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
        # stop() does NOT set _join_is_called, so it proceeds past the guard
        self.stop_proceeded = True
        with self.pool.worker_lock:
            self.pool.keep_going = False
        self.pool.bench.put(None)
        self.sentinels_added += 1

    def call_join(self):
        """Simulate join() checking the guard and proceeding."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
            self.pool._join_is_called = True
        self.join_proceeded = True
        self.pool.bench.put(None)
        self.sentinels_added += 1


def test_stop_vs_join_race():
    """Find the race where both stop() and join() proceed concurrently."""

    result = explore_interleavings(
        setup=lambda: StopVsJoinState(),
        threads=[
            lambda s: s.call_stop(),
            lambda s: s.call_join(),
        ],
        invariant=lambda s: (
            # At most one sentinel should be added (one shutdown path)
            s.sentinels_added <= 1
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 3: Double stop() race
#
# Two threads call stop() concurrently.  stop() reads _join_is_called
# under join_lock but does NOT set any flag to prevent re-entry.
# Both callers can pass the guard (since _join_is_called is False for
# both), leading to:
#   - keep_going set to False twice (harmless)
#   - Two None sentinels put on the bench (breaks semaphore accounting)
#   - Two worker.join() calls (may hang or error)
# ---------------------------------------------------------------------------

class DoubleStopState:
    """Two threads call stop() concurrently."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.sentinels_added = 0
        self.stops_that_proceeded = 0

    def call_stop_1(self):
        """First stop() call."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
        # Passed the guard
        self.stops_that_proceeded += 1
        with self.pool.worker_lock:
            self.pool.keep_going = False
        self.pool.bench.put(None)
        self.sentinels_added += 1

    def call_stop_2(self):
        """Second stop() call."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
        # Passed the guard
        self.stops_that_proceeded += 1
        with self.pool.worker_lock:
            self.pool.keep_going = False
        self.pool.bench.put(None)
        self.sentinels_added += 1


def test_double_stop_race():
    """Find the race where two concurrent stop() calls both proceed."""

    result = explore_interleavings(
        setup=lambda: DoubleStopState(),
        threads=[
            lambda s: s.call_stop_1(),
            lambda s: s.call_stop_2(),
        ],
        invariant=lambda s: (
            # Only one stop() should successfully proceed
            s.stops_that_proceeded <= 1
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 4: join() reads self.worker without a lock
#
# join() at lines 218-220 reads self.worker and calls self.worker.is_alive()
# and self.worker.join() WITHOUT holding worker_lock.  Meanwhile,
# start_worker() can replace self.worker under worker_lock.  This means
# join() might:
#   a) Read self.worker as None (hasn't started) while start_worker sets it
#   b) Read the old worker object while start_worker replaces it with a new one
#   c) Call .join() on a stale worker that's already finished
# ---------------------------------------------------------------------------

class JoinWorkerRaceState:
    """join() and start_worker() race on self.worker."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.worker_seen_by_join = "not_checked"
        self.worker_set_by_start = "not_started"
        self.join_skipped_worker = False

    def call_join_check(self):
        """Simulate join()'s worker check without locks."""
        # This mirrors what join() does at lines 218-220
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
            self.pool._join_is_called = True
        self.pool.bench.put(None)
        # The unlocked read of self.worker -- this is the bug
        worker_ref = self.pool.worker
        if worker_ref:
            self.worker_seen_by_join = id(worker_ref)
        else:
            self.worker_seen_by_join = None
            self.join_skipped_worker = True

    def call_start_worker(self):
        """start_worker() replaces self.worker under lock."""
        with self.pool.worker_lock:
            old_worker = self.pool.worker
            # Simulate spawning a new worker
            new_worker = threading.Thread(target=lambda: None)
            self.pool.worker = new_worker
            self.worker_set_by_start = id(new_worker)


def test_join_reads_worker_without_lock():
    """Find the race where join() reads self.worker while start_worker() replaces it."""

    result = explore_interleavings(
        setup=lambda: JoinWorkerRaceState(),
        threads=[
            lambda s: s.call_join_check(),
            lambda s: s.call_start_worker(),
        ],
        invariant=lambda s: (
            # If join checked the worker and start_worker ran,
            # join should have seen the current worker (or None before start)
            # Bug: join can see None (worker hasn't been set yet) even though
            # start_worker is about to set it, causing join to skip the worker.join()
            s.worker_seen_by_join == "not_checked"  # join hasn't run
            or s.worker_set_by_start == "not_started"  # start hasn't run
            or not s.join_skipped_worker  # join didn't skip
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 5: add_thread() after join() sentinel -- task lost
#
# join() sets _join_is_called=True, then puts None on the bench.
# If add_thread() is called between join setting the flag and the
# worker consuming the None, the task ends up in the queue AFTER
# the None sentinel.  The worker processes the None, loops back to
# _should_keep_going() which returns False (join_is_called=True and
# queue may appear empty at that instant), and exits.  The real task
# is left in the queue forever.
# ---------------------------------------------------------------------------

class AddAfterJoinState:
    """Task added concurrently with join() is lost."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.task_executed = False
        self.task_added = False
        self.join_sentinel_added = False

    def call_join_sequence(self):
        """Simulate join() setting flag and adding sentinel."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
            self.pool._join_is_called = True
        # Sentinel goes on the queue
        self.pool.bench.put(None)
        self.join_sentinel_added = True

    def call_add_thread(self):
        """Add a real task to the pool."""
        marker = threading.Thread(target=lambda: None)
        self.pool.add_thread(marker)
        self.task_added = True

    def simulate_worker_drain(self):
        """Worker drains the queue to see what it gets."""
        items = []
        while not self.pool.bench.empty():
            items.append(self.pool.bench.get())
        # Check: did any real tasks end up AFTER the None sentinel?
        none_idx = None
        for i, item in enumerate(items):
            if item is None:
                none_idx = i
                break
        if none_idx is not None:
            # Any items after None would be lost
            tasks_after_sentinel = [x for x in items[none_idx + 1:] if x is not None]
            if tasks_after_sentinel:
                self.task_executed = False
            else:
                self.task_executed = True
        else:
            self.task_executed = True


def test_add_thread_after_join_sentinel():
    """Find the race where a task is added after join()'s None sentinel."""

    result = explore_interleavings(
        setup=lambda: AddAfterJoinState(),
        threads=[
            lambda s: s.call_join_sequence(),
            lambda s: s.call_add_thread(),
        ],
        invariant=lambda s: (
            # After both threads run, drain the queue and check ordering.
            # We need to drain inline in the invariant since both threads finished.
            _check_add_after_join(s)
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


def _check_add_after_join(s):
    """Check whether a task ended up after the None sentinel."""
    if not s.task_added or not s.join_sentinel_added:
        return True  # Not both threads completed fully

    items = []
    while not s.pool.bench.empty():
        items.append(s.pool.bench.get())

    none_idx = None
    for i, item in enumerate(items):
        if item is None:
            none_idx = i
            break

    if none_idx is None:
        return True  # No sentinel found yet

    # Any real tasks after the sentinel would be lost by the worker
    tasks_after_sentinel = [x for x in items[none_idx + 1:] if x is not None]
    return len(tasks_after_sentinel) == 0


# ---------------------------------------------------------------------------
# Bug 6: Concurrent start_worker() race
#
# Two threads call start_worker() simultaneously.  start_worker() holds
# worker_lock, checks self.worker.is_alive(), and if False, creates a
# new thread and starts it.  This is serialized by worker_lock, so both
# can't be inside the critical section at once.  But there's a subtle
# TOCTOU: the first caller spawns a worker, releases the lock, and the
# worker starts running.  Before the worker gets a chance to do anything,
# the second caller acquires the lock, sees worker.is_alive() might be
# True (if it already started) or False (if it hasn't started yet or
# finished quickly).  If False, it spawns ANOTHER worker, overwriting
# self.worker.  Now two workers run concurrently on the same pool.
# ---------------------------------------------------------------------------

class ConcurrentStartWorkerState:
    """Two threads call start_worker() at nearly the same time."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.workers_spawned = 0
        self.worker_ids = []

    def start_worker_1(self):
        """First start_worker call."""
        with self.pool.worker_lock:
            if self.pool.worker and self.pool.worker.is_alive():
                return
            # Spawn worker
            self.workers_spawned += 1
            fake_worker = threading.Thread(target=lambda: None)
            self.pool.worker = fake_worker
            self.worker_ids.append(id(fake_worker))

    def start_worker_2(self):
        """Second start_worker call."""
        with self.pool.worker_lock:
            if self.pool.worker and self.pool.worker.is_alive():
                return
            # Spawn worker
            self.workers_spawned += 1
            fake_worker = threading.Thread(target=lambda: None)
            self.pool.worker = fake_worker
            self.worker_ids.append(id(fake_worker))


def test_concurrent_start_worker():
    """Find the race where two start_worker() calls both spawn workers."""

    result = explore_interleavings(
        setup=lambda: ConcurrentStartWorkerState(),
        threads=[
            lambda s: s.start_worker_1(),
            lambda s: s.start_worker_2(),
        ],
        invariant=lambda s: (
            # At most one worker should be spawned
            s.workers_spawned <= 1
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 7: _run_in_capsule pool tracking race
#
# _run_in_capsule does:
#   1. thread.start()
#   2. tid = thread.ident
#   3. with queue_lock: self.pool[tid] = thread
#   4. thread.join()
#   5. semaphore.release()
#
# The problem: between step 1 and step 3, the thread may complete.
# Then _kick_dead_threads runs concurrently and doesn't see this thread
# in the pool (it hasn't been added yet).  After step 3, the thread is
# already dead but sits in self.pool.  _kick_dead_threads may or may not
# clean it up depending on timing.
#
# More critically, two capsules running concurrently both call
# _kick_dead_threads in _worker_func, and they access self.pool under
# queue_lock.  But the check `not t.is_alive()` and the `del self.pool[tid]`
# happen together under the lock, so that part is safe.  The real issue
# is the growing pool dict if threads finish before being registered.
# ---------------------------------------------------------------------------

class PoolTrackingState:
    """Concurrent capsule tracking can leave zombie entries."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.completed_threads = 0
        self.pool_size_after = 0

    def capsule_1(self):
        """First capsule: add thread, thread completes, then register in pool."""
        tid_1 = 1001
        thread_1_alive = True
        # Simulate: thread runs and completes
        with self.pool.queue_lock:
            self.pool.pool[tid_1] = _FakeThread(alive=False)
        self.completed_threads += 1

    def capsule_2(self):
        """Second capsule: same pattern."""
        tid_2 = 1002
        with self.pool.queue_lock:
            self.pool.pool[tid_2] = _FakeThread(alive=False)
        self.completed_threads += 1

    def kick_dead_threads(self):
        """Simulate _kick_dead_threads running."""
        with self.pool.queue_lock:
            tids = list(self.pool.pool.keys())
            for tid in tids:
                t = self.pool.pool[tid]
                if not t.is_alive():
                    del self.pool.pool[tid]
        self.pool_size_after = len(self.pool.pool)


class _FakeThread:
    """Minimal fake thread for pool tracking tests."""
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def test_pool_tracking_zombie_entries():
    """Find the race where completed threads remain in pool dict."""

    result = explore_interleavings(
        setup=lambda: PoolTrackingState(),
        threads=[
            lambda s: s.capsule_1(),
            lambda s: s.capsule_2(),
            lambda s: s.kick_dead_threads(),
        ],
        invariant=lambda s: (
            # After kick_dead_threads runs with both capsules complete,
            # pool should be empty
            s.completed_threads < 2  # not both done yet
            or s.pool_size_after == 0
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 8: Semaphore balance after multiple None sentinels
#
# When both stop() and join() race (or double stop), multiple None
# sentinels end up on the queue.  For each None, _jump_into_the_pool
# calls semaphore.release().  But the worker already did
# semaphore.acquire() before getting the None.  So each None produces
# a matched acquire/release.  However, join() at the end does:
#   for _ in range(pool_size): semaphore.acquire()
#   for _ in range(pool_size): semaphore.release()
#
# If the semaphore was released extra times due to multiple sentinels,
# join()'s acquire loop completes too early, allowing join() to return
# before all real tasks finish.
# ---------------------------------------------------------------------------

class SemaphoreBalanceState:
    """Multiple sentinels corrupt semaphore accounting."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        # Track semaphore operations
        self.sem_releases = 0
        self.sem_acquires = 0
        self.sentinels_processed = 0

    def process_sentinel(self):
        """Simulate worker processing a None sentinel."""
        # Worker acquired semaphore already
        self.sem_acquires += 1
        th = self.pool.bench.get()
        if th is None:
            # _jump_into_the_pool(None) releases semaphore
            self.sem_releases += 1
            self.sentinels_processed += 1

    def stop_adds_sentinel(self):
        """stop() adds a None sentinel."""
        self.pool.bench.put(None)

    def join_adds_sentinel(self):
        """join() adds a None sentinel."""
        self.pool.bench.put(None)


def test_semaphore_balance_double_sentinel():
    """Find the race where multiple sentinels corrupt semaphore count."""

    result = explore_interleavings(
        setup=lambda: SemaphoreBalanceState(),
        threads=[
            lambda s: s.stop_adds_sentinel(),
            lambda s: s.join_adds_sentinel(),
            lambda s: s.process_sentinel(),
            lambda s: s.process_sentinel(),
        ],
        invariant=lambda s: (
            # Net semaphore balance should never go positive (more releases than acquires)
            # because that would allow more threads than pool_size to run
            s.sem_releases <= s.sem_acquires
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 9: stop() worker_alive TOCTOU
#
# stop() does:
#   with worker_lock:
#       if self.worker and self.worker.is_alive():
#           self.keep_going = False
#           worker_alive = True
#       else:
#           worker_alive = False
#   self.bench.put(None)
#   if worker_alive:
#       self.worker.join()   <-- uses stale worker_alive
#
# Between releasing worker_lock and calling self.worker.join(), the
# worker can finish and self.worker can be replaced by start_worker().
# stop() then joins the WRONG thread or joins a thread that's already
# dead (which returns immediately, potentially before all tasks finish).
# ---------------------------------------------------------------------------

class StopWorkerAliveTOCTOUState:
    """stop() uses stale worker_alive to decide whether to join."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.stop_saw_alive = None
        self.worker_actually_alive_when_joined = None
        self.worker_replaced = False

    def call_stop_logic(self):
        """stop()'s logic: check under lock, then join without lock."""
        with self.pool.worker_lock:
            if self.pool.worker and self.pool.worker.is_alive():
                self.pool.keep_going = False
                worker_alive = True
            else:
                worker_alive = False
        self.stop_saw_alive = worker_alive
        self.pool.bench.put(None)
        # Gap here: worker_lock released, worker might change
        if worker_alive:
            # In real code: self.worker.join()
            # Check if worker is still what we expect
            self.worker_actually_alive_when_joined = (
                self.pool.worker is not None and self.pool.worker.is_alive()
            )

    def replace_worker(self):
        """Simulate start_worker() replacing the worker object."""
        with self.pool.worker_lock:
            new_worker = _FakeThread(alive=True)
            self.pool.worker = new_worker
            self.worker_replaced = True


def test_stop_worker_alive_toctou():
    """Find the TOCTOU where stop() joins based on stale liveness check."""

    # Set up pool with a "live" worker
    def setup():
        state = StopWorkerAliveTOCTOUState()
        state.pool.worker = _FakeThread(alive=True)
        return state

    result = explore_interleavings(
        setup=setup,
        threads=[
            lambda s: s.call_stop_logic(),
            lambda s: s.replace_worker(),
        ],
        invariant=lambda s: (
            # If stop saw worker as alive but by the time it would join,
            # the worker was replaced, stop joins the wrong object
            s.stop_saw_alive is None  # stop hasn't checked yet
            or not s.stop_saw_alive  # stop saw dead, OK
            or not s.worker_replaced  # worker wasn't replaced, OK
            or s.worker_actually_alive_when_joined is not False  # joined something alive
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 10: _should_keep_going vs concurrent add_thread
#
# This is a variant of the original TOCTOU but with add_thread:
# _should_keep_going() reads bench.empty() as True under join_lock.
# Between releasing join_lock and returning False, add_thread() puts
# a task on the bench.  The worker returns False (stop), even though
# there's now a task waiting.  This task is lost forever.
#
# Note: The original test in test_tpool_real.py tests this with
# join() setting the flag.  This variant tests it with add_thread
# adding work DURING the _should_keep_going check window.
# ---------------------------------------------------------------------------

class ShouldKeepGoingVsAddThreadState:
    """Worker's _should_keep_going races with add_thread adding work."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.pool._join_is_called = True  # Join already called
        self.worker_decided_to_stop = False
        self.task_was_added = False
        self.queue_size_after = 0

    def worker_checks(self):
        """Worker calls _should_keep_going."""
        result = self.pool._should_keep_going()
        if not result:
            self.worker_decided_to_stop = True
            self.queue_size_after = self.pool.bench.qsize()

    def add_task(self):
        """Another thread adds a task."""
        dummy = threading.Thread(target=lambda: None)
        self.pool.add_thread(dummy)
        self.task_was_added = True


def test_should_keep_going_vs_add_thread():
    """Find the race where worker exits despite task being added."""

    result = explore_interleavings(
        setup=lambda: ShouldKeepGoingVsAddThreadState(),
        threads=[
            lambda s: s.worker_checks(),
            lambda s: s.add_task(),
        ],
        invariant=lambda s: (
            # If worker stopped and a task was added, queue should be empty
            # (meaning worker should have consumed it)
            not s.worker_decided_to_stop
            or not s.task_was_added
            or s.queue_size_after == 0
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 11: Non-atomic sentinels_added increment in stop/join race
#
# When stop() and join() race, both do bench.put(None). We model the
# sentinels_added counter without synchronization (as the real code
# has no coordination between these paths). The += is itself a
# read-modify-write that can lose updates, but more importantly,
# the real problem is that TWO sentinels end up in the queue when
# at most one should.
# ---------------------------------------------------------------------------

class StopJoinSentinelCountState:
    """Count sentinels to detect stop/join double-shutdown."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.total_sentinels = 0

    def stop_path(self):
        """stop() path: check guard, then send sentinel."""
        # stop()'s guard: checks _join_is_called under join_lock
        with self.pool.join_lock:
            already_joining = self.pool._join_is_called
        if already_joining:
            return
        # No flag set by stop -- this is the bug
        with self.pool.worker_lock:
            self.pool.keep_going = False
        self.pool.bench.put(None)
        self.total_sentinels += 1

    def join_path(self):
        """join() path: check guard, set flag, send sentinel."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
            self.pool._join_is_called = True
        self.pool.bench.put(None)
        self.total_sentinels += 1


def test_stop_join_double_sentinel():
    """Both stop and join send a None sentinel when racing."""

    result = explore_interleavings(
        setup=lambda: StopJoinSentinelCountState(),
        threads=[
            lambda s: s.stop_path(),
            lambda s: s.join_path(),
        ],
        invariant=lambda s: (
            # Exactly one shutdown path should add a sentinel
            s.total_sentinels <= 1
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 12: Double join() race -- _join_is_called as guard
#
# Two threads call join() concurrently. The _join_is_called flag is
# checked-and-set under join_lock, so only one proceeds. This should be
# safe.  But join() then reads self.worker WITHOUT the lock (line 218).
# If start_worker() is running concurrently and replaces self.worker,
# the winning join() caller might miss the new worker entirely.
# Also the += on sentinels is non-atomic in our model.
# ---------------------------------------------------------------------------

class DoubleJoinState:
    """Two threads call join() concurrently."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.join_proceeded_count = 0

    def call_join_1(self):
        """First join call."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
            self.pool._join_is_called = True
        self.join_proceeded_count += 1
        self.pool.bench.put(None)

    def call_join_2(self):
        """Second join call."""
        with self.pool.join_lock:
            if self.pool._join_is_called:
                return
            self.pool._join_is_called = True
        self.join_proceeded_count += 1
        self.pool.bench.put(None)


def test_double_join():
    """Verify that double join is correctly guarded (should pass)."""

    result = explore_interleavings(
        setup=lambda: DoubleJoinState(),
        threads=[
            lambda s: s.call_join_1(),
            lambda s: s.call_join_2(),
        ],
        invariant=lambda s: (
            # Only one join should proceed
            s.join_proceeded_count <= 1
        ),
        max_attempts=300,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 13: _kick_dead_threads vs _run_in_capsule pool mutation race
#
# _kick_dead_threads acquires queue_lock and iterates over pool,
# deleting dead threads.  _run_in_capsule acquires queue_lock to
# ADD a thread.  These are serialized by queue_lock, which is correct.
# But the combination of multiple capsules adding and kicking at the
# same time can lead to a pool dict that reflects an inconsistent
# state: a thread added but immediately dead, never kicked, growing
# the dict without bound.
# ---------------------------------------------------------------------------

class KickVsCapsuleState:
    """_kick_dead_threads races with multiple capsules adding threads."""

    def __init__(self):
        self.pool = WildPool(pool_size=3)
        self.threads_added = 0
        self.threads_kicked = 0

    def capsule_adds_dead_thread(self):
        """Capsule adds a thread that immediately dies."""
        tid = 2000 + self.threads_added
        with self.pool.queue_lock:
            self.pool.pool[tid] = _FakeThread(alive=False)
        self.threads_added += 1

    def kick_dead(self):
        """_kick_dead_threads runs."""
        with self.pool.queue_lock:
            tids = list(self.pool.pool.keys())
            for tid in tids:
                t = self.pool.pool[tid]
                if not t.is_alive():
                    del self.pool.pool[tid]
                    self.threads_kicked += 1


def test_kick_vs_capsule_tracking():
    """Pool dict can grow if capsules add dead threads between kicks."""

    result = explore_interleavings(
        setup=lambda: KickVsCapsuleState(),
        threads=[
            lambda s: s.capsule_adds_dead_thread(),
            lambda s: s.capsule_adds_dead_thread(),
            lambda s: s.kick_dead(),
        ],
        invariant=lambda s: (
            # After everything runs, pool should be clean
            s.threads_added < 2  # not all capsules done
            or s.threads_kicked >= s.threads_added  # all dead removed
            or len(s.pool.pool) == 0  # pool is clean
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 14: Non-atomic counter increments in stop/join sentinel tracking
#
# This directly tests that the `total_sentinels += 1` operation in
# our stop_path/join_path model is non-atomic.  Since += is a
# read-modify-write, two concurrent increments can produce 1 instead
# of 2 -- masking the double-sentinel bug.  We deliberately model this
# to show the lost-update pattern.
# ---------------------------------------------------------------------------

class NonAtomicSentinelCounterState:
    """Models the non-atomic += on a shared counter."""

    def __init__(self):
        self.counter = 0

    def increment_1(self):
        """First incrementer (models stop's sentinel count)."""
        self.counter += 1

    def increment_2(self):
        """Second incrementer (models join's sentinel count)."""
        self.counter += 1


def test_non_atomic_sentinel_counter():
    """Show that += is non-atomic and can lose updates."""

    result = explore_interleavings(
        setup=lambda: NonAtomicSentinelCounterState(),
        threads=[
            lambda s: s.increment_1(),
            lambda s: s.increment_2(),
        ],
        invariant=lambda s: s.counter == 2,
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Bug 15: Full _should_keep_going TOCTOU with three participants
#
# Extends the original bug with THREE concurrent threads:
#   - Worker 1 calls _should_keep_going()
#   - Worker 2 calls _should_keep_going()
#   - Main thread sets join and adds a task
#
# With two workers, the probability of BOTH seeing an empty queue
# right as a task is added increases.  Both may decide to stop,
# leaving the task unprocessed.
# ---------------------------------------------------------------------------

class ThreeWayShouldKeepGoingState:
    """Two workers + main thread race on _should_keep_going."""

    def __init__(self):
        self.pool = WildPool(pool_size=2)
        self.pool.keep_going = True
        self.worker1_stopped = False
        self.worker2_stopped = False
        self.items_after_1 = 0
        self.items_after_2 = 0

    def worker1(self):
        result = self.pool._should_keep_going()
        if not result:
            self.worker1_stopped = True
            self.items_after_1 = self.pool.bench.qsize()

    def worker2(self):
        result = self.pool._should_keep_going()
        if not result:
            self.worker2_stopped = True
            self.items_after_2 = self.pool.bench.qsize()

    def main_enqueue_and_join(self):
        with self.pool.join_lock:
            self.pool._join_is_called = True
        dummy = threading.Thread(target=lambda: None)
        self.pool.add_thread(dummy)


def test_three_way_should_keep_going():
    """Two workers both exit while a task sits in the queue."""

    result = explore_interleavings(
        setup=lambda: ThreeWayShouldKeepGoingState(),
        threads=[
            lambda s: s.worker1(),
            lambda s: s.worker2(),
            lambda s: s.main_enqueue_and_join(),
        ],
        invariant=lambda s: (
            # If any worker stopped, queue should be empty at that point
            (not s.worker1_stopped or s.items_after_1 == 0)
            and (not s.worker2_stopped or s.items_after_2 == 0)
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Sweep and Reproduce tests for the most interesting bugs
# ---------------------------------------------------------------------------

def test_stop_vs_join_sweep_seeds():
    """Sweep seeds for the stop vs join double-sentinel bug."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: StopJoinSentinelCountState(),
            threads=[
                lambda s: s.stop_path(),
                lambda s: s.join_path(),
            ],
            invariant=lambda s: s.total_sentinels <= 1,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_double_stop_sweep_seeds():
    """Sweep seeds for the double stop race."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: DoubleStopState(),
            threads=[
                lambda s: s.call_stop_1(),
                lambda s: s.call_stop_2(),
            ],
            invariant=lambda s: s.stops_that_proceeded <= 1,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_three_way_should_keep_going_sweep():
    """Sweep seeds for the three-way _should_keep_going TOCTOU."""

    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: ThreeWayShouldKeepGoingState(),
            threads=[
                lambda s: s.worker1(),
                lambda s: s.worker2(),
                lambda s: s.main_enqueue_and_join(),
            ],
            invariant=lambda s: (
                (not s.worker1_stopped or s.items_after_1 == 0)
                and (not s.worker2_stopped or s.items_after_2 == 0)
            ),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_stop_vs_join_reproduce():
    """Find and reproduce the stop vs join double-sentinel bug."""

    result = explore_interleavings(
        setup=lambda: StopJoinSentinelCountState(),
        threads=[
            lambda s: s.stop_path(),
            lambda s: s.join_path(),
        ],
        invariant=lambda s: s.total_sentinels <= 1,
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    if not result.counterexample:
        print("No counterexample found -- skipping reproduction")
        return

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: StopJoinSentinelCountState(),
            threads=[
                lambda s: s.stop_path(),
                lambda s: s.join_path(),
            ],
        )
        is_bug = state.total_sentinels > 1
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        print(
            f"  Run {i + 1}: sentinels={state.total_sentinels} [{status}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


def test_double_stop_reproduce():
    """Find and reproduce the double stop race."""

    result = explore_interleavings(
        setup=lambda: DoubleStopState(),
        threads=[
            lambda s: s.call_stop_1(),
            lambda s: s.call_stop_2(),
        ],
        invariant=lambda s: s.stops_that_proceeded <= 1,
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    if not result.counterexample:
        print("No counterexample found -- skipping reproduction")
        return

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: DoubleStopState(),
            threads=[
                lambda s: s.call_stop_1(),
                lambda s: s.call_stop_2(),
            ],
        )
        is_bug = state.stops_that_proceeded > 1
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        print(
            f"  Run {i + 1}: stops_proceeded={state.stops_that_proceeded} [{status}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


def test_three_way_toctou_reproduce():
    """Find and reproduce the three-way _should_keep_going TOCTOU."""

    result = explore_interleavings(
        setup=lambda: ThreeWayShouldKeepGoingState(),
        threads=[
            lambda s: s.worker1(),
            lambda s: s.worker2(),
            lambda s: s.main_enqueue_and_join(),
        ],
        invariant=lambda s: (
            (not s.worker1_stopped or s.items_after_1 == 0)
            and (not s.worker2_stopped or s.items_after_2 == 0)
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )

    if not result.counterexample:
        print("No counterexample found -- skipping reproduction")
        return

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: ThreeWayShouldKeepGoingState(),
            threads=[
                lambda s: s.worker1(),
                lambda s: s.worker2(),
                lambda s: s.main_enqueue_and_join(),
            ],
        )
        w1_bug = state.worker1_stopped and state.items_after_1 > 0
        w2_bug = state.worker2_stopped and state.items_after_2 > 0
        is_bug = w1_bug or w2_bug
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        print(
            f"  Run {i + 1}: w1_stopped={state.worker1_stopped} q1={state.items_after_1} "
            f"w2_stopped={state.worker2_stopped} q2={state.items_after_2} [{status}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ---------------------------------------------------------------------------
# Main: run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("EXHAUSTIVE TPOOL CONCURRENCY BUG EXPLORATION")
    print("=" * 70)

    print("\n--- Bug 1: stop() vs _should_keep_going() TOCTOU ---")
    test_stop_vs_should_keep_going()

    print("\n--- Bug 2: stop() vs join() race ---")
    test_stop_vs_join_race()

    print("\n--- Bug 3: Double stop() race ---")
    test_double_stop_race()

    print("\n--- Bug 4: join() reads self.worker without lock ---")
    test_join_reads_worker_without_lock()

    print("\n--- Bug 5: add_thread() after join() sentinel ---")
    test_add_thread_after_join_sentinel()

    print("\n--- Bug 6: Concurrent start_worker() ---")
    test_concurrent_start_worker()

    print("\n--- Bug 7: Pool tracking zombie entries ---")
    test_pool_tracking_zombie_entries()

    print("\n--- Bug 8: Semaphore balance with double sentinel ---")
    test_semaphore_balance_double_sentinel()

    print("\n--- Bug 9: stop() worker_alive TOCTOU ---")
    test_stop_worker_alive_toctou()

    print("\n--- Bug 10: _should_keep_going vs add_thread ---")
    test_should_keep_going_vs_add_thread()

    print("\n--- Bug 11: stop/join double sentinel count ---")
    test_stop_join_double_sentinel()

    print("\n--- Bug 12: Double join() (should be safe) ---")
    test_double_join()

    print("\n--- Bug 13: _kick_dead_threads vs capsule tracking ---")
    test_kick_vs_capsule_tracking()

    print("\n--- Bug 14: Non-atomic sentinel counter (lost update) ---")
    test_non_atomic_sentinel_counter()

    print("\n--- Bug 15: Three-way _should_keep_going TOCTOU ---")
    test_three_way_should_keep_going()

    print("\n" + "=" * 70)
    print("SWEEP TESTS")
    print("=" * 70)

    print("\n--- Sweep: stop vs join double sentinel ---")
    test_stop_vs_join_sweep_seeds()

    print("\n--- Sweep: double stop ---")
    test_double_stop_sweep_seeds()

    print("\n--- Sweep: three-way _should_keep_going ---")
    test_three_way_should_keep_going_sweep()

    print("\n" + "=" * 70)
    print("REPRODUCE TESTS")
    print("=" * 70)

    print("\n--- Reproduce: stop vs join double sentinel ---")
    test_stop_vs_join_reproduce()

    print("\n--- Reproduce: double stop ---")
    test_double_stop_reproduce()

    print("\n--- Reproduce: three-way _should_keep_going TOCTOU ---")
    test_three_way_toctou_reproduce()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
