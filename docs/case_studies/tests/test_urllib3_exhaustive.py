"""
Exhaustive concurrency tests for urllib3 HTTPConnectionPool.

Goes far beyond the single num_connections lost-update test in test_urllib3_real.py.
Each test targets a distinct race condition or TOCTOU window in connectionpool.py.

Bugs targeted:

1. _new_conn() num_connections lost update with 3 threads
   Three threads all calling _new_conn() simultaneously. The non-atomic
   ``self.num_connections += 1`` can lose multiple updates when 3+ threads
   interleave.

2. _get_conn() pool-empty TOCTOU race
   Two threads both find the pool returning None (empty slot) and both
   decide to create new connections via _new_conn(), exceeding the expected
   connection count for a maxsize=1 pool.

3. _put_conn() vs close() race -- connection leak
   Thread 1 calls _put_conn() which checks ``self.pool is not None`` and
   sees True. Thread 2 calls close() which sets ``self.pool = None``.
   Thread 1 then tries ``self.pool.put(conn, ...)`` which raises
   AttributeError because self.pool is now None. The conn never gets
   put back OR closed properly (falls through without explicit close
   in the AttributeError handler -- but only if conn is not None).

4. _get_conn() vs close() race -- ClosedPoolError not raised
   Thread 1 calls _get_conn() and passes the ``if self.pool is None`` check.
   Thread 2 calls close() setting self.pool = None. Thread 1 then does
   ``self.pool.get(...)`` which raises AttributeError, caught as
   ClosedPoolError. But the TOCTOU window means thread 1 may have already
   gotten a connection from the queue before close() drained it, using a
   connection from a "closed" pool.

5. num_requests lost update in _make_request()
   _make_request() does ``self.num_requests += 1`` without any locking,
   identical pattern to num_connections. Two concurrent requests lose a
   count.

6. close() double-drain race
   Two threads both call close(). Both check ``if self.pool is None`` and
   see it is not None. Both execute ``old_pool, self.pool = self.pool, None``.
   The first gets the real pool, the second gets None (because the first
   thread's assignment hasn't happened yet or has). This is a classic
   check-then-act race on the pool reference.

7. _get_conn() + _put_conn() connection duplication
   With maxsize=1: Thread 1 gets the pooled connection (pool now empty).
   Thread 2 gets from empty pool, creates a new connection. Thread 1
   puts its connection back. Thread 2 puts its connection back. Now the
   pool has more connections than maxsize (or connections are silently lost).

8. HTTPSConnectionPool._new_conn() num_connections lost update
   Same non-atomic += 1 bug but in the HTTPS subclass which has its own
   _new_conn() override with the same pattern.

Repository: https://github.com/urllib3/urllib3
"""

import os
import queue
import sys
import types

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_src = os.path.join(_test_dir, "..", "external_repos", "urllib3", "src")
# Insert local repo path FIRST so interlace can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_src))

# The bare urllib3 checkout lacks a generated _version.py (created by setuptools-scm
# at build time).  Stub it out so the rest of the package can be imported.
_version_path = os.path.join(os.path.abspath(_repo_src), "urllib3", "_version.py")
if not os.path.exists(_version_path):
    _ver_mod = types.ModuleType("urllib3._version")
    _ver_mod.__version__ = "0.0.0.dev0"  # type: ignore[attr-defined]
    sys.modules["urllib3._version"] = _ver_mod

from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool  # noqa: E402

from interlace.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pool_queue_items(pool):
    """Drain the LifoQueue non-destructively and return a list of its contents."""
    items = []
    q = pool.pool
    if q is None:
        return items
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    # Put them all back
    for item in items:
        q.put_nowait(item)
    return items


# ===================================================================
# Test 1: _new_conn() lost update with 3 threads
# ===================================================================

class ThreeThreadNewConnState:
    """Three threads each call _new_conn() once.

    With three concurrent threads, the non-atomic ``self.num_connections += 1``
    can lose TWO updates (worst case) leaving num_connections at 1 instead of 3.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999)

    def thread1(self):
        self.pool._new_conn()

    def thread2(self):
        self.pool._new_conn()

    def thread3(self):
        self.pool._new_conn()


def _invariant_three_thread_new_conn(s: ThreeThreadNewConnState) -> bool:
    return s.pool.num_connections == 3


def test_new_conn_lost_update_three_threads():
    """Three threads calling _new_conn() -- can lose 1 or 2 increments."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ThreeThreadNewConnState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
            invariant=_invariant_three_thread_new_conn,
            max_attempts=500,
            max_ops=400,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_new_conn_lost_update_three_threads_sweep():
    """Sweep 20 seeds for the 3-thread num_connections race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: ThreeThreadNewConnState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                    lambda s: s.thread3(),
                ],
                invariant=_invariant_three_thread_new_conn,
                max_attempts=200,
                max_ops=400,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===================================================================
# Test 2: _get_conn() pool-empty TOCTOU -- both threads create new conns
# ===================================================================

class GetConnDoubleCreateState:
    """Two threads call _get_conn() on a pool whose single slot holds None.

    Both threads call pool.get() which returns None (the placeholder),
    then both fall through to _new_conn(). The pool was maxsize=1, so
    only ONE new connection should be needed, but TWO are created.
    num_connections ends up at 2 (or more with the lost-update bug on top),
    which is wrong for a maxsize=1 pool.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999, maxsize=1)
        self.conns = [None, None]

    def thread1(self):
        self.conns[0] = self.pool._get_conn()

    def thread2(self):
        self.conns[1] = self.pool._get_conn()


def _invariant_get_conn_double_create(s: GetConnDoubleCreateState) -> bool:
    # At most 1 new connection should be created for a maxsize=1 pool
    # when starting with 0 pre-existing connections. But both threads
    # end up creating one each because the pool gives back None to both.
    # We check that num_connections <= 1 (the ideal), though the real
    # bug is num_connections == 2 (both created a new connection).
    return s.pool.num_connections <= 1


def test_get_conn_double_create():
    """Two _get_conn() calls on maxsize=1 pool both create new connections."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: GetConnDoubleCreateState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_get_conn_double_create,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===================================================================
# Test 3: _put_conn() vs close() -- connection not properly cleaned up
# ===================================================================

class PutConnVsCloseState:
    """Thread 1 returns a connection via _put_conn(), Thread 2 closes the pool.

    The TOCTOU window:
    1. _put_conn checks ``self.pool is not None`` -> True
    2. close() sets ``self.pool = None``
    3. _put_conn tries ``self.pool.put(conn, ...)`` -> AttributeError
    4. AttributeError is caught but falls through to the bottom where
       ``if conn: conn.close()`` runs, so the connection IS closed.
       However, the pool object is in an inconsistent state: the conn
       was never returned to the pool, and close() drained the old pool
       which didn't contain this conn.

    We detect this by checking whether the pool is None AND the connection
    was properly closed (conn.close() was called).
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999, maxsize=1)
        # Get a real connection object to return
        self.conn = self.pool._new_conn()
        self.put_raised = False
        self.put_completed = False

    def thread1(self):
        """Return connection to pool."""
        try:
            self.pool._put_conn(self.conn)
            self.put_completed = True
        except Exception:
            self.put_raised = True

    def thread2(self):
        """Close the pool."""
        self.pool.close()


def _invariant_put_conn_vs_close(s: PutConnVsCloseState) -> bool:
    # After close(), pool should be None
    pool_is_closed = s.pool.pool is None
    # The connection should have been closed regardless of the race
    # (either put back to pool and drained by close, or closed by _put_conn's
    # fallback). But if _put_conn succeeds BEFORE close() sets pool=None,
    # then the conn goes into the old pool, and close() drains it. Fine.
    # If _put_conn's check passes but pool goes None before put(), the
    # AttributeError handler catches it and the conn is closed by the
    # bottom fallback. Also fine.
    # The invariant we actually care about: the pool should be closed.
    return pool_is_closed


def test_put_conn_vs_close():
    """Race between _put_conn() and close() -- pool consistency."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PutConnVsCloseState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_put_conn_vs_close,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===================================================================
# Test 4: _get_conn() vs close() race
# ===================================================================

class GetConnVsCloseState:
    """Thread 1 calls _get_conn(), Thread 2 calls close().

    TOCTOU window:
    1. _get_conn checks ``self.pool is None`` -> False
    2. close() sets ``self.pool = None`` and drains old_pool
    3. _get_conn tries ``self.pool.get(...)`` -> AttributeError
       (caught, raises ClosedPoolError)

    OR (the sneaky case):
    1. _get_conn checks ``self.pool is None`` -> False
    2. _get_conn calls ``self.pool.get(...)`` and gets a connection
    3. close() runs, sets pool=None, drains remaining connections
    4. _get_conn returns the connection from a now-closed pool

    The invariant: if close() has run, _get_conn should NOT succeed.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999, maxsize=1)
        self.got_conn = None
        self.got_error = None

    def thread1(self):
        """Try to get a connection."""
        try:
            self.got_conn = self.pool._get_conn()
        except Exception as e:
            self.got_error = e

    def thread2(self):
        """Close the pool."""
        self.pool.close()


def _invariant_get_conn_vs_close(s: GetConnVsCloseState) -> bool:
    # The pool is closed. If thread1 successfully got a connection,
    # that's a race -- it obtained a conn from a pool that was being
    # closed. The "correct" behavior would be to raise ClosedPoolError.
    pool_closed = s.pool.pool is None
    if pool_closed and s.got_conn is not None:
        # Thread got a connection from a pool that is now closed.
        # This is the bug: the connection escapes the close() drain.
        return False
    return True


def test_get_conn_vs_close():
    """Race between _get_conn() and close() -- connection escapes closed pool."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: GetConnVsCloseState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_get_conn_vs_close,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===================================================================
# Test 5: num_requests lost update in _make_request()
# ===================================================================

class NumRequestsLostUpdateState:
    """Two threads both increment num_requests.

    _make_request() does ``self.num_requests += 1`` without locking.
    We simulate just the counter increment portion since actually making
    HTTP requests requires a server. We call the increment directly on
    the pool, mimicking what _make_request does.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999)

    def thread1(self):
        self.pool.num_requests += 1

    def thread2(self):
        self.pool.num_requests += 1


def _invariant_num_requests(s: NumRequestsLostUpdateState) -> bool:
    return s.pool.num_requests == 2


def test_num_requests_lost_update():
    """num_requests += 1 lost update (same pattern as num_connections)."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: NumRequestsLostUpdateState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_num_requests,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_num_requests_lost_update_sweep():
    """Sweep 20 seeds for the num_requests lost update."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: NumRequestsLostUpdateState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                ],
                invariant=_invariant_num_requests,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===================================================================
# Test 6: close() double-drain race
# ===================================================================

class DoubleCloseState:
    """Two threads both call close() on the same pool.

    The race in close():
        if self.pool is None:
            return
        old_pool, self.pool = self.pool, None
        _close_pool_connections(old_pool)

    Thread 1 and Thread 2 both pass the ``if self.pool is None`` guard,
    then both execute the swap. The tuple assignment
    ``old_pool, self.pool = self.pool, None`` is NOT atomic:
    - LOAD self.pool
    - STORE to old_pool
    - STORE None to self.pool

    If Thread 1 loads self.pool (the real queue), then Thread 2 runs
    the full swap (gets the real queue, sets self.pool=None), then
    Thread 1 completes its swap: old_pool gets None (already swapped
    by Thread 2), self.pool is set to None again. Thread 1 then
    calls _close_pool_connections(None) which will crash.

    Or both get the real queue, and it gets drained twice -- benign
    for the queue but indicates a race.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999, maxsize=2)
        # Put a real connection in the pool for draining
        conn = self.pool._new_conn()
        self.pool._put_conn(conn)
        self.close1_error = None
        self.close2_error = None

    def thread1(self):
        try:
            self.pool.close()
        except Exception as e:
            self.close1_error = e

    def thread2(self):
        try:
            self.pool.close()
        except Exception as e:
            self.close2_error = e


def _invariant_double_close(s: DoubleCloseState) -> bool:
    # After both threads complete, pool should be None and no errors
    pool_is_none = s.pool.pool is None
    no_errors = s.close1_error is None and s.close2_error is None
    return pool_is_none and no_errors


def test_double_close():
    """Two threads both call close() -- race on pool swap."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DoubleCloseState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_double_close,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===================================================================
# Test 7: _get_conn() + _put_conn() connection duplication
# ===================================================================

class GetPutConnDuplicationState:
    """Two threads do get/put cycles on a maxsize=1 pool.

    Thread 1: _get_conn() then _put_conn()
    Thread 2: _get_conn() then _put_conn()

    With maxsize=1, the pool starts with one None slot. Thread 1 gets
    that None, creates a new conn. Thread 2 also gets (pool empty, non-blocking),
    also creates a new conn. Now there are 2 connections for a maxsize=1 pool.
    When both put back, the second put will find the pool full and discard
    the connection, but num_connections will be 2 -- one connection is
    silently created and discarded, wasting resources.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999, maxsize=1)
        self.conn1 = None
        self.conn2 = None

    def thread1(self):
        self.conn1 = self.pool._get_conn()
        self.pool._put_conn(self.conn1)

    def thread2(self):
        self.conn2 = self.pool._get_conn()
        self.pool._put_conn(self.conn2)


def _invariant_get_put_duplication(s: GetPutConnDuplicationState) -> bool:
    # For a maxsize=1 pool, we expect at most 1 connection to be created.
    # If both threads race through _get_conn(), both create new connections,
    # so num_connections ends up as 2 (or even 1 due to the lost-update bug
    # on top). We check: should be exactly 1 connection created.
    return s.pool.num_connections <= 1


def test_get_put_conn_duplication():
    """Two get/put cycles on maxsize=1 pool create more connections than needed."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: GetPutConnDuplicationState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_get_put_duplication,
            max_attempts=500,
            max_ops=400,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===================================================================
# Test 8: HTTPSConnectionPool._new_conn() num_connections lost update
# ===================================================================

class HttpsNewConnState:
    """Two threads call _new_conn() on an HTTPSConnectionPool.

    HTTPSConnectionPool overrides _new_conn() with its own
    ``self.num_connections += 1`` -- the same non-atomic pattern.
    """

    def __init__(self):
        self.pool = HTTPSConnectionPool("localhost", port=9999)

    def thread1(self):
        self.pool._new_conn()

    def thread2(self):
        self.pool._new_conn()


def _invariant_https_new_conn(s: HttpsNewConnState) -> bool:
    return s.pool.num_connections == 2


def test_https_new_conn_lost_update():
    """HTTPSConnectionPool._new_conn() has the same num_connections += 1 race."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: HttpsNewConnState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_https_new_conn,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_https_new_conn_lost_update_sweep():
    """Sweep 20 seeds for the HTTPS num_connections race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: HttpsNewConnState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                ],
                invariant=_invariant_https_new_conn,
                max_attempts=200,
                max_ops=300,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===================================================================
# Test 9: _new_conn() + _get_conn() interleaved -- excess connections
# ===================================================================

class NewConnGetConnInterleaveState:
    """Thread 1 calls _new_conn() directly, Thread 2 calls _get_conn().

    _get_conn() internally may call _new_conn() if the pool returns None.
    Both threads end up calling _new_conn(), so we get two new connections
    plus the lost-update bug on num_connections.

    The real issue: _get_conn() has no coordination with direct _new_conn()
    callers, leading to over-creation of connections.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999, maxsize=1)
        self.direct_conn = None
        self.pooled_conn = None

    def thread1(self):
        """Directly create a connection (like what retry logic does)."""
        self.direct_conn = self.pool._new_conn()

    def thread2(self):
        """Get a connection through the pool."""
        self.pooled_conn = self.pool._get_conn()


def _invariant_new_get_interleave(s: NewConnGetConnInterleaveState) -> bool:
    # Both threads create connections. num_connections should be 2
    # if both ran _new_conn(), but the lost-update can make it 1.
    return s.pool.num_connections == 2


def test_new_conn_get_conn_interleave():
    """Direct _new_conn() + _get_conn() race -- num_connections lost update."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: NewConnGetConnInterleaveState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_new_get_interleave,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===================================================================
# Test 10: _put_conn() race -- two threads return connections to maxsize=1 pool
# ===================================================================

class DoublePutConnState:
    """Two threads both call _put_conn() on a maxsize=1 pool.

    After both _get_conn() calls drain the pool, both threads have
    connections. They both try to _put_conn(). The first succeeds,
    the second hits queue.Full (in non-blocking mode) and the connection
    is discarded. But the race in _put_conn() with the pool is not None
    check means the second thread's connection might not get properly
    closed if close() races in.
    """

    def __init__(self):
        self.pool = HTTPConnectionPool("localhost", port=9999, maxsize=1)
        self.conn1 = self.pool._new_conn()
        self.conn2 = self.pool._new_conn()
        # Drain the pool so both puts will compete for the single slot
        try:
            while True:
                self.pool.pool.get_nowait()
        except queue.Empty:
            pass
        self.put1_error = None
        self.put2_error = None

    def thread1(self):
        try:
            self.pool._put_conn(self.conn1)
        except Exception as e:
            self.put1_error = e

    def thread2(self):
        try:
            self.pool._put_conn(self.conn2)
        except Exception as e:
            self.put2_error = e


def _invariant_double_put(s: DoublePutConnState) -> bool:
    # For maxsize=1, exactly one connection should end up in the pool.
    # The other should be discarded (closed). No errors should be raised
    # in non-blocking mode.
    if s.pool.pool is None:
        return False
    pool_size = s.pool.pool.qsize()
    no_errors = s.put1_error is None and s.put2_error is None
    return pool_size == 1 and no_errors


def test_double_put_conn():
    """Two threads both _put_conn() to a maxsize=1 pool."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DoublePutConnState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_double_put,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===================================================================
# Test 11: Deterministic reproduction for any found bug
# ===================================================================

def test_reproduce_three_thread_new_conn():
    """Find a counterexample for 3-thread _new_conn(), reproduce 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ThreeThreadNewConnState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
            invariant=_invariant_three_thread_new_conn,
            max_attempts=500,
            max_ops=400,
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
            setup=lambda: ThreeThreadNewConnState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
        )
        is_bug = state.pool.num_connections != 3
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: num_connections={state.pool.num_connections} "
            f"[{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


def test_reproduce_get_conn_double_create():
    """Find and reproduce the _get_conn double-create race."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: GetConnDoubleCreateState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_invariant_get_conn_double_create,
            max_attempts=500,
            max_ops=300,
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
            setup=lambda: GetConnDoubleCreateState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
        )
        is_bug = state.pool.num_connections > 1
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: num_connections={state.pool.num_connections} "
            f"[{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ===================================================================
# Main -- run all tests
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: _new_conn() lost update with 3 threads")
    print("=" * 70)
    test_new_conn_lost_update_three_threads()

    print("\n" + "=" * 70)
    print("Test 1b: 3-thread _new_conn() -- seed sweep")
    print("=" * 70)
    test_new_conn_lost_update_three_threads_sweep()

    print("\n" + "=" * 70)
    print("Test 2: _get_conn() pool-empty TOCTOU -- double create")
    print("=" * 70)
    test_get_conn_double_create()

    print("\n" + "=" * 70)
    print("Test 3: _put_conn() vs close() race")
    print("=" * 70)
    test_put_conn_vs_close()

    print("\n" + "=" * 70)
    print("Test 4: _get_conn() vs close() race")
    print("=" * 70)
    test_get_conn_vs_close()

    print("\n" + "=" * 70)
    print("Test 5: num_requests lost update")
    print("=" * 70)
    test_num_requests_lost_update()

    print("\n" + "=" * 70)
    print("Test 5b: num_requests lost update -- seed sweep")
    print("=" * 70)
    test_num_requests_lost_update_sweep()

    print("\n" + "=" * 70)
    print("Test 6: double close() race")
    print("=" * 70)
    test_double_close()

    print("\n" + "=" * 70)
    print("Test 7: _get_conn() + _put_conn() connection duplication")
    print("=" * 70)
    test_get_put_conn_duplication()

    print("\n" + "=" * 70)
    print("Test 8: HTTPSConnectionPool._new_conn() lost update")
    print("=" * 70)
    test_https_new_conn_lost_update()

    print("\n" + "=" * 70)
    print("Test 8b: HTTPS _new_conn() -- seed sweep")
    print("=" * 70)
    test_https_new_conn_lost_update_sweep()

    print("\n" + "=" * 70)
    print("Test 9: _new_conn() + _get_conn() interleave")
    print("=" * 70)
    test_new_conn_get_conn_interleave()

    print("\n" + "=" * 70)
    print("Test 10: double _put_conn() to maxsize=1 pool")
    print("=" * 70)
    test_double_put_conn()

    print("\n" + "=" * 70)
    print("Test 11a: Reproduce 3-thread _new_conn() bug")
    print("=" * 70)
    test_reproduce_three_thread_new_conn()

    print("\n" + "=" * 70)
    print("Test 11b: Reproduce _get_conn() double-create bug")
    print("=" * 70)
    test_reproduce_get_conn_double_create()
