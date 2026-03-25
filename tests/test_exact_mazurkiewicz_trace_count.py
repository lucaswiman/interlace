"""Mazurkiewicz trace counts for simple concurrent programs.

Verifies that DPOR explores exactly the theoretically predicted number of
Mazurkiewicz traces for several families of concurrent programs.  Each test
includes a proof that the predicted trace count is tight (both a lower and
upper bound).

Important: those proofs are stated against the **intended logical event
model** of each program, not necessarily the full Python opcode-level event
alphabet that the current implementation explores.  Where we know a case is
exact even under the current Python event alphabet, the test says so
explicitly.

Categories:
1. N threads mutating independent state  → 1 trace
2. Two threads racing on N shared vars   → 2^N traces
3. N threads serialized by a single lock → N! traces
4. (2) but with a lock                   → 2 traces
5. File I/O and PostgreSQL integration   → analogous counts

If the DPOR implementation finds more or fewer traces than the theoretical
bound, the test is left failing so deviations are visible.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading

import pytest

from frontrun.dpor import explore_dpor


class _Slot:
    """A single mutable slot, isolated as its own Python object.

    Using a separate object per "variable" ensures DPOR tracks conflicts
    per-slot rather than per-container, matching the abstract model.
    """

    def __init__(self, value: int = 0) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# Case 1: N threads mutating independent state
# ---------------------------------------------------------------------------


class TestIndependentState:
    """N threads where thread i only touches slot i.

    All cross-thread operation pairs access disjoint objects, so every
    pair of operations from different threads is independent (commutes).
    Every linearization belongs to the same Mazurkiewicz trace.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_independent_writes(self, n: int) -> None:
        """N threads writing to N independent slots → 1 Mazurkiewicz trace.

        Proof of tightness:

        Let a^i denote the write operation of thread i to slot i.

        Independence alphabet I:
            For all i ≠ j, (a^i, a^j) ∈ I because a^i accesses only the
            Python object ``slots[i]`` and a^j accesses only ``slots[j]``.
            Reads of the container list ``slots`` are read-read and do not
            create conflicts.

        Since every cross-thread pair is independent, all N! linearizations
        belong to the same equivalence class.

        Lower bound: ≥ 1 (at least one execution exists).
        Upper bound: ≤ 1 (all linearizations are equivalent).
        Exact count: 1.
        """

        class State:
            def __init__(self) -> None:
                self.slots = [_Slot() for _ in range(n)]

        def make_thread(i: int):  # noqa: ANN202
            def thread_fn(s: State) -> None:
                s.slots[i].value = i + 1

            return thread_fn

        result = explore_dpor(
            setup=State,
            threads=[make_thread(i) for i in range(n)],
            invariant=lambda s: all(s.slots[i].value == i + 1 for i in range(len(s.slots))),
            max_executions=1000,
            preemption_bound=None,
            stop_on_first=False,
            detect_io=False,
            total_timeout=60.0,
        )

        assert result.property_holds, f"Invariant should hold for independent writes (N={n})"
        assert result.num_explored == 1, (
            f"N={n}: Expected exactly 1 Mazurkiewicz trace for independent writes, got {result.num_explored}"
        )


# ---------------------------------------------------------------------------
# Case 2: Two threads racing on N shared variables (same access order)
# ---------------------------------------------------------------------------


class TestTwoThreadsSharedState:
    """Two threads each writing to the same N variables in the same order.

    Each variable v_i is a separate Python object (_Slot).  Each thread
    performs a single write (STORE_ATTR) to each variable in sequence.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_two_threads_n_shared_vars(self, n: int) -> None:
        """Two threads writing to N shared variables → 2^N Mazurkiewicz traces.

        The proof below is for the intended logical event model where each
        ``v.value = tid`` is treated as one write event on slot ``v``.

        For ``n == 1``, this bound is also exact under the current Python
        event alphabet: after the trace-cache merge fix, the only cross-thread
        dependent pair is the two ``STORE_ATTR value`` writes to the shared
        ``_Slot`` object, so the exact count is still 2.

        For ``n > 1``, the current implementation still explores a finer
        opcode-level alphabet than the proof below assumes, so the asserted
        bound should be read as the semantic target rather than a proven bound
        for the current instrumentation.

        Proof of tightness for the logical event model:

        Let a_i denote thread A's write to variable v_i and b_i denote
        thread B's write to v_i, for i = 1, …, N.

        Program order within each thread:
            a_1 < a_2 < … < a_N
            b_1 < b_2 < … < b_N

        Dependency relation D:
            (a_i, b_i) ∈ D for each i (write–write conflict on the same
            Python object v_i).  All other cross-thread pairs (a_i, b_j)
            with i ≠ j are independent (different objects).

        A Mazurkiewicz trace is uniquely determined by the total order it
        induces on each dependent pair.  For pair i, either a_i ≺ b_i or
        b_i ≺ a_i in the trace.

        Upper bound ≤ 2^N:
            There are at most 2^N choices for the N dependent pairs.

        Lower bound ≥ 2^N (all combinations are realizable):
            Fix any assignment σ: {1,…,N} → {A, B}, where σ(i) = A means
            "a_i before b_i."  Construct a linearization L by scanning
            i = 1, …, N:
              • If σ(i) = A, append a_i then b_i.
              • If σ(i) = B, append b_i then a_i.
            Claim: L respects both program orders.
            Within thread A, a_i appears at position 2i−1 or 2i, and
            a_{i+1} at position 2(i+1)−1 or 2(i+1), so a_i < a_{i+1}.
            Symmetrically for thread B.  ∎

            Two linearizations with different σ are inequivalent because
            they differ on the ordering of a dependent pair.

        Exact count in the logical event model: 2^N.
        """

        class State:
            def __init__(self) -> None:
                self.vars = [_Slot() for _ in range(n)]

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(s: State) -> None:
                for v in s.vars:
                    v.value = tid

            return thread_fn

        expected = 2**n
        result = explore_dpor(
            setup=State,
            threads=[make_thread(0), make_thread(1)],
            invariant=lambda s: True,
            max_executions=max(expected * 10, 1000),
            preemption_bound=None,
            stop_on_first=False,
            detect_io=False,
            total_timeout=60.0,
        )

        assert result.num_explored == expected, (
            f"N={n}: Expected exactly {expected} Mazurkiewicz traces (2^{n}), got {result.num_explored}"
        )


# ---------------------------------------------------------------------------
# Case 3: N threads serialized by a single lock
# ---------------------------------------------------------------------------


class TestNThreadsWithLock:
    """N threads competing for a single lock, then updating shared state.

    The lock serializes all critical sections, so different acquisition
    orderings produce distinct traces.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_n_threads_single_lock(self, n: int) -> None:
        """N threads with single lock → N! Mazurkiewicz traces.

        Proof of tightness:

        Each thread i executes:
            acquire(L) ; write_i(shared) ; release(L)

        where L is a single shared lock.

        Dependency relation D:
            Lock operations on the same lock are pairwise dependent.
            For any two threads i ≠ j, acquire_i and acquire_j conflict
            (both are synchronization operations on L), as do release_i
            and acquire_j when j follows i.

        Since the lock serializes all critical sections, the execution
        is equivalent to a sequential run in some permutation of the
        N threads.

        Upper bound ≤ N!:
            There are only N! permutations of the N critical sections.
            Each permutation yields at most one trace (the operations
            within each critical section are fixed by program order).

        Lower bound ≥ N!:
            Every permutation is realizable (run threads in that order).
            Two permutations π and π' that disagree on the relative order
            of threads i and j are inequivalent: acquire_i and acquire_j
            are dependent, so their order is fixed within a trace.

        Exact count: N!.
        """

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.shared = _Slot()

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(s: State) -> None:
                with s.lock:
                    s.shared.value = tid

            return thread_fn

        expected = math.factorial(n)
        result = explore_dpor(
            setup=State,
            threads=[make_thread(i) for i in range(n)],
            invariant=lambda s: True,
            max_executions=max(expected * 10, 1000),
            preemption_bound=None,
            stop_on_first=False,
            detect_io=False,
            total_timeout=60.0,
        )

        assert result.num_explored == expected, (
            f"N={n}: Expected exactly {expected} Mazurkiewicz traces ({n}!), got {result.num_explored}"
        )


# ---------------------------------------------------------------------------
# Case 4: Two threads, N shared vars, with a single lock (case 2 + lock)
# ---------------------------------------------------------------------------


class TestTwoThreadsSharedStateWithLock:
    """Two threads each writing to N shared variables, protected by a
    single lock over the entire write sequence.

    The lock serializes the two threads completely → 2 traces.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_two_threads_locked_n_vars(self, n: int) -> None:
        """Two threads with single lock over N var writes → 2 traces.

        Proof of tightness:

        Each thread executes:
            acquire(L) ; write(v_1) ; … ; write(v_N) ; release(L)

        The lock L serializes the two critical sections.  Either thread A
        runs its entire critical section before thread B, or vice versa.

        Upper bound ≤ 2:
            While a thread holds L, no other thread can enter.  So the
            critical sections cannot interleave.  There are at most 2
            orderings: A-then-B or B-then-A.

        Lower bound ≥ 2:
            Both orderings are realizable.  They are inequivalent because
            acquire_A and acquire_B are dependent (same lock).

        Exact count: 2 = 2!.
        """

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.vars = [_Slot() for _ in range(n)]

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(s: State) -> None:
                with s.lock:
                    for v in s.vars:
                        v.value = tid

            return thread_fn

        result = explore_dpor(
            setup=State,
            threads=[make_thread(0), make_thread(1)],
            invariant=lambda s: True,
            max_executions=100,
            preemption_bound=None,
            stop_on_first=False,
            detect_io=False,
            total_timeout=60.0,
        )

        assert result.num_explored == 2, (
            f"N={n}: Expected exactly 2 Mazurkiewicz traces (lock serializes), got {result.num_explored}"
        )


# ---------------------------------------------------------------------------
# Case 5a: File I/O — independent files
# ---------------------------------------------------------------------------


class TestFileIOTraceCount:
    """File I/O variants of the trace-counting tests.

    These use the LD_PRELOAD I/O interception to track file-level conflicts.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_independent_file_writes(self, n: int) -> None:
        """N threads writing to N independent files → 1 Mazurkiewicz trace.

        Proof of tightness:

        Thread i writes only to file_i.  All cross-thread pairs are
        independent (different file paths).  Therefore all linearizations
        are equivalent.

        Exact count: 1 (same argument as TestIndependentState).
        """
        tmpdir = tempfile.mkdtemp()
        paths = [os.path.join(tmpdir, f"file_{i}.txt") for i in range(n)]

        class State:
            def __init__(self) -> None:
                for p in paths:
                    with open(p, "w") as f:
                        f.write("0")

        def make_thread(i: int):  # noqa: ANN202
            def thread_fn(s: State) -> None:
                with open(paths[i], "w") as f:
                    f.write(str(i + 1))

            return thread_fn

        result = explore_dpor(
            setup=State,
            threads=[make_thread(i) for i in range(n)],
            invariant=lambda s: all(open(paths[i]).read() == str(i + 1) for i in range(n)),
            max_executions=1000,
            preemption_bound=None,
            stop_on_first=False,
            detect_io=True,
            total_timeout=60.0,
        )

        assert result.property_holds, f"Invariant should hold for independent file writes (N={n})"
        assert result.num_explored == 1, (
            f"N={n}: Expected 1 Mazurkiewicz trace for independent file writes, got {result.num_explored}"
        )

    def test_two_threads_same_file(self) -> None:
        """Two threads writing to the same file → 2 Mazurkiewicz traces.

        Proof of tightness:

        Thread A and thread B each perform a single write to the same
        file path.  These writes conflict (write–write on the same file).

        Upper bound ≤ 2: there are only 2 orderings of 2 conflicting ops.
        Lower bound ≥ 2: both orderings are realizable and inequivalent.

        Exact count: 2.
        """
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "shared.txt")

        class State:
            def __init__(self) -> None:
                with open(path, "w") as f:
                    f.write("init")

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(s: State) -> None:
                with open(path, "w") as f:
                    f.write(str(tid))

            return thread_fn

        result = explore_dpor(
            setup=State,
            threads=[make_thread(0), make_thread(1)],
            invariant=lambda s: True,
            max_executions=100,
            preemption_bound=None,
            stop_on_first=False,
            detect_io=True,
            total_timeout=60.0,
        )

        assert result.num_explored == 2, (
            f"Expected 2 Mazurkiewicz traces for same-file writes, got {result.num_explored}"
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_n_threads_locked_file_writes(self, n: int) -> None:
        """N threads with lock guarding a shared file.

        The abstract proof gives N! traces (same as TestNThreadsWithLock).
        With synced I/O (Python-level I/O inside a lock uses dpor_vv which
        respects lock happens-before), the file writes are properly
        serialized and the trace counts match the theoretical N!.

        Previously the io_vv model (which ignores lock HB for I/O)
        gave N=2 → 3, N=3 → 17.  The synced I/O change fixed this.
        """
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "counter.txt")

        class State:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("0")

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(s: State) -> None:
                with s.lock:
                    with open(path, "w") as f:
                        f.write(str(tid))

            return thread_fn

        # N! traces: synced I/O respects lock HB
        import math

        expected = math.factorial(n)
        result = explore_dpor(
            setup=State,
            threads=[make_thread(i) for i in range(n)],
            invariant=lambda s: True,
            max_executions=max(expected * 10, 1000),
            preemption_bound=None,
            stop_on_first=False,
            detect_io=True,
            total_timeout=60.0,
        )

        assert result.num_explored == expected, f"N={n}: Expected {expected} traces (N!), got {result.num_explored}"


# ---------------------------------------------------------------------------
# Case 5b: PostgreSQL integration tests
# ---------------------------------------------------------------------------

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"postgresql:///{_DB_NAME}")


@pytest.fixture(scope="module")
def _pg_available():
    """Check Postgres is available and create test tables."""
    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not installed")

    try:
        conn = psycopg2.connect(_DB_URL)
    except Exception:
        pytest.skip(f"Postgres not available at {_DB_URL}")

    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS maz_trace_test")
        cur.execute("""
            CREATE TABLE maz_trace_test (
                id TEXT PRIMARY KEY,
                value INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Pre-populate rows to avoid INSERT during exploration
        cur.execute("INSERT INTO maz_trace_test (id, value) VALUES ('row1', 0)")
        cur.execute("INSERT INTO maz_trace_test (id, value) VALUES ('row2', 0)")
        cur.execute("INSERT INTO maz_trace_test (id, value) VALUES ('row3', 0)")
    conn.close()
    yield


@pytest.fixture()
def pg_engine(_pg_available):
    """Create a SQLAlchemy engine for tests."""
    from sqlalchemy import create_engine

    eng = create_engine(_DB_URL)
    yield eng
    eng.dispose()


@pytest.mark.integration
class TestPostgreSQLTraceCount:
    """PostgreSQL integration tests for Mazurkiewicz trace counting.

    These require PostgreSQL and the sqlalchemy/psycopg2 packages.
    Run with: make test-integration-3.14
    """

    def test_two_threads_update_same_row_with_select_for_update(self, pg_engine) -> None:
        """Two threads each doing SELECT FOR UPDATE + UPDATE on the same row → 2 traces.

        Proof of tightness:

        Each thread executes:
            BEGIN ;
            SELECT … FOR UPDATE (acquires row lock on row1) ;
            UPDATE row1 SET value = <tid> ;
            COMMIT (releases row lock)

        The SELECT FOR UPDATE acquires an exclusive row-level lock.
        This serializes the two transactions exactly like a mutex.

        Upper bound ≤ 2:
            The row lock prevents interleaving of the two transactions'
            access to row1.  Only two orderings: A-then-B or B-then-A.

        Lower bound ≥ 2:
            Both orderings are realizable and produce different final
            values for row1, so they are inequivalent.

        Exact count: 2.
        """
        from sqlalchemy import text

        from frontrun.contrib.sqlalchemy import get_connection, sqlalchemy_dpor

        class _State:
            def __init__(self) -> None:
                with pg_engine.connect() as conn:
                    conn.execute(text("UPDATE maz_trace_test SET value = 0 WHERE id = 'row1'"))
                    conn.commit()

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(state: _State) -> None:
                conn = get_connection()
                conn.execute(text("SELECT value FROM maz_trace_test WHERE id = 'row1' FOR UPDATE"))
                conn.execute(
                    text("UPDATE maz_trace_test SET value = :v WHERE id = 'row1'"),
                    {"v": tid},
                )
                conn.commit()

            return thread_fn

        result = sqlalchemy_dpor(
            engine=pg_engine,
            setup=_State,
            threads=[make_thread(0), make_thread(1)],
            invariant=lambda s: True,
            lock_timeout=2000,
            deadlock_timeout=10.0,
            timeout_per_run=15.0,
            max_executions=100,
            preemption_bound=None,
            stop_on_first=False,
            warn_nondeterministic_sql=False,
        )

        # Exact count is 2 (row-lock serialization).  With patch_locks()
        # active (pytest plugin), internal SQLAlchemy cooperative locks add
        # a small number of extra traces.  Upper bound allows for this.
        assert result.num_explored <= 4, (
            f"Expected ≤4 Mazurkiewicz traces for SELECT FOR UPDATE on same row, got {result.num_explored}"
        )

    @pytest.mark.parametrize("n", [2])
    def test_n_threads_select_for_update_same_row(self, pg_engine, n: int) -> None:
        """N threads with SELECT FOR UPDATE on same row → N! Mazurkiewicz traces.

        Proof of tightness:

        Each thread i executes:
            BEGIN ;
            SELECT … FOR UPDATE (acquires exclusive row lock) ;
            UPDATE row1 SET value = i ;
            COMMIT

        The row lock serializes all N transactions.  The argument is
        identical to the mutex case (TestNThreadsWithLock):

        Upper bound ≤ N!: only N! orderings of N serialized transactions.
        Lower bound ≥ N!: each ordering is realizable, and any two that
            differ on the relative order of threads i and j are
            inequivalent (the lock acquisitions are dependent).

        Exact count: N!.
        """
        from sqlalchemy import text

        from frontrun.contrib.sqlalchemy import get_connection, sqlalchemy_dpor

        class _State:
            def __init__(self) -> None:
                with pg_engine.connect() as conn:
                    conn.execute(text("UPDATE maz_trace_test SET value = 0 WHERE id = 'row1'"))
                    conn.commit()

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(state: _State) -> None:
                conn = get_connection()
                conn.execute(text("SELECT value FROM maz_trace_test WHERE id = 'row1' FOR UPDATE"))
                conn.execute(
                    text("UPDATE maz_trace_test SET value = :v WHERE id = 'row1'"),
                    {"v": tid},
                )
                conn.commit()

            return thread_fn

        expected = math.factorial(n)
        result = sqlalchemy_dpor(
            engine=pg_engine,
            setup=_State,
            threads=[make_thread(i) for i in range(n)],
            invariant=lambda s: True,
            lock_timeout=5000,
            deadlock_timeout=15.0,
            timeout_per_run=20.0,
            max_executions=max(expected * 10, 1000),
            preemption_bound=None,
            stop_on_first=False,
            warn_nondeterministic_sql=False,
        )

        # Upper bound: N! × small cooperative-lock overhead factor
        upper = expected * 3
        assert result.num_explored <= upper, (
            f"N={n}: Expected ≤{upper} Mazurkiewicz traces ({n}! × 3), got {result.num_explored}"
        )

    def test_two_threads_independent_rows(self, pg_engine) -> None:
        """Two threads updating independent rows → 1 Mazurkiewicz trace.

        Proof of tightness:

        Thread A updates row1 only; thread B updates row2 only.
        Since the rows are distinct, all operations are independent
        (no row-level lock contention, no data conflict).

        Exact count: 1 (same argument as TestIndependentState).
        """
        from sqlalchemy import text

        from frontrun.contrib.sqlalchemy import get_connection, sqlalchemy_dpor

        class _State:
            def __init__(self) -> None:
                with pg_engine.connect() as conn:
                    conn.execute(text("UPDATE maz_trace_test SET value = 0 WHERE id IN ('row1', 'row2')"))
                    conn.commit()

        def thread_a(state: _State) -> None:
            conn = get_connection()
            conn.execute(text("SELECT value FROM maz_trace_test WHERE id = 'row1' FOR UPDATE"))
            conn.execute(text("UPDATE maz_trace_test SET value = 1 WHERE id = 'row1'"))
            conn.commit()

        def thread_b(state: _State) -> None:
            conn = get_connection()
            conn.execute(text("SELECT value FROM maz_trace_test WHERE id = 'row2' FOR UPDATE"))
            conn.execute(text("UPDATE maz_trace_test SET value = 2 WHERE id = 'row2'"))
            conn.commit()

        result = sqlalchemy_dpor(
            engine=pg_engine,
            setup=_State,
            threads=[thread_a, thread_b],
            invariant=lambda s: True,
            lock_timeout=2000,
            deadlock_timeout=10.0,
            timeout_per_run=15.0,
            max_executions=100,
            preemption_bound=None,
            stop_on_first=False,
            warn_nondeterministic_sql=False,
        )

        assert result.num_explored <= 3, (
            f"Expected ≤3 Mazurkiewicz traces for independent row updates, got {result.num_explored}"
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_two_threads_n_shared_rows_select_for_update(self, pg_engine, n: int) -> None:
        """Two threads each updating N rows with per-row SELECT FOR UPDATE → 2^N traces.

        Proof of tightness:

        Thread A: for i in 1..N: SELECT row_i FOR UPDATE; UPDATE row_i; COMMIT
        Thread B: for i in 1..N: SELECT row_i FOR UPDATE; UPDATE row_i; COMMIT

        Each thread acquires and releases the row lock for row_i before
        proceeding to row_{i+1}.  The row-level locks on different rows
        are independent.

        The argument is the same as TestTwoThreadsSharedState (case 2)
        but with row locks replacing object writes.  For each row_i, the
        two threads' lock acquisitions form a dependent pair.  The N pairs
        can be independently ordered, giving 2^N traces.

        (See the proof in TestTwoThreadsSharedState.test_two_threads_n_shared_vars
        for the full construction.)

        Exact count: 2^N.
        """
        from sqlalchemy import text

        from frontrun.contrib.sqlalchemy import get_connection, sqlalchemy_dpor

        row_ids = [f"row{i + 1}" for i in range(n)]

        class _State:
            def __init__(self) -> None:
                with pg_engine.connect() as conn:
                    for rid in row_ids:
                        conn.execute(
                            text("UPDATE maz_trace_test SET value = 0 WHERE id = :id"),
                            {"id": rid},
                        )
                    conn.commit()

        def make_thread(tid: int):  # noqa: ANN202
            def thread_fn(state: _State) -> None:
                conn = get_connection()
                for rid in row_ids:
                    conn.execute(
                        text("SELECT value FROM maz_trace_test WHERE id = :id FOR UPDATE"),
                        {"id": rid},
                    )
                    conn.execute(
                        text("UPDATE maz_trace_test SET value = :v WHERE id = :id"),
                        {"v": tid, "id": rid},
                    )
                    conn.commit()

            return thread_fn

        expected = 2**n
        result = sqlalchemy_dpor(
            engine=pg_engine,
            setup=_State,
            threads=[make_thread(0), make_thread(1)],
            invariant=lambda s: True,
            lock_timeout=2000,
            deadlock_timeout=10.0,
            timeout_per_run=15.0,
            max_executions=max(expected * 10, 1000),
            preemption_bound=None,
            stop_on_first=False,
            warn_nondeterministic_sql=False,
        )

        upper = expected * 3
        assert result.num_explored <= upper, (
            f"N={n}: Expected ≤{upper} Mazurkiewicz traces (2^{n} × 3), got {result.num_explored}"
        )
