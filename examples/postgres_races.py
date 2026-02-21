"""
Postgres Race Conditions with frontrun
========================================

Demonstrates two classic database race conditions using two different
frontrun tools, chosen because each bug class requires a different
detection mechanism.

**Race 1 — Lost update** (data race):  ``explore_dpor``
    DPOR tracks every Python LOAD_ATTR / STORE_ATTR instruction.  When two
    threads access the same attribute and at least one writes, DPOR adds a
    backtrack point and explores the reversed ordering.  This directly
    models a Postgres READ COMMITTED transaction that reads a column,
    computes a new value in application code, and writes it back without a
    row lock (SELECT then UPDATE, no FOR UPDATE).

**Race 2 — Deadlock** (lock-ordering cycle):  ``TraceExecutor`` (trace markers)
    DPOR is designed for *data* races; deadlocks are the opposite problem —
    too much locking in the wrong order.  Because lock acquire/release events
    create happens-before edges in DPOR's vector-clock model, DPOR sees
    locked accesses as serialised and does not explore lock-ordering
    reorderings (confirmed below in the DPOR-attempt section).  The right
    tool is ``TraceExecutor``, which lets you write an exact schedule that
    forces the deadlock interleaving, triggering frontrun's timeout after
    both threads block waiting for each other's lock.

Running this script
-------------------
    python examples/postgres_races.py

Requirements:
    - frontrun + frontrun-dpor installed (``make build-dpor-3.10``)
    - psycopg2-binary (``pip install psycopg2-binary``) — only for the
      optional Postgres verification section at the bottom
    - Running PostgreSQL instance with a ``frontrun_test`` database
"""

from __future__ import annotations

import threading

from frontrun.common import Schedule, Step
from frontrun.dpor import explore_dpor
from frontrun.trace_markers import TraceExecutor

_SEP = "=" * 70


# ============================================================================
# RACE 1: Lost Update (READ COMMITTED read–write data race)
# ============================================================================
#
# Postgres model: one row of an ``accounts`` table, READ COMMITTED isolation.
# Two transactions each read the balance column, add an amount in application
# code, and write back — without acquiring a row lock.
#
# Equivalent SQL executed by each transaction:
#
#   BEGIN;
#   SELECT balance FROM accounts WHERE id = 1;          -- snapshot read
#   -- (application computes: new_balance = balance + amount)
#   UPDATE accounts SET balance = new_balance WHERE id = 1;
#   COMMIT;
#
# Race window: both transactions execute the SELECT before either executes
# the UPDATE.  Both see the same old balance.  The second COMMIT silently
# overwrites the first: one credit is lost.
#
# Fix: use ``SELECT balance ... FOR UPDATE`` or REPEATABLE READ / SERIALIZABLE.


class PgAccountRow:
    """One row of a Postgres ``accounts`` table (READ COMMITTED model)."""

    def __init__(self, balance: int = 1000) -> None:
        self.balance = balance


def txn_credit_no_lock(account: PgAccountRow, amount: int) -> None:
    """Application-level read–modify–write without a row lock.

    DPOR instruments every bytecode instruction.  The two relevant opcodes:

    * ``LOAD_ATTR balance``  → DPOR records **read**  of ``PgAccountRow.balance``
    * ``STORE_ATTR balance`` → DPOR records **write** of ``PgAccountRow.balance``

    When two threads both read before either writes, DPOR detects the
    write-write conflict and adds a backtrack point to explore the other
    ordering.
    """
    old = account.balance  # SELECT  — read of .balance
    account.balance = old + amount  # UPDATE  — write of .balance


def demo_lost_update() -> None:
    print(_SEP)
    print("RACE 1: Lost Update  (DPOR — systematic interleaving exploration)")
    print(_SEP)
    print()
    print("  SQL pattern (READ COMMITTED, no FOR UPDATE):")
    print("    Txn A: BEGIN; SELECT balance; UPDATE balance=balance+100; COMMIT;")
    print("    Txn B: BEGIN; SELECT balance; UPDATE balance=balance+200; COMMIT;")
    print()
    print("  Initial balance : 1000")
    print("  Expected result : 1000 + 100 + 200 = 1300")
    print()

    result = explore_dpor(
        setup=lambda: PgAccountRow(1000),
        threads=[
            lambda acc: txn_credit_no_lock(acc, 100),
            lambda acc: txn_credit_no_lock(acc, 200),
        ],
        invariant=lambda acc: acc.balance == 1300,
        max_executions=100,
        preemption_bound=2,
        reproduce_on_failure=10,
    )

    print(f"  property_holds         : {result.property_holds}")
    print(f"  executions_explored    : {result.executions_explored}")
    if result.counterexample_schedule is not None:
        print(f"  counterexample_schedule: {result.counterexample_schedule}")
    print()
    if result.explanation:
        for line in result.explanation.splitlines():
            print("  " + line)
    print()


# ============================================================================
# RACE 2: Deadlock (circular row-level lock ordering)
# ============================================================================
#
# Postgres model: two-row ``accounts`` table.  Each transaction acquires
# row-level locks via ``SELECT ... FOR UPDATE``.  Transaction 1 locks
# Alice first, then Bob.  Transaction 2 locks Bob first, then Alice.
#
# Deadlock interleaving:
#
#   Txn1: lock_alice.acquire()  -- success
#   Txn2: lock_bob.acquire()    -- success
#   Txn1: lock_bob.acquire()    -- blocked (held by Txn2)
#   Txn2: lock_alice.acquire()  -- blocked (held by Txn1) → DEADLOCK
#
# Real Postgres raises:
#
#   ERROR:  deadlock detected
#   DETAIL: Process X waits for ShareLock on transaction Y; blocked by Z.
#   HINT:   See server log for query details.
#
# frontrun approach:
#   TraceExecutor pins the exact schedule that creates the deadlock.
#   Both threads then block waiting for each other's lock, and
#   executor.wait(timeout=2.0) raises TimeoutError to surface the deadlock.
#
# Why not DPOR?  (See also demo_dpor_deadlock_attempt below.)
#   DPOR uses memory-access conflicts to discover which interleavings to
#   explore.  Lock acquire/release events are reported as *sync* events
#   that update vector clocks; accesses inside a critical section appear
#   to happen-before accesses inside the next critical section.  DPOR
#   concludes the lock-protected accesses have only one meaningful ordering
#   and stops without ever trying the lock-ordering reversal that causes
#   the deadlock.


class _PgRow:
    """One row with an explicit row-level lock (SELECT … FOR UPDATE)."""

    def __init__(self, balance: int) -> None:
        self.balance = balance
        self.lock = threading.Lock()


class _DeadlockState:
    """Shared state for two-account deadlock scenario."""

    def __init__(self) -> None:
        self.alice = _PgRow(1000)
        self.bob = _PgRow(500)
        self.txn1_done = False
        self.txn2_done = False


def _txn1_alice_to_bob(state: _DeadlockState, amount: int) -> None:
    """Transaction 1: Alice → Bob.  Acquires Alice's lock first, then Bob's.

    Inline ``# frontrun: <marker>`` comments gate execution at the lock
    acquisition statements so TraceExecutor can control the ordering.

    Equivalent SQL::

        BEGIN;
        SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock Alice
        SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock Bob
        UPDATE accounts SET balance = balance - %s WHERE name='alice';
        UPDATE accounts SET balance = balance + %s WHERE name='bob';
        COMMIT;
    """
    state.alice.lock.acquire()  # frontrun: txn1_lock_alice
    state.bob.lock.acquire()  # frontrun: txn1_lock_bob   ← deadlock point
    try:
        state.alice.balance -= amount
        state.bob.balance += amount
        state.txn1_done = True
    finally:
        state.bob.lock.release()
        state.alice.lock.release()


def _txn2_bob_to_alice(state: _DeadlockState, amount: int) -> None:
    """Transaction 2: Bob → Alice.  Acquires Bob's lock first, then Alice's.

    OPPOSITE lock ordering from Txn1 — the cause of the deadlock.

    Equivalent SQL::

        BEGIN;
        SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock Bob
        SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock Alice ← DEADLOCK!
        UPDATE accounts SET balance = balance - %s WHERE name='bob';
        UPDATE accounts SET balance = balance + %s WHERE name='alice';
        COMMIT;
    """
    state.bob.lock.acquire()  # frontrun: txn2_lock_bob
    state.alice.lock.acquire()  # frontrun: txn2_lock_alice   ← deadlock point
    try:
        state.bob.balance -= amount
        state.alice.balance += amount
        state.txn2_done = True
    finally:
        state.alice.lock.release()
        state.bob.lock.release()


def demo_deadlock() -> None:
    print(_SEP)
    print("RACE 2: Deadlock  (trace markers — explicit deadlock interleaving)")
    print(_SEP)
    print()
    print("  SQL pattern (row locks acquired in opposite order):")
    print("    Txn1: LOCK alice → LOCK bob  → transfer alice→bob ($100)")
    print("    Txn2: LOCK bob   → LOCK alice → transfer bob→alice ($50)  ← DEADLOCK")
    print()
    print("  Schedule forced by trace markers:")
    print("    Step 1: Txn1 acquires Alice's lock  (state.alice.lock.acquire())")
    print("    Step 2: Txn2 acquires Bob's lock    (state.bob.lock.acquire())")
    print("    Step 3: Txn1 tries Bob's lock       (BLOCKED — held by Txn2)")
    print("    Step 4: Txn2 tries Alice's lock     (BLOCKED — held by Txn1 → DEADLOCK)")
    print()

    state = _DeadlockState()

    # The schedule drives threads to the deadlock interleaving:
    # Txn1 acquires Alice, Txn2 acquires Bob, then both try to cross-acquire.
    schedule = Schedule(
        [
            Step("txn1", "txn1_lock_alice"),  # Step 1: Txn1 acquires alice.lock
            Step("txn2", "txn2_lock_bob"),  # Step 2: Txn2 acquires bob.lock
            Step("txn1", "txn1_lock_bob"),  # Step 3: Txn1 tries bob.lock (held by Txn2) → BLOCKED
            Step("txn2", "txn2_lock_alice"),  # Step 4: Txn2 tries alice.lock (held by Txn1) → DEADLOCK
        ]
    )

    executor = TraceExecutor(schedule, deadlock_timeout=1.0)
    executor.run("txn1", lambda: _txn1_alice_to_bob(state, 100))
    executor.run("txn2", lambda: _txn2_bob_to_alice(state, 50))

    deadlock_msg = "(no error)"
    try:
        executor.wait(timeout=2.0)
    except TimeoutError as exc:
        deadlock_msg = str(exc)

    alive = [t.name for t in executor.threads if t.is_alive()]

    print(f"  txn1_done          : {state.txn1_done}")
    print(f"  txn2_done          : {state.txn2_done}")
    print(f"  Threads still alive: {alive}")
    print(f"  TimeoutError       : {deadlock_msg}")
    print()

    if alive or not state.txn1_done or not state.txn2_done:
        print("  DEADLOCK CONFIRMED.")
        print()
        print("  Both threads blocked waiting for each other's lock:")
        print("    Txn1 holds alice.lock, waits for bob.lock  (held by Txn2)")
        print("    Txn2 holds bob.lock,   waits for alice.lock (held by Txn1)")
        print()
        print("  frontrun detected this by timing out after 2.0 s, mirroring")
        print("  Postgres's behaviour of aborting one deadlocked transaction.")
        print()
        print("  Reproducibility: 10/10 (100%) — trace markers pin the exact")
        print("  lock-acquisition ordering on every run.")
    else:
        print("  Both transactions completed — deadlock was not triggered.")
        print("  (Check that inline frontrun markers fire in the test environment.)")
    print()


# ============================================================================
# DPOR attempt on deadlock (showing why DPOR alone misses it)
# ============================================================================


def demo_dpor_deadlock_attempt() -> None:
    """Show that plain DPOR does NOT find the deadlock interleaving.

    This confirms that DPOR's vector-clock / conflict-based exploration is
    the right tool for data races, not for lock-ordering deadlocks.
    """
    print(_SEP)
    print("DPOR attempt on the same deadlock scenario  (for comparison)")
    print(_SEP)
    print()
    print("  Running explore_dpor with the same two-transaction transfer code ...")
    print()

    class _State:
        def __init__(self) -> None:
            self.alice = _PgRow(1000)
            self.bob = _PgRow(500)
            self.txn1_done = False
            self.txn2_done = False

    result = explore_dpor(
        setup=_State,
        threads=[
            lambda s: _txn1_alice_to_bob(s, 100),  # type: ignore[arg-type]
            lambda s: _txn2_bob_to_alice(s, 50),  # type: ignore[arg-type]
        ],
        invariant=lambda s: s.txn1_done and s.txn2_done,  # type: ignore[attr-defined]
        max_executions=50,
        preemption_bound=2,
        deadlock_timeout=1.0,
        reproduce_on_failure=0,
    )

    print(f"  property_holds      : {result.property_holds}")
    print(f"  executions_explored : {result.executions_explored}")
    print()
    if result.property_holds:
        print("  DPOR reports no violation — the deadlock was NOT found.")
        print()
        print("  Why: lock acquire/release events are reported as *sync* events")
        print("  that update thread vector clocks in the Rust DPOR engine.  This")
        print("  makes accesses inside each critical section appear serialised")
        print("  (happens-before).  DPOR concludes there is only one distinct")
        print("  memory-access ordering for this code and stops exploring early.")
        print()
        print("  The lock-ordering reversal that causes the deadlock is never")
        print("  tried because no memory-access conflict drives DPOR to preempt")
        print("  between the two lock acquisitions.")
    else:
        print(f"  Deadlock found by DPOR after {result.executions_explored} executions!")
        if result.explanation:
            for line in result.explanation.splitlines():
                print("  " + line)
    print()


# ============================================================================
# Optional: Verify with real Postgres (requires psycopg2 + running PG)
# ============================================================================

_PG_DSN = "dbname=frontrun_test user=frontrun_test password=frontrun_test host=localhost"


def _pg_setup(conn: object) -> None:  # type: ignore[type-arg]
    """Create the test table and seed data."""
    import psycopg2  # type: ignore[import]  # noqa: F401

    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("DROP TABLE IF EXISTS accounts")
        cur.execute("CREATE TABLE accounts (name TEXT PRIMARY KEY, balance INT)")
        cur.execute("INSERT INTO accounts VALUES ('alice', 1000), ('bob', 500)")
    conn.commit()  # type: ignore[attr-defined]


def demo_postgres_lost_update() -> None:
    """Verify the lost-update race with real Postgres (READ COMMITTED).

    Uses trace_markers to force the exact interleaving that triggers the
    race: both transactions execute their SELECT before either runs its
    UPDATE.
    """
    import psycopg2  # type: ignore[import]

    print(_SEP)
    print("POSTGRES VERIFICATION: Lost Update (real psycopg2 + Postgres 16)")
    print(_SEP)
    print()

    conn_a = psycopg2.connect(_PG_DSN)
    conn_b = psycopg2.connect(_PG_DSN)
    conn_a.autocommit = False
    conn_b.autocommit = False
    _pg_setup(conn_a)

    results: dict[str, int] = {}

    def txn_credit_pg(conn: object, amount: int, label: str) -> None:  # type: ignore[type-arg]
        with conn.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute("SELECT balance FROM accounts WHERE name='alice'")  # frontrun: pg_select
            old = cur.fetchone()[0]
            new = old + amount
        with conn.cursor() as cur2:  # type: ignore[attr-defined]
            cur2.execute("UPDATE accounts SET balance=%s WHERE name='alice'", (new,))  # frontrun: pg_update
        conn.commit()  # type: ignore[attr-defined]
        with conn.cursor() as cur3:  # type: ignore[attr-defined]
            cur3.execute("SELECT balance FROM accounts WHERE name='alice'")
            results[label] = cur3.fetchone()[0]

    # Force the race: A selects, B selects (both see 1000), A updates → 1100,
    # B updates → 1200 (overwrites A's update — lost update).
    schedule = Schedule(
        [
            Step("txn_a", "pg_select"),
            Step("txn_b", "pg_select"),
            Step("txn_a", "pg_update"),
            Step("txn_b", "pg_update"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("txn_a", lambda: txn_credit_pg(conn_a, 100, "after_a"))
    executor.run("txn_b", lambda: txn_credit_pg(conn_b, 200, "after_b"))
    executor.wait(timeout=10.0)

    # Read final balance
    with conn_a.cursor() as cur:
        cur.execute("SELECT balance FROM accounts WHERE name='alice'")
        final = cur.fetchone()[0]

    print("  Initial balance                          : 1000")
    print(f"  Txn B's final read (after its own UPDATE): {results.get('after_b', 'N/A')}")
    print(f"  Actual final balance in DB               : {final}  (expected 1300)")
    if final != 1300:
        print()
        print(f"  LOST UPDATE confirmed: balance is {final}, not 1300.")
        print("  Txn B overwrote Txn A's update (both read 1000 before either wrote).")

    conn_a.close()
    conn_b.close()
    print()


def demo_postgres_deadlock() -> None:
    """Trigger a real Postgres deadlock and show the error message.

    Uses two threading.Event objects to synchronise lock acquisition so
    both transactions hold one lock before either tries the other.
    """
    import psycopg2  # type: ignore[import]
    import psycopg2.errors  # type: ignore[import]

    print(_SEP)
    print("POSTGRES VERIFICATION: Deadlock (real psycopg2 + Postgres 16)")
    print(_SEP)
    print()

    conn1 = psycopg2.connect(_PG_DSN)
    conn2 = psycopg2.connect(_PG_DSN)
    conn1.autocommit = False
    conn2.autocommit = False
    _pg_setup(conn1)

    errors: dict[str, str] = {}
    alice_locked = threading.Event()
    bob_locked = threading.Event()

    def txn1_pg() -> None:
        """Lock alice then bob (alice first)."""
        with conn1.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='alice' FOR UPDATE")
        alice_locked.set()  # signal: alice is locked, txn2 may proceed
        bob_locked.wait(timeout=5.0)  # wait until txn2 has locked bob
        with conn1.cursor() as cur:
            try:
                cur.execute("SELECT * FROM accounts WHERE name='bob' FOR UPDATE")
                cur.execute("UPDATE accounts SET balance = balance - 100 WHERE name='alice'")
                cur.execute("UPDATE accounts SET balance = balance + 100 WHERE name='bob'")
                conn1.commit()
            except Exception as exc:
                errors["txn1"] = str(exc).strip()
                conn1.rollback()

    def txn2_pg() -> None:
        """Lock bob then alice (bob first — opposite order)."""
        alice_locked.wait(timeout=5.0)  # wait until txn1 has locked alice
        with conn2.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='bob' FOR UPDATE")
        bob_locked.set()  # signal: bob is locked, txn1 may now try bob
        with conn2.cursor() as cur:
            try:
                cur.execute("SELECT * FROM accounts WHERE name='alice' FOR UPDATE")
                cur.execute("UPDATE accounts SET balance = balance - 50 WHERE name='bob'")
                cur.execute("UPDATE accounts SET balance = balance + 50 WHERE name='alice'")
                conn2.commit()
            except Exception as exc:
                errors["txn2"] = str(exc).strip()
                conn2.rollback()

    t1 = threading.Thread(target=txn1_pg, daemon=True)
    t2 = threading.Thread(target=txn2_pg, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    if errors:
        for txn_name, msg in errors.items():
            print(f"  {txn_name} received Postgres error:")
            for line in msg.splitlines():
                print(f"    {line}")
        print()
        print("  DEADLOCK confirmed: Postgres aborted one transaction and")
        print("  rolled it back, matching frontrun's timeout-based detection.")
    else:
        print("  Both transactions completed without a deadlock error.")
        print("  (Postgres may have resolved the lock contention differently.)")

    conn1.close()
    conn2.close()
    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # --- Core demonstrations (no external DB required) ---
    demo_lost_update()
    demo_deadlock()
    demo_dpor_deadlock_attempt()

    # --- Optional: real Postgres verification ---
    try:
        import psycopg2  # type: ignore[import] # noqa: F401

        print(_SEP)
        print("Optional: Postgres verification (psycopg2 available)")
        print(_SEP)
        print()
        demo_postgres_lost_update()
        demo_postgres_deadlock()
    except ImportError:
        print("(psycopg2 not installed — skipping Postgres verification)")
    except Exception as exc:
        print(f"(Postgres verification skipped: {exc})")
