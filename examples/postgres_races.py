"""
Postgres Race Conditions — full integration test with psycopg2
==============================================================

Two classic database race conditions reproduced against a live Postgres
instance using real psycopg2 connections and frontrun scheduling tools.

Race 1 — Lost Update (READ COMMITTED data race)
    Two concurrent transactions each SELECT a row, compute a new value in
    application code, and UPDATE without a row lock.  The second COMMIT
    silently overwrites the first credit — one update is lost.

    Tool: TraceExecutor (trace_markers)
    frontrun's sys.settrace-based scheduler forces the exact interleaving:
    both transactions run their SELECT before either runs their UPDATE.
    The ``# frontrun: <marker>`` comment on each psycopg2 execute call is
    the synchronisation point that the schedule steps control.

Race 2 — Deadlock (circular row-lock ordering)
    Two transactions lock rows in opposite order.  Each transaction holds
    one row-level lock and waits for the other's lock — a circular wait.
    Postgres detects the cycle and raises ``deadlock detected``.

    Tool: threading.Event coordination
    Two threading.Event objects guarantee that Txn1 holds Alice's lock
    before Txn2 tries it, and Txn2 holds Bob's lock before Txn1 tries it.
    This deterministically forces the deadlock interleaving on every run.

Why not DPOR for psycopg2?
    DPOR instruments Python bytecode (LOAD_ATTR / STORE_ATTR).  psycopg2
    is a C extension that drives libpq directly — its execute() and
    fetchone() calls do not go through Python attribute access, so DPOR
    cannot observe or control them.  Use TraceExecutor (line-level
    sys.settrace) to control *when* psycopg2 calls run; use
    threading.Event to coordinate lock-ordering for deadlock scenarios.

Running::

    python examples/postgres_races.py

Requirements:
    pip install psycopg2-binary
    # Postgres 16 with a frontrun_test database:
    createuser -s frontrun_test
    createdb -O frontrun_test frontrun_test
    psql -c "ALTER USER frontrun_test WITH PASSWORD 'frontrun_test';"
"""

from __future__ import annotations

import threading

import psycopg2

from frontrun.bytecode import explore_interleavings
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor

_PG_DSN = "dbname=frontrun_test user=frontrun_test password=frontrun_test host=localhost"
_SEP = "=" * 70


# ---------------------------------------------------------------------------
# Shared setup helper
# ---------------------------------------------------------------------------


def _pg_setup(conn: psycopg2.extensions.connection) -> None:
    """(Re-)create the accounts table and seed two rows."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS accounts")
        cur.execute("CREATE TABLE accounts (name TEXT PRIMARY KEY, balance INT)")
        cur.execute("INSERT INTO accounts VALUES ('alice', 1000), ('bob', 500)")
    conn.commit()


# ============================================================================
# Race 1: Lost Update (READ COMMITTED read–write data race)
# ============================================================================
#
# SQL pattern executed by each transaction (READ COMMITTED, no FOR UPDATE):
#
#   BEGIN;
#   SELECT balance FROM accounts WHERE name='alice';   -- snapshot read
#   -- application code: new_balance = old_balance + amount
#   UPDATE accounts SET balance = <new_balance> WHERE name='alice';
#   COMMIT;
#
# Race window: both transactions execute their SELECT before either runs
# their UPDATE.  Both read the same old balance (1000).  The second COMMIT
# overwrites the first: one credit is lost.
#
# Expected: 1000 + 100 + 200 = 1300
# Actual (after race): 1200  (Txn B overwrote Txn A)
#
# frontrun approach
# -----------------
# TraceExecutor uses sys.settrace to watch for inline ``# frontrun: <marker>``
# comments.  A ``Step(thread, marker)`` in the schedule means "run <thread>
# until it is about to execute the line tagged <marker>, then pause".
#
# Schedule that triggers the lost update:
#   Step("txn_a", "pg_select")   → Txn A paused just before its SELECT
#   Step("txn_b", "pg_select")   → Txn B paused just before its SELECT
#   Step("txn_a", "pg_update")   → Txn A runs SELECT (reads 1000), paused before UPDATE
#   Step("txn_b", "pg_update")   → Txn B runs SELECT (reads 1000), paused before UPDATE
#   (schedule exhausted)         → both run UPDATE + COMMIT freely
#
# Txn A updates to 1100 and commits; Txn B updates to 1200 and commits,
# overwriting Txn A's result.  Final balance: 1200, not 1300.
#
# Fix: use SELECT … FOR UPDATE to hold a row lock, or raise isolation to
# REPEATABLE READ or SERIALIZABLE.


def demo_lost_update() -> None:
    """Reproduce a READ COMMITTED lost update against real Postgres."""
    print(_SEP)
    print("Race 1: Lost Update  (TraceExecutor + psycopg2 + Postgres)")
    print(_SEP)
    print()
    print("  SQL pattern (READ COMMITTED, no FOR UPDATE):")
    print("    Txn A: BEGIN; SELECT balance; -- compute 1000+100=1100")
    print("                  UPDATE balance=1100; COMMIT;")
    print("    Txn B: BEGIN; SELECT balance; -- compute 1000+200=1200")
    print("                  UPDATE balance=1200; COMMIT;  ← overwrites Txn A")
    print()
    print("  Initial balance : 1000")
    print("  Expected result : 1000 + 100 + 200 = 1300")
    print()

    conn_a = psycopg2.connect(_PG_DSN)
    conn_b = psycopg2.connect(_PG_DSN)
    conn_a.autocommit = False
    conn_b.autocommit = False
    _pg_setup(conn_a)

    results: dict[str, int] = {}
    errors: dict[str, str] = {}

    def txn_a() -> None:
        """Txn A: credit +100 to Alice (SELECT then UPDATE, no row lock)."""
        try:
            with conn_a.cursor() as cur:
                cur.execute("SELECT balance FROM accounts WHERE name='alice'")  # frontrun: pg_select
                old = cur.fetchone()[0]
                new = old + 100
            with conn_a.cursor() as cur2:
                cur2.execute("UPDATE accounts SET balance=%s WHERE name='alice'", (new,))  # frontrun: pg_update
            conn_a.commit()
            with conn_a.cursor() as cur3:
                cur3.execute("SELECT balance FROM accounts WHERE name='alice'")
                results["txn_a"] = cur3.fetchone()[0]
        except Exception as exc:
            errors["txn_a"] = str(exc).strip()
            conn_a.rollback()

    def txn_b() -> None:
        """Txn B: credit +200 to Alice (SELECT then UPDATE, no row lock)."""
        try:
            with conn_b.cursor() as cur:
                cur.execute("SELECT balance FROM accounts WHERE name='alice'")  # frontrun: pg_select
                old = cur.fetchone()[0]
                new = old + 200
            with conn_b.cursor() as cur2:
                cur2.execute("UPDATE accounts SET balance=%s WHERE name='alice'", (new,))  # frontrun: pg_update
            conn_b.commit()
            with conn_b.cursor() as cur3:
                cur3.execute("SELECT balance FROM accounts WHERE name='alice'")
                results["txn_b"] = cur3.fetchone()[0]
        except Exception as exc:
            errors["txn_b"] = str(exc).strip()
            conn_b.rollback()

    # The schedule forces: both SELECTs run before either UPDATE.
    schedule = Schedule(
        [
            Step("txn_a", "pg_select"),  # pause Txn A just before SELECT
            Step("txn_b", "pg_select"),  # pause Txn B just before SELECT
            Step("txn_a", "pg_update"),  # Txn A runs SELECT (sees 1000), pause before UPDATE
            Step("txn_b", "pg_update"),  # Txn B runs SELECT (sees 1000), pause before UPDATE
            # (schedule done) → both run UPDATE + COMMIT freely
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("txn_a", txn_a)
    executor.run("txn_b", txn_b)
    executor.wait(timeout=10.0)

    # Read authoritative final balance
    with conn_a.cursor() as cur:
        cur.execute("SELECT balance FROM accounts WHERE name='alice'")
        final = cur.fetchone()[0]

    conn_a.close()
    conn_b.close()

    if errors:
        for name, msg in errors.items():
            print(f"  {name} error: {msg}")
        print()

    print("  Initial balance                          : 1000")
    print(f"  Txn B's read (after its own UPDATE)      : {results.get('txn_b', 'N/A')}")
    print(f"  Actual final balance in DB               : {final}  (expected 1300)")
    print()
    if final != 1300:
        print(f"  LOST UPDATE confirmed: balance is {final}, not 1300.")
        print("  Txn B overwrote Txn A's update.")
        print("  Both transactions read 1000 before either wrote back.")
        print()
        print("  Reproducibility: 100% — the Schedule deterministically forces")
        print("  both SELECTs to run before either UPDATE on every execution.")
    else:
        print("  No lost update observed (both updates applied correctly).")
    print()


# ============================================================================
# Race 1b: Lost Update — random bytecode-level interleaving (bytecode fuzzing)
# ============================================================================
#
# explore_interleavings generates random opcode-level schedules and runs both
# psycopg2 transactions under controlled interleaving.  At each Python bytecode
# instruction, the scheduler decides which thread executes next.
#
# psycopg2 releases the GIL during network I/O (libpq sends the query and
# waits for Postgres over a C-level socket).  While one thread is blocked in
# that C call, the other can execute Python opcodes — including its own
# cur.execute(SELECT).  This gives the scheduler real interleaving leverage
# across the SELECT/UPDATE race window, even though psycopg2 is a C extension.
#
# The invariant is checked by querying Postgres after both transactions commit.
# The setup resets the row to balance=1000 before each attempt using an
# autocommit admin connection.  The two transaction connections are reused
# across all attempts to avoid the cost of creating 2×N connections.
#
# Why not DPOR here?
#   DPOR tracks LOAD_ATTR / STORE_ATTR on Python objects.  psycopg2's execute()
#   and fetchone() are C calls that bypass Python attribute access — DPOR has
#   no events to drive backtracking from.  Random exploration (this function)
#   and trace markers (demo_lost_update above) both work; DPOR does not.


def demo_lost_update_bytecode_fuzz() -> None:
    """Find the lost-update race with random bytecode-level interleaving."""
    print(_SEP)
    print("Race 1b: Lost Update  (bytecode fuzzing — explore_interleavings + psycopg2)")
    print(_SEP)
    print()
    print("  Generating random opcode-level schedules and running both psycopg2")
    print("  transactions under controlled interleaving.  Checking final balance")
    print("  after each attempt.  Expected 1300; lost update produces 1200.")
    print()

    # Admin connection (autocommit) for data reset between attempts and for
    # the invariant check.  Two persistent transaction connections are reused
    # across all attempts so we pay the connection-creation cost only once.
    admin = psycopg2.connect(_PG_DSN)
    admin.autocommit = True
    conn_a = psycopg2.connect(_PG_DSN)
    conn_b = psycopg2.connect(_PG_DSN)
    conn_a.autocommit = False
    conn_b.autocommit = False

    # One-time table setup via the admin connection.
    with admin.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS accounts")
        cur.execute("CREATE TABLE accounts (name TEXT PRIMARY KEY, balance INT)")
        cur.execute("INSERT INTO accounts VALUES ('alice', 1000)")

    class _State:
        """Per-attempt state: reset the row, expose the reused connections."""

        def __init__(self) -> None:
            # Roll back any open transaction left by a previous attempt,
            # then reset the balance to 1000 for a clean slate.
            conn_a.rollback()
            conn_b.rollback()
            with admin.cursor() as cur:
                cur.execute("UPDATE accounts SET balance=1000 WHERE name='alice'")
            self.conn_a = conn_a
            self.conn_b = conn_b

    def _txn(conn: psycopg2.extensions.connection, amount: int) -> None:
        """READ COMMITTED credit: SELECT balance, add amount, UPDATE — no row lock."""
        with conn.cursor() as cur:
            cur.execute("SELECT balance FROM accounts WHERE name='alice'")
            old = cur.fetchone()[0]
            new = old + amount
        with conn.cursor() as cur2:
            cur2.execute("UPDATE accounts SET balance=%s WHERE name='alice'", (new,))
        conn.commit()

    def _invariant(state: _State) -> bool:
        with admin.cursor() as cur:
            cur.execute("SELECT balance FROM accounts WHERE name='alice'")
            return cur.fetchone()[0] == 1300  # type: ignore[no-any-return]

    result = explore_interleavings(
        setup=_State,
        threads=[
            lambda s: _txn(s.conn_a, 100),
            lambda s: _txn(s.conn_b, 200),
        ],
        invariant=_invariant,
        max_attempts=50,
        seed=42,
        detect_io=False,  # psycopg2 uses C-level sockets; Python socket patch is irrelevant
        deadlock_timeout=15.0,  # generous timeout — psycopg2 holds the GIL during C calls
        timeout_per_run=60.0,
        reproduce_on_failure=5,
    )

    admin.close()
    conn_a.close()
    conn_b.close()

    print(f"  property_holds      : {result.property_holds}")
    print(f"  attempts_explored   : {result.num_explored}")
    if result.counterexample is not None:
        repro = ""
        if result.reproduction_attempts:
            repro = f"  ({result.reproduction_successes}/{result.reproduction_attempts} reproductions)"
        print(f"  counterexample found after {result.num_explored} attempt(s){repro}")
    print()
    if result.explanation:
        for line in result.explanation.splitlines():
            print("  " + line)
        print()
    if not result.property_holds:
        print("  LOST UPDATE confirmed via bytecode fuzzing.")
        print("  The random schedule interleaved both SELECTs before either UPDATE.")
    else:
        print("  No lost update found in the explored interleavings.")
    print()


# ============================================================================
# Race 2: Deadlock (circular row-lock ordering)
# ============================================================================
#
# SQL pattern executed by each transaction (SELECT … FOR UPDATE):
#
#   -- Transaction 1 (alice → bob)
#   BEGIN;
#   SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock alice
#   SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock bob
#   UPDATE accounts SET balance = balance - 100 WHERE name='alice';
#   UPDATE accounts SET balance = balance + 100 WHERE name='bob';
#   COMMIT;
#
#   -- Transaction 2 (bob → alice) — OPPOSITE lock order
#   BEGIN;
#   SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock bob
#   SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock alice ← DEADLOCK!
#   ...
#
# Postgres detects the cycle and raises:
#   ERROR:  deadlock detected
#   DETAIL: Process X waits for ShareLock on transaction Y; blocked by Z.
#
# frontrun approach
# -----------------
# Two threading.Event objects guarantee the deadlock-causing ordering:
#
#   1. Txn1 locks Alice → sets alice_locked event
#   2. Txn2 waits for alice_locked, then locks Bob → sets bob_locked event
#   3. Txn1 waits for bob_locked, then tries to lock Bob  ← BLOCKED (Txn2 holds it)
#   4. Txn2 tries to lock Alice                           ← BLOCKED (Txn1 holds it) → DEADLOCK
#
# Postgres detects the cycle and aborts one transaction with "deadlock detected".
#
# Fix: always acquire row locks in a globally consistent order, e.g.:
#   SELECT * FROM accounts WHERE name = ANY(ARRAY['alice','bob'])
#   ORDER BY name FOR UPDATE;   -- alphabetical: alice before bob, always


def demo_deadlock() -> None:
    """Trigger a real Postgres deadlock and capture the error message."""
    print(_SEP)
    print("Race 2: Deadlock  (threading.Event coordination + psycopg2 + Postgres)")
    print(_SEP)
    print()
    print("  SQL pattern (FOR UPDATE, opposite lock order):")
    print("    Txn1: LOCK alice → LOCK bob   → transfer alice→bob ($100)")
    print("    Txn2: LOCK bob   → LOCK alice → transfer bob→alice ($50)")
    print()
    print("  Interleaving forced by threading.Event coordination:")
    print("    Step 1: Txn1 acquires alice row-lock  (SELECT … FOR UPDATE)")
    print("    Step 2: Txn2 acquires bob   row-lock  (SELECT … FOR UPDATE)")
    print("    Step 3: Txn1 tries bob   row-lock  ← BLOCKED (held by Txn2)")
    print("    Step 4: Txn2 tries alice row-lock  ← BLOCKED (held by Txn1) → DEADLOCK")
    print()

    conn1 = psycopg2.connect(_PG_DSN)
    conn2 = psycopg2.connect(_PG_DSN)
    conn1.autocommit = False
    conn2.autocommit = False
    _pg_setup(conn1)

    errors: dict[str, str] = {}
    alice_locked = threading.Event()  # Txn1 sets after locking alice
    bob_locked = threading.Event()  # Txn2 sets after locking bob

    def txn1() -> None:
        """Txn1: alice → bob transfer.  Locks alice first, then bob."""
        with conn1.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='alice' FOR UPDATE")
        alice_locked.set()  # signal: alice row lock held by Txn1
        bob_locked.wait(timeout=5.0)  # wait: ensure Txn2 holds bob before we try it
        with conn1.cursor() as cur:
            try:
                cur.execute("SELECT * FROM accounts WHERE name='bob' FOR UPDATE")
                cur.execute("UPDATE accounts SET balance = balance - 100 WHERE name='alice'")
                cur.execute("UPDATE accounts SET balance = balance + 100 WHERE name='bob'")
                conn1.commit()
            except Exception as exc:
                errors["txn1"] = str(exc).strip()
                conn1.rollback()

    def txn2() -> None:
        """Txn2: bob → alice transfer.  Locks bob first, then alice (OPPOSITE order)."""
        alice_locked.wait(timeout=5.0)  # wait: ensure Txn1 holds alice before we lock bob
        with conn2.cursor() as cur:
            cur.execute("SELECT * FROM accounts WHERE name='bob' FOR UPDATE")
        bob_locked.set()  # signal: bob row lock held by Txn2; Txn1 will now try bob → BLOCKED
        with conn2.cursor() as cur:
            try:
                cur.execute("SELECT * FROM accounts WHERE name='alice' FOR UPDATE")
                cur.execute("UPDATE accounts SET balance = balance - 50 WHERE name='bob'")
                cur.execute("UPDATE accounts SET balance = balance + 50 WHERE name='alice'")
                conn2.commit()
            except Exception as exc:
                errors["txn2"] = str(exc).strip()
                conn2.rollback()

    t1 = threading.Thread(target=txn1, name="txn1", daemon=True)
    t2 = threading.Thread(target=txn2, name="txn2", daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    conn1.close()
    conn2.close()

    if errors:
        for txn_name, msg in errors.items():
            print(f"  {txn_name} received Postgres error:")
            for line in msg.splitlines():
                print(f"    {line}")
        print()
        print("  DEADLOCK confirmed: Postgres aborted one transaction and rolled")
        print("  it back, freeing the other transaction to complete.")
        print()
        print("  Reproducibility: 100% — the threading.Event coordination")
        print("  guarantees the deadlock-causing lock-ordering on every run.")
    else:
        print("  Both transactions completed without a deadlock error.")
        print("  (Check that the threading.Event coordination is working correctly.)")
    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    demo_lost_update()
    demo_lost_update_bytecode_fuzz()
    demo_deadlock()
