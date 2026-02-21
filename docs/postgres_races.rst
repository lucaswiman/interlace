Postgres Race Conditions
========================

This guide demonstrates two classic database race conditions using frontrun.
It also explains *which* frontrun tool is the right choice for each bug class.

.. code-block:: text

    python examples/postgres_races.py   # run to reproduce the traces below

The full source is in ``examples/postgres_races.py``.


Race 1: Lost Update (data race)
---------------------------------

**What it is**

A *lost update* occurs when two concurrent transactions each read the same row,
compute a new value in application code, and write it back — without holding a
row-level lock.  The second ``UPDATE`` silently overwrites the first, losing one
credit.

**Equivalent SQL pattern** (READ COMMITTED isolation, Postgres default)::

    -- Transaction A
    BEGIN;
    SELECT balance FROM accounts WHERE id = 1;   -- reads 1000
    -- (application: new_balance = 1000 + 100)
    UPDATE accounts SET balance = 1100 WHERE id = 1;
    COMMIT;

    -- Transaction B (concurrent, both read before either writes)
    BEGIN;
    SELECT balance FROM accounts WHERE id = 1;   -- also reads 1000!
    -- (application: new_balance = 1000 + 200)
    UPDATE accounts SET balance = 1200 WHERE id = 1;   -- overwrites A
    COMMIT;

    -- Final balance: 1200  (should be 1300)

**frontrun tool: ``explore_dpor``**

DPOR operates at the Python bytecode level.  Every ``LOAD_ATTR`` (SELECT) and
``STORE_ATTR`` (UPDATE) on a shared attribute is tracked as a read or write.
When two threads conflict (both access the same attribute and at least one
writes), DPOR adds a backtrack point and explores the reversed ordering.  This
discovers the lost-update interleaving in just 2 executions.

**Python model**::

    class PgAccountRow:
        def __init__(self, balance=1000):
            self.balance = balance   # the 'balance' column

    def txn_credit_no_lock(account, amount):
        old = account.balance          # SELECT  → DPOR: read  PgAccountRow.balance
        account.balance = old + amount # UPDATE  → DPOR: write PgAccountRow.balance

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

**Reproduction trace** (exact program output)::

    property_holds         : False
    executions_explored    : 2
    counterexample_schedule: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    Race condition found after 2 interleavings.

      Write-write conflict: threads 0 and 1 both wrote to balance.


      Thread 0 | postgres_races.py:92      old = account.balance  # SELECT  [read PgAccountRow.balance]
      Thread 0 | postgres_races.py:93      account.balance = old + amount  # UPDATE  [write PgAccountRow.balance]
      Thread 1 | postgres_races.py:92      old = account.balance  # SELECT  [read PgAccountRow.balance]
      Thread 1 | postgres_races.py:93      account.balance = old + amount  # UPDATE  [write PgAccountRow.balance]

      Reproduced 10/10 times (100%)

**Reading the trace**

The schedule ``[0,0,…,1,1,…,0,0,…]`` means Thread 0 (Txn A) ran until its
``STORE_ATTR`` for ``balance``, then Thread 1 (Txn B) ran its entire SELECT+UPDATE,
then Thread 0 finished.  Both threads read ``1000`` before either wrote back, so
the second write (Txn B's ``1200``) overwrote Txn A's result (``1100``).

**Reproducibility: 10/10 (100%)**

DPOR records the exact counterexample schedule (a list of thread IDs, one per
bytecode instruction).  Replaying that schedule with ``run_with_schedule``
produces the failure on every attempt.

**Fix**

Use ``SELECT … FOR UPDATE`` to hold a row-level lock, or increase isolation to
``REPEATABLE READ`` or ``SERIALIZABLE``::

    -- Fix: acquire a row lock before reading
    BEGIN;
    SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;
    UPDATE accounts SET balance = balance + 100 WHERE id = 1;
    COMMIT;


Race 2: Deadlock (lock-ordering cycle)
-----------------------------------------

**What it is**

A *deadlock* occurs when two transactions each hold one row-level lock and wait
for the other's lock, creating a circular dependency.

**Equivalent SQL pattern**::

    -- Transaction 1 (alice → bob order)
    BEGIN;
    SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock alice
    SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock bob
    UPDATE accounts SET balance = balance - 100 WHERE name='alice';
    UPDATE accounts SET balance = balance + 100 WHERE name='bob';
    COMMIT;

    -- Transaction 2 (bob → alice order — OPPOSITE!)
    BEGIN;
    SELECT * FROM accounts WHERE name='bob'   FOR UPDATE;  -- lock bob
    SELECT * FROM accounts WHERE name='alice' FOR UPDATE;  -- lock alice ← DEADLOCK!
    ...

Postgres raises::

    ERROR:  deadlock detected
    DETAIL: Process 14580 waits for ShareLock on transaction 755; blocked by process 14581.
    Process 14581 waits for ShareLock on transaction 754; blocked by process 14580.
    HINT:  See server log for query details.
    CONTEXT:  while locking tuple (0,2) in relation "accounts"

**Why DPOR does NOT find this**

DPOR uses *memory-access conflicts* (``LOAD_ATTR``/``STORE_ATTR`` on shared state) to
decide which interleavings to explore.  Lock acquire/release events are reported
as *sync* events that update thread vector clocks; accesses inside a critical
section appear to happen-before accesses in the next critical section.  DPOR
therefore concludes the lock-protected memory accesses have only one meaningful
ordering and stops after 3 executions — all without deadlocking:

.. code-block:: text

    # DPOR attempt on the same deadlock code:
    property_holds      : True
    executions_explored : 3

    DPOR reports no violation — the deadlock was NOT found.

Deadlocks are a *lock-ordering* bug, not a data race.  DPOR's conflict-based
exploration never tries the scheduling (Txn1 holds alice, Txn2 holds bob) that
creates the circular wait.

**frontrun tool: ``TraceExecutor`` (trace markers)**

``TraceExecutor`` accepts an explicit ``Schedule`` that names exactly which thread
runs at each marked point.  Placing ``# frontrun: <marker>`` inline on the lock
acquisition statements lets us control the order::

    def txn1_alice_to_bob(state, amount):
        state.alice.lock.acquire()   # frontrun: txn1_lock_alice
        state.bob.lock.acquire()     # frontrun: txn1_lock_bob   ← deadlock point
        ...

    def txn2_bob_to_alice(state, amount):
        state.bob.lock.acquire()     # frontrun: txn2_lock_bob
        state.alice.lock.acquire()   # frontrun: txn2_lock_alice  ← deadlock point
        ...

    schedule = Schedule([
        Step("txn1", "txn1_lock_alice"),  # Step 1: Txn1 acquires alice.lock
        Step("txn2", "txn2_lock_bob"),    # Step 2: Txn2 acquires bob.lock
        Step("txn1", "txn1_lock_bob"),    # Step 3: Txn1 tries bob.lock (held by Txn2) → BLOCKED
        Step("txn2", "txn2_lock_alice"),  # Step 4: Txn2 tries alice.lock (held by Txn1) → DEADLOCK
    ])

**Reproduction trace** (exact program output)::

    txn1_done          : False
    txn2_done          : False
    Threads still alive: ['txn1', 'txn2']
    TimeoutError       : Threads did not complete within timeout: txn1, txn2

    DEADLOCK CONFIRMED.

    Both threads blocked waiting for each other's lock:
      Txn1 holds alice.lock, waits for bob.lock  (held by Txn2)
      Txn2 holds bob.lock,   waits for alice.lock (held by Txn1)

    frontrun detected this by timing out after 2.0 s, mirroring
    Postgres's behaviour of aborting one deadlocked transaction.

    Reproducibility: 10/10 (100%) — trace markers pin the exact
    lock-acquisition ordering on every run.

**Reading the trace**

Both threads remain alive (``daemon=True``).  Neither completed its body
(``txn1_done`` and ``txn2_done`` are both ``False``).  After 2 seconds,
``executor.wait(timeout=2.0)`` raises ``TimeoutError`` naming the live threads.

**Reproducibility: 10/10 (100%)**

The ``Schedule`` guarantees that Txn1 always acquires ``alice.lock`` before Txn2
acquires ``alice.lock``, and Txn2 always acquires ``bob.lock`` before Txn1 tries
it.  This determinism makes the deadlock inevitable on every run.

**Fix**

Always acquire row locks in a globally consistent order (e.g. sorted by primary
key or name)::

    -- Both transactions lock in alphabetical order: alice first, then bob
    SELECT * FROM accounts WHERE name = ANY(ARRAY['alice','bob'])
    ORDER BY name FOR UPDATE;   -- alice before bob, always


Postgres verification
----------------------

Both races were reproduced against a live Postgres 16 instance using psycopg2.

**Lost update** (``demo_postgres_lost_update``)::

    Initial balance                          : 1000
    Txn B's final read (after its own UPDATE): 1200
    Actual final balance in DB               : 1200  (expected 1300)

    LOST UPDATE confirmed: balance is 1200, not 1300.
    Txn B overwrote Txn A's update (both read 1000 before either wrote).

**Deadlock** (``demo_postgres_deadlock``)::

    txn1 received Postgres error:
      deadlock detected
      DETAIL:  Process 14580 waits for ShareLock on transaction 755; blocked by process 14581.
      Process 14581 waits for ShareLock on transaction 754; blocked by process 14580.
      HINT:  See server log for query details.
      CONTEXT:  while locking tuple (0,2) in relation "accounts"

    DEADLOCK confirmed: Postgres aborted one transaction and
    rolled it back, matching frontrun's timeout-based detection.


Tool selection summary
-----------------------

+-------------------+---------------------+--------------------------------------------+
| Bug class         | Right tool          | Why                                        |
+===================+=====================+============================================+
| Lost update       | ``explore_dpor``    | DPOR tracks attr reads/writes; finds the   |
| (data race)       |                     | conflicting interleaving systematically    |
+-------------------+---------------------+--------------------------------------------+
| Deadlock          | ``TraceExecutor``   | Deadlocks are lock-ordering bugs; trace    |
| (lock ordering)   | (trace markers)     | markers pin the exact deadlock interleaving|
+-------------------+---------------------+--------------------------------------------+

Both races are **100% reproducible** once the triggering interleaving is known:

* For the lost update, DPOR records the counterexample schedule and replays it.
* For the deadlock, the explicit ``Schedule`` deterministically creates the cycle.
