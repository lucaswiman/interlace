SQL Conflict Detection -- Technical Details
============================================

Frontrun's SQL conflict detection intercepts ``cursor.execute()`` calls,
parses the SQL, and reports per-table (or per-row) resource IDs to the
DPOR engine.  This replaces the coarse endpoint-level detection where
all SQL to the same ``(host, port)`` collapses to a single conflict
point.

This document walks through the internal data flow, the refinement
hierarchy that keeps the system sound, formal verification with TLA+,
the design decision not to use z3/SMT solvers, and handling of
PostgreSQL isolation levels.


How a SQL Statement Becomes a DPOR Conflict
-------------------------------------------

Consider two threads running against the same PostgreSQL database.
Thread A executes:

.. code-block:: sql

   SELECT * FROM accounts WHERE id = 1;

Thread B executes:

.. code-block:: sql

   UPDATE accounts SET balance = 0 WHERE id = 2;

Without SQL-level detection, both statements produce ``send()``/``recv()``
on the same socket (``localhost:5432``).  DPOR sees a write-write conflict
on a single ``ObjectId`` and explores O(n!) interleavings -- none of which
can actually race, because the two statements touch different rows on
different tables.

With SQL detection enabled, the data flows through six stages:

**Stage 1 -- SQL parsing** (``_sql_parsing.py``).
``parse_sql_access(sql)`` returns ``(read_tables, write_tables,
lock_intent, tx_op)``.  A regex fast-path handles ~90% of ORM-generated
SQL (single-table SELECT/INSERT/UPDATE/DELETE).  Complex SQL (CTEs,
subqueries, UNION, MERGE) falls back to sqlglot for full AST analysis.
Transaction control statements (BEGIN, COMMIT, ROLLBACK, SAVEPOINT) are
detected and returned as ``tx_op`` with no table sets.

**Stage 2 -- Parameter resolution** (``_sql_params.py``).
ORM queries are parameterized (``WHERE id = %s``), so sqlglot sees
placeholders, not values.  ``resolve_parameters()`` substitutes the
actual Python values before AST analysis.  All five PEP 249 paramstyles
are supported: qmark (``?``), numeric (``:1``), named (``:name``),
format (``%s``), and pyformat (``%(name)s``).  The resolved SQL is only
used for predicate extraction -- it is never executed.

**Stage 3 -- Predicate extraction** (``_sql_predicates.py``).
``extract_equality_predicates()`` parses the WHERE clause and extracts
equality and IN-list predicates on primary key columns:

.. code-block:: python

   # Input (after parameter resolution):
   "SELECT * FROM accounts WHERE id = 42 AND region = 'us-east'"

   # Output:
   [EqualityPredicate("id", "42"), EqualityPredicate("region", "us-east")]

When both operations have complete PK predicates, disjointness is
checked with ``frozenset.isdisjoint()`` -- O(k) for k PK columns.

**Stage 4 -- ObjectId derivation**.
The resource ID is a string like ``"sql:accounts"`` (table-level) or
``"sql:accounts:(('id','42'),)"`` (row-level).  This string is hashed
via the existing ``_make_object_key()`` to produce a ``u64`` ObjectId
for the Rust DPOR engine.  No Rust changes are needed -- SQL tables are
just I/O objects with table-derived ObjectIds.

**Stage 5 -- Access kind mapping**.
SELECT maps to ``AccessKind::Read``.  INSERT maps to ``Write``.  UPDATE
and DELETE map to both ``Read`` and ``Write`` (conservative).  SELECT
FOR UPDATE maps to ``Write``.  The Rust engine's
``ObjectState::dependent_accesses`` implements the standard conflict
rules: Read-Read is independent, everything else (RW, WR, WW) conflicts.

**Stage 6 -- Endpoint suppression** (``_sql_cursor.py``).
While the original ``cursor.execute()`` runs, a context manager sets
``_sql_suppress = True`` in thread-local storage and adds the OS thread
ID to a shared ``_suppress_tids`` set.  This prevents the coarser
socket-level detection from double-reporting the same operation --
both in Python (``_report_socket_io`` checks TLS) and in the LD_PRELOAD
bridge (``_PreloadBridge.listener`` checks the shared set).


The Refinement Hierarchy
------------------------

The system maintains a strict refinement chain:

.. code-block:: text

   Row-level  <  Table-level  <  Endpoint-level
   (finest)                      (coarsest)

Each layer is a *sound refinement* of the one below:

- **Endpoint-level** treats all SQL to the same host:port as conflicting.
  Every pair of operations is a conflict.  This is trivially sound but
  explores many unnecessary interleavings.

- **Table-level** parses SQL to extract table names.  Operations on
  different tables are independent.  Same-table operations use the
  Adya Read/Write classification.  This is sound because different-table
  operations genuinely cannot conflict at the SQL level.

- **Row-level** (when both operations have complete PK predicates)
  treats same-table, different-PK operations as independent.  This is
  sound because equality predicates on the full primary key select
  provably disjoint rows.

When parsing fails, the system falls back to the next coarser layer.
Parse failure returns empty table sets, so endpoint-level suppression
never fires and the socket-level detection remains in effect.  When
parameter resolution fails, row-level falls back to table-level.  The
key property: **fallback is always in the conservative direction.**


Transaction Grouping
--------------------

SQL operations within a ``BEGIN`` ... ``COMMIT`` block are grouped
atomically.  When ``_intercept_execute()`` sees ``BEGIN``, it sets
``_in_transaction = True`` and starts buffering I/O reports in
``_tx_buffer``.  Subsequent SQL operations append to the buffer instead
of reporting immediately.  At ``COMMIT``, the buffer is flushed to the
reporter.  At ``ROLLBACK``, the buffer is discarded.  ``SAVEPOINT`` and
``ROLLBACK TO`` manage partial rollback via buffer truncation.

During a transaction, the DPOR scheduler skips scheduling (does not
yield to other threads), ensuring all SQL operations within the
transaction appear atomic.  This prevents false positives from
intermediate transaction states that would be rolled back on failure.


Concrete Example: SELECT-then-UPDATE Lost Update
-------------------------------------------------

This walkthrough traces exactly what happens when two threads race on
the same row, showing both the buggy case and the corrected version.

**Setup:** Two threads each increment a counter stored in a database row.

.. code-block:: python

   # Thread A and Thread B both execute this:
   def handle_login(engine):
       with Session(engine) as session:
           user = session.get(User, 1)                     # SELECT * FROM users WHERE id = 1
           user.login_count = user.login_count + 1          # Python-side increment
           session.commit()                                 # UPDATE users SET login_count = <n> WHERE id = 1; COMMIT

**The buggy interleaving:**

.. code-block:: text

   Thread A: BEGIN
   Thread A: SELECT * FROM users WHERE id = 1     -> login_count = 0
   Thread B: BEGIN
   Thread B: SELECT * FROM users WHERE id = 1     -> login_count = 0
   Thread A: UPDATE users SET login_count = 1 WHERE id = 1
   Thread A: COMMIT                                -- login_count is now 1
   Thread B: UPDATE users SET login_count = 1 WHERE id = 1
   Thread B: COMMIT                                -- login_count is still 1, not 2!

**What DPOR sees internally:**

1. Thread A's ``session.get(User, 1)`` triggers ``cursor.execute("SELECT
   * FROM users WHERE users.id = %(pk)s", {"pk": 1})``.

2. ``_intercept_execute()`` calls ``parse_sql_access(...)`` which returns
   ``(read_tables={"users"}, write_tables=set(), lock_intent=None,
   tx_op=None)``.

3. Parameter resolution converts ``%(pk)s`` to ``1``, producing
   ``"SELECT * FROM users WHERE users.id = 1"``.

4. Predicate extraction yields ``[EqualityPredicate("id", "1")]``.

5. The reporter is called with
   ``("sql:users:(('id', '1'),)", "read")``.

6. Thread B's identical SELECT reports the same resource ID with
   ``"read"`` -- the DPOR engine sees Read-Read on the same ObjectId,
   which is **independent** (no conflict).  This is correct: two
   concurrent SELECTs on the same row cannot conflict.

7. Thread A's ``session.commit()`` triggers
   ``UPDATE users SET login_count = 1 WHERE users.id = 1``, which reports
   ``("sql:users:(('id', '1'),)", "write")``.

8. Now the engine sees a Read-Write conflict between Thread B's prior
   read and Thread A's write on the same ObjectId.  DPOR explores the
   alternative ordering.

**The fix -- SELECT FOR UPDATE:**

.. code-block:: python

   def handle_login_safe(engine):
       with Session(engine) as session:
           user = session.execute(
               select(User).where(User.id == 1).with_for_update()
           ).scalar_one()                                   # SELECT ... FOR UPDATE
           user.login_count = user.login_count + 1
           session.commit()

.. code-block:: text

   Thread A: BEGIN
   Thread A: SELECT * FROM users WHERE id = 1 FOR UPDATE   -> login_count = 0 (row locked)
   Thread B: BEGIN
   Thread B: SELECT * FROM users WHERE id = 1 FOR UPDATE   -> BLOCKED (waits for row lock)
   Thread A: UPDATE users SET login_count = 1 WHERE id = 1
   Thread A: COMMIT                                         -- login_count = 1, lock released
   Thread B: (unblocked) login_count = 1                    -> reads current value
   Thread B: UPDATE users SET login_count = 2 WHERE id = 1
   Thread B: COMMIT                                         -- login_count = 2, correct!

**What changes in DPOR's view:**

When ``parse_sql_access()`` sees ``FOR UPDATE``, it returns
``lock_intent="UPDATE"``.  The ``_intercept_execute()`` function then
reports the SELECT as a **write** instead of a read:
``("sql:users:(('id', '1'),)", "write")``.  Now both Thread A's and
Thread B's initial statements are writes on the same ObjectId -- a
Write-Write conflict -- so DPOR immediately explores both orderings.

This correctly models the database's behavior: ``SELECT FOR UPDATE``
acquires a row-level exclusive lock, serializing access to the row.


Anomaly Classification
----------------------

When DPOR finds a failing interleaving involving SQL, the anomaly
classifier (``_sql_anomaly.py``) identifies the specific isolation
anomaly.  It builds a Dependency Serialization Graph (DSG) from the SQL
trace events and classifies cycles by their edge types:

- **Lost update** -- two threads both read then write the same resource
- **Write skew** -- RW-only cycle across different tables
- **Dirty read** -- WR edge (thread reads uncommitted data from another)
- **Non-repeatable read** -- same thread reads the same resource twice
  with another thread's write in between
- **Phantom read** -- same thread reads a table twice with another
  thread's INSERT/DELETE in between
- **Write-write** -- WW cycle (concurrent writes without coordination)


PostgreSQL Isolation Levels
---------------------------

Frontrun's SQL conflict detection **does not model PostgreSQL isolation
levels** (READ COMMITTED, REPEATABLE READ, SERIALIZABLE), and this is
deliberate.

**Why not?** The DPOR engine operates at the *logical scheduling* level:
it controls which thread runs next and observes which shared resources
each thread touches.  It does not simulate the database's internal
concurrency control.  The actual isolation semantics are enforced by
PostgreSQL itself during each execution.

This means:

1. **Frontrun finds bugs that exist under the configured isolation
   level.**  If your database runs at READ COMMITTED (the PostgreSQL
   default), frontrun explores interleavings that are possible under
   READ COMMITTED.  The invariant checks real database state after real
   SQL execution against the real database, so any anomaly frontrun
   reports actually happened.

2. **Frontrun does not predict which anomalies would or would not occur
   under a different isolation level.**  It does not have a model of
   PostgreSQL's MVCC, snapshot isolation, or SSI internals.

3. **The anomaly classifier labels what happened, not what's
   theoretically possible.**  When frontrun classifies an anomaly as
   "lost update" or "write skew", it is describing the pattern of
   operations in the *actual failing interleaving* -- not making claims
   about whether PostgreSQL's isolation level permits or prevents that
   pattern in general.

This design is correct for a testing tool: the goal is to find bugs in
*your code* (missing locks, incorrect transaction boundaries, stale
reads), not to verify PostgreSQL's isolation implementation.  Your code
runs at a specific isolation level, and frontrun tests whether your code
is correct at that level by trying all meaningful interleavings.

If you want to test behavior at different isolation levels, change the
database configuration and re-run frontrun.  Each run tests the code
against the configured level.


Formal Verification with TLA+
------------------------------

Three TLA+ specifications in ``specs/`` verify the core correctness
properties.  All pass exhaustive model checking via TLC.

Spec 1: SqlConflictRefinement (Static Conflict Rules)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exhaustively checks all combinations of 2 operations x 4 statement
types x 2 tables x 2 PK values x {parsed, unparsed} x
{params_resolved, params_unresolved} (3,200 states).

Thirteen invariants verify:

.. list-table::
   :widths: 5 25 50
   :header-rows: 1

   * - #
     - Invariant
     - What it checks
   * - 1
     - ``TableLevelSoundness``
     - Same table + at least one write -> table conflict is TRUE
   * - 2
     - ``TableRefinesEndpoint``
     - Table conflict -> endpoint conflict (refinement direction)
   * - 3
     - ``RowRefinesTable``
     - Row conflict -> table conflict (refinement chain)
   * - 4
     - ``RowRefinesEndpoint``
     - Row conflict -> endpoint conflict (transitivity)
   * - 5
     - ``ParseFailureFallback``
     - Parse failure -> both table and row report conflict
   * - 6
     - ``ReadReadIndependence``
     - SELECT + SELECT + parsed -> no table conflict
   * - 7
     - ``DifferentTablesIndependent``
     - Different tables + parsed -> no table conflict
   * - 8
     - ``SameRowConflict``
     - Same table + same PK + write -> row conflict
   * - 9
     - ``DifferentRowsIndependent``
     - Same table + different PK + parsed -> no row conflict
   * - 10
     - ``SuppressionCorrectness``
     - Parsed operation always has non-empty access kinds
   * - 11
     - ``ConflictSymmetry``
     - Conflict is symmetric in both arguments
   * - 12
     - ``UnresolvedParamsFallback``
     - Unresolved params + same table + write -> row conflict (conservative)
   * - 13
     - ``ResolvedParamsEnableRowLevel``
     - Resolved params + different PK -> no row conflict (precise)

Invariants 12--13 verify parameter resolution (Algorithm 1.5): failed
resolution never claims row-independence, while successful resolution
enables finer-grained detection.

Spec 2: DporSqlScheduling (Dynamic Scheduling Soundness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models an actual database as a map from ``(table, pk) -> value``.
Applies every pair of operations in both orders (A;B vs B;A) and
verifies that the conflict detector is sound: if it says "no conflict",
then both orders produce the same database state.

Seven invariants, checked across 2,048 states:

.. list-table::
   :widths: 5 25 50
   :header-rows: 1

   * - #
     - Invariant
     - What it checks
   * - 1
     - ``TableSoundness``
     - Table conflict FALSE -> both orders produce same DB state
   * - 2
     - ``RowSoundness``
     - Row conflict FALSE -> both orders produce same DB state
   * - 3
     - ``RefinementChain``
     - Row conflicts <= table conflicts <= endpoint conflicts
   * - 4
     - ``NoMissedBugs_Table``
     - Different DB states -> table conflict TRUE
   * - 5
     - ``NoMissedBugs_Row``
     - Different DB states -> row conflict TRUE
   * - 6
     - ``UnresolvedParamsSoundness``
     - Unresolved + row conflict FALSE -> orders commute
   * - 7
     - ``ResolvedParamsSoundness``
     - Resolved + row conflict FALSE -> orders commute

**``NoMissedBugs`` is the strongest result.** For any two operations
where execution order matters (different DB states), the detector
reports a conflict.  This guarantees DPOR will explore the interleaving.

Spec 3: SuppressionSafety (Endpoint Suppression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models the suppression mechanism where SQL-level reporting disables
endpoint-level reporting.  Non-deterministically generates *arbitrary*
parser outputs (including incorrect ones) and verifies safety across
32,768 states:

1. ``SuppressionSafe`` -- correct parser -> ground-truth conflict is
   detected
2. ``ParseFailureConservative`` -- parse failure -> suppression doesn't
   fire -> endpoint-level always conflicts
3. ``EmptyParseNoSuppress`` -- parsed but no tables (e.g. BEGIN/COMMIT)
   -> no suppression

Running the Specs
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd specs/
   curl -sL -o tla2tools.jar \
     https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

   java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC SqlConflictRefinement -deadlock -nowarning
   java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC DporSqlScheduling -deadlock -nowarning
   java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC SuppressionSafety -deadlock -nowarning


Why Not z3/SMT for Row-Level Conflicts
---------------------------------------

The decision to use value comparison and set disjointness rather than
an SMT solver (z3) for row-level conflict detection is deliberate:

**Coverage.** ORM-generated SQL is overwhelmingly equality lookups
(``WHERE id = ?``) and IN-lists (``WHERE id IN (?, ?, ?)``).  These
cover ~95% of real-world row-level queries.  Simple set operations
handle both correctly.

**Hot-path performance.** ``_intercept_execute`` runs on *every* SQL
statement.  z3 adds ~50 ms per satisfiability check versus nanoseconds
for ``frozenset.isdisjoint()``.  At scale (hundreds of SQL statements
per test), this would dominate execution time.

**Dependency weight.** ``z3-solver`` is ~200 MB.  For a testing library
this is disproportionate.  It would need to be optional, meaning the
equality/IN-list path would still exist as a fallback -- adding
complexity without deleting code.

**Encoding complexity.** Translating SQL types (VARCHAR, DECIMAL,
timestamps, NULL three-valued logic) into z3 sorts is non-trivial and
error-prone.  SQL NULL semantics alone (``NULL = NULL`` is ``NULL``, not
``TRUE``) require careful handling that is easy to get wrong.

**Safe fallback.** Unhandled predicates (ranges, OR, BETWEEN,
subqueries) fall back to table-level conflict detection.  This is
conservative (may explore unnecessary interleavings) but never misses
real conflicts.

If range predicate support becomes important, a lightweight
interval-arithmetic approach (no z3) could handle
``WHERE id > X AND id < Y`` vs ``WHERE id > Z`` with simple numeric
comparison.

The disjointness check is O(k) where k is the number of PK columns:

.. code-block:: python

   def pk_predicates_disjoint(preds_a, preds_b) -> bool:
       a_map = {p.column: values(p) for p in preds_a}
       b_map = {p.column: values(p) for p in preds_b}
       for col in a_map:
           if col in b_map and a_map[col].isdisjoint(b_map[col]):
               return True
       return False


Wire Protocol Parsing (C-Level Drivers)
---------------------------------------

For C-extension drivers like psycopg2/libpq that call ``send()``
directly (bypassing Python's DBAPI layer), the LD_PRELOAD library
includes a Rust-based PostgreSQL wire protocol parser
(``crates/io/src/sql_extract.rs``).

It recognizes two PostgreSQL message types:

- **Simple Query** (``'Q'``) -- message type byte + i32 length +
  null-terminated SQL string
- **Extended Query / Parse** (``'P'``) -- message type byte + i32
  length + statement name + query string + parameter info

When the ``send()`` hook intercepts a buffer matching these patterns,
it extracts the SQL text and writes an enriched event to the pipe.
The Python-side bridge can then parse the SQL and report at table level,
even for C-extension drivers that never go through ``cursor.execute()``.


References
----------

- Adya, Liskov, O'Neil.
  `Generalized Isolation Level Definitions <http://pmg.csail.mit.edu/papers/icde00.pdf>`_.
  ICDE 2000.
- Berenson et al.
  `A Critique of ANSI SQL Isolation Levels <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf>`_.
  SIGMOD 1995.
- Kingsbury, Alvaro.
  `Elle <https://github.com/jepsen-io/elle>`_.
  VLDB 2021.
- Cui et al.
  `IsoRel <https://dl.acm.org/doi/10.1145/3728953>`_.
  ACM 2025.
- Jiang et al.
  `TxCheck <https://www.usenix.org/system/files/osdi23-jiang.pdf>`_.
  OSDI 2023.
- Mao.
  `sqlglot <https://github.com/tobymao/sqlglot>`_.
