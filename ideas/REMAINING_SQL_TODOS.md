# Remaining SQL TODOs

Outstanding work items for SQL conflict detection. Everything below is
optional refinement -- the core system (table-level, row-level, wire
protocol, anomaly classification, transaction grouping, async drivers,
psycopg3, connection pooling, etc.) is complete and verified.

---

## Medium Priority

### Cross-Table Foreign Key Analysis
Schema introspection to detect FK dependencies between tables.
Currently, `INSERT INTO orders (user_id, …)` and `DELETE FROM users
WHERE id = ?` are classified as independent (different tables), but a
FK constraint creates a real conflict.

**What's needed:**
- Query `information_schema.referential_constraints` (PostgreSQL/MySQL)
  at first connection
- Build FK dependency graph: `{orders -> users, shipments -> orders}`
- At conflict detection: if Op1 touches T1 and Op2 touches T2 with
  T1 -> T2 via FK, mark as dependent
- Manual FK registration via `frontrun/_schema.py` already exists;
  automatic introspection is the remaining piece

**Estimated effort:** ~150 lines + 25 tests

---

## Low Priority

### Stored Procedure Analysis
Intercept `CREATE PROCEDURE`/`CREATE FUNCTION`, parse their bodies,
cache `{sp_name -> {read_tables, write_tables}}`. At `CALL` or
function invocation, use cached access instead of endpoint-level.

Rare in modern Python ORMs -- most code uses direct SQL.

**Estimated effort:** ~200 lines + 40 tests

### Generated & Computed Columns
Schema introspection to identify `GENERATED ALWAYS AS` columns.
Exclude from row-level predicate matching (can't be set by user).
Informational only; minimal impact on conflict detection.

**Estimated effort:** ~30 lines + 5 tests

### Window Function Handling
Recognize `OVER (PARTITION BY ...)` clauses and fall back to
table-level when present (all rows in the partition are
interdependent). Currently safe -- window functions already extract
tables correctly, just miss the partition semantics.

**Estimated effort:** ~20 lines + 3 tests

---

## Future Research / Hard Problems

### Handling Non-Deterministic Resource IDs (Sequences & UUIDs)
DPOR relies on deterministic resource IDs across runs to match events.
If a test inserts a row and gets ID=1 in run A and ID=2 in run B
(because the DB sequence wasn't reset), the resource IDs change:
- Run A: `sql:users:(('id', '1'),)`
- Run B: `sql:users:(('id', '2'),)`

The DPOR scheduler will fail to map the events from Run A to Run B,
breaking replay and exploration. This also applies to randomly generated
UUIDs.

**Approaches:**

1.  **Enforced Determinism (Recommended):**
    Document that users *must* reset sequences (`TRUNCATE ... RESTART IDENTITY`
    in PostgreSQL) and seed PRNGs for UUIDs. This is standard practice for
    reproducible testing but burdens the user.

2.  **Relative/Offset IDs:**
    At the start of a test, query the current sequence value ($S_0$).
    For all subsequent operations, report IDs as $ID - S_0$.
    *Risks:* Gaps in sequences, pre-existing data, multiple sequences per table.

3.  **Automatic Detection (Strict Mode):**
    DPOR can detect when a resource ID (e.g., `sql:users:(('id', '1'),)`)
    changes between runs for the same logical step in the execution trace.
    If detected, throw an informative error message explaining that the
    test is non-deterministic and needs sequence resetting or PRNG seeding.
    This can be disabled via `explore_dpor(..., strict_determinism=False)`.

4.  **Symbolic ID Tracking:**
    Intercept the `INSERT` that generates the ID, assign it a symbolic name
    (`$sym1`), and track that value's flow through the Python application.
    *Difficulty:* Requires full taint analysis of Python memory to know that
    variable `user_id` holds the value `1` which corresponds to `$sym1`.
    Extremely hard.

5.  **Heuristic Matching / Anonymization:**
    If row-level detection sees an integer ID, could it anonymize it?
    No -- `WHERE id = 1` and `WHERE id = 2` *must* conflict if they refer to
    the same row, but *must not* conflict if they refer to different rows.
    We cannot simply ignore the value.

**Recommendation:**
Stick to approach #1 (Enforced Determinism) for now. Add a documentation
section explaining why `TRUNCATE` alone is insufficient for DPOR (unlike
standard tests where different IDs don't matter). A sample reproduction
of this failure can be found in `tests/test_sql_nondeterministic_ids.py`.
