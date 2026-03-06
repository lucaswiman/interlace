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
