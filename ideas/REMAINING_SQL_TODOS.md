# Remaining SQL TODOs

Outstanding work items for SQL conflict detection. Everything below is
optional refinement -- the core system (table-level, row-level, wire
protocol, anomaly classification, transaction grouping, async drivers,
psycopg3, connection pooling, etc.) is complete and verified.

---

## High Priority

### Autoincrement / Sequence Non-Determinism — ✅ DONE

Implemented in `frontrun/_sql_insert_tracker.py`:

1. **Post-INSERT ID Capture with Logical Aliases.** ✅ After INSERT,
   `cursor.lastrowid` is captured and mapped to indexical aliases like
   `sql:users:t0_ins0`.  Downstream operations resolve concrete IDs
   to these aliases automatically.

2. **Sequence-as-Resource.** ✅ Each INSERT reports a write to
   `sql:<table>:seq`, ensuring DPOR explores orderings of concurrent
   INSERTs.

3. **Fallback Warning.** ✅ `NondeterministicSQLError` raised only when
   `lastrowid` capture fails (e.g. psycopg2 without RETURNING).
   Controlled by `warn_nondeterministic_sql=True`.

**Remaining:** RETURNING clause injection for PostgreSQL drivers
(psycopg2/psycopg3) where `lastrowid` is unavailable.

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
