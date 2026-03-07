# Remaining SQL TODOs

Outstanding work items for SQL conflict detection. Everything below is
optional refinement -- the core system (table-level, row-level, wire
protocol, anomaly classification, transaction grouping, async drivers,
psycopg3, connection pooling, etc.) is complete and verified.

---

## High Priority

### Autoincrement / Sequence Non-Determinism

When threads concurrently INSERT into tables with autoincrement PKs,
the assigned IDs depend on execution order.  This makes downstream
queries, invariant assertions, and row-level ObjectIds unstable across
interleavings.  See `ideas/sql_conflics/opus_autoincr_proposal.md` for
full analysis.

**Approach (three complementary pieces):**

1. **Post-INSERT ID Capture with Logical Aliases.**  After a patched
   `cursor.execute()` runs an INSERT, inspect `cursor.lastrowid` (or
   RETURNING clause result) to learn the assigned ID.  Map
   `(thread_id, table, insert_seq)` to the concrete ID.  Subsequent
   operations referencing that ID get a stable logical ObjectId like
   `sql:users:logical_insert_0_thread_a`.  (~200 lines + 30 tests)

2. **Sequence-as-Resource.**  Treat the autoincrement counter as a DPOR
   resource.  INSERT to a table with autoincrement reports a write to
   `sql:seq:<table>_<col>`.  `currval()`/`lastval()` calls report a
   read.  Pairs with `information_schema.columns` introspection for
   detection.  (~50 lines + 10 tests)

3. **Nondeterministic-SQL Warning Mode (default on).**  When
   `explore_dpor` or `explore_interleavings` detects SQL resources
   (INSERTs to tables, especially with autoincrement), fail early with a
   clear warning telling users to keep tests deterministic by
   pre-allocating IDs or using explicit PKs in test setup.  Controlled
   by `warn_nondeterministic_sql=True` (default).  Users who understand
   the implications can set it to `False`.

**Estimated total effort:** ~300 lines + 50 tests

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
