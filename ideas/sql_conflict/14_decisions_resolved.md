# Decisions Resolved

| Question | Decision | Rationale |
|----------|----------|-----------|
| Parse in Python or Rust? | **Python** (sqlglot + regex) | Parsing happens in `cursor.execute()` which is Python-side. No IPC needed. The Rust engine just sees `ObjectId`s. |
| sqlparse vs sqlglot? | **sqlglot** (with regex fast-path) | sqlparse can't extract tables from JOINs/subqueries. sqlglot has full AST + column qualification. Regex handles the 90% simple case. |
| New Rust engine methods? | **No** | Existing `report_io_access(exec, tid, obj_key, kind)` is sufficient. SQL tables are just I/O objects with table-derived `ObjectId`s. |
| Suppress LD_PRELOAD too? | **Yes** | Via `_suppress_tids` shared set (Algorithm 3). Otherwise libpq's `send()` still creates endpoint-level conflicts. |
| Row-level: z3 or equality-only? | **Equality-only for Phase 2** | Covers the common case (PK lookups). z3 adds a heavy dependency for marginal gain on range predicates. |
| Transaction boundaries? | **Deferred** | Current model: each SQL statement is an independent access. Good enough for lost-update and write-skew detection. Transaction grouping is Phase 4 work. |
| Parameterized queries? | **Full substitution** (Algorithm 1.5) | Resolve placeholders with actual values before AST analysis. All five PEP 249 paramstyles supported. Paramstyle read from driver module at patch time. Resolution failure falls back to table-level (safe). |

---

## Unresolved Questions & TODOs

| Question | Status | Rationale | TODO |
|----------|--------|-----------|------|
| SELECT FOR UPDATE semantics? | **TODO** | Locking intent is not modeled; treated as regular read. Soundness maintained but overly pessimistic. | See [03_algorithm_1_sql_parsing.md#select-for-update-for-share-locking-semantics](03_algorithm_1_sql_parsing.md#todo-select-for-update--for-share-locking-semantics) |
| LOCK TABLE statement? | **TODO** | DDL statement for explicit row/table locking. Not parsed; falls back to endpoint-level. | See [03_algorithm_1_sql_parsing.md#lock-table-statement-support](03_algorithm_1_sql_parsing.md#todo-lock-table-statement-support) |
| Advisory lock tracking? | **TODO** | Function-level locking (PostgreSQL `pg_advisory_lock`). Only socket-level detection; lock ID not tracked. | See [03_algorithm_1_sql_parsing.md#advisory-locks-postgresql-mysql](03_algorithm_1_sql_parsing.md#todo-advisory-locks-postgresql-mysql) |
| UNION handling? | **TODO** | Currently conservative (all tables → writes). Should recognize UNION as read-only composition. | See [03_algorithm_1_sql_parsing.md#union-handling-overly-conservative](03_algorithm_1_sql_parsing.md#todo-union-handling-overly-conservative) |
| Cross-table FK dependencies? | **TODO (Phase 4)** | Foreign key relationships invisible. Schema introspection needed. | See [03_algorithm_1_sql_parsing.md#cross-table-foreign-key-dependencies](03_algorithm_1_sql_parsing.md#todo-cross-table-foreign-key-dependencies) |
| Transaction grouping? | **TODO (Phase 4)** | Statement-level granularity misses transaction atomicity. Deferred as optimization. | See [03_algorithm_1_sql_parsing.md#transaction-boundaries-not-tracked](03_algorithm_1_sql_parsing.md#todo-transaction-boundaries-not-tracked) |
| Stored procedures? | **TODO (Advanced)** | Dynamic SQL opaque; no introspection. Low priority for modern Python ORMs. | See [03_algorithm_1_sql_parsing.md#stored-procedures--dynamic-sql](03_algorithm_1_sql_parsing.md#todo-stored-procedures--dynamic-sql) |
| Temporal/versioned tables? | **TODO (Low priority)** | FOR SYSTEM_TIME clauses not parsed. Rare; affects only time-domain queries. | See [03_algorithm_1_sql_parsing.md#temporal-tables--system-versioning](03_algorithm_1_sql_parsing.md#todo-temporal-tables--system-versioning) |
| Generated columns? | **TODO (Low priority)** | Computed columns not marked in ObjectId. Minimal impact on conflict detection. | See [03_algorithm_1_sql_parsing.md#generatedcomputed-columns](03_algorithm_1_sql_parsing.md#todo-generatedcomputed-columns) |
| Window functions? | **TODO (Low priority)** | PARTITION BY semantics invisible. Conservative fall-back to table-level sufficient. | See [03_algorithm_1_sql_parsing.md#window-functions--partitioning](03_algorithm_1_sql_parsing.md#todo-window-functions--partitioning) |
