# Decisions Resolved

| Question | Decision | Rationale |
|----------|----------|-----------|
| Parse in Python or Rust? | **Python** (sqlglot + regex) | Parsing happens in `cursor.execute()` which is Python-side. No IPC needed. The Rust engine just sees `ObjectId`s. |
| sqlparse vs sqlglot? | **sqlglot** (with regex fast-path) | sqlparse can't extract tables from JOINs/subqueries. sqlglot has full AST + column qualification. Regex handles the 90% simple case. |
| New Rust engine methods? | **No** | Existing `report_io_access(exec, tid, obj_key, kind)` is sufficient. SQL tables are just I/O objects with table-derived `ObjectId`s. |
| Suppress LD_PRELOAD too? | **Yes** | Via `_suppress_tids` shared set + `_suppress_endpoint_io()` context manager (Algorithm 3). Otherwise libpq's `send()` still creates endpoint-level conflicts. |
| Row-level: z3 or equality/IN-list? | **Equality + IN-list (no z3)** | Equality and IN-list disjointness via `frozenset.isdisjoint()` covers ~95% of ORM queries. z3 adds ~50ms/check, ~200MB dependency, and complex type encoding for marginal gain on range predicates. See [13_phased_implementation.md#design-note-why-not-z3smt-for-row-level-conflicts](13_phased_implementation.md#design-note-why-not-z3smt-for-row-level-conflicts). |
| Transaction boundaries? | **✅ Implemented** | Transaction grouping via `_io_tls._tx_buffer` with `BEGIN`/`COMMIT`/`ROLLBACK`/`SAVEPOINT` support. DPOR scheduler skips interleaving during transactions. |
| Parameterized queries? | **Full substitution** (Algorithm 1.5) | Resolve placeholders with actual values before AST analysis. All five PEP 249 paramstyles supported. Paramstyle read from driver module at patch time. Resolution failure falls back to table-level (safe). |
| Cursor patching strategy? | **Factory injection + class patching** | C-extension cursors (sqlite3, psycopg2) use subclass injection via `connect()` factory args. Pure-Python drivers (pymysql) use direct class patching. |

---

## Unresolved Questions & TODOs

| Question | Status | Rationale | TODO |
|----------|--------|-----------|------|
| Cross-table FK dependencies? | **TODO (Phase 6)** | Foreign key relationships invisible. Schema introspection needed. | See [13_phased_implementation.md#todo-cross-table-foreign-key-analysis](13_phased_implementation.md#todo-cross-table-foreign-key-analysis) |
| Stored procedures? | **TODO (Advanced)** | Dynamic SQL opaque; no introspection. Low priority for modern Python ORMs. | See [13_phased_implementation.md#todo-stored-procedure-analysis](13_phased_implementation.md#todo-stored-procedure-analysis) |
| Temporal/versioned tables? | **TODO (Low priority)** | FOR SYSTEM_TIME clauses not parsed. Rare; affects only time-domain queries. | See [13_phased_implementation.md#todo-temporal-table-support](13_phased_implementation.md#todo-temporal-table-support) |
| Generated columns? | **TODO (Low priority)** | Computed columns not marked in ObjectId. Minimal impact on conflict detection. | See [13_phased_implementation.md#todo-generated--computed-columns](13_phased_implementation.md#todo-generated--computed-columns) |
| Window functions? | **TODO (Low priority)** | PARTITION BY semantics invisible. Conservative fall-back to table-level sufficient. | See [03_algorithm_1_sql_parsing.md#todo-window-functions--partitioning](03_algorithm_1_sql_parsing.md#todo-window-functions--partitioning) |
