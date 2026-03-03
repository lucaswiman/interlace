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
