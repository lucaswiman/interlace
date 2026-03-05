# Phased Implementation

## Phase 1: Table-Level Detection (MVP) — ✅ Done

**Goal:** Two threads on different tables are independent. Two threads on the same table use correct Read/Write classification.

Implementation split into modular files (original plan called for single `_sql_detection.py`):

- `frontrun/_sql_parsing.py` — ✅ ~290 lines (Algorithm 1: regex + sqlglot parsing + lock intent + tx control)
- `frontrun/_sql_cursor.py` — ✅ ~410 lines (Algorithm 2: cursor patching + suppression + transaction grouping)
- `frontrun/_io_detection.py` — ✅ +3 lines (Algorithm 3: `_sql_suppress` check in `_report_socket_io`)
- `frontrun/dpor.py` — ✅ +6 lines (import + `patch_sql`/`unpatch_sql` + `is_tid_suppressed` in `_PreloadBridge.listener`)
- `frontrun/bytecode.py` — ✅ +3 lines (`patch_sql`/`unpatch_sql` in `BytecodeShuffler._patch_io`/`_unpatch_io`)
- `pyproject.toml` — ✅ `sqlglot>=20.0` in optional `[sql]` extra
- Algorithm 4 (ObjectId derivation) — ✅ no new code needed, uses existing `_make_object_key` via `sql:{table}` resource IDs
- `tests/test_sql_parsing.py` — ✅ 514 lines (91 tests)
- `tests/test_sql_cursor.py` — ✅ 1236 lines (77 tests)

## Phase 2: Row-Level Detection — ✅ Done

**Goal:** Two threads on the same table but different rows (identified by PK equality predicates) are independent. Parameterized queries are resolved before predicate extraction.

- `frontrun/_sql_params.py` — ✅ 128 lines (Algorithm 1.5: parameter resolution, all 5 PEP 249 paramstyles)
- `frontrun/_sql_predicates.py` — ✅ ~100 lines (Algorithm 5: `EqualityPredicate`, `extract_equality_predicates()`, `can_use_row_level()`, `pk_predicates_disjoint()`)
- `frontrun/_sql_cursor.py` — ✅ updated `_intercept_execute()` with row-level predicate integration (`resolve_parameters` + `extract_equality_predicates` + `_sql_resource_id`)
- `tests/test_sql_params.py` — ✅ 856 lines (123 tests)
- `tests/test_sql_predicates.py` — ✅ ~230 lines (41 tests)
- `tests/test_integration_orm.py` — ✅ ~265 lines (ORM lost-update integration tests: trace markers, bytecode exploration, DPOR, naive threading)

## Phase 3: Wire Protocol Parsing — ✅ Done

**Goal:** Catch C-level SQL (libpq `send()`) that bypasses DBAPI.

- `crates/io/src/sql_extract.rs` — ✅ ~210 lines (Algorithm 6: `extract_pg_query()` for Simple Query 'Q' and Parse 'P' messages, 16 unit tests)
- `crates/io/src/lib.rs` — ✅ integrated SQL extraction into `send()` hook (both Linux LD_PRELOAD and macOS DYLD_INSERT_LIBRARIES)

## Phase 4: Anomaly Classification — ✅ Done

**Goal:** When DPOR finds a failing interleaving involving SQL, classify it as a specific isolation anomaly (lost update, write skew, dirty read, etc.).

- `frontrun/_sql_anomaly.py` — ✅ ~200 lines (DSG construction + cycle classification)
- `frontrun/common.py` — ✅ extend `InterleavingResult` with anomaly metadata
- `frontrun/dpor.py` — ✅ integrated `classify_sql_anomaly` into result processing

---

## Phase 5: Advanced SQL Features — ✅ Done

### SELECT FOR UPDATE & Lock Semantics — ✅ Done
- Extract `FOR UPDATE` / `FOR SHARE` intent in `_sql_parsing.py`
- Return `lock_intent: Literal['NONE', 'UPDATE', 'SHARE']` from `parse_sql_access()`
- `_sql_cursor.py` uses `lock_intent` to set access kind (UPDATE → write, SHARE → read)

### LOCK TABLE Statement — ✅ Done
- Regex fast-path detects `LOCK TABLE ... IN ... MODE`
- Correctly parses table name + lock mode (EXCLUSIVE, SHARED, etc.)

### Advisory Lock Detection — ✅ Done
- `_sql_parsing.py` (via `sqlglot`) recognizes advisory lock function calls
- Extracts lock ID/name from literals
- Maps advisory lock IDs to DPOR ObjectIds: `sql:advisory_lock:{lock_id}`

### UNION / INTERSECT / EXCEPT Optimization — ✅ Done
- Handlers for `exp.Union`, `exp.Intersect`, `exp.Except` in `_sqlglot_parse()`
- Recognized as read-only compositions (all branches are selects)
- Returns all tables from all branches as reads

---

## Phase 6: Further Improvements (TODO)

### TODO: Cross-Table Foreign Key Analysis (Phase 6)
**Priority:** Medium (affects multi-table transactions)
**Scope:**
- Schema introspection: query `information_schema.referential_constraints` (PostgreSQL, MySQL) or equivalent
- Build FK dependency graph on first connection: `{orders → users, shipments → orders}`
- Cache in thread-local storage
- At conflict detection: if Op1 touches T1, Op2 touches T2, and path T1 → T2 exists, mark as dependent

**Estimated effort:** ~150 lines + 25 tests

---

### TODO: Transaction Grouping (Phase 6) — ✅ Done
**Priority:** Medium (optimization, improves search space)
**Scope:**
- Track `BEGIN` / `START TRANSACTION` / `COMMIT` / `ROLLBACK` / `SAVEPOINT`
- Group SQL operations into transaction-level ObjectIds: Implemented via buffering in `_io_tls._tx_buffer` and flushing at `COMMIT`.
- Modify DPOR suppression to suppress entire transactions atomically: Implemented in `frontrun/dpor.py` by skipping scheduling when `_in_transaction` is true.

**Estimated effort:** ~80 lines (Python) + 20 tests + Rust engine changes (Note: Rust changes not needed as atomicity handled in Python scheduler).

---

### TODO: Stored Procedure Analysis (Phase 7, Low Priority)
**Priority:** Very Low (rare in Python; most code uses direct SQL)
**Scope:**
- Intercept `CREATE PROCEDURE` / `CREATE FUNCTION` statements
- Parse their bodies and extract table access
- Cache: `{sp_name → {read_tables, write_tables}}`
- At `CALL` or function invocation, use cached access instead of endpoint-level

**Estimated effort:** ~200 lines + 40 tests

---

### TODO: Temporal Table Support (Phase 7, Very Low Priority)
**Priority:** Very Low (rare; specialized SQL)
**Scope:**
- Detect `FOR SYSTEM_TIME` clauses in SELECT
- Extract temporal predicate (`AS OF`, `BETWEEN`, `ALL`, etc.)
- Add temporal dimension to ObjectIds: `table:pk:time_bucket`
- Conflict detection: historical queries don't conflict with current writes

**Estimated effort:** ~40 lines + 10 tests

---

### TODO: Generated & Computed Columns (Phase 7, Very Low Priority)
**Priority:** Very Low (informational only; minimal impact)
**Scope:**
- Schema introspection: query `information_schema.columns` for `GENERATED` clause
- Mark computed columns in ObjectId derivation
- Exclude from row-level predicate matching (can't be set by user)

**Estimated effort:** ~30 lines + 5 tests

---

### TODO: psycopg3 (psycopg) Driver Support (Phase 6)
**Priority:** Medium (psycopg3 is the modern PostgreSQL driver, increasingly common)
**Scope:**
- Patch `psycopg.Cursor.execute` and `psycopg.Cursor.executemany`
- Patch `psycopg.AsyncCursor.execute` for async support
- Factory injection pattern (C-extension cursor, like psycopg2)
- Handle psycopg3's `$1` placeholder style (server-side prepared statements) in parameter resolution
- psycopg3 uses `format` paramstyle for client-side binding, `$1` for server-side

**Estimated effort:** ~80 lines + 15 tests

---

### TODO: Async Driver Support (Phase 6)
**Priority:** Medium (async is increasingly standard in Python web frameworks)
**Scope:**
- `asyncpg` — PostgreSQL async driver (C-extension, uses binary protocol directly)
- `aiosqlite` — async wrapper around sqlite3
- `aiomysql` — async MySQL driver
- `psycopg.AsyncCursor` — psycopg3 async mode
- Requires async-aware interception (wrapper must be `async def`)
- `_suppress_endpoint_io()` context manager needs async variant or must work across await boundaries

**Challenge:** asyncpg bypasses DBAPI entirely (uses PostgreSQL binary protocol). May need wire-protocol approach (Algorithm 6) rather than cursor patching.

**Estimated effort:** ~200 lines + 30 tests

---

### TODO: Connection Pooling Awareness (Phase 7, Low Priority)
**Priority:** Low (pooled connections mostly transparent to cursor patching)
**Scope:**
- SQLAlchemy `create_engine()` pool management
- pgbouncer / PgPool-II — external connection poolers
- Per-connection TLS tracking (pool may share connections across logical sessions)
- Transaction state may leak across pool checkout/checkin boundaries

**Mitigation:** Current cursor-level patching works correctly with pooled connections since we intercept at `cursor.execute()`, not `connect()`. The main risk is transaction state (`_in_transaction`) persisting across pool recycling.

**Estimated effort:** ~40 lines + 10 tests (mostly defensive cleanup at pool boundaries)

---

### TODO: Multi-Statement SQL Strings (Phase 7, Low Priority)
**Priority:** Low (rare in ORM-generated SQL; mainly manual `cursor.execute()`)
**Scope:**
- Current parser strips trailing `;` and parses only the first statement
- `cursor.execute("INSERT INTO a ...; DELETE FROM b ...")` only reports `a`
- Need to split on `;` (respecting string literals) and parse each statement
- Report union of all table accesses

**Estimated effort:** ~30 lines + 10 tests

---

### TODO: Phantom Read Detection in Anomaly Classifier (Phase 7)
**Priority:** Low (phantoms require range predicates, currently table-level)
**Scope:**
- `_sql_anomaly.py` classifies: dirty_read, write_skew, lost_update, write_write, non_repeatable_read
- Missing: phantom reads (a thread re-executes a range query and gets different rows due to concurrent INSERT/DELETE)
- Requires range predicate support to distinguish from non-repeatable reads
- Per Berenson et al. (1995): phantom = new rows appear; non-repeatable = existing rows change

**Estimated effort:** ~50 lines + 15 tests (blocked on range predicate support)

---

## Design Note: Why Not z3/SMT for Row-Level Conflicts

The decision to use value comparison and set disjointness rather than an SMT solver (z3) for row-level conflict detection is deliberate:

1. **Coverage vs. cost.** ORM SQL is overwhelmingly equality lookups (`WHERE id = ?`) and IN-lists (`WHERE id IN (?, ?, ?)`). These cover ~95% of real-world row-level queries. Simple set operations handle both correctly.

2. **Hot-path performance.** `_intercept_execute` runs on *every* SQL statement. z3 adds ~50ms per satisfiability check vs nanoseconds for `frozenset.isdisjoint()`. At scale (hundreds of SQL statements per test), this would dominate execution time.

3. **Dependency weight.** `z3-solver` is ~200MB. For a testing library, this is disproportionate. It would need to be optional, meaning we'd still need the equality/IN-list fallback path — so we wouldn't delete code, just add complexity.

4. **Encoding complexity.** Translating SQL types (VARCHAR, DECIMAL, timestamps, NULL three-valued logic) into z3 sorts is non-trivial and error-prone. SQL NULL semantics alone (NULL = NULL is NULL, not TRUE) require careful handling that's easy to get wrong.

5. **Safe fallback.** Unhandled predicates (ranges, OR, BETWEEN, subqueries) fall back to table-level conflict detection. This is conservative (may explore unnecessary interleavings) but never misses real conflicts.

If range predicate support becomes important, a lightweight interval-arithmetic approach (no z3) could handle `WHERE id > X AND id < Y` vs `WHERE id > Z` with simple numeric comparison. This would be a future Phase 7+ item.

---

## Summary: TODO Effort & Priority

| Phase | Task | Priority | Effort | Impact | Status |
|-------|------|----------|--------|--------|--------|
| 6 | Foreign key dependencies | 🟡 Medium | 150 lines + 25 tests | Multi-table correctness | **TODO** |
| 6 | Transaction grouping | 🟡 Medium | 80 lines + 20 tests | Search space optimization | ✅ **Done** |
| 6 | psycopg3 driver support | 🟡 Medium | 80 lines + 15 tests | Modern PostgreSQL driver | **TODO** |
| 6 | Async driver support | 🟡 Medium | 200 lines + 30 tests | Async web frameworks | **TODO** |
| 7 | Stored procedures | 🔴 Very Low | 200 lines + 40 tests | Rare in Python ORMs | **TODO** |
| 7 | Temporal tables | 🔴 Very Low | 40 lines + 10 tests | Specialized SQL | **TODO** |
| 7 | Computed columns | 🔴 Very Low | 30 lines + 5 tests | Informational | **TODO** |
| 7 | Connection pooling awareness | 🔴 Low | 40 lines + 10 tests | Pool state leakage | **TODO** |
| 7 | Multi-statement SQL strings | 🔴 Low | 30 lines + 10 tests | Manual SQL only | **TODO** |
| 7 | Phantom read detection | 🔴 Low | 50 lines + 15 tests | Blocked on range predicates | **TODO** |

**Legend:** 🟢 High-impact | 🟡 Medium-impact | 🔴 Low-impact
