# Phased Implementation

## Phase 1: Table-Level Detection (MVP) — ✅ Done

**Goal:** Two threads on different tables are independent. Two threads on the same table use correct Read/Write classification.

Implementation split into modular files (original plan called for single `_sql_detection.py`):

- `frontrun/_sql_parsing.py` — ✅ 173 lines (Algorithm 1: regex + sqlglot parsing)
- `frontrun/_sql_cursor.py` — ✅ 275 lines (Algorithm 2: cursor patching + partial Algorithm 3: suppression)
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

## Phase 4: Anomaly Classification

**Goal:** When DPOR finds a failing interleaving involving SQL, classify it as a specific isolation anomaly (lost update, write skew, dirty read, etc.).

Files:
- `frontrun/_sql_anomaly.py` — new, ~200 lines (DSG construction + cycle classification)
- `frontrun/common.py` — extend `InterleavingResult` with anomaly metadata

---

## Phase 5+: Advanced SQL Features (TODO)

### TODO: SELECT FOR UPDATE & Lock Semantics (Phase 5)
**Priority:** Medium (common in ORMs using explicit locking)
**Scope:**
- Extract `FOR UPDATE` / `FOR SHARE` intent in `_sql_parsing.py`
- Return `lock_intent: Literal['NONE', 'UPDATE', 'SHARE']` from `parse_sql_access()`
- Modify `_sql_cursor.py` to track lock ownership per ObjectId
- Update DPOR suppression logic: exclusive lock blocks all, shared lock blocks exclusives

**Estimated effort:** ~50 lines + 20 tests

**Files to modify:**
- `frontrun/_sql_parsing.py` — add `lock_intent` to return tuple
- `frontrun/_sql_cursor.py` — track lock acquisition/release per ObjectId
- `tests/test_sql_parsing.py` — add tests for `SELECT FOR UPDATE` / `FOR SHARE`

---

### TODO: LOCK TABLE Statement (Phase 5)
**Priority:** Low (rarely used in Python applications)
**Scope:**
- Extend regex fast-path to detect `LOCK TABLE ... IN ... MODE`
- Parse table name + lock mode (EXCLUSIVE, SHARED, etc.)
- Integrate with lock semantics from Phase 5

**Estimated effort:** ~30 lines + 15 tests

**Files to modify:**
- `frontrun/_sql_parsing.py` — add `LOCK TABLE` detection + parsing
- `tests/test_sql_parsing.py` — add tests for LOCK TABLE variants

---

### TODO: Advisory Lock Detection (Phase 5)
**Priority:** Low-Medium (PostgreSQL-specific, used in some applications)
**Scope:**
- Extend `crates/io/src/sql_extract.rs` to recognize advisory lock function calls
- Parse function name + lock ID from PostgreSQL protocol
- Map advisory lock IDs to DPOR ObjectIds: `advisory_lock:{lock_id}`
- Integrate with lock semantics from Phase 5

**Estimated effort:** ~100 lines (Rust) + 30 tests

**Files to modify:**
- `crates/io/src/sql_extract.rs` — add advisory lock parsing
- `crates/io/src/lib.rs` — emit `advisory_lock:{id}` as ObjectId
- `tests/` — add integration tests with `pg_advisory_lock()`

---

### TODO: UNION / INTERSECT / EXCEPT Optimization (Phase 5)
**Priority:** Low (easy to implement, low impact)
**Scope:**
- Add explicit handlers for `exp.Union`, `exp.Intersect`, `exp.Except` in `_sqlglot_parse()`
- Recognize as read-only compositions (all branches are selects)
- Return all tables from all branches as reads (not writes)

**Estimated effort:** ~20 lines + 8 tests

**Files to modify:**
- `frontrun/_sql_parsing.py` — add `Union` / `Intersect` / `Except` handlers
- `tests/test_sql_parsing.py` — add tests for set operations

---

### TODO: Cross-Table Foreign Key Analysis (Phase 6)
**Priority:** Medium (affects multi-table transactions)
**Scope:**
- Schema introspection: query `information_schema.referential_constraints` (PostgreSQL, MySQL) or equivalent
- Build FK dependency graph on first connection: `{orders → users, shipments → orders}`
- Cache in thread-local storage
- At conflict detection: if Op1 touches T1, Op2 touches T2, and path T1 → T2 exists, mark as dependent

**Estimated effort:** ~150 lines + 25 tests

**Files to modify:**
- `frontrun/_sql_cursor.py` — add schema introspection + FK caching
- `frontrun/dpor.py` — integrate FK dependencies into conflict detection
- `tests/test_integration_orm.py` — add FK-related test cases

---

### TODO: Transaction Grouping (Phase 6)
**Priority:** Medium (optimization, improves search space)
**Scope:**
- Track `BEGIN` / `START TRANSACTION` / `COMMIT` / `ROLLBACK` / `SAVEPOINT`
- Group SQL operations into transaction-level ObjectIds
- Modify DPOR suppression to suppress entire transactions atomically
- Requires changes to DPOR engine (grouping logic)

**Estimated effort:** ~80 lines (Python) + 20 tests + Rust engine changes

**Files to modify:**
- `frontrun/_sql_cursor.py` — add transaction boundary tracking
- `crates/dpor/src/engine.rs` — add transaction-level grouping
- `tests/test_integration_orm.py` — add transaction-specific tests

---

### TODO: Stored Procedure Analysis (Phase 7, Low Priority)
**Priority:** Very Low (rare in Python; most code uses direct SQL)
**Scope:**
- Intercept `CREATE PROCEDURE` / `CREATE FUNCTION` statements
- Parse their bodies and extract table access
- Cache: `{sp_name → {read_tables, write_tables}}`
- At `CALL` or function invocation, use cached access instead of endpoint-level

**Estimated effort:** ~200 lines + 40 tests

**Challenge:** Dynamic SQL in procedures (string concatenation) is opaque; may require heuristics or user hints.

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

## Summary: TODO Effort & Priority

| Phase | Task | Priority | Effort | Impact | Status |
|-------|------|----------|--------|--------|--------|
| 5 | SELECT FOR UPDATE semantics | 🟡 Medium | 50 lines + 20 tests | Correct lock modeling | **TODO** |
| 5 | LOCK TABLE statement | 🔴 Low | 30 lines + 15 tests | Rare; explicit locking | **TODO** |
| 5 | Advisory lock detection | 🟡 Low-Medium | 100 lines (Rust) + 30 tests | PostgreSQL-specific | **TODO** |
| 5 | UNION/INTERSECT/EXCEPT | 🔴 Low | 20 lines + 8 tests | Easy; low impact | **TODO** |
| 6 | Foreign key dependencies | 🟡 Medium | 150 lines + 25 tests | Multi-table correctness | **TODO** |
| 6 | Transaction grouping | 🟡 Medium | 80 lines + 20 tests | Search space optimization | **TODO** |
| 7 | Stored procedures | 🔴 Very Low | 200 lines + 40 tests | Rare in Python ORMs | **TODO** |
| 7 | Temporal tables | 🔴 Very Low | 40 lines + 10 tests | Specialized SQL | **TODO** |
| 7 | Computed columns | 🔴 Very Low | 30 lines + 5 tests | Informational | **TODO** |

**Legend:** 🟢 High-impact | 🟡 Medium-impact | 🔴 Low-impact
