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
