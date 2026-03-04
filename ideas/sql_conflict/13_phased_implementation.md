# Phased Implementation

## Phase 1: Table-Level Detection (MVP)

**Goal:** Two threads on different tables are independent. Two threads on the same table use correct Read/Write classification.

Implementation split into modular files (original plan called for single `_sql_detection.py`):

Done:
- `frontrun/_sql_parsing.py` — ✅ 173 lines (Algorithm 1: regex + sqlglot parsing)
- `frontrun/_sql_cursor.py` — ✅ 275 lines (Algorithm 2: cursor patching + partial Algorithm 3: suppression)
- `tests/test_sql_parsing.py` — ✅ 514 lines (91 tests)
- `tests/test_sql_cursor.py` — ✅ 1236 lines (77 tests)

Remaining:
- `frontrun/_io_detection.py` — 3-line change (Algorithm 3: add `_sql_suppress` check to `_report_socket_io`)
- `frontrun/dpor.py` — 4-line change (import and call `patch_sql` / `unpatch_sql`)
- `frontrun/bytecode.py` — 4-line change (call `patch_sql()` / `unpatch_sql()`)
- `pyproject.toml` — add `sqlglot` to optional `[sql]` extra
- Algorithm 4 (ObjectId derivation) — no new code needed, uses existing `_make_object_key` via reporter

## Phase 2: Row-Level Detection

**Goal:** Two threads on the same table but different rows (identified by PK equality predicates) are independent. Parameterized queries are resolved before predicate extraction.

Done:
- `frontrun/_sql_params.py` — ✅ 128 lines (Algorithm 1.5: parameter resolution, all 5 PEP 249 paramstyles)
- `tests/test_sql_params.py` — ✅ 856 lines (123 tests)

Remaining:
- Algorithm 5 (row-level predicates): `EqualityPredicate`, `extract_equality_predicates()`, `pk_predicates_disjoint()` (~100 lines)
- Integration of parameter resolution into cursor patching's execute wrapper
- Tests for predicate extraction + disjointness (~120 lines)

## Phase 3: Wire Protocol Parsing

**Goal:** Catch C-level SQL (libpq `send()`) that bypasses DBAPI.

Files:
- `crates/io/src/sql_extract.rs` — new, ~80 lines (Algorithm 6)
- `crates/io/src/lib.rs` — integrate SQL extraction into `send()` hook

## Phase 4: Anomaly Classification

**Goal:** When DPOR finds a failing interleaving involving SQL, classify it as a specific isolation anomaly (lost update, write skew, dirty read, etc.).

Files:
- `frontrun/_sql_anomaly.py` — new, ~200 lines (DSG construction + cycle classification)
- `frontrun/common.py` — extend `InterleavingResult` with anomaly metadata
