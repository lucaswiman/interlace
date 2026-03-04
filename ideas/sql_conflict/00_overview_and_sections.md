# SQL Resource Conflict Detection — Implementation Plan
## Master Index & Status Tracker

**Quick navigation to all sections of the SQL conflict detection design.**

---

## Table of Contents

### Entry Point
- **This file (00)** — Master overview, status tracking, dependencies

### Core Design Documents (Read in Order)
1. **[01_problem.md](01_problem.md)** — Problem statement (why we need SQL-level conflict detection)
2. **[02_architecture_overview.md](02_architecture_overview.md)** — High-level design overview and data flow
3. **[16_file_by_file_changes.md](16_file_by_file_changes.md)** — Summary of files to create/modify (high-level scope)

### Core Algorithms (Phase 1: Table-Level Detection)
4. **[03_algorithm_1_sql_parsing.md](03_algorithm_1_sql_parsing.md)** — SQL Read/Write Set Extraction
   - 1a. Regex Fast-Path (90% of cases)
   - 1b. sqlglot Full Parser (fallback)
   - 1c. Combined Entry Point
5. **[05_algorithm_2_cursor_patching.md](05_algorithm_2_cursor_patching.md)** — DBAPI Cursor Monkey-Patching
   - Patch strategy for known drivers (psycopg2, sqlite3, pymysql, etc.)
   - Implementation pattern reusing `_io_detection.py`
6. **[06_algorithm_3_endpoint_suppression.md](06_algorithm_3_endpoint_suppression.md)** — Endpoint Suppression
   - Thread-local suppression flag
   - C-level LD_PRELOAD bridge integration
7. **[07_algorithm_4_objectid_derivation.md](07_algorithm_4_objectid_derivation.md)** — ObjectId Derivation
   - How SQL table names map to DPOR ObjectIds
   - Reusing existing `_make_object_key` infrastructure

### Phase 2: Row-Level Detection (Optional, for precision)
8. **[04_algorithm_1_5_parameter_resolution.md](04_algorithm_1_5_parameter_resolution.md)** — Parameter Resolution
   - Substituting placeholders with actual values (required for Phase 2)
   - All five PEP 249 paramstyles supported
   - 1.5a. Value Conversion
   - 1.5b. Paramstyle Detection
   - 1.5c. Resolution Functions
   - 1.5d. End-to-End Example
   - 1.5e. Limitations
9. **[08_algorithm_5_row_level_predicates.md](08_algorithm_5_row_level_predicates.md)** — Row-Level Predicate Extraction
   - 5a. Predicate Extraction via sqlglot
   - 5b. Row-Level ObjectId
   - 5c. Soundness Decision Function
   - 5d. Row-Level Disjointness (equality-only, no z3)

### Phase 3: Wire Protocol Parsing (Advanced)
10. **[09_algorithm_6_wire_protocol.md](09_algorithm_6_wire_protocol.md)** — Wire Protocol SQL Extraction
    - PostgreSQL Simple Query Protocol
    - PostgreSQL Extended Query Protocol
    - Rust extraction implementation in crates/io/

### Integration & Validation
11. **[10_integration_points.md](10_integration_points.md)** — Integration Points
    - Changes to dpor.py (explore_dpor, _setup_dpor_tls, flush logic)
    - No changes needed to Rust engine
12. **[11_correctness_argument.md](11_correctness_argument.md)** — Correctness Argument
    - Soundness (no missed bugs)
    - Completeness (false positives and limitations)
13. **[12_test_plan.md](12_test_plan.md)** — Test Plan
    - Unit tests for parser, parameter resolution, predicates
    - Integration tests for ORM SQL detection
14. **[13_phased_implementation.md](13_phased_implementation.md)** — Phased Implementation
    - Phase 1: Table-Level Detection (MVP) — ~300 lines code
    - Phase 2: Row-Level Detection — ~270 lines code
    - Phase 3: Wire Protocol Parsing — ~80 lines code
    - Phase 4: Anomaly Classification — ~200 lines code (deferred)

### Decision & Verification
15. **[14_decisions_resolved.md](14_decisions_resolved.md)** — Design Decisions
    - Why Python parsing + regex fast-path
    - Why sqlglot over sqlparse
    - Why equality-only (no z3) for Phase 2
    - Why full parameter resolution
16. **[15_formal_verification.md](15_formal_verification.md)** — Formal Verification (TLA+)
    - Spec 1: SqlConflictRefinement (13 invariants, 3,200 states)
    - Spec 2: DporSqlScheduling (7 invariants, 2,048 states)
    - Spec 3: SuppressionSafety (3 invariants, 32,768 states)
    - All specs pass — design is proven correct
17. **[17_references.md](17_references.md)** — References
    - Academic papers (Adya isolation levels, Elle, IsoRel, TxCheck)
    - Tools (sqlglot, datafusion-sqlparser)

---

## Phase Status

| Phase | Goal | Algorithms | Status | Est. Code |
|-------|------|-----------|--------|-----------|
| **1** | Table-level detection (MVP) | 1, 2, 3, 4 | ✅ Done | ~300 lines |
| **2** | Row-level detection (precision) | 1.5, 5 | ⚙ In Progress (1.5, 5 done; integration pending) | ~270 lines |
| **3** | Wire protocol parsing (C-level) | 6 | ✓ Designed | ~80 lines |
| **4** | Isolation anomaly classification | — | ✓ Designed | ~200 lines |

**Next step:** Integrate Algorithm 5 predicate extraction into the cursor patching execute wrapper (wire row-level ObjectIds into `_intercept_execute`), then add integration tests (`test_integration_orm.py`).

---

## Algorithm Dependency Graph

```
Algorithm 1 (SQL Read/Write Parsing)
  ├─→ Algorithm 2 (Cursor Patching) — calls parse_sql_access()
  ├─→ Algorithm 3 (Endpoint Suppression) — suppression triggered by Algorithm 2
  └─→ Algorithm 4 (ObjectId Derivation) — uses Algorithm 1 output

Algorithm 1.5 (Parameter Resolution) — REQUIRED for Phase 2
  └─→ Algorithm 5 (Row-Level Predicates) — extracts predicates from resolved SQL

Algorithm 6 (Wire Protocol Parsing)
  └─→ Used for C-extension drivers (libpq) not caught by Algorithm 2
```

**Execution order:**
1. **Phase 1:** Algorithms 1 → 2 → 3 → 4 (enables table-level conflict detection)
2. **Phase 2:** Algorithm 1.5 → Algorithm 5 (enables row-level conflict detection)
3. **Phase 3:** Algorithm 6 (catches C-level SQL)
4. **Phase 4:** Anomaly classification (not yet designed)

---

## Implementation Checklist

### Phase 1: Table-Level Detection (MVP)

- [x] ~~Create `frontrun/_sql_detection.py`~~ → Split into modular files:
  - [x] Implement Algorithm 1a: `_regex_parse()` → `frontrun/_sql_parsing.py` (173 lines)
  - [x] Implement Algorithm 1b: `_sqlglot_parse()` (with optional sqlglot) → `frontrun/_sql_parsing.py`
  - [x] Implement Algorithm 1c: `parse_sql_access()` combined entry point → `frontrun/_sql_parsing.py`
  - [x] Implement Algorithm 2: `_make_patched_execute()`, `patch_sql()`, `unpatch_sql()` → `frontrun/_sql_cursor.py` (275 lines)
  - [x] Implement Algorithm 3 (partial): `_suppress_tids` set + `is_tid_suppressed()` → `frontrun/_sql_cursor.py`

- [x] Modify `frontrun/_io_detection.py` (+3 lines)
  - [x] Add `_sql_suppress` check to `_report_socket_io()`

- [x] Modify `frontrun/dpor.py` (+6 lines)
  - [x] Import `patch_sql` / `unpatch_sql` / `is_tid_suppressed`
  - [x] Call `patch_sql()` / `unpatch_sql()` in `DporBytecodeRunner._patch_io()` / `_unpatch_io()`
  - [x] Add `is_tid_suppressed()` check in `_PreloadBridge.listener()`

- [x] Modify `frontrun/bytecode.py` (+3 lines)
  - [x] Call `patch_sql()` / `unpatch_sql()` in `BytecodeShuffler._patch_io()` / `_unpatch_io()`

- [x] Modify `pyproject.toml`
  - [x] Add `sqlglot>=20.0` to optional `[sql]` extra

- [x] ~~Create `tests/test_sql_detection.py`~~ → Split into modular test files:
  - [x] TestRegexParse (44 tests) → `tests/test_sql_parsing.py` (514 lines, 91 tests total)
  - [x] TestSqlglotParse (16 tests, guarded with importorskip) → `tests/test_sql_parsing.py`
  - [x] TestParseSqlAccess (24 tests) → `tests/test_sql_parsing.py`
  - [x] TestCursorPatching (77 tests) → `tests/test_sql_cursor.py` (1236 lines)

### Phase 2: Row-Level Detection

- [x] ~~Add Algorithm 1.5 to `frontrun/_sql_detection.py`~~ → `frontrun/_sql_params.py` (128 lines)
  - [x] `_python_to_sql_literal()` — value conversion
  - [x] `resolve_parameters()` — placeholder substitution (all 5 paramstyles)
  - [x] Helper functions: `_resolve_positional()`, `_resolve_numeric()`, `_resolve_named()`, `_resolve_pyformat()`
  - [x] Tests: `tests/test_sql_params.py` (856 lines, 123 tests)

- [x] ~~Add Algorithm 5 to `frontrun/_sql_detection.py`~~ → `frontrun/_sql_predicates.py` (~100 lines)
  - [x] `EqualityPredicate` dataclass
  - [x] `extract_equality_predicates()` — WHERE clause parsing
  - [x] `can_use_row_level()` — decision function
  - [x] `pk_predicates_disjoint()` — row disjointness check
  - [x] Tests: `tests/test_sql_predicates.py` (41 tests)

- [ ] Integrate row-level predicates into `_sql_cursor.py` `_intercept_execute()`
  - [ ] Wire `extract_equality_predicates()` + `pk_predicates_disjoint()` into ObjectId derivation

### Phase 3: Wire Protocol Parsing

- [ ] Create `crates/io/src/sql_extract.rs` (~80 lines)
  - [ ] `extract_pg_query()` — PostgreSQL protocol parsing
  - [ ] Support Simple Query ('Q') and Parse ('P') messages

- [ ] Modify `crates/io/src/lib.rs`
  - [ ] Integrate SQL extraction into `send()` hook
  - [ ] Write SQL-enriched events to pipe

### Phase 4: Anomaly Classification (Deferred)

- [ ] Create `frontrun/_sql_anomaly.py` (~200 lines)
  - [ ] DSG (Dependency Serialization Graph) construction
  - [ ] Cycle classification (lost update, write skew, dirty read, phantom, etc.)

- [ ] Extend `frontrun/common.py`
  - [ ] Add anomaly metadata to `InterleavingResult`

---

## File Map

| File | Location | Lines | Status | Purpose |
|------|----------|-------|--------|---------|
| `_sql_parsing.py` | `frontrun/` | 173 | ✅ Done | SQL read/write set extraction (regex + sqlglot) |
| `_sql_params.py` | `frontrun/` | 128 | ✅ Done | Parameter resolution (all 5 PEP 249 paramstyles) |
| `_sql_cursor.py` | `frontrun/` | 275 | ✅ Done | DBAPI cursor monkey-patching + suppression |
| `test_sql_parsing.py` | `tests/` | 514 | ✅ Done | 91 tests for SQL parsing |
| `test_sql_params.py` | `tests/` | 856 | ✅ Done | 123 tests for parameter resolution |
| `test_sql_cursor.py` | `tests/` | 1236 | ✅ Done | 77 tests for cursor patching |
| `_sql_predicates.py` | `frontrun/` | ~100 | ✅ Done | Row-level predicate extraction (Algorithm 5) |
| `test_sql_predicates.py` | `tests/` | ~230 | ✅ Done | 41 tests for predicate extraction + disjointness |
| `_io_detection.py` | `frontrun/` | +3 | ✅ Done | Add `_sql_suppress` check to `_report_socket_io` |
| `dpor.py` | `frontrun/` | +6 | ✅ Done | `patch_sql`/`unpatch_sql` + `is_tid_suppressed` in bridge |
| `bytecode.py` | `frontrun/` | +3 | ✅ Done | `patch_sql`/`unpatch_sql` in BytecodeShuffler |
| `pyproject.toml` | root | +3 | ✅ Done | Add `sqlglot>=20.0` to `[sql]` extra |
| `sql_extract.rs` | `crates/io/src/` | ~80 | ⬜ Phase 3 | Wire protocol parsing |
| `_sql_anomaly.py` | `frontrun/` | ~200 | ⬜ Phase 4 | Anomaly classification |

---

## Key Design Properties

✓ **Soundness** — Never claims independence when a conflict exists
- Parse failure → fall back to endpoint-level (safe)
- UPDATE/DELETE classified as Read+Write (conservative)
- Table-level is a strict refinement of endpoint-level

✓ **Completeness** — Reduces false positives (spurious interleavings)
- Table-level: independent only if different tables
- Row-level (Phase 2): independent if different PKs + full parameters resolved
- Cross-table dependencies (FKs): known limitation, addressed in Phase 4

✓ **Performance** — Negligible overhead vs. network RTT
- Regex fast-path: O(n) with 6 scans
- sqlglot: O(n) with higher constant (AST construction)
- Parameter resolution: O(n) + O(k) for k parameter values

✓ **Formality** — Proven correct via exhaustive TLA+ model checking
- 3 specs, 38,016 states total
- All invariants pass, no violations

---

## Quick Start

1. **Read** → [01_problem.md](01_problem.md) and [02_architecture_overview.md](02_architecture_overview.md)
2. **Understand** → [03_algorithm_1_sql_parsing.md](03_algorithm_1_sql_parsing.md) (core parsing logic)
3. **Review** → [10_integration_points.md](10_integration_points.md) (how it integrates with dpor.py)
4. **Verify** → [11_correctness_argument.md](11_correctness_argument.md) (why it's sound)
5. **Implement** → [13_phased_implementation.md](13_phased_implementation.md) (Phase 1 first)
6. **Test** → [12_test_plan.md](12_test_plan.md) (comprehensive test suite)

---

## Document History

- **Designed**: 2026-03-03
- **Implementation started**: 2026-03-03
  - Algorithm 1 (SQL parsing): `frontrun/_sql_parsing.py` — 91 tests passing
  - Algorithm 1.5 (Parameter resolution): `frontrun/_sql_params.py` — 123 tests passing
  - Algorithm 2 (Cursor patching): `frontrun/_sql_cursor.py` — 77 tests passing (includes partial Algorithm 3 suppression)
  - **Note:** Implementation split into separate modules (`_sql_parsing.py`, `_sql_params.py`, `_sql_cursor.py`) instead of a single `_sql_detection.py`. These can be consolidated later if desired.
- **Phase 1 completed + Phase 2 predicates**: 2026-03-04
  - Algorithm 3 (Endpoint suppression): `_io_detection.py` `_sql_suppress` check + `_PreloadBridge.listener` `is_tid_suppressed` check
  - Algorithm 4 (ObjectId derivation): no new code — uses existing `_make_object_key` via `sql:{table}` resource IDs
  - Phase 1 integration: `dpor.py` and `bytecode.py` call `patch_sql()`/`unpatch_sql()`; `pyproject.toml` adds `[sql]` extra
  - Algorithm 5 (Row-level predicates): `frontrun/_sql_predicates.py` — 41 tests passing
- **Verified**: All 3 TLA+ specs pass exhaustive model checking
- **Last Updated**: 2026-03-04

---

## Notes for Implementers

- **Parameter resolution** is critical for Phase 2 row-level detection. Without it, parameterized queries fall back to table-level.
- **Suppression thread-safety** is important: use `threading.get_native_id()` + shared lock, not just thread-local TLS.
- **Regex fast-path** handles ~90% of ORM-generated SQL — keep it in Phase 1 for performance.
- **sqlglot** is optional via import-try-except; system is sound if it fails to parse (falls back to endpoint-level).
- **Phase 2 row-level** is optional but valuable for reducing false positives in same-table, multi-row workloads.
- **Phase 3 wire protocol** is for C-extension drivers; most Python code uses DBAPI so Phase 1 covers most cases.
- **TLA+ specs** are in `specs/` directory; see 15_formal_verification.md for how to run them.
