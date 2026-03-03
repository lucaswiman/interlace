# Phased Implementation

## Phase 1: Table-Level Detection (MVP)

**Goal:** Two threads on different tables are independent. Two threads on the same table use correct Read/Write classification.

Files:
- `frontrun/_sql_detection.py` — new, ~200 lines (Algorithms 1, 2, 3)
- `frontrun/_io_detection.py` — 3-line change (suppression check)
- `frontrun/dpor.py` — 4-line change (patch/unpatch calls)
- `frontrun/bytecode.py` — 4-line change (patch/unpatch calls)
- `pyproject.toml` — add `sqlglot` to extras
- `tests/test_sql_detection.py` — new, ~150 lines

## Phase 2: Row-Level Detection

**Goal:** Two threads on the same table but different rows (identified by PK equality predicates) are independent. Parameterized queries are resolved before predicate extraction.

Files:
- `frontrun/_sql_detection.py` — add ~150 lines (Algorithms 1.5, 5a-5d)
- `tests/test_sql_detection.py` — add ~120 lines (parameter resolution + predicate tests)

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
