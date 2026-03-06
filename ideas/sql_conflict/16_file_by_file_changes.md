# File-by-File Changes

## Completed (2026-03-03)

### `frontrun/_sql_parsing.py` (new file) — ✅

SQL read/write set extraction with lock intent and transaction control.
- `_strip_quotes()`, `_regex_parse()`, `_sqlglot_parse()`, `parse_sql_access()`
- Returns 4-tuple: `(read_tables, write_tables, lock_intent, tx_op)`
- Regex fast-path handles ~90% of ORM SQL; sqlglot fallback for complex queries
- Lock intent: FOR UPDATE, FOR SHARE, LOCK TABLE with mode detection
- Transaction control: BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE, ROLLBACK TO
- Advisory locks: pg_advisory_lock, get_lock → `advisory_lock:{lock_id}`
- Tests: `tests/test_sql_parsing.py`

### `frontrun/_sql_params.py` (new file) — ✅

Parameter resolution for all 5 PEP 249 paramstyles. 128 lines.
- `_python_to_sql_literal()`, `resolve_parameters()`
- Resolvers: `_resolve_positional()`, `_resolve_numeric()`, `_resolve_named()`, `_resolve_pyformat()`
- Tests: `tests/test_sql_params.py`

### `frontrun/_sql_cursor.py` (new file) — ✅

DBAPI cursor monkey-patching + endpoint suppression + transaction grouping.
- `patch_sql()`, `unpatch_sql()`, `_intercept_execute()`, `is_tid_suppressed()`
- Factory injection for C-extension cursors (sqlite3, psycopg2)
- Direct class patching for pure-Python drivers (pymysql)
- `_suppress_endpoint_io()` context manager for dual suppression
- Transaction buffering: BEGIN → buffer, COMMIT → flush, ROLLBACK → discard
- SAVEPOINT/ROLLBACK TO support via buffer truncation
- Tests: `tests/test_sql_cursor.py`

**Note:** Original plan called for a single `_sql_detection.py`; implementation uses focused modules.

## Completed (2026-03-04)

### `frontrun/_io_detection.py` (modify) — ✅

Added `_sql_suppress` check to `_report_socket_io`. +3 lines.
- Early return when `_io_tls._sql_suppress` is True (set by `_sql_cursor._intercept_execute`)
- Prevents duplicate endpoint-level reports when SQL-level detection already reported table accesses

### `frontrun/dpor.py` (modify) — ✅

Integrated SQL patching and transaction atomicity into DPOR scheduler.
- Import `is_tid_suppressed`, `patch_sql`, `unpatch_sql` from `_sql_cursor`
- Import `classify_sql_anomaly` from `_sql_anomaly`
- Call `patch_sql()` / `unpatch_sql()` in `DporBytecodeRunner._patch_io()` / `_unpatch_io()`
- Add `is_tid_suppressed(event.tid)` check in `_PreloadBridge.listener()` to skip LD_PRELOAD events
- Skip scheduling when `_in_transaction` is True (transaction atomicity)
- Classify SQL anomaly in result processing

### `frontrun/bytecode.py` (modify) — ✅

Integrated SQL patching into random bytecode explorer. +3 lines.
- Import `patch_sql`, `unpatch_sql` from `_sql_cursor`
- Call `patch_sql()` / `unpatch_sql()` in `BytecodeShuffler._patch_io()` / `_unpatch_io()`

### `pyproject.toml` (modify) — ✅

Added `sqlglot>=20.0` to optional `[sql]` extra.

### `frontrun/_sql_predicates.py` (new file) — ✅

Row-level predicate extraction (Algorithm 5). ~100 lines.
- `EqualityPredicate` frozen dataclass (`column`, `value`)
- `extract_equality_predicates()` — WHERE clause parsing via sqlglot (AND conjuncts only)
- `can_use_row_level()` — decision function (both operations must have full PK predicates)
- `pk_predicates_disjoint()` — O(k) check for different PK values on shared columns
- Tests: `tests/test_sql_predicates.py`

### `frontrun/_sql_cursor.py` (modify) — ✅

Integrated row-level predicate extraction into `_intercept_execute()`:
- Added imports for `resolve_parameters` and `extract_equality_predicates` (with try/except fallbacks)
- Added `_sql_resource_id()` helper to build resource IDs with optional predicates
- For single-table non-executemany queries: resolves parameters → extracts predicates → includes in resource ID
- Updated `_make_traced_cursor_class()` to accept and propagate `paramstyle`
- Updated all patching functions to pass correct paramstyle (qmark for sqlite3, pyformat for psycopg2, format for pymysql)

### `tests/test_integration_orm.py` (new file) — ✅

ORM lost-update integration tests. Requires SQLAlchemy + PostgreSQL.
- `TestOrmTraceMarkers` — deterministic reproduction via trace markers
- `TestOrmBytecodeExploration` — automatic detection via random bytecode schedules
- `TestOrmDpor` — systematic exploration via DPOR + LD_PRELOAD
- `TestOrmNaiveThreading` — demonstrates intermittent nature of the race

### `crates/io/src/sql_extract.rs` (new file) — ✅

PostgreSQL wire protocol SQL extraction (~210 lines, Algorithm 6).
- `extract_pg_query()` — parses Simple Query ('Q') and Parse ('P') messages
- 16 unit tests covering: valid messages, truncated headers/bodies, invalid UTF-8, non-query messages

### `crates/io/src/lib.rs` (modify) — ✅

Integrated SQL extraction into send() hook.
- Added `mod sql_extract;` declaration
- Linux `send()` hook: try SQL extraction before `report_io()`, emit `sql_write` event if SQL found
- macOS `frontrun_send()` hook: same logic

### `frontrun/_sql_anomaly.py` (new file) — ✅

SQL isolation anomaly classification (~300 lines).
- `SqlAnomaly` frozen dataclass (kind, summary, tables, threads)
- `classify_sql_anomaly()` — main entry point
- `_extract_sql_events()` — filter trace events for SQL I/O
- `_build_dsg()` — Dependency Serialization Graph construction (WR, RW, WW edges)
- `_find_cycle()` — DFS cycle detection
- `_check_non_repeatable_read()` — pre-DSG pattern check
- `_check_lost_update()` — pre-DSG pattern check
- Classification: dirty_read, write_skew, lost_update, write_write, non_repeatable_read
- Tests: `tests/test_sql_anomaly.py`

### `frontrun/common.py` (modify) — ✅

Extended `InterleavingResult` with `sql_anomaly: SqlAnomaly | None` field.

## Completed (2026-03-06)

### `frontrun/_schema.py` (new file) — ✅

Schema registry for Foreign Key constraints. ~50 lines.
- `ForeignKey` dataclass (`table`, `column`, `ref_table`, `ref_column`)
- `Schema` class (`add_foreign_key`, `get_fks`)
- Global `register_schema()` / `get_schema()`

### `frontrun/_sql_cursor.py` (modify) — ✅

Integrated FK dependency logic into `_intercept_execute()`.
- Query `get_schema()` for FKs on write tables
- Synthesize implicit `read` operations for referenced tables
- Attempt to map row-level predicates from write to read (e.g. `orders.user_id=1` → `users.id=1`)
- Fall back to table-level read if predicates cannot be mapped

### `tests/test_foreign_keys.py` (new file) — ✅

Tests for FK dependency detection.
- `test_fk_dependency_table_level` — verifies implicit read of parent table
- `test_fk_dependency_row_level` — verifies row-level granularity preserved
