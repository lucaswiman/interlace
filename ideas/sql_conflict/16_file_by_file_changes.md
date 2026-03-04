# File-by-File Changes

## Completed (2026-03-03)

### `frontrun/_sql_parsing.py` (new file) — ✅

SQL read/write set extraction. 173 lines.
- `_strip_quotes()`, `_regex_parse()`, `_sqlglot_parse()`, `parse_sql_access()`
- Regex fast-path handles ~90% of ORM SQL; sqlglot fallback for complex queries
- Tests: `tests/test_sql_parsing.py` (514 lines, 91 tests)

### `frontrun/_sql_params.py` (new file) — ✅

Parameter resolution for all 5 PEP 249 paramstyles. 128 lines.
- `_python_to_sql_literal()`, `resolve_parameters()`
- Resolvers: `_resolve_positional()`, `_resolve_numeric()`, `_resolve_named()`, `_resolve_pyformat()`
- Tests: `tests/test_sql_params.py` (856 lines, 123 tests)

### `frontrun/_sql_cursor.py` (new file) — ✅

DBAPI cursor monkey-patching + endpoint suppression infrastructure. 275 lines.
- `patch_sql()`, `unpatch_sql()`, `_make_patched_execute()`, `is_tid_suppressed()`
- Factory-injection for C-extension cursors (sqlite3), direct patching for pure-Python drivers
- Tests: `tests/test_sql_cursor.py` (1236 lines, 77 tests)

**Note:** Original plan called for a single `_sql_detection.py`; implementation uses four focused modules.

## Completed (2026-03-04)

### `frontrun/_io_detection.py` (modify) — ✅

Added `_sql_suppress` check to `_report_socket_io`. +3 lines.
- Early return when `_io_tls._sql_suppress` is True (set by `_sql_cursor._intercept_execute`)
- Prevents duplicate endpoint-level reports when SQL-level detection already reported table accesses

### `frontrun/dpor.py` (modify) — ✅

Integrated SQL patching into DPOR scheduler. +6 lines.
- Import `is_tid_suppressed`, `patch_sql`, `unpatch_sql` from `_sql_cursor`
- Call `patch_sql()` / `unpatch_sql()` in `DporBytecodeRunner._patch_io()` / `_unpatch_io()`
- Add `is_tid_suppressed(event.tid)` check in `_PreloadBridge.listener()` to skip LD_PRELOAD events when SQL-level detection already reported

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
- Tests: `tests/test_sql_predicates.py` (~230 lines, 41 tests)

## Completed (2026-03-04, cont.)

### `frontrun/_sql_cursor.py` (modify) — ✅

Integrated row-level predicate extraction into `_intercept_execute()`:
- Added imports for `resolve_parameters` and `extract_equality_predicates` (with try/except fallbacks)
- Added `_sql_resource_id()` helper to build resource IDs with optional predicates
- Extended `_intercept_execute()` with `is_executemany` and `paramstyle` keyword args
- For single-table non-executemany queries: resolves parameters → extracts predicates → includes in resource ID
- Updated `_make_traced_cursor_class()` to accept and propagate `paramstyle`
- Updated all patching functions to pass correct paramstyle (qmark for sqlite3, pyformat for psycopg2, format for pymysql)

### `tests/test_integration_orm.py` (new file) — ✅

ORM lost-update integration tests (~265 lines). Requires SQLAlchemy + PostgreSQL.
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

## Remaining

### `frontrun/_sql_anomaly.py` (new file) — ⬜ Phase 4

DSG construction + cycle classification for isolation anomaly detection.

### `frontrun/common.py` (modify) — ⬜ Phase 4

Extend `InterleavingResult` with anomaly metadata.
