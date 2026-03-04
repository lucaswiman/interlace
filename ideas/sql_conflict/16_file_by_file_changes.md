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

**Note:** Original plan called for a single `_sql_detection.py`; implementation uses three focused modules.

## Remaining

### `frontrun/_io_detection.py` (modify) — ⬜

Add `_sql_suppress` check to `_report_socket_io` and `_traced_open`.

### `frontrun/dpor.py` (modify) — ⬜

Call `patch_sql()` / `unpatch_sql()` alongside existing `patch_io()` / `unpatch_io()`.

### `frontrun/bytecode.py` (modify) — ⬜

Same — call `patch_sql()` / `unpatch_sql()` for the random explorer.

### `pyproject.toml` (modify) — ⬜

Add `sqlglot` to optional `[sql]` extra; add to integration test extra.

### `tests/test_integration_orm.py` (modify) — ⬜

Add tests for table-level conflict reduction.
