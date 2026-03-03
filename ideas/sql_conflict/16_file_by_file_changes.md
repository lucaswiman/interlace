# File-by-File Changes

## 1. `frontrun/_sql_detection.py` (new file)

Core module. ~150 lines.

## 2. `frontrun/_io_detection.py` (modify)

Add `_sql_suppress` check to `_report_socket_io` and `_traced_open`.

## 3. `frontrun/dpor.py` (modify)

Call `patch_sql()` / `unpatch_sql()` alongside existing `patch_io()` / `unpatch_io()`.

## 4. `frontrun/bytecode.py` (modify)

Same — call `patch_sql()` / `unpatch_sql()` for the random explorer.

## 5. `pyproject.toml` (modify)

Add `sqlglot` to optional `[sql]` extra; add to integration test extra.

## 6. `tests/test_sql_detection.py` (new file)

Unit tests for the parser (no database needed).

## 7. `tests/test_integration_orm.py` (modify)

Add tests for table-level conflict reduction.
