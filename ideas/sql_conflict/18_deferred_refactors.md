# Deferred Refactors — ✅ All Completed

Issues identified during the 2026-03-06 code review. All have been resolved.

---

## 1. Return tuple → NamedTuple — ✅ Done

**Files:** `_sql_parsing.py`, `_sql_cursor.py`, all test files

**What changed:**
- `SqlAccessResult` is now defined only in `_sql_parsing.py` as a `NamedTuple`
  with typed fields (including `LockIntent | None` and `TxControl | None`).
- Removed the duplicate `SqlAccessResult` definition from `_sql_cursor.py`.
- Added a 6th field `ast: Any | None` (defaulting to `None`) to carry the
  pre-parsed sqlglot AST through to predicate extraction.
- Updated all ~80 call sites in test files to use `*_` unpacking for
  forward-compatible destructuring.

---

## 2. Stringly-typed `tx_op` and `lock_intent` — ✅ Done

**Files:** `_sql_parsing.py`, `_sql_cursor.py`, all test files

**What changed:**
- Introduced `LockIntent` enum (`UPDATE`, `SHARE`) replacing raw strings.
- Introduced `TxOp` enum (`BEGIN`, `COMMIT`, `ROLLBACK`) for simple tx control.
- Introduced frozen dataclasses `Savepoint(name)`, `RollbackTo(name)`,
  `Release(name)` for compound operations (replacing colon-delimited strings).
- Defined `TxControl = TxOp | Savepoint | RollbackTo | Release` union type.
- Updated `_sql_cursor._intercept_execute()` to use `is` checks for `TxOp`
  enum members and `isinstance` checks for `Savepoint`/`RollbackTo`/`Release`.
- Updated all test assertions from `== "UPDATE"` to `is LockIntent.UPDATE` etc.

---

## 3. Double sqlglot parse (parse_sql_access + extract_row_level_access) — ✅ Done

**Files:** `_sql_cursor.py`, `_sql_parsing.py`, `_sql_predicates.py`

**What changed:**
- Implemented Option 1 from the original plan: `SqlAccessResult` now includes
  an `ast` field populated by `_sqlglot_parse()` for single-statement SQL.
- `extract_row_level_access()` and `extract_equality_predicates()` accept an
  optional `ast` keyword argument; when provided, they skip `sqlglot.parse_one()`.
- `_intercept_execute()` passes `access.ast` to `extract_row_level_access()`
  when parameters are not resolved (the common case for simple WHERE queries).
- When parameters are resolved, the SQL string changes so a fresh parse is
  needed — the AST passthrough is skipped in that case.

---

## 4. Multi-statement SQL semantic mismatch — ✅ Addressed

**Files:** `_sql_parsing.py`

**What changed:**
- `_sqlglot_parse()` now only attaches the AST to the result for
  single-statement SQL (checked via `len([e for e in expressions if e is not None]) == 1`).
- For multi-statement SQL, `ast` is `None`, which forces `extract_row_level_access()`
  to re-parse (correct behavior since the merged result doesn't carry per-statement ASTs).
- The existing `len(all_tables) == 1` guard in `_intercept_execute()` continues to
  prevent incorrect row-level predicate extraction for multi-table results.
- This is the conservative approach noted in the original plan: multi-statement
  SQL falls back to table-level detection, which is sound.
