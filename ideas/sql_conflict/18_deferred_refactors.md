# Deferred Refactors

Issues identified during the 2026-03-06 code review that are valid but too large
or risky to address in-line. Each entry explains the problem, why it was deferred,
and what a fix would look like.

---

## 1. Return tuple → NamedTuple

**Files:** `_sql_parsing.py`, `_sql_cursor.py`, all test files

`parse_sql_access()` returns a 5-element tuple:

```python
tuple[set[str], set[str], str | None, str | None, dict[str, str] | None]
```

Every call site unpacks positionally (`r, w, lock, tx, temporal = …`), and
the fallback stub in `_sql_cursor.py` must mirror the full signature. Adding
a 6th element will require touching every call site again.

**Fix:** Replace with a `NamedTuple` or `@dataclass`:

```python
class SqlAccessResult(NamedTuple):
    read_tables: set[str]
    write_tables: set[str]
    lock_intent: str | None
    tx_op: str | None
    temporal_clauses: dict[str, str] | None
```

**Why deferred:** Touches ~80+ call sites across production code and tests.
Mechanical but high-churn change best done in isolation.

---

## 2. Stringly-typed `tx_op` and `lock_intent`

**Files:** `_sql_parsing.py`, `_sql_cursor.py`

`tx_op` uses raw strings like `"BEGIN"`, `"COMMIT"`, `"ROLLBACK"`,
`"SAVEPOINT:name"`, `"ROLLBACK_TO:name"`, `"RELEASE:name"`. The compound
forms use colon-delimited string splitting. `lock_intent` uses `"UPDATE"` /
`"SHARE"` compared with `==`.

A typo in any of these strings produces a silent bug.

**Fix:** Use `enum.Enum` for simple cases, and a small dataclass for compound
operations:

```python
class TxOp(enum.Enum):
    BEGIN = "BEGIN"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"

@dataclass(frozen=True)
class Savepoint:
    name: str

@dataclass(frozen=True)
class RollbackTo:
    name: str
```

**Why deferred:** Systemic change across parsing and cursor modules.
Requires updating all producers and consumers simultaneously.

---

## 3. Double sqlglot parse (parse_sql_access + extract_row_level_access)

**Files:** `_sql_cursor.py`, `_sql_parsing.py`, `_sql_predicates.py`

In `_intercept_execute`, the SQL is parsed once by `parse_sql_access()`
(which may invoke `sqlglot.parse()` via `_sqlglot_parse`) and then parsed
again by `extract_row_level_access()` (which calls `sqlglot.parse_one()`).
Each sqlglot parse costs ~0.5–2 ms depending on complexity.

**Fix options:**

1. Have `parse_sql_access` optionally return the parsed AST alongside the
   result tuple, so `extract_row_level_access` can accept a pre-parsed AST.
2. Cache the most recent parse result in a module-level variable (keyed by
   SQL string identity).
3. Merge predicate extraction into `_sqlglot_parse` so it happens in the
   same pass.

Option 3 is cleanest but couples parsing and predicate extraction. Option 1
is lowest risk.

**Why deferred:** Requires API restructuring across module boundaries. The
current overhead is acceptable for typical ORM workloads (most SQL hits the
regex fast-path and never reaches sqlglot).

---

## 4. Multi-statement SQL semantic mismatch

**Files:** `_sql_parsing.py`, `_sql_cursor.py`

`_sqlglot_parse` now uses `sqlglot.parse()` (multi-statement) and merges all
statements' reads/writes into single sets, taking "the last" `tx_op`. But
`_intercept_execute` processes one `execute()` call at a time, and
row-level predicate extraction (`pred_rows`) is computed once for all tables
combined.

For `"INSERT INTO a VALUES(1); DELETE FROM b WHERE id=2"`, the merged result
loses the per-statement distinction needed for correct row-level predicates.
The `len(all_tables) == 1` guard partially mitigates this (multi-table
results skip row-level), but the semantic mismatch is a latent issue.

**Fix:** Either:
- Split multi-statement SQL in `_intercept_execute` and process each
  statement independently (correct but complex).
- Return per-statement results from `_sqlglot_parse` and let the caller
  decide how to merge.

**Why deferred:** Multi-statement `cursor.execute()` calls are rare in
practice (ORMs always send single statements). The current behavior is
conservative (falls back to table-level), not incorrect.
