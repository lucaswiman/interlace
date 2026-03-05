# Algorithm 1: SQL Read/Write Set Extraction

Two implementations, chosen at runtime: a regex fast-path for the 90% case (simple single-table statements), and sqlglot for everything else.

All functions return a 4-tuple: `(read_tables, write_tables, lock_intent, tx_op)`.

- `lock_intent`: `"UPDATE"` | `"SHARE"` | `None` — for SELECT FOR UPDATE/SHARE and LOCK TABLE
- `tx_op`: `"BEGIN"` | `"COMMIT"` | `"ROLLBACK"` | `"SAVEPOINT:{name}"` | `"ROLLBACK_TO:{name}"` | `"RELEASE:{name}"` | `None`

## 1a. Regex Fast-Path

```python
import re

# Precompiled patterns — matches the leading keyword + first table name.
# Handles optional schema qualification (schema.table) and quoted identifiers.
_IDENT = r'(?:"[^"]+"|`[^`]+`|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)'
_WS = r"[\s\n]+"

_RE_SELECT    = re.compile(r"\bSELECT\b", re.I)
_RE_INSERT    = re.compile(rf"\bINSERT{_WS}INTO{_WS}({_IDENT})", re.I)
_RE_UPDATE    = re.compile(rf"\bUPDATE{_WS}({_IDENT}){_WS}SET\b", re.I)
_RE_DELETE    = re.compile(rf"\bDELETE{_WS}FROM{_WS}({_IDENT})", re.I)
_RE_FROM      = re.compile(rf"\bFROM{_WS}({_IDENT})", re.I)
_RE_JOIN      = re.compile(rf"\bJOIN{_WS}({_IDENT})", re.I)
_RE_LITERAL   = re.compile(r"'[^']*'")
_RE_FOR_UPDATE = re.compile(r"\bFOR" + _WS + r"UPDATE\b", re.I)
_RE_FOR_SHARE  = re.compile(r"\bFOR" + _WS + r"SHARE\b", re.I)
_RE_LOCK_TABLE = re.compile(rf"\bLOCK{_WS}TABLE{_WS}({_IDENT})", re.I)
_RE_LOCK_MODE  = re.compile(rf"\bIN{_WS}(.+){_WS}MODE\b", re.I)
_RE_TX_BEGIN      = re.compile(r"^\s*(BEGIN|START\s+TRANSACTION|...)\b", re.I)
_RE_TX_COMMIT     = re.compile(r"^\s*(COMMIT|END|...)\b", re.I)
_RE_TX_ROLLBACK   = re.compile(r"^\s*ROLLBACK\b", re.I)
_RE_TX_SAVEPOINT  = re.compile(r"^\s*SAVEPOINT\s+(\w+)\b", re.I)
_RE_TX_RELEASE    = re.compile(r"^\s*RELEASE\s+(SAVEPOINT\s+)?(\w+)\b", re.I)
_RE_TX_ROLLBACK_TO = re.compile(r"^\s*ROLLBACK\s+TO\s+(SAVEPOINT\s+)?(\w+)\b", re.I)
```

**Key additions vs. original design:**

1. **Transaction control detection** — BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE, ROLLBACK TO are detected first and return `tx_op` (no tables).
2. **Literal stripping** — `_RE_LITERAL.sub(" ", stripped)` removes string literals before regex matching, preventing false positives from SQL like `WHERE name = 'FROM table'`.
3. **Lock intent detection** — FOR UPDATE, FOR SHARE, LOCK TABLE with mode detection.
4. **Extended bail-out** — now bails to sqlglot for `EXISTS`, `IN (`, subqueries in DELETE/UPDATE, and advisory lock functions in addition to WITH/UNION/INTERSECT/EXCEPT/MERGE/RETURNING.

```python
def _regex_parse(sql: str) -> tuple[set[str], set[str], str | None, str | None] | None:
    """Fast-path: extract tables from simple single-statement SQL.

    Returns (read_tables, write_tables, lock_intent, tx_op) or None if the SQL
    is too complex and needs the full parser.
    """
    # Transaction control checked first
    # Bail-out for: WITH, UNION, INTERSECT, EXCEPT, MERGE, RETURNING, EXISTS, IN (
    # Also bails for subqueries in DELETE/UPDATE and advisory lock functions
    # Strips string literals before table extraction
    # Detects LOCK TABLE ... IN ... MODE
    # Detects FOR UPDATE / FOR SHARE
    ...
```

## 1b. sqlglot Full Parser (Fallback)

```python
def _sqlglot_parse(sql: str) -> tuple[set[str], set[str], str | None, str | None] | None:
    """Full parser: handles CTEs, subqueries, UNION, MERGE, etc."""
```

**Key additions vs. original design:**

1. **Transaction control** — handles `exp.Transaction` (BEGIN), `exp.Commit`, `exp.Rollback`
2. **Lock intent** — extracts `exp.Lock` node from SELECT for FOR UPDATE/SHARE
3. **Advisory locks** — recognizes `pg_advisory_lock`, `pg_advisory_xact_lock`, `pg_advisory_lock_shared`, `pg_advisory_xact_lock_shared`, `get_lock` function calls. Extracts lock ID from literal arguments. Maps to `advisory_lock:{lock_id}` in write set.
4. **UNION/INTERSECT/EXCEPT** — handled as read-only compositions, collecting all tables from all branches.

## 1c. Combined Entry Point

```python
def parse_sql_access(sql: str) -> tuple[set[str], set[str], str | None, str | None]:
    """Extract (read_tables, write_tables, lock_intent, tx_op) from a SQL statement.

    Uses regex fast-path for simple statements, falls back to sqlglot
    for complex SQL. Returns empty sets if parsing fails entirely
    (endpoint-level I/O detection remains as fallback).
    """
    # Fast path: covers ~90% of ORM-generated SQL
    result = _regex_parse(sql)
    if result is not None:
        return result

    # Full parser for complex SQL
    result = _sqlglot_parse(sql)
    if result is not None:
        return result

    # Parse failure: return empty sets → endpoint-level fallback
    return set(), set(), None, None
```

**Complexity:** Regex fast-path is O(n) in SQL length with ~8 regex scans (more than original due to lock/tx patterns). sqlglot parse_one is O(n) but with higher constant factor (AST construction). Both are negligible vs. network RTT to the database.

---

## Remaining Gaps & TODOs

### TODO: Cross-Table Foreign Key Dependencies
**Issue:** Schema relationships are invisible to the parser:
```sql
-- Thread A:
INSERT INTO orders (user_id, amount) VALUES (42, 100);

-- Thread B:
DELETE FROM users WHERE id = 42;  -- FK constraint violated!
```
Parser sees: Thread A writes `orders`, Thread B writes `users` → independent.
But if FK `orders.user_id → users.id` exists, Thread B's DELETE fails (constraint violation).

**Impact:**
- False negatives (missed conflicts): Two threads operating on related tables are reported independent, but FK constraints create real conflicts
- Limited to multi-table workloads with foreign keys

**Fix (Phase 6):**
- Add schema introspection: query `information_schema.referential_constraints` (or equivalent for other DBs)
- Build FK dependency graph: `{orders → users, shipments → orders}` etc.
- At conflict detection time: if Op1 touches table T1 and Op2 touches table T2, and T1 → T2 via FK, classify as conflicting
- Cache FK graph in DPOR tls (per-connection)
- Add tests: ORM relationships, Alembic migrations with FKs

**Estimated effort:** ~150 lines + 25 tests

---

### ~~TODO: Transaction Boundaries Not Tracked~~ ✅ Done
Transaction grouping is now implemented in `_sql_cursor.py` via `_intercept_execute()`:
- `BEGIN` → sets `_io_tls._in_transaction = True`, initializes buffer
- `COMMIT` → flushes buffered I/O reports to reporter
- `ROLLBACK` → discards buffer
- `SAVEPOINT` → records buffer position for partial rollback
- `ROLLBACK TO` → truncates buffer to savepoint position
- `RELEASE` → removes savepoint marker
- DPOR scheduler skips interleaving within transactions via `_in_transaction` flag

---

### TODO: Temporal Tables & System Versioning
**Issue:** Some databases (PostgreSQL, MySQL 8.0+) support temporal/versioned tables:
```sql
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR,
  valid_from TIMESTAMP GENERATED ALWAYS AS ROW START,
  valid_to TIMESTAMP GENERATED ALWAYS AS ROW END,
  FOR SYSTEM_TIME BETWEEN valid_from AND valid_to
) WITH SYSTEM VERSIONING;

SELECT * FROM users FOR SYSTEM_TIME AS OF '2024-01-01';
```
The parser treats these as regular table access, unaware of the time-dimension semantics.

**Impact:**
- False positives: Queries at different "times" are reported as conflicting, even if they select disjoint time windows
- Rarely used, but valid in some applications

**Fix:**
- Add `FOR SYSTEM_TIME` clause parsing
- Extract temporal predicate: `AS OF '2024-01-01'`, `BETWEEN`, `ALL`, `VERSIONED BETWEEN`
- Add temporal dimension to row-level ObjectIds: `table:pk:time_bucket`
- Match against concurrent writes (which write to current `now()`, not historical times)

**Estimated effort:** ~40 lines + 10 tests (low priority)

---

### TODO: Generated/Computed Columns
**Issue:** Some ORMs and databases use computed columns:
```sql
CREATE TABLE orders (
  id INT PRIMARY KEY,
  qty INT,
  unit_price DECIMAL,
  total DECIMAL GENERATED ALWAYS AS (qty * unit_price)
);
```
When `total` is updated, the actual SQL does not include `total = ...` (it's computed). Parser should be aware.

**Impact:**
- Minimal: row-level predicate extraction might incorrectly assume `total` is user-settable

**Fix:**
- Schema introspection: fetch `GENERATED` clause for all columns
- Mark computed columns in ObjectId derivation (informational only)
- Ignore computed columns in row-level predicate matching

**Estimated effort:** ~30 lines + 5 tests (low priority)

---

### TODO: RETURNING Clause with Multiple Tables
**Issue:** PostgreSQL `RETURNING` in `UPDATE...FROM`:
```sql
UPDATE orders SET status = 'shipped' FROM shipments
WHERE orders.id = shipments.order_id
RETURNING orders.id, shipments.id;
```
The `RETURNING` clause lists outputs but doesn't affect read/write semantics. However, some ORMs use `RETURNING` to fetch results.

**Impact:**
- Minimal: current handler correctly identifies `orders` (write) + `shipments` (read)
- `RETURNING` doesn't change the classification

**Fix:**
- Already handled: `_regex_parse` bails on `RETURNING`, sqlglot handles it correctly
- Add test for `RETURNING` with multi-table UPDATE

**Estimated effort:** ~5 lines + 2 tests

---

### TODO: Window Functions & Partitioning
**Issue:** Window functions reference tables but have implicit grouping semantics:
```sql
SELECT id, name, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rank
FROM employees;
```
Parser extracts `employees` as a read, which is correct, but the PARTITION BY semantics are invisible.

**Impact:**
- Minimal: row-level detection would be overly conservative (all rows involved in window function are interdependent)

**Fix:**
- Recognize `OVER (PARTITION BY ...)` clauses
- If window function present, fall back to table-level (not row-level)
- Mark in ObjectId: `table:window_function` to distinguish from regular table access

**Estimated effort:** ~20 lines + 3 tests (low priority)

---

### TODO: Prepared Statements & Caching
**Issue:** When using prepared statements with parameter placeholders, different parameter values can affect row-level conflict detection:
```python
stmt = cursor.prepare("SELECT * FROM users WHERE id = ? FOR UPDATE")
cursor.execute(stmt, (1,))  # locks row 1
cursor.execute(stmt, (2,))  # locks row 2 — independent!
```
Current implementation resolves parameters at execute-time, which is correct. However, caching the resolved ObjectId across multiple executions of the same prepared statement with different parameters could miss conflicts.

**Impact:**
- Minimal: current implementation resolves at execute-time, no caching issue
- Potential issue if future optimization adds ObjectId caching

**Fix:**
- Ensure `_sql_cursor.py` resolves parameters fresh for each `execute()` call
- Add test: repeated prepared statement execution with different parameters
- Document in `_sql_params.py`: never cache ObjectIds across executions

**Estimated effort:** ~10 lines + 3 tests

---

### TODO: Stored Procedures & Dynamic SQL
**Issue:** Stored procedures and dynamic SQL (`EXECUTE`, `PREPARE`, PL/pgSQL, T-SQL) are opaque:
```sql
CALL sp_update_user(1, 'new_name');  -- Rust calls unknown SQL inside
EXECUTE 'SELECT * FROM ' || table_name;  -- table_name determined at runtime
```
Parser cannot determine which tables are accessed without executing the procedure.

**Impact:**
- False negatives: Stored procedures are treated as endpoint-level (socket) I/O, not table-level
- Missing optimization: stored proc calls could expose table access via introspection

**Fix (Advanced):**
- Intercept stored procedure definitions and parse their SQL bodies
- Cache computed table access sets: `sp_update_user → {users:write}`
- At call-site, substitute cached access instead of endpoint-level

**Estimated effort:** ~200 lines + 40 tests (very low priority, high complexity)

---

### TODO: Multi-Dialect SQL Differences
**Issue:** The parser must handle SQL from multiple dialects (PostgreSQL, MySQL, SQLite, etc.), but some syntax differs:
- PostgreSQL: `RETURNING`, `ON CONFLICT`, `FOR UPDATE SKIP LOCKED`
- MySQL: `ON DUPLICATE KEY UPDATE`, `SELECT ... INTO @var`
- SQLite: `AUTOINCREMENT`, `INSERT OR REPLACE`

**Impact:**
- Parse failures for dialect-specific syntax → fallback to endpoint-level

**Fix:**
- Extend sqlglot dialect support: already handles 30+ dialects
- Add integration tests for each supported dialect
- Document supported dialects in README

**Estimated effort:** ~0 lines (sqlglot already handles) + 30 tests

---

### TODO: Performance Optimization: Regex Pattern Precompilation
**Issue:** Current regex patterns are compiled at module load time, which is good. However, the patterns could be optimized:
- `_IDENT` pattern is complex; could use a simpler DFA-based approach
- Multiple passes over the SQL string (~8 regex scans) could be combined into a single state machine

**Impact:**
- Minimal: regex overhead is negligible vs. network RTT
- Readable code is more important than micro-optimization

**Fix (Optimization, not required):**
- Profile regex performance on large SQL strings (>100 KB)
- Consider Rust regex engine via PyO3 if needed
- Benchmark: regex fast-path vs sqlglot on typical ORM SQL

**Estimated effort:** ~50 lines + benchmarks (low priority)
