# Algorithm 1: SQL Read/Write Set Extraction

Two implementations, chosen at runtime: a regex fast-path for the 90% case (simple single-table statements), and sqlglot for everything else.

## 1a. Regex Fast-Path

```python
import re

# Precompiled patterns — matches the leading keyword + first table name.
# Handles optional schema qualification (schema.table) and quoted identifiers.
_IDENT = r'(?:"[^"]+"|`[^`]+`|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)'
_WS = r'[\s\n]+'

_RE_SELECT    = re.compile(rf'\bSELECT\b', re.I)
_RE_INSERT    = re.compile(rf'\bINSERT{_WS}INTO{_WS}({_IDENT})', re.I)
_RE_UPDATE    = re.compile(rf'\bUPDATE{_WS}({_IDENT}){_WS}SET\b', re.I)
_RE_DELETE    = re.compile(rf'\bDELETE{_WS}FROM{_WS}({_IDENT})', re.I)
_RE_FROM      = re.compile(rf'\bFROM{_WS}({_IDENT})', re.I)
_RE_JOIN      = re.compile(rf'\bJOIN{_WS}({_IDENT})', re.I)

def _strip_quotes(name: str) -> str:
    """Remove surrounding quotes/backticks and extract table from schema.table."""
    if name.startswith(('"', '`')):
        name = name[1:-1]
    # Take last component: "public"."users" → users
    return name.rsplit('.', 1)[-1]

def _regex_parse(sql: str) -> tuple[set[str], set[str]] | None:
    """Fast-path: extract tables from simple single-statement SQL.

    Returns (read_tables, write_tables) or None if the SQL is too complex
    (subqueries, CTEs, UNION, MERGE) and needs the full parser.
    """
    stripped = sql.strip().rstrip(';').strip()

    # Bail to full parser for complex SQL
    upper = stripped.upper()
    if any(kw in upper for kw in ('WITH ', 'UNION', 'INTERSECT', 'EXCEPT', 'MERGE', 'RETURNING')):
        return None

    read: set[str] = set()
    write: set[str] = set()

    m_insert = _RE_INSERT.search(stripped)
    if m_insert:
        write.add(_strip_quotes(m_insert.group(1)))
        # Source tables in INSERT ... SELECT ... FROM
        for m in _RE_FROM.finditer(stripped, m_insert.end()):
            read.add(_strip_quotes(m.group(1)))
        return read, write

    m_update = _RE_UPDATE.search(stripped)
    if m_update:
        tbl = _strip_quotes(m_update.group(1))
        write.add(tbl)
        read.add(tbl)  # WHERE reads the target
        # Subquery tables in FROM/JOIN (UPDATE ... FROM ... syntax)
        for m in _RE_FROM.finditer(stripped, m_update.end()):
            t = _strip_quotes(m.group(1))
            if t not in write:
                read.add(t)
        return read, write

    m_delete = _RE_DELETE.search(stripped)
    if m_delete:
        tbl = _strip_quotes(m_delete.group(1))
        write.add(tbl)
        read.add(tbl)
        return read, write

    if _RE_SELECT.search(stripped):
        for m in _RE_FROM.finditer(stripped):
            read.add(_strip_quotes(m.group(1)))
        for m in _RE_JOIN.finditer(stripped):
            read.add(_strip_quotes(m.group(1)))
        return read, write

    return None  # unknown statement type → fall through
```

## 1b. sqlglot Full Parser (Fallback)

```python
def _sqlglot_parse(sql: str) -> tuple[set[str], set[str]] | None:
    """Full parser: handles CTEs, subqueries, UNION, MERGE, etc."""
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return None

    try:
        ast = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError:
        return None  # unparseable → fall back to endpoint-level

    write: set[str] = set()
    read: set[str] = set()

    if isinstance(ast, exp.Insert):
        tbl = ast.find(exp.Table)
        if tbl:
            write.add(tbl.name)
        # Source tables (everything after the target)
        if ast.expression:  # the SELECT source
            for t in ast.expression.find_all(exp.Table):
                if t.name not in write:
                    read.add(t.name)
        return read, write

    if isinstance(ast, exp.Update):
        tbl = ast.this
        if isinstance(tbl, exp.Table):
            write.add(tbl.name)
            read.add(tbl.name)
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write

    if isinstance(ast, exp.Delete):
        tbl = ast.this
        if isinstance(tbl, exp.Table):
            write.add(tbl.name)
            read.add(tbl.name)
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write

    if isinstance(ast, exp.Select):
        for t in ast.find_all(exp.Table):
            read.add(t.name)
        return read, write

    if isinstance(ast, exp.Merge):
        target = ast.this
        if isinstance(target, exp.Table):
            write.add(target.name)
            read.add(target.name)
        using = ast.find(exp.Table, bfs=False)
        # All non-target tables are read sources
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write

    # DDL, GRANT, etc. — conservatively treat as write
    for t in ast.find_all(exp.Table):
        write.add(t.name)
    return read, write
```

## 1c. Combined Entry Point

```python
def parse_sql_access(sql: str) -> tuple[set[str], set[str]]:
    """Extract (read_tables, write_tables) from a SQL statement.

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
    return set(), set()
```

**Complexity:** Regex fast-path is O(n) in SQL length with ~6 regex scans. sqlglot parse_one is O(n) but with higher constant factor (AST construction). Both are negligible vs. network RTT to the database.

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

**Fix (Phase 4):**
- Add schema introspection: query `information_schema.referential_constraints` (or equivalent for other DBs)
- Build FK dependency graph: `{orders → users, shipments → orders}` etc.
- At conflict detection time: if Op1 touches table T1 and Op2 touches table T2, and T1 → T2 via FK, classify as conflicting
- Cache FK graph in DPOR tls (per-connection)
- Add tests: ORM relationships, Alembic migrations with FKs

**Estimated effort:** ~150 lines + 25 tests

---

### TODO: Transaction Boundaries Not Tracked
**Issue:** Current model treats each SQL statement independently:
```python
# Thread A:
cursor.execute("BEGIN")
cursor.execute("SELECT * FROM accounts WHERE id = 1")  # read
cursor.execute("UPDATE accounts SET balance = ...")     # write
cursor.execute("COMMIT")

# Thread B:
cursor.execute("BEGIN")
cursor.execute("SELECT * FROM accounts WHERE id = 1")  # read
cursor.execute("UPDATE accounts SET balance = ...")     # write
cursor.execute("COMMIT")
```
DPOR sees 4 separate operations with potential interleavings. In reality, the `BEGIN...COMMIT` bundle is atomic.

**Impact:**
- Explosion of search space: unnecessary interleavings between statements in committed transactions
- Missed optimization: can suppress threads as a group during transaction

**Fix (Phase 4):**
- Track transaction open/close via `cursor.execute("BEGIN")`, `cursor.execute("COMMIT")`, `cursor.execute("ROLLBACK")`
- Group SQL operations into transaction-level ObjectIds: `tx:1`, `tx:2`
- Modify DPOR suppression: suppress entire transaction at once
- Add tests: explicit transactions, savepoints, nested transactions

**Estimated effort:** ~80 lines + 20 tests

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
- Multiple passes over the SQL string (6 regex scans) could be combined into a single state machine

**Impact:**
- Minimal: regex overhead is negligible vs. network RTT
- Readable code is more important than micro-optimization

**Fix (Optimization, not required):**
- Profile regex performance on large SQL strings (>100 KB)
- Consider Rust regex engine via PyO3 if needed
- Benchmark: regex fast-path vs sqlglot on typical ORM SQL

**Estimated effort:** ~50 lines + benchmarks (low priority)
