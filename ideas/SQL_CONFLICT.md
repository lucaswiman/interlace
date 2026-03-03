# SQL Resource Conflict Detection — Implementation Plan

## Problem

All SQL to the same `(host, port)` collapses to a single DPOR `ObjectId`. Two threads doing `INSERT INTO logs` and `SELECT * FROM users` appear to conflict because both produce `send()`/`recv()` to `localhost:5432`. DPOR explores O(n!) interleavings that can never actually race. Similarly, two concurrent `SELECT`s on the same table look like write-write conflicts (because they're both "writes to the socket").

**Fix:** Parse SQL at the DBAPI layer, derive per-table `ObjectId`s with correct `AccessKind` (Read/Write), and suppress the coarser endpoint-level reports.

---

## Architecture Overview

```
cursor.execute(sql, params)
    │
    ├── _patched_execute() intercepts             ← frontrun/_sql_detection.py (new)
    │     │
    │     ├── parse_sql_access(sql)               ← sqlglot or regex fast-path
    │     │     returns (read_tables, write_tables)
    │     │
    │     ├── for table in read_tables:
    │     │     io_reporter(f"sql:{table}", "read")
    │     ├── for table in write_tables:
    │     │     io_reporter(f"sql:{table}", "write")
    │     │
    │     ├── _io_tls._sql_suppress = True        ← suppress endpoint-level
    │     ├── original_execute(sql, params)        ← actual DB call
    │     └── _io_tls._sql_suppress = False
    │
    └── socket.send() → LD_PRELOAD send()
          │
          ├── _report_socket_io() checks _sql_suppress
          │     → SKIPPED (sql-level already reported)
          │
          └── _PreloadBridge.listener() checks _sql_suppress
                → SKIPPED
```

The io_reporter callback is the *same* one already installed by `_setup_dpor_tls` in `dpor.py:1445`. SQL detection just calls it with `"sql:{table}"` resource IDs instead of `"socket:host:port"`. No changes to the Rust engine or PyO3 bindings are needed — the existing `report_io_access(execution, thread_id, object_key, kind)` interface is sufficient.

---

## File-by-File Changes

### 1. `frontrun/_sql_detection.py` (new file)

Core module. ~150 lines.

### 2. `frontrun/_io_detection.py` (modify)

Add `_sql_suppress` check to `_report_socket_io` and `_traced_open`.

### 3. `frontrun/dpor.py` (modify)

Call `patch_sql()` / `unpatch_sql()` alongside existing `patch_io()` / `unpatch_io()`.

### 4. `frontrun/bytecode.py` (modify)

Same — call `patch_sql()` / `unpatch_sql()` for the random explorer.

### 5. `pyproject.toml` (modify)

Add `sqlglot` to optional `[sql]` extra; add to integration test extra.

### 6. `tests/test_sql_detection.py` (new file)

Unit tests for the parser (no database needed).

### 7. `tests/test_integration_orm.py` (modify)

Add tests for table-level conflict reduction.

---

## Algorithm 1: SQL Read/Write Set Extraction

Two implementations, chosen at runtime: a regex fast-path for the 90% case (simple single-table statements), and sqlglot for everything else.

### 1a. Regex Fast-Path

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

### 1b. sqlglot Full Parser (Fallback)

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

### 1c. Combined Entry Point

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

## Algorithm 2: DBAPI Cursor Monkey-Patching

Follows the exact pattern of `_io_detection.py`. Key insight: psycopg2's cursor is a C extension — `cursor.execute` is a C function. We can still monkey-patch at the class level because Python attribute lookup goes through the MRO, and assigning to the class replaces the descriptor.

```python
# frontrun/_sql_detection.py

import threading
from frontrun._io_detection import _io_tls, get_io_reporter

_ORIGINAL_EXECUTE: dict[type, Any] = {}
_ORIGINAL_EXECUTEMANY: dict[type, Any] = {}
_sql_patched = False

def _make_patched_execute(original):
    """Create a patched execute that intercepts SQL before calling the original."""

    def _patched_execute(self, operation, parameters=None, *args, **kwargs):
        reporter = get_io_reporter()
        reported = False

        if reporter is not None and isinstance(operation, str):
            read_tables, write_tables = parse_sql_access(operation)
            if read_tables or write_tables:
                reported = True
                for table in read_tables:
                    reporter(f"sql:{table}", "read")
                for table in write_tables:
                    reporter(f"sql:{table}", "write")

        # Suppress endpoint-level I/O for this call if SQL-level succeeded
        if reported:
            _io_tls._sql_suppress = True
        try:
            if parameters is not None:
                return original(self, operation, parameters, *args, **kwargs)
            return original(self, operation, *args, **kwargs)
        finally:
            if reported:
                _io_tls._sql_suppress = False

    return _patched_execute


# Target drivers: (module_path, class_name, method_name)
_CURSOR_TARGETS = [
    ("psycopg2.extensions", "cursor", "execute"),
    ("psycopg2.extensions", "cursor", "executemany"),
    ("psycopg.cursor", "Cursor", "execute"),
    ("psycopg.cursor_async", "AsyncCursor", "execute"),
    ("sqlite3", "Cursor", "execute"),
    ("sqlite3", "Cursor", "executemany"),
    ("pymysql.cursors", "Cursor", "execute"),
    ("pymysql.cursors", "Cursor", "executemany"),
]


def patch_sql() -> None:
    """Monkey-patch DBAPI cursor.execute() for known drivers."""
    global _sql_patched
    if _sql_patched:
        return

    import importlib
    for module_path, class_name, method_name in _CURSOR_TARGETS:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            original = getattr(cls, method_name)
            key = (cls, method_name)
            _ORIGINAL_EXECUTE[key] = original
            setattr(cls, method_name, _make_patched_execute(original))
        except (ImportError, AttributeError):
            pass  # driver not installed — skip silently

    _sql_patched = True


def unpatch_sql() -> None:
    """Restore original DBAPI cursor methods."""
    global _sql_patched
    if not _sql_patched:
        return
    for (cls, method_name), original in _ORIGINAL_EXECUTE.items():
        setattr(cls, method_name, original)
    _ORIGINAL_EXECUTE.clear()
    _sql_patched = False
```

---

## Algorithm 3: Endpoint Suppression

Modify `_io_detection.py:_report_socket_io` and `_PreloadBridge.listener`:

```python
# In _io_detection.py — modify _report_socket_io:

def _report_socket_io(sock: socket.socket, kind: str) -> None:
    """Report a socket I/O event to the per-thread reporter, if installed."""
    # Skip if SQL-level detection already reported for this cursor.execute call
    if getattr(_io_tls, "_sql_suppress", False):
        return
    reporter = get_io_reporter()
    if reporter is not None:
        resource_id = _socket_resource_id(sock)
        if resource_id is not None:
            reporter(resource_id, kind)
```

For LD_PRELOAD events (C-level `send()`/`recv()` from libpq), the suppression is trickier because the `_PreloadBridge` listener runs on a *different* thread (the pipe reader). The thread that called `cursor.execute()` has `_sql_suppress=True` in its TLS, but the LD_PRELOAD event arrives on the dispatcher thread.

**Solution:** Use the existing `_PreloadBridge._tid_to_dpor` mapping. When `_patched_execute` sets `_sql_suppress`, also store the OS thread ID in a shared set. The bridge listener checks this set:

```python
# In _sql_detection.py:
_suppress_tids: set[int] = set()  # OS thread IDs currently in sql-suppress mode
_suppress_lock = threading.Lock()  # real lock, not cooperative

# In _patched_execute, around the original call:
tid = threading.get_native_id()
with _suppress_lock:
    _suppress_tids.add(tid)
try:
    return original(self, operation, parameters)
finally:
    with _suppress_lock:
        _suppress_tids.discard(tid)
    _io_tls._sql_suppress = False

# In _PreloadBridge.listener (dpor.py), add early exit:
def listener(self, event):
    if not self._active:
        return
    if event.kind == "close":
        return
    # Skip if this thread's cursor.execute() already reported at SQL level
    from frontrun._sql_detection import is_tid_suppressed
    if is_tid_suppressed(event.tid):
        return
    # ... existing logic ...
```

---

## Algorithm 4: ObjectId Derivation

Uses the existing `_make_object_key` function from `dpor.py:569`:

```python
def _make_object_key(obj_id: int, name: Any) -> int:
    """Create a non-negative u64 object key for the Rust engine."""
    return hash((obj_id, name)) & 0xFFFFFFFFFFFFFFFF
```

For SQL, the resource_id is `"sql:{table}"`. The io_reporter in `_setup_dpor_tls` already computes:

```python
object_key = _make_object_key(hash(resource_id), resource_id)
```

So `"sql:accounts"` and `"sql:users"` get different `ObjectId`s. Two threads on different tables → different `ObjectId`s → no conflict → DPOR skips the interleaving.

Two threads on the same table with one reading and one writing → same `ObjectId` `hash("sql:accounts")`, but one reports `kind="read"` and the other `kind="write"` → the Rust engine's `ObjectState::dependent_accesses` correctly identifies this as a RW conflict → DPOR explores the interleaving.

Two threads both doing `SELECT` on the same table → same `ObjectId`, both `kind="read"` → `dependent_accesses(Read, thread_id)` returns only writes by other threads → no writes → no conflict → DPOR skips.

**This is the key correctness property: the Adya Read/Write classification maps directly to `AccessKind::Read`/`AccessKind::Write`, and `ObjectState::dependent_accesses` implements exactly the right conflict rules.**

---

## Algorithm 5: Row-Level Predicate Extraction (Phase 2)

Extract WHERE clause predicates from the parsed SQL and encode them as part of the ObjectId, enabling two operations on the same table but different rows to be independent.

### 5a. Predicate Extraction via sqlglot

```python
from sqlglot import exp

@dataclass(frozen=True)
class EqualityPredicate:
    """A simple WHERE column = literal predicate."""
    column: str
    value: str  # string representation of the literal

def extract_equality_predicates(sql: str) -> list[EqualityPredicate]:
    """Extract simple equality predicates from a SQL WHERE clause.

    Only extracts conjuncts of the form `column = literal` (ANDed together).
    Returns empty list for OR, IN, BETWEEN, subqueries, function calls, etc.
    """
    import sqlglot
    try:
        ast = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError:
        return []

    where = ast.find(exp.Where)
    if where is None:
        return []

    predicates: list[EqualityPredicate] = []
    predicate_expr = where.this

    # Flatten ANDs into individual conjuncts
    conjuncts: list[exp.Expression]
    if isinstance(predicate_expr, exp.And):
        conjuncts = list(predicate_expr.flatten())
    else:
        conjuncts = [predicate_expr]

    for conjunct in conjuncts:
        if not isinstance(conjunct, exp.EQ):
            continue  # skip non-equality predicates
        left, right = conjunct.this, conjunct.expression
        # Normalize: column on left, literal on right
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            predicates.append(EqualityPredicate(left.name, right.this))
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            predicates.append(EqualityPredicate(right.name, left.this))

    return predicates
```

### 5b. Row-Level ObjectId

When predicates are available, derive a finer-grained ObjectId that encodes the table *and* the row predicate:

```python
def sql_row_object_key(table: str, predicates: list[EqualityPredicate]) -> int:
    """Derive ObjectId from table + WHERE equality predicates.

    If predicates are present, the key includes them — so
    "UPDATE accounts WHERE id=1" and "SELECT accounts WHERE id=2"
    get different ObjectIds and are independent.

    If no predicates, falls back to table-level key.
    """
    if not predicates:
        # No predicates → table-level granularity
        return _make_object_key(hash(f"sql:{table}"), f"sql:{table}")

    # Sort predicates for deterministic hashing
    pred_key = tuple(sorted((p.column, p.value) for p in predicates))
    resource_id = f"sql:{table}:{pred_key}"
    return _make_object_key(hash(resource_id), resource_id)
```

### 5c. Soundness: When Row-Level is Safe

Row-level ObjectIds are sound (no missed conflicts) when:

1. The WHERE clause is a conjunction of equalities on the *primary key* columns.
2. Both operations have complete primary key predicates.

If either operation has no WHERE clause (full table scan), range predicates, OR conditions, or predicates on non-key columns, we must fall back to table-level (conservative, correct).

**Decision function:**

```python
def can_use_row_level(
    table: str,
    predicates_a: list[EqualityPredicate],
    predicates_b: list[EqualityPredicate],
    pk_columns: set[str] | None,  # from schema, or None if unknown
) -> bool:
    """Can we safely use row-level ObjectIds for these two operations?"""
    if not predicates_a or not predicates_b:
        return False  # one has no WHERE → full table scan
    if pk_columns is None:
        return False  # unknown schema → conservative

    cols_a = {p.column for p in predicates_a}
    cols_b = {p.column for p in predicates_b}

    # Both must have equality predicates on ALL primary key columns
    return pk_columns <= cols_a and pk_columns <= cols_b
```

### 5d. Row-Level Disjointness (No z3 Needed for Equalities)

When `can_use_row_level` is true, two operations are independent iff their PK predicates select different rows:

```python
def pk_predicates_disjoint(
    preds_a: list[EqualityPredicate],
    preds_b: list[EqualityPredicate],
) -> bool:
    """Are two sets of PK equality predicates provably disjoint?

    True if any shared column has different values.
    E.g., (id=1) vs (id=2) → True (disjoint).
         (id=1, region='us') vs (id=1, region='eu') → True.
         (id=1) vs (id=1) → False (same row).
    """
    a_map = {p.column: p.value for p in preds_a}
    b_map = {p.column: p.value for p in preds_b}
    for col in a_map:
        if col in b_map and a_map[col] != b_map[col]:
            return True
    return False
```

This is an O(k) check where k = number of PK columns. No SMT solver needed. z3 is only needed for range predicates (`WHERE id > 10` vs `WHERE id < 5`), which can be deferred to a later phase.

---

## Algorithm 6: Wire Protocol SQL Extraction (LD_PRELOAD Enhancement)

For C-extension drivers (psycopg2 uses libpq, which calls `send()` directly), the DBAPI monkey-patch may not fire. The LD_PRELOAD library already intercepts `send()` buffers. We can parse the PostgreSQL wire protocol to extract SQL:

### 6a. PostgreSQL Simple Query Protocol

```
Byte1('Q')        — message type
Int32             — message length (including self)
String            — the SQL query text (null-terminated)
```

### 6b. PostgreSQL Extended Query Protocol (Prepared Statements)

```
Parse:    Byte1('P') Int32-len String-name String-query Int16-nparams ...
Bind:     Byte1('B') Int32-len String-portal String-stmt Int16-nformats ...
Execute:  Byte1('E') Int32-len String-portal Int32-maxrows
```

### 6c. Extraction in Rust (crates/io/)

```rust
// crates/io/src/sql_extract.rs

/// Extract SQL query text from a PostgreSQL wire protocol buffer.
/// Returns None if the buffer doesn't contain a recognizable query message.
pub fn extract_pg_query(buf: &[u8]) -> Option<&str> {
    if buf.is_empty() {
        return None;
    }
    match buf[0] {
        b'Q' => {
            // Simple query: 'Q' + i32 len + null-terminated SQL
            if buf.len() < 5 {
                return None;
            }
            let len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
            if buf.len() < 1 + len {
                return None;
            }
            let sql_bytes = &buf[5..1 + len - 1]; // exclude null terminator
            std::str::from_utf8(sql_bytes).ok()
        }
        b'P' => {
            // Parse message: 'P' + i32 len + name(str0) + query(str0) + i16 nparams
            if buf.len() < 5 {
                return None;
            }
            let len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
            if buf.len() < 1 + len {
                return None;
            }
            let payload = &buf[5..1 + len];
            // Skip statement name (null-terminated)
            let name_end = payload.iter().position(|&b| b == 0)?;
            let query_start = name_end + 1;
            let remaining = &payload[query_start..];
            let query_end = remaining.iter().position(|&b| b == 0)?;
            std::str::from_utf8(&remaining[..query_end]).ok()
        }
        _ => None,
    }
}
```

Then in the LD_PRELOAD `send()` hook, after intercepting the buffer, attempt SQL extraction. If successful, write SQL-enriched events to the pipe instead of raw socket events. The Python-side `_PreloadBridge.listener` can then parse the SQL and report at table level.

**This is Phase 3 work.** The DBAPI monkey-patching (Phase 1) covers most cases. Wire protocol parsing is only needed for C-extension drivers that bypass the Python DBAPI layer entirely (rare in practice — even psycopg2's `cursor.execute()` goes through the Python method).

---

## Integration Points

### In `dpor.py` — `explore_dpor()` function

```python
# At the top of explore_dpor(), alongside existing patch calls:
from frontrun._sql_detection import patch_sql, unpatch_sql

# In the setup block (around line 1640):
if detect_io:
    patch_io()
    patch_sql()  # NEW

# In the teardown block:
finally:
    if detect_io:
        unpatch_io()
        unpatch_sql()  # NEW
```

### In `dpor.py` — `_setup_dpor_tls()` method

No changes needed. The existing `_io_reporter` closure (line 1445) already handles any `resource_id` string. When `_patched_execute` calls `reporter("sql:accounts", "write")`, it flows through the same path:

```python
def _io_reporter(resource_id: str, kind: str) -> None:
    object_key = _make_object_key(hash(resource_id), resource_id)
    pending: list[tuple[int, str]] = _dpor_tls.pending_io
    pending.append((object_key, kind))
```

The `object_key` is derived from `hash("sql:accounts")` instead of `hash("socket:127.0.0.1:5432")`, giving table-level granularity automatically.

### In `dpor.py` — I/O flush logic (line 377)

No changes needed. The existing flush logic already handles the pending I/O events correctly:

```python
if _pending_io and getattr(_dpor_tls, "lock_depth", 0) == 0:
    for _obj_key, _io_kind in _pending_io:
        with _elock:
            _engine.report_io_access(_execution, thread_id, _obj_key, _io_kind)
    _pending_io.clear()
```

SQL-level events go through `report_io_access` (uses `io_vv`, ignores lock-based happens-before — appropriate for I/O), same as socket events.

### In the Rust engine

**No changes.** The existing `process_io_access` → `ObjectState::record_io_access` → `dependent_accesses` pipeline handles SQL table objects identically to socket objects. The `ObjectId` is just a `u64`; it doesn't know whether it represents a socket or a table.

---

## Correctness Argument

### Soundness (No missed bugs)

The system is **sound** (never claims independence when a conflict exists) because:

1. **Parse failure → fallback.** If `parse_sql_access` returns empty sets, the endpoint-level suppression doesn't activate, and the coarser socket-level conflict detection remains in effect.

2. **Conservative classification.** UPDATE/DELETE are classified as *both* read and write on their target table. DDL is classified as write to all mentioned tables. Unknown statements return empty sets.

3. **Table-level is a strict refinement.** Any two SQL operations that touch the same table get the same `ObjectId` and are correctly classified as Read or Write. The only operations that become independent are those on *different* tables — which genuinely cannot conflict at the SQL level.

4. **Row-level requires full PK.** Row-level ObjectIds are only used when both operations have complete primary key equality predicates, guaranteeing that different PK values select provably disjoint rows.

### Completeness (No false positives)

The system reduces false positives (spurious interleavings) but does not eliminate them entirely:

1. **Table-level is conservative.** Two operations on the same table with `WHERE id=1` and `WHERE id=2` are reported as conflicting (table-level), even though they touch different rows. This is fixed by Phase 2 row-level detection.

2. **Cross-table dependencies.** Foreign key relationships are invisible. Thread A inserts into `orders` (references `users.id`), Thread B deletes from `users` — these are classified as independent (different tables), but the FK constraint could cause a real conflict. This is a known limitation; fixing it requires schema-aware analysis.

---

## Test Plan

### Unit Tests (`tests/test_sql_detection.py`)

```python
class TestRegexParse:
    def test_select(self):
        r, w = _regex_parse("SELECT id, name FROM users WHERE id = 1")
        assert r == {"users"} and w == set()

    def test_insert(self):
        r, w = _regex_parse("INSERT INTO orders (user_id, amount) VALUES (1, 100)")
        assert r == set() and w == {"orders"}

    def test_insert_select(self):
        r, w = _regex_parse("INSERT INTO archive SELECT * FROM orders")
        assert r == {"orders"} and w == {"archive"}

    def test_update(self):
        r, w = _regex_parse("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
        assert r == {"accounts"} and w == {"accounts"}

    def test_delete(self):
        r, w = _regex_parse("DELETE FROM sessions WHERE expires_at < NOW()")
        assert r == {"sessions"} and w == {"sessions"}

    def test_join(self):
        r, w = _regex_parse("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id")
        assert r == {"users", "orders"} and w == set()

    def test_cte_falls_through(self):
        assert _regex_parse("WITH cte AS (SELECT 1) SELECT * FROM cte") is None

    def test_quoted_identifiers(self):
        r, w = _regex_parse('SELECT * FROM "My Table"')
        assert r == {"My Table"} and w == set()

    def test_schema_qualified(self):
        r, w = _regex_parse("SELECT * FROM public.users")
        assert r == {"users"} and w == set()


class TestSqlglotParse:
    def test_cte(self):
        r, w = _sqlglot_parse("WITH active AS (SELECT * FROM users WHERE active) SELECT * FROM active")
        assert "users" in r

    def test_subquery_in_where(self):
        r, w = _sqlglot_parse("UPDATE accounts SET status = 'closed' WHERE id IN (SELECT account_id FROM expired)")
        assert "expired" in r and "accounts" in w

    def test_update_from(self):
        r, w = _sqlglot_parse("UPDATE t1 SET t1.col = t2.col FROM t1 JOIN t2 ON t1.id = t2.id")
        assert "t2" in r and "t1" in w


class TestCombined:
    def test_simple_uses_regex(self):
        """Simple statements should not require sqlglot."""
        r, w = parse_sql_access("SELECT * FROM users")
        assert r == {"users"}

    def test_complex_uses_sqlglot(self):
        r, w = parse_sql_access("WITH x AS (SELECT 1) SELECT * FROM x JOIN y ON x.id = y.id")
        assert "y" in r

    def test_unparseable_returns_empty(self):
        r, w = parse_sql_access("GIBBERISH NOT SQL")
        assert r == set() and w == set()


class TestPredicateExtraction:
    def test_simple_equality(self):
        preds = extract_equality_predicates("SELECT * FROM users WHERE id = 42")
        assert preds == [EqualityPredicate("id", "42")]

    def test_compound_and(self):
        preds = extract_equality_predicates("UPDATE t SET x = 1 WHERE id = 1 AND region = 'us'")
        assert EqualityPredicate("id", "1") in preds
        assert EqualityPredicate("region", "us") in preds

    def test_or_returns_empty(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = 1 OR id = 2")
        assert preds == []  # OR is not a conjunction


class TestPredicateDisjointness:
    def test_different_pk_values(self):
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("id", "2")]
        assert pk_predicates_disjoint(a, b)

    def test_same_pk_values(self):
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("id", "1")]
        assert not pk_predicates_disjoint(a, b)

    def test_composite_pk_one_differs(self):
        a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        b = [EqualityPredicate("id", "1"), EqualityPredicate("region", "eu")]
        assert pk_predicates_disjoint(a, b)
```

### Integration Tests (`tests/test_integration_orm.py`)

```python
class TestOrmSqlConflictDetection:
    """Verify that SQL-level conflict detection reduces DPOR exploration."""

    def test_different_tables_independent(self, engine):
        """Two threads writing different tables: DPOR should explore 1 execution."""
        # Thread A: INSERT INTO table_a
        # Thread B: INSERT INTO table_b
        # With SQL detection: independent → 1 execution
        # Without: same socket → many executions

    def test_same_table_write_write_conflict(self, engine):
        """Two threads writing same table: DPOR should find the conflict."""
        # Thread A: UPDATE accounts SET balance = balance - 100 WHERE id = 1
        # Thread B: UPDATE accounts SET balance = balance + 50 WHERE id = 1
        # Both write "accounts" → conflict → DPOR explores interleavings

    def test_same_table_read_read_independent(self, engine):
        """Two threads reading same table: DPOR should explore 1 execution."""
        # Thread A: SELECT * FROM users WHERE id = 1
        # Thread B: SELECT * FROM users WHERE id = 2
        # Both read "users" → no conflict → 1 execution

    def test_read_write_conflict(self, engine):
        """One reader, one writer on same table: DPOR should find conflict."""
        # Thread A: SELECT balance FROM accounts WHERE id = 1
        # Thread B: UPDATE accounts SET balance = 0 WHERE id = 1
        # RW anti-dependency → conflict
```

---

## Phased Implementation

### Phase 1: Table-Level Detection (MVP)

**Goal:** Two threads on different tables are independent. Two threads on the same table use correct Read/Write classification.

Files:
- `frontrun/_sql_detection.py` — new, ~200 lines (Algorithms 1, 2, 3)
- `frontrun/_io_detection.py` — 3-line change (suppression check)
- `frontrun/dpor.py` — 4-line change (patch/unpatch calls)
- `frontrun/bytecode.py` — 4-line change (patch/unpatch calls)
- `pyproject.toml` — add `sqlglot` to extras
- `tests/test_sql_detection.py` — new, ~150 lines

### Phase 2: Row-Level Detection

**Goal:** Two threads on the same table but different rows (identified by PK equality predicates) are independent.

Files:
- `frontrun/_sql_detection.py` — add ~80 lines (Algorithms 5a-5d)
- `tests/test_sql_detection.py` — add ~50 lines (predicate tests)

### Phase 3: Wire Protocol Parsing

**Goal:** Catch C-level SQL (libpq `send()`) that bypasses DBAPI.

Files:
- `crates/io/src/sql_extract.rs` — new, ~80 lines (Algorithm 6)
- `crates/io/src/lib.rs` — integrate SQL extraction into `send()` hook

### Phase 4: Anomaly Classification

**Goal:** When DPOR finds a failing interleaving involving SQL, classify it as a specific isolation anomaly (lost update, write skew, dirty read, etc.).

Files:
- `frontrun/_sql_anomaly.py` — new, ~200 lines (DSG construction + cycle classification)
- `frontrun/common.py` — extend `InterleavingResult` with anomaly metadata

---

## Decisions Resolved

| Question | Decision | Rationale |
|----------|----------|-----------|
| Parse in Python or Rust? | **Python** (sqlglot + regex) | Parsing happens in `cursor.execute()` which is Python-side. No IPC needed. The Rust engine just sees `ObjectId`s. |
| sqlparse vs sqlglot? | **sqlglot** (with regex fast-path) | sqlparse can't extract tables from JOINs/subqueries. sqlglot has full AST + column qualification. Regex handles the 90% simple case. |
| New Rust engine methods? | **No** | Existing `report_io_access(exec, tid, obj_key, kind)` is sufficient. SQL tables are just I/O objects with table-derived `ObjectId`s. |
| Suppress LD_PRELOAD too? | **Yes** | Via `_suppress_tids` shared set (Algorithm 3). Otherwise libpq's `send()` still creates endpoint-level conflicts. |
| Row-level: z3 or equality-only? | **Equality-only for Phase 2** | Covers the common case (PK lookups). z3 adds a heavy dependency for marginal gain on range predicates. |
| Transaction boundaries? | **Deferred** | Current model: each SQL statement is an independent access. Good enough for lost-update and write-skew detection. Transaction grouping is Phase 4 work. |

---

## References

- Adya, Liskov, O'Neil. ["Generalized Isolation Level Definitions"](http://pmg.csail.mit.edu/papers/icde00.pdf). ICDE 2000.
- Berenson et al. ["A Critique of ANSI SQL Isolation Levels"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf). SIGMOD 1995.
- Kingsbury, Alvaro. [Elle](https://github.com/jepsen-io/elle). VLDB 2021.
- Cui et al. [IsoRel](https://dl.acm.org/doi/10.1145/3728953). ACM 2025.
- Jiang et al. [TxCheck](https://www.usenix.org/system/files/osdi23-jiang.pdf). OSDI 2023.
- Mao. [sqlglot](https://github.com/tobymao/sqlglot).
- Apache. [datafusion-sqlparser-rs](https://github.com/apache/datafusion-sqlparser-rs).
