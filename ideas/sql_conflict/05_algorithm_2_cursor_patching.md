# Algorithm 2: DBAPI Cursor Monkey-Patching

Uses **factory injection** for C-extension cursors (sqlite3, psycopg2) and **direct class patching** for pure-Python drivers (pymysql). Key insight: C-extension cursor classes cannot be monkey-patched with `setattr`, so we subclass them and inject the subclass via `connect()` factory arguments.

The `paramstyle` is read from the driver module at patch time and baked into the traced cursor class, so each driver uses its native placeholder style automatically.

## Core Interception Logic

```python
# frontrun/_sql_cursor.py

def _intercept_execute(
    original_method, self, operation, parameters=None,
    *, is_executemany=False, paramstyle="format",
):
    """Intercept a single execute/executemany call.

    1. parse_sql_access(operation) → (read_tables, write_tables, lock_intent, tx_op)
    2. Handle transaction control (BEGIN/COMMIT/ROLLBACK/SAVEPOINT)
    3. Resolve parameters → extract row-level predicates
    4. Report table accesses (or buffer if in transaction)
    5. Suppress endpoint-level I/O during original call
    """
    reporter = get_io_reporter()
    if reporter is not None and isinstance(operation, str):
        read_tables, write_tables, lock_intent, tx_op = parse_sql_access(operation)

        # Transaction grouping: buffer reports during transactions
        if tx_op == "BEGIN":
            _io_tls._in_transaction = True
            _io_tls._tx_buffer = []
            _io_tls._tx_savepoints = {}
        elif tx_op == "COMMIT":
            _io_tls._in_transaction = False
            for res_id, kind in _io_tls._tx_buffer:
                reporter(res_id, kind)
            # ... clear buffer
        elif tx_op == "ROLLBACK":
            _io_tls._in_transaction = False
            # ... discard buffer
        elif tx_op.startswith("SAVEPOINT:"):
            # Record buffer position for partial rollback
        elif tx_op.startswith("ROLLBACK_TO:"):
            # Truncate buffer to savepoint position

        # Row-level predicates (single-table, non-executemany only)
        if len(all_tables) == 1 and not is_executemany:
            resolved = resolve_parameters(operation, parameters, paramstyle)
            predicates = extract_equality_predicates(resolved)

        for table in read_tables:
            kind = "write" if lock_intent == "UPDATE" else "read"
            report_or_buffer(table, kind)
        for table in write_tables:
            report_or_buffer(table, "write")

    # Suppress endpoint-level I/O
    with _suppress_endpoint_io():
        return original_method(self, operation, parameters)
```

## Patching Strategies

### sqlite3 — Factory Injection

C-extension cursors (`sqlite3.Cursor`) cannot be monkey-patched with `setattr`. Instead:

1. Create `TracedCursor(sqlite3.Cursor)` subclass with overridden `execute`/`executemany`
2. Create `TracedConnection(sqlite3.Connection)` subclass whose `cursor()` returns `TracedCursor`
3. On Python 3.14+, also override `Connection.execute`/`executemany` (C-level fast path bypasses `cursor()`)
4. Patch `sqlite3.connect` to inject `TracedConnection` as the `factory` kwarg

### psycopg2 — Cursor Factory Injection

Similar to sqlite3: create `TracedCursor(psycopg2.extensions.cursor)` and patch `psycopg2.connect` to set `cursor_factory=TracedCursor`.

### pymysql — Direct Class Patching

Pure-Python driver: directly replace `pymysql.cursors.Cursor.execute` and `executemany` with wrapped versions.

## Key Functions

```python
patch_sql() -> None         # Monkey-patch all known DBAPI drivers
unpatch_sql() -> None       # Restore original implementations
is_tid_suppressed(tid: int) -> bool  # Check thread suppression (for LD_PRELOAD bridge)
```

## Suppression Infrastructure

```python
_suppress_tids: set[int] = set()  # OS thread IDs currently in sql-suppress mode
_suppress_lock = threading.Lock()  # real lock, not cooperative

@contextlib.contextmanager
def _suppress_endpoint_io():
    """Context manager: suppress endpoint I/O for current thread."""
    tid = threading.get_native_id()
    _io_tls._sql_suppress = True
    with _suppress_lock:
        _suppress_tids.add(tid)
    try:
        yield
    finally:
        with _suppress_lock:
            _suppress_tids.discard(tid)
        _io_tls._sql_suppress = False
```
