# Algorithm 2: DBAPI Cursor Monkey-Patching

Follows the exact pattern of `_io_detection.py`. Key insight: psycopg2's cursor is a C extension — `cursor.execute` is a C function. We can still monkey-patch at the class level because Python attribute lookup goes through the MRO, and assigning to the class replaces the descriptor.

The `paramstyle` is read from the driver module at patch time and baked into the closure, so each driver uses its native placeholder style automatically.

```python
# frontrun/_sql_detection.py

import threading
from frontrun._io_detection import _io_tls, get_io_reporter

_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}
_sql_patched = False

def _make_patched_execute(original, paramstyle, *, is_executemany=False):
    """Create a patched execute that intercepts SQL before calling the original."""

    def _patched_execute(self, operation, parameters=None, *args, **kwargs):
        reporter = get_io_reporter()
        reported = False

        if reporter is not None and isinstance(operation, str):
            read_tables, write_tables = parse_sql_access(operation)
            if read_tables or write_tables:
                reported = True
                all_tables = read_tables | write_tables

                # Row-level: resolve params → extract predicates for
                # single-table operations.  Skip for executemany (each
                # row may have different params) and multi-table queries
                # (can't attribute columns to tables without aliases).
                predicates: list[EqualityPredicate] = []
                if len(all_tables) == 1 and not is_executemany:
                    if parameters is not None:
                        resolved = resolve_parameters(
                            operation, parameters, paramstyle,
                        )
                        predicates = extract_equality_predicates(resolved)
                    else:
                        predicates = extract_equality_predicates(operation)

                for table in read_tables:
                    reporter(_sql_resource_id(table, predicates), "read")
                for table in write_tables:
                    reporter(_sql_resource_id(table, predicates), "write")

        # Suppress endpoint-level I/O for this call if SQL-level succeeded
        if reported:
            tid = threading.get_native_id()
            _io_tls._sql_suppress = True
            with _suppress_lock:
                _suppress_tids.add(tid)
            try:
                if parameters is not None:
                    return original(self, operation, parameters, *args, **kwargs)
                return original(self, operation, *args, **kwargs)
            finally:
                with _suppress_lock:
                    _suppress_tids.discard(tid)
                _io_tls._sql_suppress = False
        else:
            if parameters is not None:
                return original(self, operation, parameters, *args, **kwargs)
            return original(self, operation, *args, **kwargs)

    return _patched_execute


# Target drivers: (module_path, class_name, method_name, paramstyle_module)
_CURSOR_TARGETS = [
    ("psycopg2.extensions", "cursor", "execute", "psycopg2"),
    ("psycopg2.extensions", "cursor", "executemany", "psycopg2"),
    ("psycopg.cursor", "Cursor", "execute", "psycopg"),
    ("psycopg.cursor_async", "AsyncCursor", "execute", "psycopg"),
    ("sqlite3", "Cursor", "execute", "sqlite3"),
    ("sqlite3", "Cursor", "executemany", "sqlite3"),
    ("pymysql.cursors", "Cursor", "execute", "pymysql"),
    ("pymysql.cursors", "Cursor", "executemany", "pymysql"),
]


def patch_sql() -> None:
    """Monkey-patch DBAPI cursor.execute() for known drivers."""
    global _sql_patched
    if _sql_patched:
        return

    import importlib
    for module_path, class_name, method_name, paramstyle_module in _CURSOR_TARGETS:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            original = getattr(cls, method_name)
            key = (cls, method_name)
            if key in _ORIGINAL_METHODS:
                continue

            # Read paramstyle from the driver module (PEP 249)
            driver_mod = importlib.import_module(paramstyle_module)
            paramstyle = getattr(driver_mod, "paramstyle", "format")

            _ORIGINAL_METHODS[key] = original
            is_many = "executemany" in method_name
            patched = _make_patched_execute(
                original, paramstyle, is_executemany=is_many,
            )
            setattr(cls, method_name, patched)
        except (ImportError, AttributeError):
            pass  # driver not installed — skip silently

    _sql_patched = True


def unpatch_sql() -> None:
    """Restore original DBAPI cursor methods."""
    global _sql_patched
    if not _sql_patched:
        return
    for (cls, method_name), original in _ORIGINAL_METHODS.items():
        setattr(cls, method_name, original)
    _ORIGINAL_METHODS.clear()
    _sql_patched = False
```
