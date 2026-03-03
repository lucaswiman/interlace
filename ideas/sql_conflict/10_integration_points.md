# Integration Points

## In `dpor.py` — `explore_dpor()` function

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

## In `dpor.py` — `_setup_dpor_tls()` method

No changes needed. The existing `_io_reporter` closure (line 1445) already handles any `resource_id` string. When `_patched_execute` calls `reporter("sql:accounts", "write")`, it flows through the same path:

```python
def _io_reporter(resource_id: str, kind: str) -> None:
    object_key = _make_object_key(hash(resource_id), resource_id)
    pending: list[tuple[int, str]] = _dpor_tls.pending_io
    pending.append((object_key, kind))
```

The `object_key` is derived from `hash("sql:accounts")` instead of `hash("socket:127.0.0.1:5432")`, giving table-level granularity automatically.

## In `dpor.py` — I/O flush logic (line 377)

No changes needed. The existing flush logic already handles the pending I/O events correctly:

```python
if _pending_io and getattr(_dpor_tls, "lock_depth", 0) == 0:
    for _obj_key, _io_kind in _pending_io:
        with _elock:
            _engine.report_io_access(_execution, thread_id, _obj_key, _io_kind)
    _pending_io.clear()
```

SQL-level events go through `report_io_access` (uses `io_vv`, ignores lock-based happens-before — appropriate for I/O), same as socket events.

## In the Rust engine

**No changes.** The existing `process_io_access` → `ObjectState::record_io_access` → `dependent_accesses` pipeline handles SQL table objects identically to socket objects. The `ObjectId` is just a `u64`; it doesn't know whether it represents a socket or a table.
