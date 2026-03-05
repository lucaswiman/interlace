# Integration Points

## In `dpor.py` — `DporBytecodeRunner._patch_io()` / `_unpatch_io()`

```python
from frontrun._sql_cursor import patch_sql, unpatch_sql, is_tid_suppressed
from frontrun._sql_anomaly import classify_sql_anomaly

# In _patch_io():
patch_sql()

# In _unpatch_io():
unpatch_sql()
```

## In `dpor.py` — `_PreloadBridge.listener()`

```python
# Skip if this thread's cursor.execute() already reported at SQL level
from frontrun._sql_cursor import is_tid_suppressed
if is_tid_suppressed(event.tid):
    return
```

## In `dpor.py` — `DporScheduler._report_and_wait()`

Transaction atomicity: when `_io_tls._in_transaction` is True, the scheduler skips scheduling (does not yield to other threads). This ensures all SQL operations within a `BEGIN...COMMIT` block appear as atomic.

## In `dpor.py` — Result processing

```python
if result.sql_anomaly is None:
    result.sql_anomaly = classify_sql_anomaly(recorder.events)
```

## In `bytecode.py` — `BytecodeShuffler._patch_io()` / `_unpatch_io()`

```python
from frontrun._sql_cursor import patch_sql, unpatch_sql

# In _patch_io():
patch_sql()

# In _unpatch_io():
unpatch_sql()
```

## In `dpor.py` — `_setup_dpor_tls()` method

No changes needed. The existing `_io_reporter` closure already handles any `resource_id` string. When `_intercept_execute` calls `reporter("sql:accounts", "write")`, it flows through the same path:

```python
def _io_reporter(resource_id: str, kind: str) -> None:
    object_key = _make_object_key(hash(resource_id), resource_id)
    pending: list[tuple[int, str]] = _dpor_tls.pending_io
    pending.append((object_key, kind))
```

The `object_key` is derived from `hash("sql:accounts")` instead of `hash("socket:127.0.0.1:5432")`, giving table-level granularity automatically.

## In the Rust engine

**No changes.** The existing `process_io_access` → `ObjectState::record_io_access` → `dependent_accesses` pipeline handles SQL table objects identically to socket objects. The `ObjectId` is just a `u64`; it doesn't know whether it represents a socket or a table.
