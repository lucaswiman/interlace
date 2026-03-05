# Algorithm 3: Endpoint Suppression

Two suppression mechanisms ensure endpoint-level socket I/O reports are skipped when SQL-level detection has already reported finer-grained table accesses.

## 1. Thread-Local Suppression (Python-level)

In `_io_detection.py`, `_report_socket_io` checks `_io_tls._sql_suppress`:

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

## 2. Shared Set (for LD_PRELOAD bridge)

For LD_PRELOAD events (C-level `send()`/`recv()` from libpq), the suppression is trickier because the `_PreloadBridge` listener runs on a *different* thread (the pipe reader). The thread that called `cursor.execute()` has `_sql_suppress=True` in its TLS, but the LD_PRELOAD event arrives on the dispatcher thread.

**Solution:** Use a shared set of OS thread IDs protected by a real lock. Implemented as a context manager in `_sql_cursor.py`:

```python
# In _sql_cursor.py:
_suppress_tids: set[int] = set()  # OS thread IDs currently in sql-suppress mode
_suppress_lock = threading.Lock()  # real lock, not cooperative

@contextlib.contextmanager
def _suppress_endpoint_io() -> Generator[None, None, None]:
    """Temporarily suppress endpoint-level I/O for the current thread."""
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

def is_tid_suppressed(tid: int) -> bool:
    """Check if a thread ID is currently suppressed (for LD_PRELOAD bridge)."""
    with _suppress_lock:
        return tid in _suppress_tids
```

The bridge listener checks this:

```python
# In _PreloadBridge.listener (dpor.py):
def listener(self, event):
    if not self._active:
        return
    if event.kind == "close":
        return
    # Skip if this thread's cursor.execute() already reported at SQL level
    from frontrun._sql_cursor import is_tid_suppressed
    if is_tid_suppressed(event.tid):
        return
    # ... existing logic ...
```

## 3. Transaction Atomicity Suppression

When a transaction is active (`_io_tls._in_transaction == True`), the DPOR scheduler in `dpor.py` skips scheduling (does not yield to other threads). This ensures all SQL operations within a `BEGIN...COMMIT` block appear as atomic, preventing false positives from intermediate states.

Reports are buffered in `_io_tls._tx_buffer` and only flushed to the reporter at `COMMIT`. `ROLLBACK` discards the buffer. `SAVEPOINT`/`ROLLBACK TO` manage partial rollback via buffer truncation.
