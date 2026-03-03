# Algorithm 3: Endpoint Suppression

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
