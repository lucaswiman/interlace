# Architecture Overview

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
