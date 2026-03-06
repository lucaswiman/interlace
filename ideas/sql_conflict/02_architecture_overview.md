# Architecture Overview

```
cursor.execute(sql, params)
    │
    ├── _intercept_execute() intercepts             ← frontrun/_sql_cursor.py
    │     │
    │     ├── parse_sql_access(sql)                 ← frontrun/_sql_parsing.py
    │     │     returns (read_tables, write_tables, lock_intent, tx_op)
    │     │
    │     ├── Handle transaction control:
    │     │     BEGIN  → buffer subsequent reports in _io_tls._tx_buffer
    │     │     COMMIT → flush buffer to reporter
    │     │     ROLLBACK → discard buffer
    │     │     SAVEPOINT/RELEASE/ROLLBACK TO → manage savepoint stack
    │     │
    │     ├── resolve_parameters(sql, params, paramstyle)  ← frontrun/_sql_params.py
    │     ├── extract_equality_predicates(resolved_sql)    ← frontrun/_sql_predicates.py
    │     │
    │     ├── for table in read_tables:
    │     │     report_or_buffer(f"sql:{table}:{predicates}", kind)
    │     │     kind = "write" if lock_intent == "UPDATE" else "read"
    │     ├── for table in write_tables:
    │     │     report_or_buffer(f"sql:{table}:{predicates}", "write")
    │     │
    │     ├── _suppress_endpoint_io() context manager:
    │     │     _io_tls._sql_suppress = True
    │     │     _suppress_tids.add(native_tid)
    │     │     original_execute(sql, params)        ← actual DB call
    │     │     _suppress_tids.discard(native_tid)
    │     │     _io_tls._sql_suppress = False
    │     │
    │     └── dpor.py skips scheduling when _in_transaction is True
    │
    └── socket.send() → LD_PRELOAD send()
          │
          ├── _report_socket_io() checks _sql_suppress
          │     → SKIPPED (sql-level already reported)
          │
          └── _PreloadBridge.listener() checks is_tid_suppressed(event.tid)
                → SKIPPED
```

The io_reporter callback is the *same* one already installed by `_setup_dpor_tls` in `dpor.py`. SQL detection just calls it with `"sql:{table}"` or `"sql:{table}:{predicates}"` resource IDs instead of `"socket:host:port"`. No changes to the Rust engine or PyO3 bindings are needed — the existing `report_io_access(execution, thread_id, object_key, kind)` interface is sufficient.

---

## Known Limitations

This implementation has comprehensive coverage of common SQL DML operations but intentionally defers some advanced features. See [03_algorithm_1_sql_parsing.md](03_algorithm_1_sql_parsing.md#known-limitations--todos) for detailed TODOs on each gap.

### Currently Not Modeled

| Feature | Why Deferred | Impact | Workaround |
|---------|--------------|--------|-----------|
| **Foreign key constraints** | Schema introspection not implemented | Cross-table dependencies invisible (false negatives) | Phase 6 TODO |
| **Stored procedures** | Dynamic SQL opaque | Treated as endpoint-level I/O | Phase 7 TODO |
| **Temporal tables** | Specialized SQL dialect | Rare; conservative table-level detection sufficient | Phase 7 TODO |

**Soundness property maintained:** All limitations are conservative (fall back to coarser detection), so no real conflicts are missed.

---

## Roadmap

- **Phase 1-3** (✅ Done): Table-level, row-level, wire protocol SQL detection
- **Phase 4-5** (✅ Done): Isolation anomaly classification and advanced features
- **Phase 6** (✅ Partial): Transaction grouping done; FK analysis TODO
- **Phase 6+** (📋 Documented): Further improvements — see [13_phased_implementation.md#phase-6-further-improvements-todo](13_phased_implementation.md#phase-6-further-improvements-todo)
