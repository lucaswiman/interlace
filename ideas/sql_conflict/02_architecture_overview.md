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

---

## Known Limitations

This implementation has comprehensive coverage of common SQL DML operations but intentionally defers some advanced features. See [03_algorithm_1_sql_parsing.md](03_algorithm_1_sql_parsing.md#known-limitations--todos) for detailed TODOs on each gap.

### Currently Not Modeled

| Feature | Why Deferred | Impact | Workaround |
|---------|--------------|--------|-----------|
| **SELECT FOR UPDATE / FOR SHARE** | Locking intent not semantically tracked | Conservative but correct; row-level predicates handle same-row cases | Phase 5 TODO |
| **Advisory locks** | Function calls, not SQL DML | Only socket-level detection; different lock IDs reported as conflicting | Phase 5 TODO |
| **LOCK TABLE** | DDL not implemented | Falls back to endpoint-level | Phase 5 TODO |
| **Foreign key constraints** | Schema introspection not implemented | Cross-table dependencies invisible (false negatives) | Phase 6 TODO |
| **Transaction boundaries** | Statement-level granularity sufficient for most cases | Unnecessary interleavings explored (search space explosion) | Phase 6 TODO |
| **Stored procedures** | Dynamic SQL opaque | Treated as endpoint-level I/O | Phase 7 TODO |
| **Temporal tables** | Specialized SQL dialect | Rare; conservative table-level detection sufficient | Phase 7 TODO |

**Soundness property maintained:** All limitations are conservative (fall back to coarser detection), so no real conflicts are missed.

---

## Roadmap

- **Phase 1-3** (✅ Done): Table-level, row-level, wire protocol SQL detection
- **Phase 4** (📋 Designed): Isolation anomaly classification
- **Phase 5+** (📋 Documented): Advanced SQL features — see [13_phased_implementation.md#phase-5-advanced-sql-features-todo](13_phased_implementation.md#phase-5-advanced-sql-features-todo)
