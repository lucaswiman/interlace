# Known Issues

## Async DPOR explores excess paths for independent I/O objects

The async DPOR scheduler combines two layers of tracing:

1. **Opcode-level tracing** — detects shared Python object accesses (attribute loads/stores, subscript operations, etc.)
2. **I/O-level tracing** — detects Redis key conflicts, SQL table conflicts, etc.

Even when Redis keys (or other I/O resources) are completely independent between tasks, the opcode tracer sees shared Python state — module globals, connection pool internals, closure variables, etc. — creating additional backtrack points unrelated to the key-level analysis.

**Consequence:** `explore_async_dpor()` with `detect_redis=True` on two tasks writing to completely disjoint Redis keys will explore more than 1 DPOR path. The Rust DPOR engine itself correctly handles I/O independence (verified by direct engine unit tests in `test_redis_parsing.py::TestDporEngineIoIndependence`), but the opcode tracer's shared-state detection adds false backtrack points.

**Fix (applied):** When `detect_redis=True` or `detect_sql=True`, opcode-level access reporting via `_process_opcode` is skipped entirely. The I/O-level reporters (Redis key-level, SQL table-level) capture the real conflicts. This eliminates false backtrack points from shared Python state and reduces DPOR exploration to only the interleavings that matter for I/O-level conflicts.
