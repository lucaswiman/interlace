# Known Issues

## Async DPOR explores excess paths for independent I/O objects

The async DPOR scheduler combines two layers of tracing:

1. **Opcode-level tracing** — detects shared Python object accesses (attribute loads/stores, subscript operations, etc.)
2. **I/O-level tracing** — detects Redis key conflicts, SQL table conflicts, etc.

Even when Redis keys (or other I/O resources) are completely independent between tasks, the opcode tracer sees shared Python state — module globals, connection pool internals, closure variables, etc. — creating additional backtrack points unrelated to the key-level analysis.

**Consequence:** `explore_async_dpor()` with `detect_redis=True` on two tasks writing to completely disjoint Redis keys will explore more than 1 DPOR path. The Rust DPOR engine itself correctly handles I/O independence (verified by direct engine unit tests in `test_redis_parsing.py::TestDporEngineIoIndependence`), but the opcode tracer's shared-state detection adds false backtrack points.

**Workaround:** None currently. The extra exploration is bounded by `max_executions` and does not produce false positives (the invariant still holds on all explored paths). It just means DPOR does more work than theoretically necessary for independent I/O operations.

**Potential fix:** Exclude library code (site-packages) from opcode tracing when `detect_redis=True` or `detect_sql=True` is the primary conflict source, or add a mechanism to suppress opcode-level backtracking for accesses that occur entirely within library frames. This would require changes to the `_AutoPauseCoroutine` / `_process_opcode` interaction, not the Redis/SQL analysis layer.
