# Frontrun Roadmap

Consolidated from the original `ideas/` directory. Items already implemented have been removed;
only remaining work is listed here.

## Documents

| Document | Scope |
|----------|-------|
| [dpor-improvements.md](dpor-improvements.md) | Optimal DPOR trace reduction, wakeup tree equivalence, complex SQL race detection |
| [integrations-and-detection.md](integrations-and-detection.md) | SQL/Redis/resource detection layers, transaction identity, FK analysis |
| [formal-methods.md](formal-methods.md) | TLA+/Quint integration, spec-guided exploration, counterexample replay |
| [testing-strategies.md](testing-strategies.md) | Marker schedule extensions, hybrid exploration, pytest plugin |

## Priority overview

### P0 -- High impact, unblocks other work

1. **Position-sensitive future access cache** (dpor-improvements: Fix 3) -- Best candidate for
   eliminating remaining shared-var overcounting. ~10-20% reduction in complex benchmarks.
2. **Provenance-tagged access summaries** (dpor-improvements: Fix 4) -- Infrastructure that
   enables safer per-origin merge strategies and unblocks Fix 6.
3. **Operation coalescing for unrelated tables** (dpor-improvements: Defect #15, Approach 2) --
   Enables discovery of multi-table SQL races (django-reversion). ~50-80% reduction for
   affected patterns.

### P1 -- Valuable, moderate effort

4. **Cross-table FK analysis** (integrations-and-detection) -- Schema introspection for foreign
   key dependencies. Catches referential integrity races. ~150 LOC + 25 tests.
5. **Counterexample replay from TLC** (formal-methods: 2.1) -- TLC finds invariant violation,
   frontrun replays it against real Python code. Agent-driven pipeline.
6. **Invariant assertion bridge** (formal-methods: 1.2) -- TLA+ invariants become Python
   assertions checked after every DPOR step.
7. **Async/await marker support** (testing-strategies: Extension 6) -- Extends marker-based
   scheduling to async code.
8. **Hybrid marker + bytecode exploration** (testing-strategies: Extension 3) -- Two-level
   search: coarse markers + fine bytecode within each window.

### P2 -- Nice to have, lower effort

9. **Wakeup tree equivalence checking** (dpor-improvements: Phase 4c) -- Sound optimization;
    benefit depends on workload.
10. **RETURNING clause injection** (integrations-and-detection) -- Captures autoincrement IDs
    from PostgreSQL INSERTs.
11. **sys.addaudithook integration** (integrations-and-detection) -- Zero-config I/O safety net.
12. **sys.monitoring CALL events** (integrations-and-detection) -- Lower-overhead resource
    detection on Python 3.12+.
13. **Pytest marker plugin** (testing-strategies: Extension 9) -- `@pytest.mark.frontrun_markers`
    for native test integration.
14. **Marker coverage tracking** (testing-strategies: Extension 8) -- Report which interleavings
    were actually exercised.

### P3 -- Deferred / exploratory

15. **Lock precision (Algorithms 3-6)** (dpor-improvements: Phase 5) -- Current `io_vv` model
    works; revisit only if lock-aware file I/O becomes important.
16. **Spec-guided schedule generation from TLC** (formal-methods: 2.3) -- Replace random
    exploration with TLC-enumerated behaviors.
17. **Refinement checking** (formal-methods: 3.1) -- Mathematical proof that code implements
    spec. Requires solid foundation from P1 formal-methods items.
18. **Record/replay of external state** (integrations-and-detection) -- Deterministic I/O replay.
    High complexity; most tests use isolated DBs anyway.
19. **Wire-protocol parsing at LD_PRELOAD level** (integrations-and-detection) -- For non-Python
    drivers. Niche use case.

## What was removed

The following ideas were fully implemented and removed from the roadmap:

- **SQL conflict detection** (Phases 1-4) -- `_sql_cursor.py`, `_sql_anomaly.py`, row-level predicates, endpoint suppression
- **Optimal DPOR core** -- Wakeup trees, sleep set propagation, deferred race detection, notdep sequences, trace caching
- **Trace overcounting fixes** 6, 8, 9, 10 -- Lock-aware backtracking, step-count-indexed cache, I/O wrapper suppression
- **Resource detection layers** 1, 1.5, 2 -- Socket/file patching, sys.setprofile C_CALL, SQL cursor monkey-patching
- **Property-based marker schedules** -- `marker_schedule_strategy()`, `all_marker_schedules()`, `explore_marker_interleavings()`
- **Redis key-level conflict detection** -- Sync and async
- **Cooperative threading primitives** -- Lock, RLock, Event, Semaphore, Queue wrappers
- **Deadlock detection** -- WaitForGraph with cycle detection
- **INSERT tracking with autoincrement** -- `sql:table:seq` resources
- **LD_PRELOAD I/O interception** -- Linux + macOS, pipe-based event streaming
- **Async variants** of all three exploration strategies
