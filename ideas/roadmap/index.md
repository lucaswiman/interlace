# Frontrun Roadmap

Consolidated from the original `ideas/` directory. Items already implemented have been removed;
only remaining work is listed here. Last reviewed: 2026-03-28.

## Documents

| Document | Status | Scope |
|----------|--------|-------|
| [dpor-improvements.md](dpor-improvements.md) | Active | Optimal DPOR trace reduction, wakeup tree equivalence, complex SQL race detection |
| [integrations-and-detection.md](integrations-and-detection.md) | Active | SQL/Redis/resource detection layers, transaction identity, FK analysis |
| [formal-methods.md](formal-methods.md) | Active | TLA+/Quint integration, spec-guided exploration, counterexample replay |
| [testing-strategies.md](testing-strategies.md) | Active | Marker schedule extensions, hybrid exploration, pytest plugin |
| [../search_strategies.md](../search_strategies.md) | ✅ Implemented | All 5 search strategies (DFS, bit-reversal, round-robin, stride, conflict-first) |
| [../random_dpor.md](../random_dpor.md) | Proposal | Literature survey + 4 proposals for randomized/hybrid DPOR exploration |

## Completed (removed from active priorities)

- ✅ **Position-sensitive future access cache** (dpor-improvements: Fix 3) -- Exact Mazurkiewicz
  trace counts for shared-variable tests.
- ✅ **Async/await marker support** (testing-strategies: Extension 6) -- Full `AsyncTraceExecutor`
  in `frontrun/async_trace_markers.py` with `sys.settrace`-based marker detection.
- ✅ **Search strategies** (search_strategies.md) -- All 5 strategies implemented in
  `crates/dpor/src/path.rs` and exposed via `DporEngine(..., search=...)`.
- ✅ **Provenance-tagged access summaries** (dpor-improvements: Fix 4) -- `AccessOrigin` enum
  (`PythonMemory`, `LockSynthetic`, `IoDirect`) in `crates/dpor/src/access.rs`, threaded through
  `record_access()`, `propagate_sleep()`, and future access cache. Unblocks Fix 6.
- ✅ **Operation coalescing for unrelated tables** (dpor-improvements: Defect #15, Approach 2) --
  `register_resource_group()` with cross-group intermediate detection. Automatic SQL table
  group registration in `frontrun/dpor.py`. Skips inline wakeup for cross-group intermediates.
- ✅ **WeakWrite+WeakRead merge** (dpor-improvements: Fix 5) -- `merge()` now returns `WeakWrite`
  instead of `Write`, aligning with `access_kinds_conflict()`. Origins wired through
  `accesses_are_independent()` for future per-origin policies.

## Priority overview

### P1 -- Valuable, moderate effort

1. **Cross-table FK analysis** (integrations-and-detection) -- Schema introspection for foreign
   key dependencies. Catches referential integrity races. ~150 LOC + 25 tests.
2. **Counterexample replay from TLC** (formal-methods: 2.1) -- TLC finds invariant violation,
   frontrun replays it against real Python code. Agent-driven pipeline.
3. **Invariant assertion bridge** (formal-methods: 1.2) -- TLA+ invariants become Python
   assertions checked after every DPOR step.
4. **Hybrid marker + bytecode exploration** (testing-strategies: Extension 3) -- Two-level
   search: coarse markers + fine bytecode within each window.
5. **Randomized wakeup tree ordering** (random_dpor.md: Proposal A) -- Different seeds explore
   different trace space regions. Low effort, high value for `stop_on_first=True` use cases.

### P2 -- Nice to have, lower effort

6. **Wakeup tree equivalence checking** (dpor-improvements: Phase 4c) -- Sound optimization;
    benefit depends on workload.
7. **Per-branch merge with provenance** (dpor-improvements: Fix 6) -- Infrastructure ready
    (Fix 4 + Fix 5 done). Needs concrete use case for per-origin conflict relaxation.
8. **RETURNING clause injection** (integrations-and-detection) -- Captures autoincrement IDs
    from PostgreSQL INSERTs.
9. **sys.addaudithook integration** (integrations-and-detection) -- Zero-config I/O safety net.
10. **sys.monitoring CALL events** (integrations-and-detection) -- Lower-overhead resource
    detection on Python 3.12+.
11. **Pytest marker plugin** (testing-strategies: Extension 9) -- `@pytest.mark.frontrun_markers`
    for native test integration.
12. **Marker coverage tracking** (testing-strategies: Extension 8) -- Report which interleavings
    were actually exercised.

### P3 -- Deferred / exploratory

13. **Spec-guided schedule generation from TLC** (formal-methods: 2.3) -- Replace random
    exploration with TLC-enumerated behaviors.
14. **Refinement checking** (formal-methods: 3.1) -- Mathematical proof that code implements
    spec. Requires solid foundation from P1 formal-methods items.
15. **Record/replay of external state** (integrations-and-detection) -- Deterministic I/O replay.
    High complexity; most tests use isolated DBs anyway.
16. **Wire-protocol parsing at LD_PRELOAD level** (integrations-and-detection) -- For non-Python
    drivers. Niche use case.
17. **Trace fingerprinting with coverage feedback** (random_dpor.md: Proposal B) -- Hash
    reads-from relations; adaptively skip stale backtrack points. Medium effort.
18. **Depth-biased backtrack selection** (random_dpor.md: Proposal D) -- Explore/exploit
    trade-off across search phases. Low effort.
