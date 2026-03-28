# Frontrun Roadmap

Last reviewed: 2026-03-28.

## Documents

| Document | Scope |
|----------|-------|
| [dpor-improvements.md](dpor-improvements.md) | Wakeup tree equivalence, per-step independence, redundant opcode suppression |
| [integrations-and-detection.md](integrations-and-detection.md) | SQL/Redis/resource detection layers, FK analysis |
| [formal-methods.md](formal-methods.md) | TLA+/Quint integration, spec-guided exploration, counterexample replay |
| [testing-strategies.md](testing-strategies.md) | Marker schedule extensions, hybrid exploration, pytest plugin |
| [../random_dpor.md](../random_dpor.md) | Literature survey + 4 proposals for randomized/hybrid DPOR exploration |

## Implemented (not active)

Position-sensitive future access cache (Fix 3), provenance-tagged access summaries (Fix 4),
WeakWrite+WeakRead merge (Fix 5), per-step independence check (Fix 6), operation coalescing
for SQL tables (Defect #15), all 5 search strategies, async/await marker support.

## Priority overview

### P1 -- Valuable, moderate effort

1. **Cross-table FK analysis** (integrations-and-detection) -- Schema introspection for foreign
   key dependencies. Catches referential integrity races. ~150 LOC + 25 tests.
3. **Counterexample replay from TLC** (formal-methods: 2.1) -- TLC finds invariant violation,
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
7. **RETURNING clause injection** (integrations-and-detection) -- Captures autoincrement IDs
    from PostgreSQL INSERTs.
8. **sys.addaudithook integration** (integrations-and-detection) -- Zero-config I/O safety net.
9. **sys.monitoring CALL events** (integrations-and-detection) -- Lower-overhead resource
    detection on Python 3.12+.
10. **Pytest marker plugin** (testing-strategies: Extension 9) -- `@pytest.mark.frontrun_markers`
    for native test integration.
11. **Marker coverage tracking** (testing-strategies: Extension 8) -- Report which interleavings
    were actually exercised.

### P3 -- Deferred / exploratory

12. **Spec-guided schedule generation from TLC** (formal-methods: 2.3) -- Replace random
    exploration with TLC-enumerated behaviors.
13. **Refinement checking** (formal-methods: 3.1) -- Mathematical proof that code implements
    spec. Requires solid foundation from P1 formal-methods items.
14. **Record/replay of external state** (integrations-and-detection) -- Deterministic I/O replay.
    High complexity; most tests use isolated DBs anyway.
15. **Wire-protocol parsing at LD_PRELOAD level** (integrations-and-detection) -- For non-Python
    drivers. Niche use case.
16. **Trace fingerprinting with coverage feedback** (random_dpor.md: Proposal B) -- Hash
    reads-from relations; adaptively skip stale backtrack points. Medium effort.
17. **Depth-biased backtrack selection** (random_dpor.md: Proposal D) -- Explore/exploit
    trade-off across search phases. Low effort.
