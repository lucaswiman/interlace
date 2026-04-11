# Refactoring Roadmap

This roadmap is focused on reducing code size and making the implementation easier to understand without changing the public behavior of `frontrun`.

The first phase is intentionally structural rather than feature-oriented: split the current `frontrun/dpor.py` monolith into smaller modules with clearer ownership. The later phases then build on those boundaries to remove near-duplicate code across sync and async paths.

## Goals

- Reduce the amount of logic any one file is responsible for.
- Make execution, scheduling, replay, instrumentation, and patching boundaries explicit.
- Lower the cost of future fixes by removing duplicated lifecycle code.
- Preserve the existing public API until a later, explicit API-cleanup phase.

## Non-goals

- No semantic changes to DPOR exploration behavior during phase 1.
- No user-facing API redesign during the initial refactors.
- No Rust engine rewrite as part of this roadmap.

## Phase order

1. [Phase 1: Split `frontrun/dpor.py`](./phase-1-split-dpor.md)
2. [Phase 2: Share async auto-pause machinery](./phase-2-shared-async-autopause.md)
3. [Phase 3: Consolidate framework adapters](./phase-3-framework-adapters.md)
4. [Phase 4: Replace hand-written patching with registries](./phase-4-driver-patching.md)
5. [Phase 5: Unify threaded runner lifecycle](./phase-5-threaded-runner-harness.md)

## Cross-phase constraints

- Preserve `from frontrun.dpor import explore_dpor` and the current top-level imports throughout the roadmap.
- Keep changes reviewable. Prefer one structural move per PR over one very large rewrite.
- Land behavior-preserving extraction PRs before simplification PRs.
- Keep test coverage broad for every phase:
  - `make test-3.14`
  - `make check`

## Suggested PR strategy

- PR 1: introduce new internal modules and move code with compatibility re-exports.
- PR 2: simplify imports and remove stale forwarding code once the new layout settles.
- PR 3+: start removing duplicated logic using the new internal boundaries.

The phase documents below assume that strategy.
