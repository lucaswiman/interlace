# DPOR Improvements Roadmap

Consolidated view of remaining work for the optimal DPOR implementation. Items marked ✅ are
complete; this document focuses only on unfinished work. Last reviewed: 2026-03-28.

## Overview

The Rust DPOR engine has implemented:
- Wakeup trees (Optimal-DPOR, Algorithm 2)
- Sleep set propagation through replayed prefix and to new branches (via trace caching)
- Deferred race detection and notdep sequences
- Wakeup subtree guidance during replay
- Stable object keys and per-thread step-count-indexed access caches
- Lock-aware deferred release backtracking
- Provenance-tagged access summaries (Fix 4: `AccessOrigin` enum)
- Resource grouping for SQL tables (`register_resource_group()`)

**Baseline result**: Dining philosophers (4 threads, preemption_bound=2) explores 4,076 executions with lock-only model. Exact Mazurkiewicz traces achieved for:
- All shared-variable tests (2^N traces for N vars)
- All lock-only tests
- Independent file writes

Remaining: complex multi-table SQL patterns.

---

## Phase 4: Wakeup Tree Insert Equivalence (Optimization)

**Paper ref**: Definition 6.2 (JACM'17 p.22), Lemma 6.2 (p.22)

The wakeup tree insert checks for `w' ∼[E] w` — whether an existing branch already covers the sequence. Current implementation only checks exact prefix match.

### 4c. Add equivalence checking for wakeup tree inserts

**Why**: Reduces redundant branches in the wakeup tree when two sequences are equivalent under the independence relation.

**What**: Before inserting a new sequence into the wakeup tree, check if any existing leaf explores an equivalent interleaving using independence information. Two sequences are equivalent if swapping independent events produces the same order.

**Complexity**: Medium. Requires:
- Building an equivalence oracle from explored sequences
- Efficient comparison during tree operations
- Careful handling of the independence relation at boundaries

**Impact**: Sound optimization (may reduce explored executions further, but not required for correctness). Benefit dependent on workload — most benefit for programs with many independent access patterns.

---

## Fix 7: Reduce Scheduling Points for Redundant Shared Opcodes

**Root cause**: Python opcode decomposition creates 4–7 scheduling points per semantic operation (e.g., `with lock:` → 4 points; `s.slot.value = tid` → 2 points).

**Why**: Additional positions amplify any false wakeup or conservative merge, opening unnecessary exploration branches.

**What**: Selectively suppress shared opcodes that are redundant with stronger semantic events:
- Lock metadata lookups (`LOAD_ATTR`, `LOAD_SPECIAL` on lock objects) already covered by `report_sync()`
- Container iteration reads already pinned to first access (via Fix 5)
- Immutable-object lookups already covered by `_IMMUTABLE_TYPES`

**NOT**: A global "skip potentially-written objects" pass (unsound). Only targeted, semantics-aware suppression.

**Complexity**: Low–Medium. Requires careful case analysis per opcode type to ensure soundness.

**Impact**: Optimization (reduces per-execution overhead; may not affect trace counts). Candidates:
- Skip LOAD_SPECIAL on Lock, RLock objects
- Skip STORE_ATTR on lock.internal state
- Skip LOAD_ATTR on Lock.__enter__/__exit__

---

## Proposed Trace Cache Refinements

### Fix 3: Position-Sensitive Future Access Cache ✅

**Status**: Complete. Implemented in `crates/dpor/src/path.rs`.

**Implementation**: `prev_thread_step_future: HashMap<usize, Vec<HashMap<u64, AccessKind>>>` — per-thread suffix union cache indexed by the thread's own step count. Built in `step()` by iterating each thread's scheduling points in reverse, accumulating accesses. Consumed by `future_accesses_for()` which counts how many times a thread was active before a given position to determine the correct suffix index. Used by `propagate_sleep()` for both locally-sleeping and propagated-sleeping threads.

**Result**: Eliminated the predicted overcounting. Shared-variable tests now achieve exact Mazurkiewicz trace counts:
- 2-var: 4 traces (2^2, was predicted 6 → 4)
- 3-var: 8 traces (2^3, was predicted 18 → 8)

All shared-variable, lock-only, and independent file write tests achieve exact counts.

### Fix 4: Provenance-Tagged Access Summaries ✅

**Status**: Complete. Implemented in `crates/dpor/src/access.rs` and threaded throughout.

**Implementation**: `AccessOrigin` enum with `PythonMemory`, `LockSynthetic`, `IoDirect` variants. All `active_accesses` entries are now `HashMap<u64, (AccessKind, AccessOrigin)>`. Origin is tracked through `record_access()`, `propagate_sleep()`, the future access cache, and merge logic. `AccessOrigin::merge()` preserves the strongest origin (`IoDirect` > `LockSynthetic` > `PythonMemory`).

**Result**: Infrastructure in place for per-origin merge strategies (Fix 6).

### Fix 5: Improve AccessKind Merge for WeakWrite + WeakRead ✅

**Status**: Complete. `AccessKind::merge()` now returns `WeakWrite` for `(WeakWrite, WeakRead)`,
aligning the merge with `access_kinds_conflict()` which already treats them as independent.

### Fix 6: Per-Branch Merge Investigation

**Status**: Fix 5 (merge improvement) and Fix 4 (provenance tags) are both complete. Origin
variables are wired through `accesses_are_independent()` but currently unused. The previous
regression in `test_independent_file_writes[2]` could not have been caused by the merge rule
alone (the test uses disjoint objects), suggesting the original attempt included additional
changes that caused the regression.

**Remaining**: Per-origin conflict policies in `accesses_are_independent()` — e.g., relaxing
`PythonMemory` merges while keeping `IoDirect` conservative. This is now infrastructure-ready
but needs a concrete use case that demonstrates benefit.

**Complexity**: Medium. Requires investigation before proceeding. Provenance tags are now available to distinguish I/O vs Python-memory accesses.

**Impact**: If successful, may reduce file-I/O trace counts; if reverted, no change.

---

## Defect #15: Complex SQL Race Detection

**Status**: Known defect, partially mitigated. Simple check-then-insert races are found (2 ops/thread); intermediate operations (5 ops/thread) prevent race discovery. A resource grouping workaround (`register_resource_group()`) is now implemented that allows the DPOR engine to skip inline wakeups for grouped resources, reducing backtrack explosion for multi-table patterns.

**Root cause**: DPOR explores all interleavings of intermediate operations on unrelated tables, causing the tree to grow too large to reach the critical interleaving where both check operations precede both insert operations.

**Example**: django-reversion `create_revision()` with 5–7 SQL operations per thread. Dedup check (step 3) races with another thread's dedup check, but the article UPDATE operations (step 2) create so many backtrack points that DPOR exhausts its budget (~30 interleavings) before exploring the critical interleaving.

### Approach 1: Conflict Prioritization

**What**: Prioritize backtrack points involving `:seq` resources (phantom-read conflicts) over same-row conflicts. `:seq` conflicts represent check-vs-mutate dependencies more directly.

**Complexity**: Medium. Requires conflict classification and priority queue in wakeup tree management.

**Impact**: May improve convergence for multi-table patterns where sequence numbers drive the race.

### Approach 2: Operation Coalescing for Unrelated Tables (Recommended)

**What**: Recognize that operations on table X are independent of operations on table Y (different logical resources). DPOR would avoid exploring orderings of X-ops against Y-ops from the same thread.

**Complexity**: Medium–High. Requires:
- Resource-level independence analysis (which operations touch which tables)
- Cross-thread conflict awareness (thread A's X-op conflicts with thread B's X-op but not Y-op)
- Modification to backtrack insertion to skip irrelevant conflicts

**Soundness**: Sound as long as the independence analysis is correct. The key insight: if thread T1 does `X1, Y1, X2` and thread T2 does `X3, Y2`, the only interleaving orders that matter are those involving X-ops, not the relative placement of Y1 vs Y2.

**Impact**: Likely 50–80% reduction in explored interleavings for django-reversion and similar multi-table patterns. Most promising for general case.

### Approach 3: Targeted Exploration (User-Facing)

**What**: Allow users to annotate which resources are "interesting" for race detection. DPOR would only create backtrack points for those resources.

**Complexity**: Low. API and filtering in backtrack logic.

**Impact**: Useful for users who understand their domain (e.g., "only check for races on the versions table"). Less general than Approach 2.

### Approach 4: Hybrid Random + Systematic (Fallback)

**What**: Use DPOR for small trees; fall back to random scheduling when tree exceeds threshold. Random exploration might stumble on critical interleavings faster.

**Complexity**: Low–Medium. Requires integrating random scheduler into engine.

**Impact**: Pragmatic workaround but doesn't address the root algorithmic limitation. Better as a fallback than primary fix.

---

## Priority Ranking

1. **Fix 3 (Position-sensitive cache)**: ✅ Complete.

2. **Fix 4 (Provenance tags)**: ✅ Complete. Infrastructure for Fix 6 and per-origin merge strategies.

3. **Fix 5 (WeakWrite merge)**: ✅ Complete. `merge(WeakWrite, WeakRead) → WeakWrite`.

4. **Defect #15 Approach 2 (Operation coalescing)**: ✅ Complete. Resource grouping via `register_resource_group()`.

5. **Fix 6 (Per-branch merge)**: Low priority. Infrastructure ready (Fix 4 + Fix 5). Needs a concrete use case demonstrating benefit of per-origin conflict relaxation.

6. **Phase 4c (Wakeup tree equivalence)**: Low priority. Sound optimization; benefit uncertain without empirical data.

7. **Fix 7 (Suppress redundant opcodes)**: Low priority. Optimization; no impact on trace counts.

---

## Architecture Notes

**Current separation of concerns**:
- `WakeupTree` — pure data structure
- `Branch` — per-scheduling-point state (wakeup tree, sleep set, objects)
- `Path` — DFS exploration (schedule, backtrack, step)
- `DporEngine` — race detection, vector clocks, sync events

**Where improvements apply**:
- **Phase 4c**: Modify `WakeupTree::insert()` to check equivalence before adding branches
- **Fix 3**: Extend `Path` to maintain 2D cache; use in `propagate_sleep()`
- **Fix 4**: Extend `active_accesses` tuple; thread through `record_access()` calls
- **Defect #15 Approach 2**: Add resource-level conflict analysis in `DporEngine`; modify `backtrack()` to skip unrelated conflicts

No changes needed to engine's core race detection or vector clock logic.

