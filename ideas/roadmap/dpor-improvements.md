# DPOR Improvements Roadmap

Consolidated view of remaining work for the optimal DPOR implementation. Items marked ✅ are complete; this document focuses only on unfinished work.

## Overview

The Rust DPOR engine has implemented:
- Wakeup trees (Optimal-DPOR, Algorithm 2)
- Sleep set propagation through replayed prefix and to new branches (via trace caching)
- Deferred race detection and notdep sequences
- Wakeup subtree guidance during replay
- Stable object keys and per-thread step-count-indexed access caches
- Lock-aware deferred release backtracking

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

### Fix 3: Position-Sensitive Future Access Cache

**Root cause**: `prev_thread_all_accesses` union (Fix 9) is sound but very coarse. A thread that touched object X at any point in the prior execution makes X appear "in scope" for the sleeping thread's entire future, creating false wakeups.

**Why**: Current cache prevents advanced understanding of when a thread's future work truly touches a specific object. Makes the sleep set more conservative than necessary.

**What**: Replace `prev_thread_all_accesses` with `prev_thread_step_future`, indexed by the thread's own step count within a path position:
```
prev_thread_step_future[(pos, tid)][k] = union of tid's accesses from its k-th scheduling point onward
```

**Soundness**: Remains conservative as long as each cached summary is a superset of what the thread can still do from that replay position.

**Complexity**: Medium. Requires:
- 2D indexing: (path_position, thread_step) → access_union
- Tracking thread step counts separately per position
- Careful replay semantics to ensure the cache remains valid

**Impact**: Best candidate for eliminating remaining 2-var (6 → 4?) and 3-var (18 → 8?) overcounting in shared-var tests. Likely 10–20% reduction in dining philosophers benchmark.

### Fix 4: Provenance-Tagged Access Summaries

**Root cause**: No distinction between Python-memory, lock, and I/O accesses in the path layer. Current merge policies must be conservative for all three.

**Why**: Enables targeted fixes: relax Python-memory merges (Fix 6 already partially does this) while keeping I/O origins conservative if needed.

**What**: Extend `active_accesses` entries to track origin:
```rust
enum AccessOrigin {
    PythonMemory,
    LockSynthetic,
    IoDirect,
}
// active_accesses: HashMap<u64, (AccessKind, AccessOrigin)>
```

**Complexity**: Low. Plumbing change in `record_access()` calls; enables cleaner policies in `propagate_sleep()` and merge logic.

**Impact**: Infrastructure for safer follow-on fixes. May enable per-origin merge strategies (e.g., relax Python but stay conservative for I/O).

### Fix 5: Improve AccessKind Merge for WeakWrite + WeakRead

**Root cause**: `WeakWrite` conflicts with `{Read, Write}` while `WeakRead` conflicts with `{Write}`. When both occur on the same object, merging to `Write` (full conflict) is too conservative.

**What**: Extend `AccessKind::merge()` to handle:
```
(WeakWrite, WeakRead) | (WeakRead, WeakWrite) => WeakWrite
```

**Complexity**: Trivial. One line in `merge()`.

**Impact**: Low. Only applies when both WeakWrite and WeakRead occur on the same object across different scheduling points of the same thread. Optimization (reduces false wakeups in edge cases).

### Fix 6: Per-Branch Merge Investigation

**Status**: Attempted and caused a regression in `test_independent_file_writes[2]` (independent writes got 2 instead of 1).

**Why**: Unknown. The regression needs root-cause analysis. Two possibilities:
1. Per-branch merge interacts unexpectedly with I/O access tracking
2. A pre-existing bug in file-I/O race detection was masked by the conservative merge

**Next steps**:
1. Add provenance tags (Fix 4) so per-branch merge can distinguish I/O accesses
2. Selectively relax only Python-memory merges while keeping I/O conservative
3. Investigate the file-I/O case to determine if a real bug exists

**Complexity**: Medium. Requires investigation before proceeding.

**Impact**: If successful, may reduce file-I/O trace counts; if reverted, no change.

---

## Defect #15: Complex SQL Race Detection

**Status**: Known defect. Simple check-then-insert races are found (2 ops/thread); intermediate operations (5 ops/thread) prevent race discovery.

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

1. **Phase 4c (Wakeup tree equivalence)**: Low priority. Sound optimization; benefit uncertain without empirical data.

2. **Fix 3 (Position-sensitive cache)**: High priority. Best candidate for eliminating shared-var overcounting. Estimated 10–20% reduction in complex benchmarks.

3. **Fix 4 (Provenance tags)**: High priority. Infrastructure for safer follow-on fixes (especially Fix 6).

4. **Defect #15 Approach 2 (Operation coalescing)**: Medium priority. Enables discovery of multi-table SQL races. Likely 50–80% reduction for affected patterns.

5. **Fix 6 (Per-branch merge)**: Medium priority. Blocked on provenance tags (Fix 4) and investigation. Uncertain impact.

6. **Fix 5 (WeakWrite merge)**: Low priority. Optimization; impact likely < 5%.

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

