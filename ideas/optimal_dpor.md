# Optimal DPOR Roadmap

Implementation plan for Optimal DPOR (Abdulla et al., "Source Sets: A Foundation
for Optimal Dynamic Partial Order Reduction", JACM 2017). Goal: explore exactly
one execution per Mazurkiewicz trace, eliminating redundant interleavings.

## Current state

The dining philosophers benchmark (4 threads, preemption_bound=2) explores
**14,221 executions in ~95s**. Many of these are equivalent traces.

**Where we are in the algorithm**: The current implementation is a **hybrid** —
it has wakeup trees (from Optimal-DPOR, Algorithm 2), replay-only sleep set
propagation (Algorithm 2 line 16, through the replayed prefix only), and the
basic source set check (`I[E'](v) ∩ backtrack(E')` via `contains_thread`).
Race detection is done *during* execution (from Source-DPOR, Algorithm 1).
Sleep set propagation reduces redundant interleavings where independent
actions commute (e.g., read-read on the same object). Extending propagation
to new branches and deferred race detection with notdep sequences and weak
initials are the next steps for further reduction.

## Completed

- [x] Wakeup tree data structure (`crates/dpor/src/wakeup_tree.rs`)
  - Ordered tree of thread-id sequences with insert/remove/subtree/min_thread
  - Merges shared prefixes, deduplicates sequences
  - Unit tests for all operations including complex multi-branch trees
  - **Paper ref**: Definition 6.1 (JACM'17 p.21-22, POPL'14 p.6-7)

- [x] Replace `ThreadStatus::Backtrack` with wakeup tree in `Branch`
  - `step()` picks next thread from wakeup tree (min-index for deterministic order)
  - `backtrack()` inserts single-thread sequences into wakeup tree
  - Maintains identical exploration behavior to classic DPOR (same execution count)
  - **Paper ref**: Algorithm 2 lines 14-20 (JACM'17 p.24-25)

- [x] Per-branch object access tracking
  - `active_objects: HashSet<u64>` records objects accessed at each scheduling step
  - `explored_objects: HashMap<usize, HashSet<u64>>` records per-Visited-thread objects
  - `record_access()` called from engine after each `process_access`
  - **Paper ref**: needed for independence checks in sleep set propagation
    (Algorithm 2 line 16: `Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}`)

- [x] Sleep set skeleton at branch level
  - `sleep: Vec<bool>` tracks Visited threads (equivalent to existing Visited check)
  - `backtrack()` checks sleep set before inserting into wakeup tree
  - **Paper ref**: Algorithm 2 line 13 (`sleep(E) := Sleep`) and line 20
    (`add p to sleep(E)`)

- [x] `race_object: Option<u64>` parameter plumbed through `backtrack()`
  - Engine passes the racing object ID from `process_access` / `process_io_access`
  - Currently unused (filtering disabled) but available for source set optimization
  - **Paper ref**: needed for source set filtering (Algorithm 1 line 8:
    `I[E'](v)` check; Definition 4.1 JACM'17 p.14)

- [x] Benchmark infrastructure
  - `benchmarks/bench_dining_philosophers.py` with configurable run count
  - Baseline recorded: 14,221 executions, 95.4s median

- [x] Comprehensive test suite
  - 39 Rust unit tests (wakeup tree, path, engine)
  - 3 Python engine-level tests (4-thread exhaustive, independent pairs, read-read)
  - Full test suite passes (1158 tests)

## Phase 1: Sleep set propagation

The core optimization. Without this, all other improvements are unsound.

**Paper ref**: Algorithm 2 lines 13-20 (JACM'17 p.24-25). The key line is 16:
`Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}` — when exploring E.p, only keep sleeping
threads whose next action is independent of p's action.

### 1a. Record per-thread next-action during execution

- [x] During each execution, at each scheduling point, record what object each
  enabled thread *would* access next (not just the chosen thread).
  - **Implementation chosen**: Approach (c) — replay-only propagation with
    access-kind-aware independence checks. Rather than recording next-actions
    for all enabled threads, we propagate sleeping threads' recorded accesses
    forward through the replayed prefix. This is conservative (threads are
    woken when their actions are unknown) but correct.
  - **Note on approach (c') — full propagation to new branches**: This was
    attempted and reverted. Although the paper's recursive Algorithm 2 passes
    Sleep' to all Explore calls, our iterative implementation has a subtle
    issue: propagating sleep to new branches causes `backtrack()` to skip
    threads that appear in `propagated_sleep_accesses`, which prevents DPOR
    from discovering races that require those threads to be explored at new
    positions. The root cause is that our independence check is an
    approximation — it checks object-level access compatibility, but the
    paper's `E ⊢ p♦q` requires full trace equivalence (the actions of p and
    q must commute in ALL continuations, not just based on their recorded
    accesses). Without trace caching (approach (b), as used by Concuerror),
    we cannot guarantee that a sleeping thread's equivalence class is truly
    covered at new positions. This caused 6 Python-level race detection tests
    to fail (TestDictMergeOperatorRace, TestReduceAccumulateRace, etc.) where
    DPOR incorrectly reported `property_holds=True` because it skipped
    interleavings needed to find the bug.
  - **Data structures changed**:
    - `active_objects: HashSet<u64>` → `active_accesses: HashMap<u64, AccessKind>`
    - `explored_objects` → `explored_accesses: HashMap<usize, HashMap<u64, AccessKind>>`
    - New: `propagated_sleep_accesses: HashMap<usize, HashMap<u64, AccessKind>>`
    - `record_access()` now takes `AccessKind` parameter
  - **Independence check**: `access_kinds_conflict()` function mirrors the
    conflict semantics in `ObjectState::dependent_accesses`. Read+Read is
    independent, WeakWrite+WeakWrite is independent, etc. This is crucial
    for the writer-readers scenario where read-read independence allows
    collapsing equivalent reader orderings.

  Three approaches to resolve this in stateless MC:

  **(a) Speculative execution** (expensive): Run each enabled thread for one step
  to discover its next access, then roll back. Too expensive for Python threads.

  **(b) Trace caching** (fragile): Cache each thread's next action from a
  previous execution at the same prefix. Works because prefixes are deterministic.
  **This is the approach used by Concuerror** (Section 10, JACM'17 p.31-35).

  **(c) Replay-only propagation** (practical): During replay of positions 0..K-1,
  we know all threads' actions from the previous execution trace. Propagate sleep
  sets only through the replayed prefix; beyond the backtrack point, wake all
  sleeping threads (their actions are unknown).

  **(c') Full propagation** (attempted, reverted): Extend propagation to ALL
  positions, including new branches beyond the replay prefix. Although the
  argument that sleeping threads' state is frozen seems sound, our
  object-level independence check is insufficient — it can incorrectly keep
  threads asleep at new positions, causing `backtrack()` to skip them and
  miss real races. The paper's independence relation `E ⊢ p♦q` requires
  trace-level equivalence, not just access-kind compatibility. To safely
  implement full propagation, we likely need trace caching (approach (b)),
  where we cache each thread's full next-action trace from a previous
  execution and verify it against the current execution's divergent path.
  See Concuerror's implementation (Section 10, JACM'17 p.31-35).

### 1b. Implement sleep set propagation through replayed prefix

- [x] In `Path::schedule()`, during replay (pos < branches.len()):
  - Call `propagate_sleep(pos)` which computes `Sleep'` from pos-1's sleep set
  - For each sleeping thread q: keep q sleeping if its recorded accesses
    are independent of the active thread's accesses (using `accesses_are_independent()`)
  - Store in `propagated_sleep_accesses` at the new position
  - Only propagates during REPLAY (not to new branches — see note on (c') above)
  - **Paper ref**: Algorithm 2 line 16 (JACM'17 p.24)

- [x] In `Path::step()`, when computing the initial sleep set at backtrack point:
  - `sleep(E) = {threads marked Visited at position K}`
  - Uses `explored_accesses` (not just objects) to store per-thread access info
  - This is the starting point for propagation in the next execution
  - **Paper ref**: Algorithm 2 line 20 (JACM'17 p.25)

- [x] In `Path::backtrack()`, check both local and propagated sleep:
  - Checks `sleep[thread_id]` (locally Visited threads)
  - Checks `propagated_sleep_accesses.contains_key(thread_id)` (propagated)
  - Also updated `add_conservative_backtrack()` for preemption-bounded case

### 1c. Track sleeping thread actions across replay boundary

- [x] Propagated sleep carries access info with the thread, enabling multi-hop
  propagation during replay. When a thread propagates from pos i to pos i+1,
  its access map travels with it. At pos i+1, the propagation to pos i+2 uses
  the same access info (since independence guarantees the state is unchanged).
- [x] Propagation works during REPLAY only — `schedule()` calls
  `propagate_sleep()` only in the replay code path. Propagation to new
  branches was attempted (approach (c')) but reverted because our
  object-level independence check is insufficient to guarantee soundness
  at divergent execution paths — it caused real races to be missed.
- [x] If no access info is available for a sleeping thread (e.g., a locally
  Visited thread without explored_accesses), it is woken up (conservative).
- [ ] **Future**: Upgrade to approach (b) — trace caching. This would allow
  safe propagation to new branches by caching each thread's full next-action
  trace and verifying it against the current execution. This is the approach
  used by Concuerror (Section 10, JACM'17 p.31-35). Expected improvement:
  readers(2) from 5→4, readers(4) from ~65→16 traces.

### 1d. Verification

- [x] Add test: two independent writers to different objects → still 1 execution
  with sleep sets (`test_independent_pairs_with_propagation`, also existing
  `test_independent_threads_one_execution` continues to pass)
- [x] Add test: writer-readers example (JACM'17 Fig.1 p.6-7) → `test_writer_readers_sleep_propagation`
  verifies replay-only propagation explores ≤5 interleavings (1W+2R case)
- [x] Writer-readers 1W+4R: currently explores ~65 traces with replay-only propagation.
  Full reduction to 16 traces requires propagation to new branches (Phase 2).
- [x] 2-philosopher benchmark: reduced from 6 to 3 Mazurkiewicz traces
  (`test_two_philosophers_all_orderings` updated to assert ≥3)
- [x] Add test: lastzero-style program (POPL'14 Fig.4 p.11) →
  `test_lastzero_three` models lastzero(3) with 4 threads. With replay-only
  propagation, explores ≤60 traces.
- [ ] Re-run dining philosophers benchmark: expect reduction from 14,221

## Phase 2: Sleep propagation to new branches (trace caching)

The main remaining improvement from sleep set propagation is extending it
past the replay prefix to new branches. Replay-only propagation gives
readers(2)=5, readers(4)=~65. Full propagation would give readers(2)=4,
readers(4)=16 (matching POPL'14 Table 2: source=2^N for readers(N)).

**Attempted and reverted**: A naive approach (c') that propagated sleep to
new branches using object-level independence checks was unsound. Our
`accesses_are_independent()` check (object + access-kind compatibility) is
an approximation of the paper's `E ⊢ p♦q` (full trace equivalence). At new
branches where execution diverges, the sleeping thread's actual next action
may differ from its recorded action, and keeping it asleep incorrectly
prevents `backtrack()` from exploring it. This caused 6 Python race
detection tests to fail (DPOR reported no bugs when bugs existed).

**Correct approach**: Trace caching (approach (b), as used by Concuerror,
Section 10, JACM'17 p.31-35). Cache each thread's full execution trace from
a previous run. At a new position, verify the cached trace against the
current execution: if the sleeping thread's cached next action matches what
it would actually do, propagation is safe. If the execution has diverged
such that the cached trace is invalid, wake the thread.

### 2a. Source set check via contains_thread (already implemented)

- [x] The `contains_thread` check in `backtrack()` implements `I[E'](v) ∩
  backtrack(E') = ∅` for single-step races. No additional tracking needed.
  - **Paper ref**: Algorithm 1 line 8 (JACM'17 p.16)

### 2b. Trace caching for safe propagation to new branches

- [ ] Cache per-thread execution traces (sequence of accesses) from each run
- [ ] At new branches, verify cached trace before propagating sleep
- [ ] If trace diverges (thread would access different objects), wake thread
- [ ] **Expected results**: readers(2): 5→4, readers(4): ~65→16,
  lastzero(3): 48→~30

### 2c. Verification

- [ ] Writer-readers (3 threads, 1 writer, 2 readers):
  - Current (replay-only): 5 interleavings
  - Target (with trace caching): exactly 4
- [ ] Writer-readers (5 threads, 1 writer, 4 readers):
  - Current (replay-only): ~65 interleavings
  - Target (with trace caching): exactly 16
- [ ] Lastzero(3) (4 threads, POPL'14 Fig.4 p.11):
  - Current (replay-only): 48 traces
  - Target: ~30 traces
- [ ] Dining philosophers benchmark: expect reduction from 14,221
- [x] Independent pairs (T0/T1 write X, T2/T3 write Y): remains at 4
- [x] All existing tests pass (46 Rust tests, 1158 Python tests)

## Phase 3: Deferred race detection (Optimal-DPOR)

Move from Source-DPOR (Algorithm 1) to Optimal-DPOR (Algorithm 2) race detection.

**Paper ref**: Algorithm 2 lines 1-6 (JACM'17 p.24). Race detection happens only
at maximal executions (`enabled(s[E]) = ∅`), not during execution.

### 3a. Defer race detection to end of execution

- [ ] Currently: `process_access()` calls `backtrack()` immediately on each race.
  This is Algorithm 1 style (race detection during exploration).
- [ ] Algorithm 2 style: collect races during execution, process them only when
  all threads are finished or blocked (`enabled(s[E]) = ∅`).
- [ ] Store races in `pending_races: Vec<(usize, usize, u64)>` (position, thread, object)
- [ ] In `next_execution()`, before calling `step()`, process all pending races.
- [ ] **Why this matters**: Deferred race detection allows computing `v = notdep(e, E).e'`
  (Algorithm 2 line 4) using the full execution trace, not just the prefix seen so far.
  This produces better wakeup sequences.

### 3b. Compute notdep sequences

- [ ] For each race (e ≾_E e') at position i:
  - `E' = pre(E, e)` — the execution prefix up to just before event e
  - `v = notdep(e, E).e'` — events between e and e' that are independent of e,
    followed by e'. This is the sequence to insert into `wut(E')`.
  - **Paper ref**: Algorithm 2 line 4 (JACM'17 p.24)
  - **Implementation**: Walk the execution trace from position of e to position
    of e', collecting thread IDs of events that are independent of e (disjoint
    object sets). Append the thread of e'.

### 3c. Insert notdep sequences into wakeup tree

- [ ] Instead of `wakeup.insert(&[thread_id])`, insert the full notdep sequence.
- [ ] Add the sleep set check: `sleep(E') ∩ WI[E'](v) = ∅` before inserting
  (Algorithm 2 line 5). If a sleeping thread is a weak initial of v, the
  equivalent execution has already been explored.
- [ ] **Paper ref**: Algorithm 2 lines 5-6 (JACM'17 p.24)

## Phase 4: Multi-step wakeup sequences

Full wakeup tree integration for Optimal DPOR.

**Paper ref**: Algorithm 2 lines 17-18 (JACM'17 p.24-25): `WuT' = subtree(wut(E), p)`
is passed to the recursive Explore call.

### 4a. Compute wakeup sequences for blocked threads

- [ ] When thread q is not enabled at position i (blocked on a lock):
  - Find the sequence of threads that must run to enable q
  - Example: q blocked on lock L held by p → sequence is `[p, q]`
  - **Paper ref**: Section 8 (JACM'17 p.26-29), Algorithms 3-6 handle
    disabling/blocking. Algorithm 4 (p.28) is Optimal-DPOR with locks.

### 4b. Use wakeup subtrees during replay

- [ ] In `Path::schedule()`, when creating a new branch, check if the parent's
  wakeup subtree suggests a specific thread (from a multi-step sequence).
- [ ] Pass `WuT' = subtree(wut(E), p)` as guidance for the next scheduling point.
- [ ] Override the "prefer current thread" heuristic when the wakeup tree has guidance.
- [ ] **Paper ref**: Algorithm 2 lines 8-12, 17 (JACM'17 p.24-25)

### 4c. Wakeup tree insert with equivalence checking

- [ ] The paper's insert operation (Definition 6.2, JACM'17 p.22) checks for
  `w' ∼[E] w` — whether an existing branch already covers the sequence.
- [ ] Current implementation only checks exact prefix match. This is sound
  (may insert redundant branches) but not optimal.
- [ ] Add equivalence checking: before inserting, check if any existing leaf
  explores an equivalent interleaving using independence information.
- [ ] **Paper ref**: Lemma 6.2 (JACM'17 p.22): any leaf w is the smallest node
  in the tree consistent with w after E.

## Phase 5: Locks and disabling (Algorithms 3-6)

**Paper ref**: Section 8 (JACM'17 p.26-29). The basic algorithms (1-2) assume
Assumption 3.1: processes don't disable each other. With locks, this fails.

### Current approach vs paper

The current engine uses `io_vv` (a separate vector clock without lock-based HB)
for lock operations, making them always appear concurrent. This is a pragmatic
approach that creates backtrack points at lock boundaries.

The paper's approach (Algorithms 3-4, JACM'17 p.27-28) is more precise:
- Track which threads are *enabled* vs *blocked* at each state
- Only consider races between enabled threads
- Compute wakeup sequences that unblock threads before they can participate

### 5a. Evaluate paper's lock algorithms

- [ ] Compare current `io_vv` approach with Algorithm 3/4 precision
  - Does the `io_vv` approach over-explore? Under-explore?
  - The current approach may find more interleavings than necessary (conservative)
  - **Paper ref**: Algorithm 3 (JACM'17 p.27) adds enabled-check to backtrack;
    Algorithm 4 (p.28) adds wakeup sequences for blocked threads

### 5b. Implement Algorithm 3/4 if beneficial

- [ ] Add enabled-thread tracking at each scheduling point
- [ ] Modify `backtrack()` to check if the target thread was enabled
- [ ] For blocked threads, compute enabling sequences
- [ ] **Paper ref**: Algorithm 4 line modifications (JACM'17 p.28)

## Exploration order heuristics

Independent of Optimal DPOR correctness, exploration order affects how
quickly bugs are found with `stop_on_first=True`:

- [ ] Prioritize threads at lock boundaries for deadlock detection
  - Threads about to acquire a lock are more likely to create deadlocks
  - Use wakeup tree ordering to try these first

- [ ] Depth-first vs breadth-first at backtrack points
  - Current: deepest backtrack first (DFS)
  - Alternative: shallowest backtrack first might find bugs faster
    for certain patterns (trades memory for earlier bug discovery)

## Architecture notes

The current implementation cleanly separates concerns:
- `WakeupTree` — pure data structure, no DPOR logic
- `Branch` — per-scheduling-point state (wakeup tree, sleep set, objects)
- `Path` — DFS exploration via `schedule()`, `backtrack()`, `step()`
- `DporEngine` — race detection, vector clocks, sync events
- `PyDporEngine` — PyO3 bridge to Python

Sleep set propagation is implemented in `Path::schedule()` (during replay
only) and `Path::step()` (when computing the initial sleep set at the
backtrack point). Propagation to new branches requires trace caching
(Phase 2). The engine doesn't need changes — it already passes
`race_object` through to `backtrack()`.

For Phase 3 (deferred race detection), the engine will need changes: collect
races during execution and process them in `next_execution()`.

## Known implementation gaps vs paper

1. **Race detection timing**: Current = during execution (Alg 1 style).
   Paper's Alg 2 = only at maximal executions. (Phase 3)

2. **Wakeup tree insert**: Current = exact prefix matching only.
   Paper = equivalence checking via `∼[E]` relation. (Phase 4c)

3. **notdep sequences**: Current = single thread `[q]`.
   Paper = full `notdep(e, E).e'` sequence. (Phase 3b)

4. **Sleep set propagation**: Replay-only propagation is done (Phase 1).
   Propagation to new branches requires trace caching (Phase 2). Current
   implementation uses `propagated_sleep_accesses` for multi-hop propagation
   during replay only. An attempt to extend to new branches using
   object-level independence was reverted (unsound — caused missed races).

5. **Source set filtering**: The basic check (`I[E'](v) ∩ backtrack(E')`)
   is implemented via `contains_thread`. The weak initials optimization
   (`WI` instead of `I`) requires trace equivalence computation. (Phase 3)

6. **Wakeup subtree guidance**: Current = `schedule()` ignores subtrees.
   Paper = `WuT' = subtree(wut(E), p)` guides replay. (Phase 4b)

## References

### Primary sources (checked into `ideas/`)

- Abdulla et al., "Optimal Dynamic Partial Order Reduction", POPL 2014
  (`ideas/abdulla_popl2014/`)
  - Algorithm 1 (Source-DPOR): p.6
  - Algorithm 2 (Optimal-DPOR): p.7-8
  - Correctness proofs: p.8-9 (Theorems 7.4, 7.5, 7.7)
  - Benchmarks: p.11-12 (Tables 1-3)

- Abdulla et al., "Source Sets: A Foundation for Optimal Dynamic Partial Order
  Reduction", JACM 2017 (`ideas/abdulla_jacm2017/`)
  - Framework and computation model: p.11-14 (Section 3)
  - Source sets definition: p.14-16 (Section 4, Definition 4.3)
  - Algorithm 1 (Source-DPOR): p.16 (Section 5)
  - Wakeup trees: p.21-24 (Section 6, Definitions 6.1-6.2)
  - Algorithm 2 (Optimal-DPOR): p.24-26 (Section 7)
  - Locks/disabling: p.26-29 (Section 8, Algorithms 3-6)
  - Trade-offs analysis: p.29-31 (Section 9)
  - Implementation details: p.31-35 (Section 10)
  - Benchmarks: p.35-44 (Section 11)

### Secondary references

- Kokologiannakis et al., "Stateless Model Checking for TSO and PSO", TACAS 2018
  (extends to weak memory models)
