# Optimal DPOR Roadmap

Implementation plan for Optimal DPOR (Abdulla et al., "Source Sets: A Foundation
for Optimal Dynamic Partial Order Reduction", JACM 2017). Goal: explore exactly
one execution per Mazurkiewicz trace, eliminating redundant interleavings.

## Current state

The dining philosophers benchmark (4 threads, preemption_bound=2) explores
**14,221 executions in ~95s**. Many of these are equivalent traces.

## Completed

- [x] Wakeup tree data structure (`crates/dpor/src/wakeup_tree.rs`)
  - Ordered tree of thread-id sequences with insert/remove/subtree/min_thread
  - Merges shared prefixes, deduplicates sequences
  - Unit tests for all operations including complex multi-branch trees

- [x] Replace `ThreadStatus::Backtrack` with wakeup tree in `Branch`
  - `step()` picks next thread from wakeup tree (min-index for deterministic order)
  - `backtrack()` inserts single-thread sequences into wakeup tree
  - Maintains identical exploration behavior to classic DPOR (same execution count)

- [x] Per-branch object access tracking
  - `active_objects: HashSet<u64>` records objects accessed at each scheduling step
  - `explored_objects: HashMap<usize, HashSet<u64>>` records per-Visited-thread objects
  - `record_access()` called from engine after each `process_access`

- [x] Sleep set skeleton at branch level
  - `sleep: Vec<bool>` tracks Visited threads (equivalent to existing Visited check)
  - `backtrack()` checks sleep set before inserting into wakeup tree

- [x] `race_object: Option<u64>` parameter plumbed through `backtrack()`
  - Engine passes the racing object ID from `process_access` / `process_io_access`
  - Currently unused (filtering disabled) but available for source set optimization

- [x] Benchmark infrastructure
  - `benchmarks/bench_dining_philosophers.py` with configurable run count
  - Baseline recorded: 14,221 executions, 95.4s median

- [x] Comprehensive test suite
  - 39 Rust unit tests (wakeup tree, path, engine)
  - 3 Python engine-level tests (4-thread exhaustive, independent pairs, read-read)
  - Full test suite passes (1158 tests)

## TODO: Sleep set propagation (the hard part)

The core challenge: sleep sets in **stateless** model checking require knowing
each sleeping thread's next action, but we don't run sleeping threads. In
**stateful** MC, the state is explicit and actions are computable.

- [ ] Record per-thread next-action during execution
  - At each scheduling point, for EVERY enabled thread (not just the chosen one),
    record what object it would access next. This requires either:
    - (a) Speculatively running each thread for one step (expensive), or
    - (b) Caching the action from a previous execution at the same prefix (fragile), or
    - (c) Using the execution trace to infer actions during replay (only works for
      the replayed prefix, not beyond the backtrack point)
  - Option (c) is most practical: during replay, we know all threads' actions
    because the prefix is deterministic

- [ ] Implement sleep set propagation through replayed prefix
  - At the backtrack point (position K): sleep = {Visited threads at K}
  - During replay of positions 0..K-1: propagate sleep forward
  - At position i+1: `sleep' = {q in sleep(i) : action(q, state_i) indep action(chosen_i, state_i)}`
  - Independence = `explored_objects[q] disjoint from active_objects` at position i
  - This is SOUND for the replayed prefix because the prefix is deterministic
  - Do NOT propagate beyond the backtrack point (actions are unknown there)

- [ ] Track sleeping thread actions across the replay boundary
  - At position K (backtrack point), sleeping threads have known actions from the
    previous execution. Record these in a `replay_actions: HashMap<usize, HashSet<u64>>`
  - Propagate ONE step past K using these recorded actions
  - Beyond K+1, sleeping threads must be woken (their actions are unknown)

## TODO: Source set filtering

With sleep sets working, source set filtering becomes sound:

- [ ] Enable race-object filtering in `backtrack()`
  - For each (position, object) pair, add at most ONE racing thread
  - This is safe because sleep sets prevent re-exploration of equivalent traces
  - Use `pending_race_objects: HashSet<u64>` (already removed, re-add)
  - Clear `pending_race_objects` when step() picks a new thread

- [ ] Verify with dining philosophers benchmark
  - Expected: significant reduction in execution count (from 14k to ~hundreds)
  - The filtering is the main source of exploration reduction

## TODO: Multi-step wakeup sequences

Single-thread sequences `[q]` are the simple case. For full Optimal DPOR:

- [ ] Compute wakeup sequences for races where the target thread is blocked
  - When thread q is not enabled at position i (e.g., blocked on a lock),
    compute the sequence of threads that must run to enable q
  - Example: q blocked on lock L held by p → sequence is [p, q]
    (p must release L before q can acquire it)
  - Use the execution trace to find the enabling sequence

- [ ] Insert multi-step sequences into wakeup tree
  - `wakeup.insert(&[p, q])` instead of just `&[q]`
  - The wakeup tree's subtree mechanism guides replay:
    at position i choose p, at position i+1 choose q

- [ ] Use wakeup subtrees during replay in `schedule()`
  - When creating a new branch, check if the parent's wakeup subtree
    suggests a specific thread (from a multi-step sequence)
  - Override the default "prefer current thread" heuristic

## TODO: Exploration order heuristics

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

The sleep set propagation should be implemented in `Path::schedule()` (during
replay) and `Path::step()` (when computing the initial sleep set at the
backtrack point). The engine doesn't need changes — it already passes
`race_object` through to `backtrack()`.

## References

- Abdulla et al., "Optimal Dynamic Partial Order Reduction", POPL 2014
- Abdulla et al., "Source Sets: A Foundation for Optimal Dynamic Partial Order
  Reduction", JACM 2017, Sections 6-7
- Kokologiannakis et al., "Stateless Model Checking for TSO and PSO", TACAS 2018
  (extends to weak memory models)
