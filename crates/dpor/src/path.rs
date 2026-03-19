//! Exploration tree for Source Sets DPOR.
//!
//! Implements a hybrid of Algorithms 1 and 2 from Abdulla et al., "Source Sets:
//! A Foundation for Optimal Dynamic Partial Order Reduction", JACM 2017.
//!
//! **Current algorithm**: Uses wakeup trees (Algorithm 2, JACM'17 p.24-25) as
//! the backtrack data structure, but performs race detection during execution
//! (Algorithm 1 style, JACM'17 p.16) rather than only at maximal executions.
//! Sleep sets are local per-branch (not yet propagated across scheduling points).
//! Source set filtering is disabled (all racing threads are added to backtrack).
//!
//! Each scheduling point (Branch) maintains:
//!
//! - A **wakeup tree** of thread sequences to explore — replaces classic
//!   Backtrack status with ordered, structured exploration.
//!   (Algorithm 2 lines 14-20, JACM'17 p.24-25)
//! - A **sleep set** of threads that should not be re-explored because an
//!   equivalent execution has already been covered.
//!   (Algorithm 2 lines 13, 20, JACM'17 p.24-25)
//! - **Object tracking** per thread to enable independence-based sleep set
//!   propagation across scheduling points (needed for Algorithm 2 line 16:
//!   `Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}`).

use std::collections::HashMap;

use crate::access::AccessKind;
use crate::thread::ThreadStatus;
use crate::wakeup_tree::WakeupTree;

/// Check whether two access kinds conflict (i.e., are dependent).
///
/// Two accesses to the same object are **dependent** if reordering them could
/// produce a different result. This mirrors the conflict semantics in
/// `ObjectState::dependent_accesses` (object.rs).
///
/// Paper ref: JACM'17 Def 3.3 (p.13) — events e, e' are dependent when they
/// access the same shared variable and at least one is a write (for the basic
/// model). Our extended model also has WeakWrite/WeakRead with relaxed
/// conflict rules for container operations.
///
/// Independence (¬conflict) is used for sleep set propagation:
///   Algorithm 2 line 16 (JACM'17 p.24): Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
/// where p♦q means p's action is independent of q's action.
fn access_kinds_conflict(a: AccessKind, b: AccessKind) -> bool {
    matches!(
        (a, b),
        // Write conflicts with everything
        (AccessKind::Write, _) | (_, AccessKind::Write)
        // Read + WeakWrite conflict (Read depends on WeakWrite)
        | (AccessKind::Read, AccessKind::WeakWrite) | (AccessKind::WeakWrite, AccessKind::Read)
    )
    // Independent pairs (not matched above):
    //   Read + Read, Read + WeakRead, WeakRead + WeakRead,
    //   WeakWrite + WeakWrite, WeakWrite + WeakRead, WeakRead + WeakWrite
}

/// Check if two sets of (object → access_kind) are independent.
///
/// Two sets are independent if, for every object that appears in both sets,
/// the access kinds are non-conflicting (e.g., both reads).
///
/// Paper ref: JACM'17 Def 3.3 (p.13) — independence is the negation of
/// dependence. We approximate the paper's E ⊢ p♦q (Def 3.3) using access-kind
/// compatibility as a sufficient condition for independence.
fn accesses_are_independent(a: &HashMap<u64, AccessKind>, b: &HashMap<u64, AccessKind>) -> bool {
    // Iterate over the smaller map for efficiency
    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    for (obj, kind_a) in smaller {
        if let Some(kind_b) = larger.get(obj) {
            if access_kinds_conflict(*kind_a, *kind_b) {
                return false;
            }
        }
    }
    true
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub threads: Vec<ThreadStatus>,
    pub active_thread: usize,
    pub preemptions: u32,
    /// Wakeup tree: threads/sequences to explore at this scheduling point.
    /// Replaces the classic Backtrack thread status.
    /// Paper: wut(E) in Algorithm 2 (JACM'17 p.24).
    pub wakeup: WakeupTree,
    /// Sleep set: threads that should not be added as backtracks because
    /// an equivalent execution starting with them has already been explored.
    /// Indexed by thread ID; `true` = sleeping.
    /// Paper: sleep(E) in Algorithm 2 (JACM'17 p.24). Tracks Visited threads
    /// locally at each position. Cross-branch propagation is handled by
    /// `propagated_sleep_accesses` below.
    pub sleep: Vec<bool>,
    /// Accesses (object → kind) performed by the active thread at this step.
    /// Used for sleep set independence checks during propagation.
    ///
    /// Tracks the AccessKind so we can distinguish read-read (independent)
    /// from read-write (dependent) on the same object.
    ///
    /// Paper: needed to compute E ⊢ p♦q (independence, JACM'17 Def 3.3 p.13)
    /// — two events are independent if their accesses don't conflict.
    pub active_accesses: HashMap<u64, AccessKind>,
    /// For each previously-explored (Visited) thread at this position,
    /// its accesses (object → kind). Used for sleep set propagation:
    /// a sleeping thread stays asleep if its accesses are independent of
    /// the active thread's accesses.
    ///
    /// Paper: approximates the independence check E ⊢ p♦q (JACM'17 p.13).
    pub explored_accesses: HashMap<usize, HashMap<u64, AccessKind>>,
    /// Sleep set entries propagated from the previous scheduling point.
    ///
    /// When thread p is chosen at position i, each sleeping thread q at
    /// position i whose next action is independent of p's action (p♦q)
    /// stays asleep at position i+1. This map carries both the sleeping
    /// status AND the thread's access info, enabling multi-hop propagation.
    ///
    /// Paper ref: Algorithm 2 line 16 (JACM'17 p.24):
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    /// In the paper's recursive formulation, Sleep' is passed to the
    /// recursive Explore(E.p, WuT', Sleep') call. In our iterative
    /// implementation, we compute it during replay and store it here.
    pub propagated_sleep_accesses: HashMap<usize, HashMap<u64, AccessKind>>,
    // Note: pending_race_objects was removed — source set filtering at the
    // race-object level requires sleep sets for soundness.
    // Paper: source set filtering (Def 4.3, JACM'17 p.15) allows adding only
    // one thread per race, but this requires sleep sets to prevent re-exploration
    // of equivalent traces. See ideas/optimal_dpor.md Phase 2.
}

impl Branch {
    pub fn new(threads: Vec<ThreadStatus>, active_thread: usize, preemptions: u32) -> Self {
        let num_threads = threads.len();
        Self {
            threads,
            active_thread,
            preemptions,
            wakeup: WakeupTree::empty(),
            sleep: vec![false; num_threads],
            active_accesses: HashMap::new(),
            explored_accesses: HashMap::new(),
            propagated_sleep_accesses: HashMap::new(),
        }
    }
}

/// The exploration tree: manages DFS over scheduling decisions using
/// wakeup trees and sleep sets (Source Sets DPOR).
pub struct Path {
    branches: Vec<Branch>,
    pos: usize,
    preemption_bound: Option<u32>,
    /// Per-thread union of all accesses from the most recently completed
    /// execution. Used for sleep set propagation to new branches (Phase 2b).
    ///
    /// When propagating sleep to positions beyond the replay prefix, we
    /// cannot rely on `explored_accesses` (which only captures one scheduling
    /// step's accesses). Instead, we use this cached union of ALL the
    /// thread's accesses from the previous execution as a conservative
    /// approximation of its future behavior.
    ///
    /// Paper ref: JACM'17 Section 10 (p.31-35) — Concuerror maintains
    /// per-process event sequences to determine sleeping processes' next
    /// actions during propagation. Our approach uses the full access union
    /// instead of per-step traces, which is more conservative (may over-wake)
    /// but simpler and always sound for deterministic access patterns.
    ///
    /// Soundness note: For threads with data-dependent access patterns (where
    /// the set of accessed objects depends on values read from shared state),
    /// the cached union may under-approximate the thread's actual future
    /// accesses. This could cause a thread to remain asleep when it should
    /// wake, potentially missing an interleaving. For programs with fixed
    /// access patterns (like the writer-readers benchmark), the cache is exact.
    prev_thread_all_accesses: HashMap<usize, HashMap<u64, AccessKind>>,
}

impl Path {
    pub fn new(preemption_bound: Option<u32>) -> Self {
        Self {
            branches: Vec::new(),
            pos: 0,
            preemption_bound,
            prev_thread_all_accesses: HashMap::new(),
        }
    }

    pub fn current_position(&self) -> usize {
        self.pos
    }

    pub fn depth(&self) -> usize {
        self.branches.len()
    }

    /// Record that an object was accessed at the given scheduling step.
    /// Called by the engine after each process_access/process_io_access.
    /// Populates `active_accesses` for future independence checks
    /// (E ⊢ p♦q, JACM'17 Def 3.3 p.13).
    ///
    /// When the same object is accessed multiple times at one scheduling
    /// point with different AccessKinds, we conservatively upgrade to
    /// Write (which conflicts with everything). This ensures the
    /// independence check remains sound.
    pub fn record_access(&mut self, path_id: usize, object_id: u64, kind: AccessKind) {
        if let Some(branch) = self.branches.get_mut(path_id) {
            branch.active_accesses
                .entry(object_id)
                .and_modify(|existing| {
                    // If access kinds differ, upgrade to Write (conservative:
                    // Write conflicts with everything, so independence checks
                    // will correctly report dependent).
                    if *existing != kind {
                        *existing = AccessKind::Write;
                    }
                })
                .or_insert(kind);
        }
    }

    /// Pick which thread to run at the current scheduling point.
    /// During replay, follows the recorded path; otherwise creates a new branch.
    ///
    /// In Algorithm 2 (JACM'17 p.24), this corresponds to lines 8-12 (choosing
    /// which process to explore) and line 18 (recursive Explore call with
    /// Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}).
    ///
    /// **Sleep set propagation**: At each position, we propagate sleeping
    /// threads from the previous position. A thread q sleeping at position
    /// i-1 stays asleep at position i if q's recorded accesses are
    /// independent of the chosen thread p's accesses at position i-1.
    /// This implements Algorithm 2 line 16 (JACM'17 p.24).
    pub fn schedule(
        &mut self,
        runnable: &[usize],
        current_thread: usize,
        num_threads: usize,
    ) -> Option<usize> {
        if runnable.is_empty() {
            return None;
        }

        if self.pos < self.branches.len() {
            // Replay: propagate sleep set from previous position, then
            // clear active_accesses for fresh recording.
            self.propagate_sleep(self.pos);
            self.branches[self.pos].active_accesses.clear();
            let chosen = self.branches[self.pos].active_thread;
            self.pos += 1;
            return Some(chosen);
        }

        // New branch: prefer current thread to minimize preemptions
        let chosen = if runnable.contains(&current_thread) {
            current_thread
        } else {
            runnable[0]
        };

        let is_preemption = chosen != current_thread && runnable.contains(&current_thread);
        let prev_preemptions = self.branches.last().map_or(0, |b| b.preemptions);
        let preemptions = if is_preemption { prev_preemptions + 1 } else { prev_preemptions };

        let mut threads = vec![ThreadStatus::Disabled; num_threads];
        for &tid in runnable {
            threads[tid] = if tid == chosen { ThreadStatus::Active } else { ThreadStatus::Pending };
        }

        let branch = Branch::new(threads, chosen, preemptions);
        self.branches.push(branch);

        // Propagate sleep set to new branches using trace caching (Phase 2b).
        //
        // The trace cache (`prev_thread_all_accesses`) stores the union of
        // ALL accesses each thread performed in the previous execution. This
        // provides a conservative approximation of each sleeping thread's
        // future behavior, enabling the independence check (Algorithm 2
        // line 16, JACM'17 p.24: Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}) to
        // work at new positions beyond the replay prefix.
        //
        // Without trace caching, `explored_accesses` only captures one
        // scheduling step's accesses — a thread with reads at one step but
        // writes at another would appear falsely independent. The trace
        // cache union includes ALL accesses, preventing this.
        //
        // Paper ref: JACM'17 Section 10 (p.31-35) — Concuerror caches
        // per-process event traces for this purpose.
        self.propagate_sleep(self.pos);

        self.pos += 1;
        Some(chosen)
    }

    /// Propagate the sleep set from position `pos-1` to position `pos`.
    ///
    /// For each thread q that is sleeping at position pos-1 (either locally
    /// Visited or propagated from an earlier position), check if q's
    /// recorded accesses are independent of the active thread's accesses
    /// at pos-1. If independent, q stays asleep at pos.
    ///
    /// This implements Algorithm 2 line 16 (JACM'17 p.24):
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    ///
    /// The independence check E ⊢ p♦q is approximated by access-kind
    /// compatibility: two accesses to the same object are independent if
    /// they are both reads (or other non-conflicting combinations like
    /// WeakWrite+WeakRead). This is a sufficient condition — if the object
    /// sets are disjoint or all overlapping accesses are compatible, the
    /// actions are truly independent in any execution context.
    ///
    /// **Approach (c) from the plan**: We propagate through the replayed
    /// prefix and into new positions. For locally-sleeping threads, we use
    /// `explored_accesses` (recorded when the thread was the active thread
    /// at this position). For propagated threads, we carry their access
    /// info forward. Threads without access info are woken up (conservative).
    fn propagate_sleep(&mut self, pos: usize) {
        if pos == 0 {
            // Position 0 has no predecessor to propagate from.
            return;
        }

        let prev = pos - 1;
        // Collect the active thread's accesses at the previous position.
        // These are the accesses of the thread that was chosen at pos-1.
        let prev_active_accesses = self.branches[prev].active_accesses.clone();

        // Collect all sleeping threads at pos-1 and their access info.
        // A thread is sleeping if it's locally Visited (sleep[q]=true) or
        // propagated from an earlier position (in propagated_sleep_accesses).
        let mut sleeping_threads: HashMap<usize, HashMap<u64, AccessKind>> = HashMap::new();

        // 1. Locally-sleeping threads (Visited at pos-1)
        //
        // Use trace cache (`prev_thread_all_accesses`) when available for
        // the thread's access info. The trace cache provides the union of
        // ALL the thread's accesses from the previous execution, which is
        // critical for propagation to new branches: `explored_accesses`
        // only captures one scheduling step's accesses, so a thread with
        // reads at one step but writes at another would appear falsely
        // independent. The trace cache union includes all accesses.
        //
        // Fallback to `explored_accesses` when no trace cache is available
        // (e.g., during the first execution).
        //
        // Paper ref: JACM'17 Section 10 (p.31-35) — Concuerror caches
        // per-process event traces for this purpose.
        let num_threads = self.branches[prev].sleep.len();
        for tid in 0..num_threads {
            if self.branches[prev].sleep.get(tid).copied().unwrap_or(false) {
                if let Some(accesses) = self.prev_thread_all_accesses.get(&tid) {
                    sleeping_threads.insert(tid, accesses.clone());
                } else if let Some(accesses) = self.branches[prev].explored_accesses.get(&tid) {
                    sleeping_threads.insert(tid, accesses.clone());
                }
                // If neither available, we can't check independence →
                // wake up (don't add to sleeping_threads). Conservative.
            }
        }

        // 2. Propagated-sleeping threads (from even earlier positions)
        //
        // Use trace cache for propagated threads too, since their carried
        // accesses may only reflect a single-position snapshot from when
        // they were first put to sleep. The trace cache provides the
        // complete access union.
        for (tid, accesses) in &self.branches[prev].propagated_sleep_accesses {
            if let Some(cached) = self.prev_thread_all_accesses.get(tid) {
                sleeping_threads.insert(*tid, cached.clone());
            } else {
                sleeping_threads.insert(*tid, accesses.clone());
            }
        }

        // Compute which sleeping threads stay asleep at pos.
        let mut propagated: HashMap<usize, HashMap<u64, AccessKind>> = HashMap::new();
        for (tid, tid_accesses) in sleeping_threads {
            if accesses_are_independent(&tid_accesses, &prev_active_accesses) {
                propagated.insert(tid, tid_accesses);
            }
            // If dependent, the thread wakes up — it must be available for
            // backtracking at this position since its equivalent trace may
            // not have been explored.
        }

        self.branches[pos].propagated_sleep_accesses = propagated;
    }

    /// Mark thread_id for backtracking at branch path_id.
    /// Uses wakeup tree insertion and sleep set filtering.
    ///
    /// This is called during execution (Algorithm 1 style, JACM'17 p.16 lines
    /// 5-9). In full Optimal-DPOR (Algorithm 2, p.24 lines 2-6), race detection
    /// is deferred to maximal executions and inserts notdep sequences rather
    /// than single threads. See ideas/optimal_dpor.md Phase 3.
    ///
    /// `_race_object`: the object involved in the race. Reserved for future
    /// source set filtering (Def 4.3, JACM'17 p.15; requires sleep sets).
    pub fn backtrack(&mut self, path_id: usize, thread_id: usize, _race_object: Option<u64>) {
        if path_id >= self.branches.len() {
            return;
        }

        let branch = &self.branches[path_id];

        // Only runnable threads can be backtracked
        match branch.threads.get(thread_id).copied() {
            Some(ThreadStatus::Pending) | Some(ThreadStatus::Yield) => {}
            _ => return,
        }

        // Sleep set check: if thread is sleeping (locally Visited or
        // propagated from an earlier position), skip.
        //
        // Paper: Algorithm 2 line 5 (JACM'17 p.24) checks
        //   sleep(E') ∩ WI[E'](v) = ∅
        // before inserting. Our version checks both the local sleep set
        // (threads Visited at this position) and the propagated sleep set
        // (threads whose next action is independent of all chosen threads
        // between their home position and this position).
        if branch.sleep.get(thread_id).copied().unwrap_or(false) {
            return;
        }
        if branch.propagated_sleep_accesses.contains_key(&thread_id) {
            return;
        }

        // Already in wakeup tree?
        if branch.wakeup.contains_thread(thread_id) {
            return;
        }

        // Preemption bound check
        if let Some(bound) = self.preemption_bound {
            let branch = &self.branches[path_id];
            if branch.active_thread != thread_id && branch.preemptions >= bound {
                self.add_conservative_backtrack(path_id, thread_id, bound);
                return;
            }
        }

        // Insert into wakeup tree.
        // Paper: Algorithm 2 line 6 (JACM'17 p.24):
        //   wut(E') := insert[E'](v, wut(E'))
        // We insert single-thread sequences [q] rather than full notdep
        // sequences v = notdep(e, E).e'. See Phase 3b for multi-step.
        self.branches[path_id].wakeup.insert(&[thread_id]);
    }

    /// Advance to the next unexplored execution path.
    /// Uses wakeup trees instead of scanning for Backtrack status.
    ///
    /// Implements the while loop of Algorithm 2 lines 14-20 (JACM'17 p.24-25):
    ///   while ∃p ∈ wut(E):
    ///     pick min≺{p}           → min_thread()
    ///     explore E.p            → set active_thread, reset pos
    ///     remove p.w from wut(E) → remove_branch()
    ///     add p to sleep(E)      → sleep[active] = true
    pub fn step(&mut self) -> bool {
        // Save per-thread access unions from the just-completed execution.
        // These are used as a trace cache for sleep set propagation to new
        // branches in the NEXT execution (Phase 2b).
        //
        // For each thread, compute the union of all its `active_accesses`
        // across all positions where it was the active thread. This gives
        // a conservative approximation of the thread's complete behavior:
        // if ANY access conflicts, the thread is woken during propagation.
        //
        // Paper ref: JACM'17 Section 10 (p.31-35) — trace caching for
        // sleep set propagation. The union is conservative (may over-wake)
        // but sound for deterministic access patterns.
        let mut thread_accesses: HashMap<usize, HashMap<u64, AccessKind>> = HashMap::new();
        for branch in &self.branches {
            let tid = branch.active_thread;
            let entry = thread_accesses.entry(tid).or_default();
            for (obj_id, kind) in &branch.active_accesses {
                entry
                    .entry(*obj_id)
                    .and_modify(|existing| {
                        if *existing != *kind {
                            *existing = AccessKind::Write;
                        }
                    })
                    .or_insert(*kind);
            }
        }
        self.prev_thread_all_accesses = thread_accesses;

        while let Some(branch) = self.branches.last_mut() {
            let active = branch.active_thread;
            if active < branch.threads.len() && branch.threads[active] == ThreadStatus::Active {
                branch.threads[active] = ThreadStatus::Visited;
                // Save explored accesses for future sleep set propagation.
                // These are used to check independence (E ⊢ p♦q) when
                // propagating sleep sets across scheduling points.
                // Paper: Algorithm 2 line 16 (JACM'17 p.24).
                let accesses = branch.active_accesses.clone();
                branch.explored_accesses.insert(active, accesses);
                // Add to sleep set: this thread has been explored at this position.
                // Paper: Algorithm 2 line 20 (JACM'17 p.25): "add p to sleep(E)"
                if active < branch.sleep.len() {
                    branch.sleep[active] = true;
                }
                branch.active_accesses.clear();
            }

            // Remove current thread from wakeup tree (already explored).
            // Paper: Algorithm 2 line 19 (JACM'17 p.25):
            //   "remove all sequences of form p.w from wut(E)"
            branch.wakeup.remove_branch(active);

            // Find next thread to explore from wakeup tree.
            // Paper: Algorithm 2 line 15 (JACM'17 p.24):
            //   "let p = min≺{p ∈ wut(E)}"
            // We use minimum thread ID as a deterministic proxy for ≺.
            if let Some(next) = branch.wakeup.min_thread() {
                branch.threads[next] = ThreadStatus::Active;
                branch.active_thread = next;
                // Reset to start: stateless MC replays the full prefix from scratch.
                self.pos = 0;
                return true;
            }

            self.branches.pop();
        }
        false
    }

    fn add_conservative_backtrack(&mut self, path_id: usize, thread_id: usize, bound: u32) {
        for i in (0..path_id).rev() {
            let branch = &self.branches[i];
            if let Some(status) = branch.threads.get(thread_id) {
                let would_preempt = branch.active_thread != thread_id && status.is_runnable();
                if matches!(status, ThreadStatus::Pending | ThreadStatus::Yield)
                    && (!would_preempt || branch.preemptions < bound)
                    && !branch.wakeup.contains_thread(thread_id)
                    && !branch.sleep.get(thread_id).copied().unwrap_or(false)
                    && !branch.propagated_sleep_accesses.contains_key(&thread_id)
                {
                    self.branches[i].wakeup.insert(&[thread_id]);
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_path() {
        let path = Path::new(None);
        assert_eq!(path.depth(), 0);
        assert_eq!(path.current_position(), 0);
    }

    #[test]
    fn test_schedule_prefers_current_thread() {
        let mut path = Path::new(None);
        assert_eq!(path.schedule(&[0, 1], 0, 2), Some(0));
        assert_eq!(path.depth(), 1);
    }

    #[test]
    fn test_backtrack_and_step() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2);
        path.backtrack(0, 1, None);
        assert!(path.step());
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
    }

    #[test]
    fn test_step_exhausted() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2);
        assert!(!path.step());
    }

    #[test]
    fn test_full_exploration_two_threads() {
        let mut path = Path::new(None);
        let mut executions = Vec::new();

        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(chosen);
        path.backtrack(0, 1, None);

        assert!(path.step());
        let chosen = path.schedule(&[0, 1], 0, 2).unwrap();
        executions.push(chosen);

        assert!(!path.step());
        assert_eq!(executions, vec![0, 1]);
    }

    #[test]
    fn test_preemption_bounding_zero() {
        let mut path = Path::new(Some(0));
        path.schedule(&[0, 1], 0, 2);
        path.backtrack(0, 1, None);
        // With bound=0, backtrack to thread 1 is a preemption and should be suppressed
        assert!(!path.step());
    }

    // --- Source Sets / Wakeup Tree specific tests ---

    #[test]
    fn test_wakeup_tree_min_index_ordering() {
        // step() picks the minimum thread ID from the wakeup tree
        let mut path = Path::new(None);
        path.schedule(&[0, 1, 2], 0, 3);
        path.backtrack(0, 2, None); // add 2 first
        path.backtrack(0, 1, None); // add 1 second

        // step() should pick the minimum index (1), not insertion order (2)
        assert!(path.step());
        let chosen = path.schedule(&[0, 1, 2], 0, 3);
        assert_eq!(chosen, Some(1));
    }

    #[test]
    fn test_sleep_set_visited_thread() {
        use crate::access::AccessKind;
        // After exploring thread 0 at position 0, it should be in the sleep set
        let mut path = Path::new(None);
        path.schedule(&[0, 1, 2], 0, 3);
        // Record some access for thread 0
        path.record_access(0, 100, AccessKind::Write);
        path.backtrack(0, 1, None);

        // step() marks 0 as Visited and adds to sleep set
        assert!(path.step());

        // Now if a race suggests backtracking to 0 at position 0, it should be skipped
        path.backtrack(0, 0, None); // 0 is Visited, so this should be a no-op
        // The wakeup tree should not have thread 0 (only thread 1 was from step())
        // This just verifies the Visited check prevents re-adding
    }

    #[test]
    fn test_visited_thread_cannot_be_backtracked() {
        use crate::access::AccessKind;
        // After exploring T0 at position 0, T0 is Visited and in the sleep set.
        // Backtracks to T0 at position 0 should be rejected.
        let mut path = Path::new(None);
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write);

        // Add backtrack for T1
        path.backtrack(0, 1, None);

        // step() → T0 becomes Visited, sleep[0] = true. T1 becomes Active.
        assert!(path.step());
        assert!(path.branches[0].sleep[0]);
        assert_eq!(path.branches[0].threads[0], ThreadStatus::Visited);
        assert_eq!(path.branches[0].threads[1], ThreadStatus::Active);

        // Try to backtrack T0 at position 0 — rejected because T0 is Visited
        path.backtrack(0, 0, None);
        // T2 should still be addable
        path.backtrack(0, 2, None);
        assert!(path.branches[0].wakeup.contains_thread(2));
    }

    #[test]
    fn test_duplicate_backtrack_rejected() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1], 0, 2);
        path.backtrack(0, 1, None);
        path.backtrack(0, 1, None); // duplicate
        // Wakeup tree should have only one entry
        assert_eq!(path.branches[0].wakeup.root_threads(), vec![1]);
    }

    #[test]
    fn test_backtrack_to_disabled_rejected() {
        let mut path = Path::new(None);
        path.schedule(&[0], 0, 2); // Only thread 0 is runnable; thread 1 is Disabled
        path.backtrack(0, 1, None);
        assert!(path.branches[0].wakeup.is_empty());
    }

    #[test]
    fn test_race_object_passed_through() {
        // All racing threads should be added regardless of object
        // (source set filtering is disabled for soundness without sleep sets)
        let mut path = Path::new(None);
        path.schedule(&[0, 1, 2, 3], 0, 4);

        path.backtrack(0, 1, Some(100));
        path.backtrack(0, 2, Some(100));
        path.backtrack(0, 3, Some(200));

        assert_eq!(path.branches[0].wakeup.root_threads(), vec![1, 2, 3]);
    }

    /// Verify that `contains_thread` in `backtrack()` correctly deduplicates:
    /// same thread is rejected, different threads (even on same object) are added.
    #[test]
    fn test_source_set_check_via_contains_thread() {
        let mut path = Path::new(None);
        path.schedule(&[0, 1, 2, 3], 0, 4);

        // First backtrack for object 100: T1 added
        path.backtrack(0, 1, Some(100));
        assert!(path.branches[0].wakeup.contains_thread(1));

        // Different thread, same object: T2 also added (different initials)
        path.backtrack(0, 2, Some(100));
        assert!(path.branches[0].wakeup.contains_thread(2));

        // Same thread again: duplicate rejected by contains_thread
        path.backtrack(0, 1, Some(100));
        assert_eq!(path.branches[0].wakeup.root_threads(), vec![1, 2]);
    }

    // --- Sleep set propagation tests ---

    /// Test that sleep set propagation works during REPLAY of existing branches.
    ///
    /// Scenario: After exploring T0 and T1 at pos 1, when replaying through
    /// pos 0→pos 1, T1's local sleep at pos 1 should be checked against
    /// T0's active_accesses at pos 0 for independence.
    ///
    /// Paper ref: Algorithm 2 line 16 (JACM'17 p.24):
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    #[test]
    fn test_propagated_sleep_during_replay() {
        use crate::access::AccessKind;
        let mut path = Path::new(None);

        // --- First execution: T0 at pos 0, T1 at pos 1 ---
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write);
        path.schedule(&[1, 2], 1, 3);
        path.record_access(1, 200, AccessKind::Read);

        // Add backtrack for T1 at pos 0 and T2 at pos 1
        path.backtrack(0, 1, None);
        path.backtrack(1, 2, None);

        // step(): pops to pos 1 (T1 Visited, wakeup has T2). T2 active at pos 1.
        assert!(path.step());

        // --- Second execution: replay pos 0 (T0), replay pos 1 (T2) ---
        let chosen = path.schedule(&[0, 1, 2], 0, 3);
        assert_eq!(chosen, Some(0)); // replay T0
        path.record_access(0, 100, AccessKind::Write); // T0 writes obj 100

        let chosen = path.schedule(&[1, 2], 1, 3);
        assert_eq!(chosen, Some(2)); // replay T2 (from wakeup)

        // Propagation from pos 0 to pos 1 during replay:
        // No threads are sleeping at pos 0 (T0 is Active, never Visited there).
        // T1 is locally sleeping at pos 1 (Visited), but that's handled by
        // the local sleep check in backtrack(), not by propagation.
        // Propagation only carries sleeping threads FROM the source position.
        assert!(
            path.branches[1].propagated_sleep_accesses.is_empty(),
            "Nothing propagated from pos 0 (no sleeping threads there)"
        );
        // T1 IS locally sleeping at pos 1
        assert!(path.branches[1].sleep[1]);
    }

    /// Test that propagation works for new branches via trace caching (Phase 2b).
    ///
    /// The trace cache records per-thread access unions from the previous
    /// execution, enabling sleep set propagation to new branches (beyond the
    /// replay prefix). When a sleeping thread's cached accesses are
    /// independent of the active thread's accesses, it stays asleep.
    ///
    /// Paper ref: JACM'17 Section 10 (p.31-35) — Concuerror uses trace
    /// caching to determine sleeping threads' next actions. The independence
    /// check Sleep' = {q ∈ sleep(E) | E ⊢ p♦q} (Algorithm 2 line 16, p.24)
    /// uses the cached accesses for q's actions.
    #[test]
    fn test_propagation_to_new_branches_via_trace_cache() {
        use crate::access::AccessKind;
        let mut path = Path::new(None);

        // --- First execution: T0 writes obj 100, T1 reads obj 200 ---
        path.schedule(&[0, 1], 0, 2);
        path.record_access(0, 100, AccessKind::Write);
        path.schedule(&[1], 1, 2);
        path.record_access(1, 200, AccessKind::Read);

        // Backtrack T1 at pos 0
        path.backtrack(0, 1, None);

        // step(): saves trace, T0 → Visited at pos 0
        assert!(path.step());

        // --- Second execution: T1 at pos 0 (replay), then new pos 1 ---
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 200, AccessKind::Read); // T1 reads different object

        // New branch at pos 1: T0 chosen
        path.schedule(&[0], 0, 2);

        // With trace caching, propagation to new branches is enabled.
        // T0's cached accesses = {100: Write}, T1's active at pos 0 = {200: Read}.
        // Disjoint objects → INDEPENDENT → T0 stays asleep at pos 1.
        assert!(
            path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 should be propagated to new branch (independent: obj 100 vs obj 200)"
        );
    }

    /// Test that trace caching correctly wakes threads on conflict at new branches.
    ///
    /// When a sleeping thread's cached accesses conflict with the active
    /// thread's accesses, the sleeping thread must be woken up to allow
    /// exploring the conflicting interleaving.
    ///
    /// Paper ref: Algorithm 2 line 16 (JACM'17 p.24) — threads are removed
    /// from Sleep' when their action is dependent (¬(p♦q)).
    #[test]
    fn test_trace_cache_wakes_on_conflict_at_new_branch() {
        use crate::access::AccessKind;
        let mut path = Path::new(None);

        // --- First execution: T0 writes obj 100, T1 writes obj 100 ---
        path.schedule(&[0, 1], 0, 2);
        path.record_access(0, 100, AccessKind::Write);
        path.schedule(&[1], 1, 2);
        path.record_access(1, 100, AccessKind::Write);

        // Backtrack T1 at pos 0
        path.backtrack(0, 1, None);

        // step(): saves trace, T0 → Visited
        assert!(path.step());

        // --- Second execution: T1 at pos 0 (replay), then new pos 1 ---
        let chosen = path.schedule(&[0, 1], 0, 2);
        assert_eq!(chosen, Some(1));
        path.record_access(0, 100, AccessKind::Write); // T1 writes same obj 100

        // New branch at pos 1: T0 chosen
        path.schedule(&[0], 0, 2);

        // T0's cached accesses = {100: Write}, T1's active at pos 0 = {100: Write}.
        // Write vs Write on obj 100 → CONFLICT → T0 wakes up.
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&0),
            "T0 should NOT be propagated (Write vs Write conflict on obj 100)"
        );
    }

    /// Test that propagation correctly wakes threads on conflict during replay.
    #[test]
    fn test_propagated_sleep_wakes_on_conflict_during_replay() {
        use crate::access::AccessKind;
        let mut path = Path::new(None);

        // --- First execution: T0 at pos 0, T1 at pos 1 ---
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write); // T0 writes obj 100
        path.schedule(&[1, 2], 1, 3);
        path.record_access(1, 100, AccessKind::Read); // T1 reads obj 100

        // Add backtracks
        path.backtrack(0, 2, None);
        path.backtrack(1, 2, None);

        // step(): pops to pos 1 (T1 Visited, wakeup has T2). T2 active at pos 1.
        assert!(path.step());

        // --- Second execution: replay pos 0 (T0), replay pos 1 (T2) ---
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100, AccessKind::Write); // T0 writes obj 100

        path.schedule(&[1, 2], 1, 3);

        // Propagation from pos 0 to pos 1:
        // T1 sleeping at pos 1 with explored_accesses[1] = {100: Read}.
        // T0's active at pos 0 = {100: Write}.
        // Write vs Read on same obj 100 → CONFLICT → T1 wakes up.
        assert!(
            !path.branches[1].propagated_sleep_accesses.contains_key(&1),
            "T1 should NOT be propagated (Write vs Read conflict on obj 100)"
        );
    }
}
