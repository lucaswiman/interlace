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

use std::collections::{HashMap, HashSet};

use crate::thread::ThreadStatus;
use crate::wakeup_tree::WakeupTree;

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
    /// Paper: sleep(E) in Algorithm 2 (JACM'17 p.24). Currently only tracks
    /// Visited threads locally; cross-branch propagation (line 16:
    /// `Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}`) is not yet implemented.
    pub sleep: Vec<bool>,
    /// Objects accessed by the current active thread at this scheduling step.
    /// Used for sleep set independence checks during propagation.
    /// Paper: needed to compute E ⊢ p♦q (independence, JACM'17 Def 3.3 p.13)
    /// — two events are independent if their object sets are disjoint.
    pub active_objects: HashSet<u64>,
    /// For each previously-explored (Visited) thread at this position,
    /// the set of objects it accessed. Used for sleep set propagation:
    /// a sleeping thread stays asleep if its recorded objects are
    /// disjoint from the active thread's objects (independent actions).
    /// Paper: approximates the independence check E ⊢ p♦q (JACM'17 p.13)
    /// using object-set disjointness as a sufficient condition.
    pub explored_objects: HashMap<usize, HashSet<u64>>,
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
            active_objects: HashSet::new(),
            explored_objects: HashMap::new(),
        }
    }
}

/// The exploration tree: manages DFS over scheduling decisions using
/// wakeup trees and sleep sets (Source Sets DPOR).
pub struct Path {
    branches: Vec<Branch>,
    pos: usize,
    preemption_bound: Option<u32>,
}

impl Path {
    pub fn new(preemption_bound: Option<u32>) -> Self {
        Self { branches: Vec::new(), pos: 0, preemption_bound }
    }

    pub fn current_position(&self) -> usize {
        self.pos
    }

    pub fn depth(&self) -> usize {
        self.branches.len()
    }

    /// Record that an object was accessed at the given scheduling step.
    /// Called by the engine after each process_access/process_io_access.
    /// Populates `active_objects` for future independence checks
    /// (E ⊢ p♦q, JACM'17 Def 3.3 p.13).
    pub fn record_access(&mut self, path_id: usize, object_id: u64) {
        if let Some(branch) = self.branches.get_mut(path_id) {
            branch.active_objects.insert(object_id);
        }
    }

    /// Pick which thread to run at the current scheduling point.
    /// During replay, follows the recorded path; otherwise creates a new branch.
    ///
    /// In Algorithm 2 (JACM'17 p.24), this corresponds to lines 8-12 (choosing
    /// which process to explore) and line 18 (recursive Explore call). Currently
    /// does NOT pass wakeup subtrees (WuT') to guide replay — see Phase 4b.
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
            // Replay: clear active_objects for fresh recording.
            self.branches[self.pos].active_objects.clear();
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
        self.pos += 1;
        Some(chosen)
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

        // Sleep set check: if thread is sleeping (Visited), skip.
        // Paper: Algorithm 2 line 5 (JACM'17 p.24) checks
        //   sleep(E') ∩ WI[E'](v) = ∅
        // before inserting. Our simplified version just checks if the
        // thread itself is in the sleep set.
        if branch.sleep.get(thread_id).copied().unwrap_or(false) {
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
        while let Some(branch) = self.branches.last_mut() {
            let active = branch.active_thread;
            if active < branch.threads.len() && branch.threads[active] == ThreadStatus::Active {
                branch.threads[active] = ThreadStatus::Visited;
                // Save explored objects for future sleep set propagation
                // (needed for Algorithm 2 line 16: independence check E ⊢ p♦q)
                let objects = branch.active_objects.clone();
                branch.explored_objects.entry(active).or_default().extend(objects);
                // Add to sleep set: this thread has been explored at this position.
                // Paper: Algorithm 2 line 20 (JACM'17 p.25): "add p to sleep(E)"
                if active < branch.sleep.len() {
                    branch.sleep[active] = true;
                }
                branch.active_objects.clear();
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
        // After exploring thread 0 at position 0, it should be in the sleep set
        let mut path = Path::new(None);
        path.schedule(&[0, 1, 2], 0, 3);
        // Record some access for thread 0
        path.record_access(0, 100);
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
        // After exploring T0 at position 0, T0 is Visited and in the sleep set.
        // Backtracks to T0 at position 0 should be rejected.
        let mut path = Path::new(None);
        path.schedule(&[0, 1, 2], 0, 3);
        path.record_access(0, 100);

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
}
