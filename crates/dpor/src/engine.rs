//! The DPOR engine: orchestrates systematic exploration of interleavings.
//!
//! Implements race detection and happens-before tracking for DPOR.
//!
//! **Paper ref**: Abdulla et al., "Source Sets: A Foundation for Optimal
//! Dynamic Partial Order Reduction", JACM 2017.
//!
//! **Current algorithm**: Race detection is performed during execution
//! (Algorithm 1 style, JACM'17 p.16 lines 5-9), not deferred to maximal
//! executions (Algorithm 2 style, p.24 lines 1-6). Happens-before is tracked
//! via vector clocks (Section 10, JACM'17 p.34-35).

use std::collections::HashMap;

use crate::access::{Access, AccessKind};
use crate::object::{ObjectId, ObjectState};
use crate::path::Path;
use crate::thread::Thread;
use crate::vv::VersionVec;

/// Synchronization events that affect the happens-before relation.
/// Paper: happens-before is defined in Definition 3.2 (JACM'17 p.12-13).
/// Lock acquire/release create happens-before edges; thread spawn/join
/// establish causal ordering between parent and child.
#[derive(Clone, Debug)]
pub enum SyncEvent {
    LockAcquire { lock_id: u64 },
    LockRelease { lock_id: u64 },
    ThreadJoin { joined_thread: usize },
    ThreadSpawn { child_thread: usize },
}

/// Per-execution state. Reset at the start of each execution.
pub struct Execution {
    pub threads: Vec<Thread>,
    pub objects: HashMap<ObjectId, ObjectState>,
    pub active_thread: usize,
    pub lock_release_vv: HashMap<u64, VersionVec>,
    pub aborted: bool,
    pub schedule_trace: Vec<usize>,
}

impl Execution {
    pub fn new(num_threads: usize) -> Self {
        Self {
            threads: (0..num_threads).map(|i| Thread::new(i, num_threads)).collect(),
            objects: HashMap::new(),
            active_thread: 0,
            lock_release_vv: HashMap::new(),
            aborted: false,
            schedule_trace: Vec::new(),
        }
    }

    pub fn finish_thread(&mut self, thread_id: usize) {
        self.threads[thread_id].finished = true;
    }

    pub fn block_thread(&mut self, thread_id: usize) {
        self.threads[thread_id].blocked = true;
    }

    pub fn unblock_thread(&mut self, thread_id: usize) {
        self.threads[thread_id].blocked = false;
    }

    pub fn runnable_threads(&self) -> Vec<usize> {
        self.threads.iter().filter(|t| t.is_runnable()).map(|t| t.id).collect()
    }
}

/// The main DPOR engine.
pub struct DporEngine {
    path: Path,
    num_threads: usize,
    max_executions: Option<u64>,
    executions_completed: u64,
    max_branches: usize,
}

impl DporEngine {
    /// XOR mask to derive virtual object IDs for lock acquire conflict tracking.
    const LOCK_OBJECT_XOR: u64 = 0x4C4F_434B_4C4F_434B; // "LOCKLOCK"
    /// XOR mask for lock release objects (distinct from acquire).
    const LOCK_RELEASE_XOR: u64 = 0x524C_5345_524C_5345; // "RLSERLSE"

    pub fn new(
        num_threads: usize,
        preemption_bound: Option<u32>,
        max_branches: usize,
        max_executions: Option<u64>,
    ) -> Self {
        Self {
            path: Path::new(preemption_bound),
            num_threads,
            max_executions,
            executions_completed: 0,
            max_branches,
        }
    }

    pub fn begin_execution(&self) -> Execution {
        Execution::new(self.num_threads)
    }

    pub fn schedule(&mut self, execution: &mut Execution) -> Option<usize> {
        let runnable = execution.runnable_threads();
        if runnable.is_empty() {
            execution.aborted = true;
            return None;
        }
        if self.path.current_position() >= self.max_branches {
            execution.aborted = true;
            return None;
        }
        let chosen = self.path.schedule(&runnable, execution.active_thread, self.num_threads)?;
        execution.threads[chosen].dpor_vv.increment(chosen);
        execution.threads[chosen].io_vv.increment(chosen);
        execution.active_thread = chosen;
        execution.schedule_trace.push(chosen);
        Some(chosen)
    }

    /// Report a shared memory access and detect races.
    ///
    /// For each prior dependent access from another thread that is NOT
    /// ordered by happens-before, we have a **race** (JACM'17 Def 3.3 p.13:
    /// e ⋖_E e' when events are from different threads and concurrent).
    /// We add the current thread to the backtrack set at the prior access's
    /// scheduling point to explore the reversed ordering.
    ///
    /// **Note**: This performs race detection during execution (Algorithm 1
    /// style, JACM'17 p.16 lines 5-9). Algorithm 2 (p.24 lines 1-6) defers
    /// race detection to maximal executions and computes notdep sequences.
    pub fn process_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        let current_dpor_vv = execution.threads[thread_id].dpor_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        // Check ALL dependent accesses from other threads (not just the last one).
        // This ensures 3+ thread scenarios are handled correctly.
        // Paper: a race exists when e and e' are dependent and concurrent
        // (¬(e →_E e'), JACM'17 Def 3.3 p.13-14).
        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_dpor_vv) {
                self.path.backtrack(prev_access.path_id, thread_id, Some(object_id));
            }
        }

        let access = Access::new(current_path_id, current_dpor_vv, thread_id);
        object_state.record_access(access, kind);

        // Record access for sleep set independence checks.
        // Includes AccessKind so propagation can distinguish read-read
        // (independent) from read-write (dependent) on the same object.
        self.path.record_access(current_path_id, object_id, kind);
    }

    /// Like [`process_access`] but uses first-access recording semantics
    /// (keeps the earliest access per thread rather than the latest).
    /// Uses the regular `dpor_vv` for happens-before computation, so
    /// lock-based synchronization is still respected.
    ///
    /// This is useful for container-level access keys (`__cmethods__`)
    /// where a thread performs multiple writes to the same container.
    /// By keeping the first write position, DPOR can backtrack to the
    /// earliest point, enabling exploration of interleavings where
    /// another thread runs between the first and subsequent writes.
    pub fn process_first_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        let current_dpor_vv = execution.threads[thread_id].dpor_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_dpor_vv) {
                self.path.backtrack(prev_access.path_id, thread_id, Some(object_id));
            }
        }

        let access = Access::new(current_path_id, current_dpor_vv, thread_id);
        object_state.record_io_access(access, kind);

        self.path.record_access(current_path_id, object_id, kind);
    }

    /// Like [`process_access`] but uses the thread's `io_vv` instead of
    /// `dpor_vv`.  Because `io_vv` does not include lock-based
    /// happens-before edges, I/O accesses from different threads always
    /// appear potentially concurrent — even when they occur inside
    /// separate lock acquisitions.  This lets DPOR explore interleavings
    /// around file/socket operations and catch TOCTOU races.
    pub fn process_io_access(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        object_id: ObjectId,
        kind: AccessKind,
    ) {
        let current_path_id = self.path.current_position().saturating_sub(1);
        let current_io_vv = execution.threads[thread_id].io_vv.clone();

        let object_state = execution.objects.entry(object_id).or_insert_with(ObjectState::new);

        for prev_access in object_state.dependent_accesses(kind, thread_id) {
            if !prev_access.happens_before(&current_io_vv) {
                self.path.backtrack(prev_access.path_id, thread_id, Some(object_id));
            }
        }

        let access = Access::new(current_path_id, current_io_vv, thread_id);
        object_state.record_io_access(access, kind);

        self.path.record_access(current_path_id, object_id, kind);
    }

    /// Process a synchronization event (lock acquire/release, thread spawn/join).
    ///
    /// Updates vector clocks to establish happens-before edges.
    /// Paper: Definition 3.2 properties 1-7 (JACM'17 p.12-13) define valid
    /// happens-before assignments. Lock acquire/release create edges via
    /// vector clock joins; thread spawn/join do the same.
    ///
    /// **Lock handling**: The paper's Algorithms 3-4 (JACM'17 p.27-28) handle
    /// locks by relaxing Assumption 3.1 (processes can disable each other).
    /// Our approach is different: we use a separate `io_vv` vector clock that
    /// omits lock HB edges, making lock operations always appear concurrent
    /// for backtracking purposes. This is conservative (may over-explore)
    /// but catches multi-lock races that pure HB-based tracking would miss.
    pub fn process_sync(
        &mut self,
        execution: &mut Execution,
        thread_id: usize,
        event: SyncEvent,
    ) {
        match event {
            SyncEvent::LockAcquire { lock_id } => {
                // Join with the releasing thread's vector clock to establish
                // happens-before: release(L) →_E acquire(L).
                // Paper: Def 3.2 property 4 (JACM'17 p.12) — linearizations
                // must preserve happens-before and reach the same state.
                if let Some(release_vv) = execution.lock_release_vv.get(&lock_id) {
                    let release_vv = release_vv.clone();
                    execution.threads[thread_id].causality.join(&release_vv);
                    execution.threads[thread_id].dpor_vv.join(&release_vv);
                }
                // Report lock acquire as an I/O access (Write to virtual lock
                // object).  This uses io_vv (no lock-based HB) so that lock
                // operations on the same lock by different threads always
                // appear concurrent — creating backtrack points at lock
                // boundaries.  First-access semantics ensure the backtrack
                // targets the earliest lock position, which is critical for
                // multi-lock race detection: it lets DPOR explore orderings
                // where thread B runs between thread A's two critical sections
                // on different locks.
                let lock_obj_id = lock_id ^ Self::LOCK_OBJECT_XOR;
                self.process_io_access(
                    execution,
                    thread_id,
                    lock_obj_id,
                    AccessKind::Write,
                );
            }
            SyncEvent::LockRelease { lock_id } => {
                // Store dpor_vv (not causality) so that lock-based
                // happens-before edges carry meaningful scheduling
                // information.  The acquiring thread's dpor_vv will be
                // joined with this, ordering data accesses inside the
                // same critical section correctly.
                let vv = execution.threads[thread_id].dpor_vv.clone();
                execution.lock_release_vv.insert(lock_id, vv);
                // Report lock release as an I/O access to a SEPARATE
                // virtual object (distinct from the acquire object).
                // This creates backtrack points at lock release positions,
                // which is critical for multi-lock races: the "gap"
                // between two critical sections starts at the release.
                // Using a different XOR constant ensures the release
                // object doesn't alias with the acquire object, so
                // first-access semantics track them independently.
                let lock_rel_obj_id = lock_id ^ Self::LOCK_RELEASE_XOR;
                self.process_io_access(
                    execution,
                    thread_id,
                    lock_rel_obj_id,
                    AccessKind::Write,
                );
            }
            SyncEvent::ThreadJoin { joined_thread } => {
                let joined_causality = execution.threads[joined_thread].causality.clone();
                let joined_dpor_vv = execution.threads[joined_thread].dpor_vv.clone();
                let joined_io_vv = execution.threads[joined_thread].io_vv.clone();
                execution.threads[thread_id].causality.join(&joined_causality);
                execution.threads[thread_id].dpor_vv.join(&joined_dpor_vv);
                execution.threads[thread_id].io_vv.join(&joined_io_vv);
            }
            SyncEvent::ThreadSpawn { child_thread } => {
                let parent_causality = execution.threads[thread_id].causality.clone();
                let parent_dpor_vv = execution.threads[thread_id].dpor_vv.clone();
                let parent_io_vv = execution.threads[thread_id].io_vv.clone();
                execution.threads[child_thread].causality.join(&parent_causality);
                execution.threads[child_thread].dpor_vv.join(&parent_dpor_vv);
                execution.threads[child_thread].io_vv.join(&parent_io_vv);
            }
        }
    }

    pub fn next_execution(&mut self) -> bool {
        self.executions_completed += 1;
        if let Some(max) = self.max_executions {
            if self.executions_completed >= max {
                return false;
            }
        }
        self.path.step()
    }

    pub fn executions_completed(&self) -> u64 {
        self.executions_completed
    }

    pub fn tree_depth(&self) -> usize {
        self.path.depth()
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_threads_no_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut execution = engine.begin_execution();

        let t0 = engine.schedule(&mut execution).unwrap();
        assert_eq!(t0, 0);
        engine.process_access(&mut execution, 0, 1, AccessKind::Write);
        execution.finish_thread(0);

        let t1 = engine.schedule(&mut execution).unwrap();
        assert_eq!(t1, 1);
        engine.process_access(&mut execution, 1, 2, AccessKind::Write);
        execution.finish_thread(1);

        assert!(!engine.next_execution());
        assert_eq!(engine.executions_completed(), 1);
    }

    #[test]
    fn test_two_threads_write_write_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();
            let first = engine.schedule(&mut execution).unwrap();
            engine.process_access(&mut execution, first, 1, AccessKind::Write);
            execution.finish_thread(first);

            let second = engine.schedule(&mut execution).unwrap();
            engine.process_access(&mut execution, second, 1, AccessKind::Write);
            execution.finish_thread(second);

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        assert_eq!(exec_count, 2);
    }

    #[test]
    fn test_read_read_no_conflict() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut execution = engine.begin_execution();

        let first = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, first, 1, AccessKind::Read);
        execution.finish_thread(first);

        let second = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, second, 1, AccessKind::Read);
        execution.finish_thread(second);

        assert!(!engine.next_execution());
    }

    #[test]
    fn test_counter_lost_update() {
        #[derive(Clone, Debug)]
        struct State {
            counter: i64,
            local: [i64; 2],
        }

        let mut engine = DporEngine::new(2, None, 100_000, None);
        let thread_ops = vec![
            vec![(0u64, AccessKind::Read), (0, AccessKind::Write)],
            vec![(0u64, AccessKind::Read), (0, AccessKind::Write)],
        ];
        let mut found_bug = false;

        loop {
            let mut execution = engine.begin_execution();
            let mut state = State { counter: 0, local: [0, 0] };
            let mut pcs = vec![0usize; 2];

            loop {
                for i in 0..2 {
                    if pcs[i] >= thread_ops[i].len() {
                        execution.finish_thread(i);
                    }
                }
                if execution.runnable_threads().is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                let pc = pcs[chosen];
                if pc >= thread_ops[chosen].len() {
                    break;
                }
                let (obj_id, kind) = thread_ops[chosen][pc];
                engine.process_access(&mut execution, chosen, obj_id, kind);

                // Apply the step
                match (chosen, pc) {
                    (0, 0) => state.local[0] = state.counter,
                    (0, 1) => state.counter = state.local[0] + 1,
                    (1, 0) => state.local[1] = state.counter,
                    (1, 1) => state.counter = state.local[1] + 1,
                    _ => unreachable!(),
                }
                pcs[chosen] += 1;
            }

            if state.counter != 2 {
                found_bug = true;
            }

            if !engine.next_execution() {
                break;
            }
        }

        assert!(found_bug);
    }

    #[test]
    fn test_independent_threads_one_execution() {
        let mut engine = DporEngine::new(2, None, 1000, None);
        let mut execution = engine.begin_execution();

        let t0 = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, t0, 0, AccessKind::Write);
        execution.finish_thread(t0);

        let t1 = engine.schedule(&mut execution).unwrap();
        engine.process_access(&mut execution, t1, 1, AccessKind::Write);
        execution.finish_thread(t1);

        assert!(!engine.next_execution());
        assert_eq!(engine.executions_completed(), 1);
    }

    #[test]
    fn test_three_threads_write_conflict() {
        // Regression test: with the old single-last-access tracking,
        // Thread A's access would be overwritten by Thread B's, so the
        // conflict between A and C was never explored. The per-thread
        // map ensures all conflicts are detected.
        let mut engine = DporEngine::new(3, None, 1000, None);
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();

            // Each thread writes to the same object (id=1)
            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(&mut execution, chosen, 1, AccessKind::Write);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // 3 threads all writing to the same object: 3! = 6 orderings
        assert_eq!(exec_count, 6);
    }

    /// Model of the multi-lock race (defect #11):
    ///
    /// Thread A: lock(cL), write(count), unlock(cL), lock(sL), write(sum), unlock(sL)
    /// Thread B: lock(cL), read(count), unlock(cL), lock(sL), read(sum), unlock(sL)
    ///
    /// The bug: B can interleave between A's two critical sections, seeing
    /// count=1 but sum=0.  DPOR must explore the ordering where A goes first
    /// on count_lock but B goes first on sum_lock.
    #[test]
    fn test_multi_lock_race_detected() {
        let mut engine = DporEngine::new(2, None, 100_000, None);

        // Object IDs
        const COUNT: u64 = 1;
        const SUM: u64 = 2;
        const CL: u64 = 100;  // count_lock
        const SL: u64 = 200;  // sum_lock

        // Thread operations: (object_id, access_kind, is_lock_acquire, is_lock_release, lock_id)
        // We model lock acquire/release as sync events + io accesses.
        #[derive(Clone)]
        enum Op {
            LockAcquire(u64),
            Access(u64, AccessKind),
            LockRelease(u64),
        }

        let thread_a = vec![
            Op::LockAcquire(CL),
            Op::Access(COUNT, AccessKind::Write),
            Op::LockRelease(CL),
            Op::LockAcquire(SL),
            Op::Access(SUM, AccessKind::Write),
            Op::LockRelease(SL),
        ];
        let thread_b = vec![
            Op::LockAcquire(CL),
            Op::Access(COUNT, AccessKind::Read),
            Op::LockRelease(CL),
            Op::LockAcquire(SL),
            Op::Access(SUM, AccessKind::Read),
            Op::LockRelease(SL),
        ];
        let thread_ops = [thread_a, thread_b];

        let mut found_bug = false;
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();
            let mut count = 0i64;
            let mut sum = 0i64;
            let mut observed_count: Option<i64> = None;
            let mut observed_sum: Option<i64> = None;
            let mut pcs = [0usize; 2];
            let mut locks_held: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();

            loop {
                // Mark finished threads
                for i in 0..2 {
                    if pcs[i] >= thread_ops[i].len() {
                        execution.finish_thread(i);
                    }
                }
                if execution.runnable_threads().is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                let pc = pcs[chosen];
                if pc >= thread_ops[chosen].len() {
                    break;
                }

                match &thread_ops[chosen][pc] {
                    Op::LockAcquire(lock_id) => {
                        if let Some(&holder) = locks_held.get(lock_id) {
                            if holder != chosen {
                                // Lock is held by other thread — block
                                execution.block_thread(chosen);
                                continue; // don't advance PC
                            }
                        }
                        locks_held.insert(*lock_id, chosen);
                        engine.process_sync(
                            &mut execution, chosen,
                            SyncEvent::LockAcquire { lock_id: *lock_id },
                        );
                    }
                    Op::Access(obj_id, kind) => {
                        engine.process_access(&mut execution, chosen, *obj_id, *kind);
                        // Apply state changes
                        match (chosen, *obj_id, *kind) {
                            (0, 1, AccessKind::Write) => count = 1,
                            (0, 2, AccessKind::Write) => sum = 10,
                            (1, 1, AccessKind::Read) => observed_count = Some(count),
                            (1, 2, AccessKind::Read) => observed_sum = Some(sum),
                            _ => {}
                        }
                    }
                    Op::LockRelease(lock_id) => {
                        locks_held.remove(lock_id);
                        engine.process_sync(
                            &mut execution, chosen,
                            SyncEvent::LockRelease { lock_id: *lock_id },
                        );
                        // Unblock any thread waiting for this lock
                        for tid in 0..2 {
                            if tid != chosen && execution.threads[tid].blocked {
                                if pcs[tid] < thread_ops[tid].len() {
                                    if let Op::LockAcquire(lid) = &thread_ops[tid][pcs[tid]] {
                                        if lid == lock_id {
                                            execution.unblock_thread(tid);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                pcs[chosen] += 1;
            }

            // Check invariant: if B observed count, it should be consistent
            if let (Some(c), Some(s)) = (observed_count, observed_sum) {
                if c > 0 && s != c * 10 {
                    found_bug = true;
                }
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        assert!(
            found_bug,
            "DPOR should detect the multi-lock race (count=1, sum=0 interleaving). \
             Explored {exec_count} executions without finding it."
        );
    }

    #[test]
    fn test_three_threads_read_write_conflict() {
        // Thread A reads, Thread B reads, Thread C writes — all same object.
        // With the old implementation, Thread A's read could be lost.
        let mut engine = DporEngine::new(3, None, 1000, None);
        let mut exec_count = 0;
        let thread_kinds = [AccessKind::Read, AccessKind::Read, AccessKind::Write];

        loop {
            let mut execution = engine.begin_execution();

            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(&mut execution, chosen, 1, thread_kinds[chosen]);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Read-Read is independent, but Read-Write and Write-Read conflict.
        // C(write) conflicts with both A(read) and B(read), so C must be
        // explored in all positions relative to both A and B.
        // Expected: 3 orderings (C before both, C between, C after both)
        assert!(exec_count >= 3, "expected at least 3 executions, got {exec_count}");
    }

    /// Four threads each doing one write to the same object.
    /// Verifies that the wakeup tree correctly handles 4-way write conflicts.
    /// All 4! = 24 orderings are distinct traces and must be explored.
    #[test]
    fn test_four_threads_write_conflict() {
        let mut engine = DporEngine::new(4, None, 10000, None);
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();

            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(&mut execution, chosen, 1, AccessKind::Write);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // 4 threads all writing to the same object: 4! = 24 orderings
        assert_eq!(exec_count, 24);
    }

    /// Two independent pairs: T0/T1 write X, T2/T3 write Y.
    /// Should explore 2 * 2 = 4 orderings (independent pairs don't cross).
    #[test]
    fn test_two_independent_pairs() {
        let mut engine = DporEngine::new(4, None, 10000, None);
        let mut exec_count = 0;
        // T0 and T1 write object 1; T2 and T3 write object 2
        let thread_objects = [1u64, 1, 2, 2];

        loop {
            let mut execution = engine.begin_execution();

            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(
                    &mut execution,
                    chosen,
                    thread_objects[chosen],
                    AccessKind::Write,
                );
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Two independent pairs: 2! * 2! = 4 orderings
        assert_eq!(exec_count, 4);
    }

    /// Model of dining philosophers (simplified):
    /// 2 threads, each locks two resources in opposite order.
    /// Thread 0: write(A), write(B)
    /// Thread 1: write(B), write(A)
    /// Should find all interleavings including the deadlock-prone one.
    #[test]
    fn test_two_philosophers_all_orderings() {
        let mut engine = DporEngine::new(2, None, 10000, None);
        let thread_ops: Vec<Vec<(u64, AccessKind)>> = vec![
            vec![(1, AccessKind::Write), (2, AccessKind::Write)],
            vec![(2, AccessKind::Write), (1, AccessKind::Write)],
        ];
        let mut exec_count = 0;
        let mut traces = Vec::new();

        loop {
            let mut execution = engine.begin_execution();
            let mut pcs = vec![0usize; 2];

            loop {
                for i in 0..2 {
                    if pcs[i] >= thread_ops[i].len() {
                        execution.finish_thread(i);
                    }
                }
                if execution.runnable_threads().is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                let pc = pcs[chosen];
                if pc >= thread_ops[chosen].len() {
                    break;
                }
                let (obj_id, kind) = thread_ops[chosen][pc];
                engine.process_access(&mut execution, chosen, obj_id, kind);
                pcs[chosen] += 1;
            }

            traces.push(execution.schedule_trace.clone());
            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Without sleep set propagation, classic DPOR explores 6 orderings.
        // With propagation, write(A)♦write(B) (different objects) allows
        // commuting across thread boundaries, reducing to 3 Mazurkiewicz
        // traces (the feasible orderings of A-writes and B-writes):
        //   1. T0A < T1A and T0B < T1B  →  [0,0,1,1]
        //   2. T0A < T1A and T1B < T0B  →  [0,1,1,0] ≡ [1,0,0,1] etc.
        //   3. T1A < T0A and T1B < T0B  →  [1,1,0,0]
        // (The 4th combination T1A<T0A, T0B<T1B creates a cycle and is infeasible.)
        assert!(
            exec_count >= 3,
            "expected at least 3 executions for 2-philosopher, got {exec_count}"
        );
    }

    /// Writer-Readers: T0 writes x, T1 reads x, T2 reads x.
    ///
    /// There are exactly 4 distinct Mazurkiewicz traces (modulo the
    /// independence of read-read on the same object):
    ///
    ///   1. W-R1-R2  (≡ W-R2-R1 since R1♦R2)
    ///   2. R1-W-R2
    ///   3. R2-W-R1
    ///   4. R1-R2-W  (≡ R2-R1-W since R1♦R2)
    ///
    /// Without sleep set propagation, classic DPOR explores 5+ executions
    /// because it doesn't recognize that e.g. R1-R2-W and R2-R1-W are
    /// equivalent traces (the reads commute).
    ///
    /// With sleep set propagation (Algorithm 2 line 16, JACM'17 p.24:
    ///   Sleep' = {q ∈ sleep(E) | E ⊢ p♦q}
    /// ), after exploring R2-W-R1 with T0 visited/sleeping at position 0,
    /// when we explore T2 at position 0 (T2-...), T1 is also sleeping and
    /// independent of T2 (read♦read), so T1 stays asleep. This prevents
    /// exploring both T2-T0-T1 and T2-T1-T0, collapsing them into one.
    ///
    /// Paper ref: JACM'17 Section 11 Table 1 (p.36): the "readers" benchmark
    /// with n=3 has 4 source-set traces vs more for classic DPOR.
    #[test]
    fn test_writer_readers_sleep_propagation() {
        let mut engine = DporEngine::new(3, None, 10000, None);
        let thread_kinds = [AccessKind::Write, AccessKind::Read, AccessKind::Read];
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();

            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(&mut execution, chosen, 1, thread_kinds[chosen]);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // With replay-only sleep set propagation (approach (c)):
        // - Full propagation (replay + new branches) gives 4 traces (optimal)
        // - Replay-only propagation gives 5 traces (one redundant trace
        //   because reader-reader propagation to new branches is disabled)
        // - Without propagation: 5+ traces
        // Full optimality (4 traces) requires propagation to new branches,
        // which needs trace caching (approach (b)) for soundness.
        assert!(
            exec_count <= 5,
            "writer-readers (1W + 2R) should explore at most 5 traces, got {exec_count}"
        );
    }

    /// Five threads: T0 writes x, T1-T4 all read x.
    ///
    /// The distinct Mazurkiewicz traces are determined by where the write
    /// appears relative to the reads. Since all reads are mutually
    /// independent (read♦read), the optimal number of traces is 5
    /// (one per position of W in the sequence).
    ///
    /// With Phase 1 sleep set propagation alone, we get 16 traces
    /// (= sum of C(4,k) for k=0..4 — the binomial coefficients counting
    /// how many readers appear before W). Full optimality (5 traces)
    /// requires source set filtering (Phase 2, JACM'17 Def 4.3 p.15):
    /// adding only one thread per race prevents the combinatorial blowup
    /// of reader orderings.
    ///
    /// Paper ref: JACM'17 Table 1 (p.36), "readers" benchmark.
    #[test]
    fn test_writer_four_readers_sleep_propagation() {
        let mut engine = DporEngine::new(5, None, 100_000, None);
        let thread_kinds = [
            AccessKind::Write,
            AccessKind::Read,
            AccessKind::Read,
            AccessKind::Read,
            AccessKind::Read,
        ];
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();

            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(&mut execution, chosen, 1, thread_kinds[chosen]);
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // With replay-only propagation (approach (c)), the count is ~65.
        // Full propagation (to new branches) reduces to 16 but risks
        // unsound blocking (see tricky_races test failures).
        // Optimal = 5 (requires source set filtering, Phase 2).
        // 5! = 120 would be the worst case without any DPOR.
        assert!(
            exec_count < 120,
            "writer-readers (1W + 4R) should be less than 5!=120, got {exec_count}"
        );
    }

    /// Sleep set propagation must not break independent-pair reduction.
    /// T0/T1 write X, T2/T3 write Y — two independent pairs should still
    /// explore exactly 2! × 2! = 4 orderings with propagation enabled.
    #[test]
    fn test_independent_pairs_with_propagation() {
        let mut engine = DporEngine::new(4, None, 10000, None);
        let thread_objects = [1u64, 1, 2, 2];
        let mut exec_count = 0;

        loop {
            let mut execution = engine.begin_execution();

            loop {
                let runnable = execution.runnable_threads();
                if runnable.is_empty() {
                    break;
                }
                let chosen = match engine.schedule(&mut execution) {
                    Some(t) => t,
                    None => break,
                };
                engine.process_access(
                    &mut execution,
                    chosen,
                    thread_objects[chosen],
                    AccessKind::Write,
                );
                execution.finish_thread(chosen);
            }

            exec_count += 1;
            if !engine.next_execution() {
                break;
            }
        }

        // Independent pairs: 2! × 2! = 4 orderings — same as without propagation
        assert_eq!(exec_count, 4);
    }
}
