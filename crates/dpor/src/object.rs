//! Shared object tracking for DPOR.

use std::collections::HashMap;

use crate::access::{Access, AccessKind};

/// Opaque integer ID for shared objects.
pub type ObjectId = u64;

/// Tracks per-thread accesses to a shared object for DPOR.
///
/// Maintains per-thread maps of the most recent read and the most recent
/// write.  A **Write** by another thread depends on *both* the latest
/// read and the latest write from each other thread, because the
/// backtrack points differ: backtracking at a read position allows the
/// adversary to interleave between a read and a subsequent write on the
/// same object (TOCTOU bugs), while backtracking at the write position
/// only reorders complete read-write pairs.
#[derive(Clone, Debug)]
pub struct ObjectState {
    /// Per-thread most recent read access.
    per_thread_read: HashMap<usize, Access>,
    /// Per-thread most recent write access.
    per_thread_write: HashMap<usize, Access>,
    /// Per-thread most recent weak-write access.
    per_thread_weak_write: HashMap<usize, Access>,
    /// Per-thread most recent weak-read access.
    per_thread_weak_read: HashMap<usize, Access>,
}

impl ObjectState {
    pub fn new() -> Self {
        Self {
            per_thread_read: HashMap::new(),
            per_thread_write: HashMap::new(),
            per_thread_weak_write: HashMap::new(),
            per_thread_weak_read: HashMap::new(),
        }
    }

    /// Returns all accesses that the given `kind` by `current_thread` depends on.
    ///
    /// - A **Read** depends on writes from *other* threads (reads are independent).
    /// - A **Write** depends on both reads and writes from *other* threads.
    ///   Returning both ensures DPOR creates backtrack points at read
    ///   positions (for TOCTOU detection) and write positions (for
    ///   write-write ordering).
    pub fn dependent_accesses(&self, kind: AccessKind, current_thread: usize) -> Vec<&Access> {
        match kind {
            AccessKind::Read => {
                // Read depends on Write and WeakWrite from other threads.
                let mut result: Vec<&Access> = Vec::new();
                for (tid, access) in &self.per_thread_write {
                    if *tid != current_thread {
                        result.push(access);
                    }
                }
                for (tid, access) in &self.per_thread_weak_write {
                    if *tid != current_thread {
                        let dominated = self.per_thread_write.get(tid).is_some_and(|w| {
                            w.path_id == access.path_id
                        });
                        if !dominated {
                            result.push(access);
                        }
                    }
                }
                result
            }
            AccessKind::Write => {
                // Write depends on Read, Write, WeakWrite, and WeakRead.
                let mut result: Vec<&Access> = Vec::new();
                for (tid, access) in &self.per_thread_read {
                    if *tid != current_thread {
                        result.push(access);
                    }
                }
                for (tid, access) in &self.per_thread_write {
                    if *tid != current_thread {
                        let dominated = self.per_thread_read.get(tid).is_some_and(|r| {
                            r.path_id == access.path_id
                        });
                        if !dominated {
                            result.push(access);
                        }
                    }
                }
                for (tid, access) in &self.per_thread_weak_write {
                    if *tid != current_thread {
                        let dominated_by_read = self.per_thread_read.get(tid).is_some_and(|r| {
                            r.path_id == access.path_id
                        });
                        let dominated_by_write = self.per_thread_write.get(tid).is_some_and(|w| {
                            w.path_id == access.path_id
                        });
                        if !dominated_by_read && !dominated_by_write {
                            result.push(access);
                        }
                    }
                }
                for (tid, access) in &self.per_thread_weak_read {
                    if *tid != current_thread {
                        let dominated_by_read = self.per_thread_read.get(tid).is_some_and(|r| {
                            r.path_id == access.path_id
                        });
                        let dominated_by_write = self.per_thread_write.get(tid).is_some_and(|w| {
                            w.path_id == access.path_id
                        });
                        let dominated_by_ww = self.per_thread_weak_write.get(tid).is_some_and(|w| {
                            w.path_id == access.path_id
                        });
                        if !dominated_by_read && !dominated_by_write && !dominated_by_ww {
                            result.push(access);
                        }
                    }
                }
                result
            }
            AccessKind::WeakWrite => {
                // WeakWrite depends on Read and Write, but NOT WeakWrite
                // or WeakRead.
                let mut result: Vec<&Access> = Vec::new();
                for (tid, access) in &self.per_thread_read {
                    if *tid != current_thread {
                        result.push(access);
                    }
                }
                for (tid, access) in &self.per_thread_write {
                    if *tid != current_thread {
                        let dominated = self.per_thread_read.get(tid).is_some_and(|r| {
                            r.path_id == access.path_id
                        });
                        if !dominated {
                            result.push(access);
                        }
                    }
                }
                result
            }
            AccessKind::WeakRead => {
                // WeakRead depends only on Write (not Read, WeakWrite, or
                // other WeakRead).
                self.per_thread_write
                    .iter()
                    .filter(|(tid, _)| **tid != current_thread)
                    .map(|(_, access)| access)
                    .collect()
            }
        }
    }

    pub fn record_access(&mut self, access: Access, kind: AccessKind) {
        let thread_id = access.thread_id;
        match kind {
            AccessKind::Read => {
                self.per_thread_read.insert(thread_id, access);
            }
            AccessKind::Write => {
                self.per_thread_write.insert(thread_id, access);
            }
            AccessKind::WeakWrite => {
                self.per_thread_weak_write.insert(thread_id, access);
            }
            AccessKind::WeakRead => {
                self.per_thread_weak_read.insert(thread_id, access);
            }
        }
    }

    /// Like [`record_access`] but keeps the **first** (earliest) access for
    /// each thread rather than overwriting with the latest.  Used for I/O
    /// objects where the earliest position creates the most useful backtrack
    /// target (e.g. between a SELECT and UPDATE in a database transaction).
    pub fn record_io_access(&mut self, access: Access, kind: AccessKind) {
        let thread_id = access.thread_id;
        match kind {
            AccessKind::Read => {
                self.per_thread_read.entry(thread_id).or_insert(access);
            }
            AccessKind::Write => {
                self.per_thread_write.entry(thread_id).or_insert(access);
            }
            AccessKind::WeakWrite => {
                self.per_thread_weak_write.entry(thread_id).or_insert(access);
            }
            AccessKind::WeakRead => {
                self.per_thread_weak_read.entry(thread_id).or_insert(access);
            }
        }
    }
}

impl Default for ObjectState {
    fn default() -> Self {
        Self::new()
    }
}
