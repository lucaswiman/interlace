//! DPOR access tracking.
//!
//! Records shared memory accesses for race detection. Two accesses form a
//! **race** (JACM'17 Def 3.3 p.13-14) when they are from different threads,
//! at least one is a write, they access the same object, and they are not
//! ordered by the happens-before relation.

use crate::vv::VersionVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
    /// Like `Write` but does NOT conflict with other `WeakWrite`s or
    /// `WeakRead`s.
    ///
    /// Used for container-level subscript writes (`STORE_SUBSCR`): these
    /// should conflict with C-method reads (iteration, `len()`) but two
    /// subscript writes on different keys of the same container should
    /// NOT conflict with each other.
    WeakWrite,
    /// Like `Read` but does NOT conflict with `WeakWrite`s or other
    /// `WeakRead`s.
    ///
    /// Used for `LOAD_ATTR` on mutable values: loading a container just
    /// to subscript it shouldn't conflict with subscript writes on
    /// disjoint keys, but should still conflict with C-method writes
    /// (append, clear, etc.).
    WeakRead,
}

impl AccessKind {
    /// Merge two access kinds into a single kind whose conflict set is a
    /// superset of both inputs' conflict sets.  Used by trace caching and
    /// per-branch access recording when the same object is accessed with
    /// different kinds.
    ///
    /// The key improvement over the naive "if different, upgrade to Write"
    /// is that `Read` subsumes `WeakRead`: Read conflicts with {Write,
    /// WeakWrite}, WeakRead conflicts with {Write} — so Read already
    /// covers WeakRead's conflicts.  Upgrading Read+WeakRead to Write
    /// would make the thread appear to *write* the object, causing
    /// spurious sleep-set wakeups when another thread reads it.
    pub fn merge(self, other: Self) -> Self {
        if self == other {
            return self;
        }
        match (self, other) {
            // Read subsumes WeakRead (Read ⊇ WeakRead in conflict sets)
            (AccessKind::Read, AccessKind::WeakRead)
            | (AccessKind::WeakRead, AccessKind::Read) => AccessKind::Read,
            // Write subsumes everything
            _ => AccessKind::Write,
        }
    }
}

/// Records an access to a shared object for DPOR dependency detection.
/// The `path_id` identifies the scheduling point (position in the execution
/// sequence E), used to determine where to insert into wakeup trees.
/// The `dpor_vv` captures the happens-before state at the time of access,
/// used to check for races: e ⋖_E e' when ¬(e →_E e') (JACM'17 p.13-14).
#[derive(Clone, Debug)]
pub struct Access {
    pub path_id: usize,
    pub dpor_vv: VersionVec,
    pub thread_id: usize,
}

impl Access {
    pub fn new(path_id: usize, dpor_vv: VersionVec, thread_id: usize) -> Self {
        Self { path_id, dpor_vv, thread_id }
    }

    /// Check if this access happens-before the given vector clock.
    /// Paper: e →_E e' (Def 3.2, JACM'17 p.12). When this returns false,
    /// the two events are concurrent and form a race (Def 3.3 p.13-14).
    pub fn happens_before(&self, later_vv: &VersionVec) -> bool {
        self.dpor_vv.partial_le(later_vv)
    }
}
