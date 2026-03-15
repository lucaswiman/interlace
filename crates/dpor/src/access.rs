//! DPOR access tracking.

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

/// Records an access to a shared object for DPOR dependency detection.
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

    pub fn happens_before(&self, later_vv: &VersionVec) -> bool {
        self.dpor_vv.partial_le(later_vv)
    }
}
