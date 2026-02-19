//! Shared object tracking for DPOR.

use crate::access::{Access, AccessKind};

/// Opaque integer ID for shared objects.
pub type ObjectId = u64;

/// Tracks the last accesses to a shared object for DPOR.
#[derive(Clone, Debug)]
pub struct ObjectState {
    pub last_access: Option<Access>,
    pub last_write_access: Option<Access>,
}

impl ObjectState {
    pub fn new() -> Self {
        Self { last_access: None, last_write_access: None }
    }

    /// Returns the last dependent access for the given kind.
    /// - Read depends on last Write (reads are independent).
    /// - Write depends on any prior access.
    pub fn last_dependent_access(&self, kind: AccessKind) -> Option<&Access> {
        match kind {
            AccessKind::Read => self.last_write_access.as_ref(),
            AccessKind::Write => self.last_access.as_ref(),
        }
    }

    pub fn record_access(&mut self, access: Access, kind: AccessKind) {
        if kind == AccessKind::Write {
            self.last_write_access = Some(access.clone());
        }
        self.last_access = Some(access);
    }
}

impl Default for ObjectState {
    fn default() -> Self {
        Self::new()
    }
}
