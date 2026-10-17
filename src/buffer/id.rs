// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

/// Unique identifier for a buffer
///
/// Used to track buffer location and lifecycle across the three-layer architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId {
    /// Unique identifier (typically derived from tensor pointer or layer/weight index)
    pub id: u64,
}

impl BufferId {
    pub fn new(id: u64) -> Self {
        Self { id }
    }

    /// Create from a raw pointer (common for tensor weights)
    pub fn from_ptr<T>(ptr: *const T) -> Self {
        Self { id: ptr as u64 }
    }
}
