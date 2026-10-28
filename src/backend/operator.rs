// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Operator abstraction for compute operations.

use std::fmt;

/// Unique operator identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperatorId(pub usize);

impl fmt::Display for OperatorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Operator({})", self.0)
    }
}

/// Result of operator execution.
pub type OperatorResult<T> = Result<T, OperatorError>;

/// Execution cost estimate.
#[derive(Clone, Copy, Debug)]
pub struct Cost {
    /// Estimated compute time in microseconds.
    pub compute_us: f64,
    /// Estimated memory bandwidth in bytes.
    pub memory_bytes: u64,
    /// Memory reads in bytes.
    pub read_bytes: u64,
    /// Memory writes in bytes.
    pub write_bytes: u64,
}

impl Cost {
    /// Create a zero cost.
    pub fn zero() -> Self {
        Self { compute_us: 0.0, memory_bytes: 0, read_bytes: 0, write_bytes: 0 }
    }

    /// Estimate execution time given memory bandwidth (GB/s).
    pub fn estimate_time(&self, bandwidth_gbps: f64) -> f64 {
        if bandwidth_gbps == 0.0 {
            return self.compute_us;
        }

        // Memory-bound time
        let memory_time =
            (self.memory_bytes as f64) / (bandwidth_gbps * 1e9) * 1e6;

        // Return max of compute and memory time
        self.compute_us.max(memory_time)
    }
}

/// Error from operator execution.
#[derive(Clone, Debug, thiserror::Error)]
pub enum OperatorError {
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidInputShape { expected: String, actual: String },

    #[error("Invalid output shape: expected {expected}, got {actual}")]
    InvalidOutputShape { expected: String, actual: String },

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Operator trait for compute operations.
pub trait Operator: Send + Sync + fmt::Debug {
    /// Get the operator ID.
    fn id(&self) -> OperatorId;

    /// Get the operator name.
    fn name(&self) -> &str;

    /// Get the input shapes.
    fn input_shapes(&self) -> Vec<Vec<usize>>;

    /// Get the output shapes.
    fn output_shapes(&self) -> Vec<Vec<usize>>;
}
