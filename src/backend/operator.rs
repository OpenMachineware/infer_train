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

/// Quantization type for weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    IQ4_NL,
    IQ4_XS,
    IQ3_XXS,
    IQ3_S,
}

/// Core operators for transformer inference.
#[derive(Clone, Debug)]
pub enum CoreOperator {
    /// RMS normalization.
    RmsNorm { id: OperatorId, hidden_size: usize, eps: f32 },

    /// Linear layer (matrix multiplication).
    Linear {
        id: OperatorId,
        in_features: usize,
        out_features: usize,
        quant_type: QuantType,
    },

    /// RoPE position encoding.
    Rope { id: OperatorId, head_dim: usize, max_seq_len: usize, freq_base: f32 },

    /// Flash Attention.
    FlashAttention {
        id: OperatorId,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    },

    /// SwiGLU activation (gate * silu(up)).
    SwiGLU { id: OperatorId, intermediate_size: usize },

    /// MoE router (compute expert weights).
    MoERouter { id: OperatorId, n_experts: usize, top_k: usize },

    /// MoE expert computation.
    MoEExpert {
        id: OperatorId,
        expert_id: usize,
        in_features: usize,
        intermediate_size: usize,
        quant_type: QuantType,
    },

    /// Softmax.
    Softmax { id: OperatorId, size: usize },

    /// Add operation (broadcast).
    Add { id: OperatorId, shape: Vec<usize> },

    /// Mul operation (broadcast).
    Mul { id: OperatorId, shape: Vec<usize> },
}

impl CoreOperator {
    /// Get the operator ID.
    pub fn op_id(&self) -> OperatorId {
        match self {
            CoreOperator::RmsNorm { id, .. } => *id,
            CoreOperator::Linear { id, .. } => *id,
            CoreOperator::Rope { id, .. } => *id,
            CoreOperator::FlashAttention { id, .. } => *id,
            CoreOperator::SwiGLU { id, .. } => *id,
            CoreOperator::MoERouter { id, .. } => *id,
            CoreOperator::MoEExpert { id, .. } => *id,
            CoreOperator::Softmax { id, .. } => *id,
            CoreOperator::Add { id, .. } => *id,
            CoreOperator::Mul { id, .. } => *id,
        }
    }
}

impl Operator for CoreOperator {
    fn id(&self) -> OperatorId {
        self.op_id()
    }

    fn name(&self) -> &str {
        match self {
            CoreOperator::RmsNorm { .. } => "RmsNorm",
            CoreOperator::Linear { .. } => "Linear",
            CoreOperator::Rope { .. } => "Rope",
            CoreOperator::FlashAttention { .. } => "FlashAttention",
            CoreOperator::SwiGLU { .. } => "SwiGLU",
            CoreOperator::MoERouter { .. } => "MoERouter",
            CoreOperator::MoEExpert { .. } => "MoEExpert",
            CoreOperator::Softmax { .. } => "Softmax",
            CoreOperator::Add { .. } => "Add",
            CoreOperator::Mul { .. } => "Mul",
        }
    }

    fn input_shapes(&self) -> Vec<Vec<usize>> {
        match self {
            CoreOperator::RmsNorm { hidden_size, .. } => {
                vec![vec![*hidden_size], vec![*hidden_size]]
            }
            CoreOperator::Linear { in_features, out_features, .. } => {
                vec![vec![*in_features], vec![*in_features, *out_features]]
            }
            CoreOperator::Rope { head_dim, .. } => {
                vec![vec![*head_dim]]
            }
            CoreOperator::FlashAttention { head_dim, n_heads, .. } => {
                vec![
                    vec![*n_heads, *head_dim], // Q
                    vec![*n_heads, *head_dim], // K
                    vec![*n_heads, *head_dim], // V
                ]
            }
            CoreOperator::SwiGLU { intermediate_size, .. } => {
                vec![vec![*intermediate_size], vec![*intermediate_size]]
            }
            CoreOperator::MoERouter { n_experts, .. } => {
                vec![vec![*n_experts]]
            }
            CoreOperator::MoEExpert { in_features, .. } => {
                vec![vec![*in_features]]
            }
            CoreOperator::Softmax { size, .. } => {
                vec![vec![*size]]
            }
            CoreOperator::Add { shape, .. }
            | CoreOperator::Mul { shape, .. } => {
                vec![shape.clone(), shape.clone()]
            }
        }
    }

    fn output_shapes(&self) -> Vec<Vec<usize>> {
        match self {
            CoreOperator::RmsNorm { hidden_size, .. } => {
                vec![vec![*hidden_size]]
            }
            CoreOperator::Linear { out_features, .. } => {
                vec![vec![*out_features]]
            }
            CoreOperator::Rope { head_dim, .. } => {
                vec![vec![*head_dim]]
            }
            CoreOperator::FlashAttention { head_dim, n_heads, .. } => {
                vec![vec![*n_heads, *head_dim]]
            }
            CoreOperator::SwiGLU { intermediate_size, .. } => {
                vec![vec![*intermediate_size]]
            }
            CoreOperator::MoERouter { n_experts: _, top_k, .. } => {
                vec![vec![*top_k], vec![*top_k]] // expert indices and weights
            }
            CoreOperator::MoEExpert { in_features, .. } => {
                vec![vec![*in_features]]
            }
            CoreOperator::Softmax { size, .. } => {
                vec![vec![*size]]
            }
            CoreOperator::Add { shape, .. }
            | CoreOperator::Mul { shape, .. } => {
                vec![shape.clone()]
            }
        }
    }
}
