// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Operation types for compute graph

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    // Element-wise
    Add,
    Mul,
    Div,
    Scale,

    // Normalization
    RMSNorm,
    LayerNorm,

    // Activation
    Silu,
    Gelu,
    Relu,

    // Attention
    Softmax,
    Rope,
    FlashAttn,

    // Matrix operations
    MatMul,
    MatVec,

    // Reshaping
    Reshape,
    Transpose,
    Contiguous,

    // Copy/Movement
    Copy,

    // MoE
    MulMatId,
    GetRows,
}

impl OpType {
    /// Get the number of inputs for this operation
    pub fn input_count(&self) -> usize {
        match self {
            OpType::Add | OpType::Mul | OpType::Div => 2,
            OpType::Scale => 2,
            OpType::RMSNorm => 2, // input, weight
            OpType::LayerNorm => 2,
            OpType::Silu | OpType::Gelu | OpType::Relu => 1,
            OpType::Softmax => 1,
            OpType::Rope => 2,      // input, freqs
            OpType::FlashAttn => 4, // q, k, v, mask (optional: sinks, kv_max)
            OpType::MatMul => 2,
            OpType::MatVec => 2,
            OpType::Reshape => 1,
            OpType::Transpose => 1,
            OpType::Contiguous => 1,
            OpType::Copy => 1,
            OpType::MulMatId => 3, // weights, ids, experts
            OpType::GetRows => 2,  // src, indices
        }
    }

    /// Check if this operation can be fused with others
    pub fn is_fusable(&self) -> bool {
        matches!(
            self,
            OpType::Add
                | OpType::Mul
                | OpType::Scale
                | OpType::Silu
                | OpType::Gelu
                | OpType::Relu
        )
    }
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::Add => write!(f, "Add"),
            OpType::Mul => write!(f, "Mul"),
            OpType::Div => write!(f, "Div"),
            OpType::Scale => write!(f, "Scale"),
            OpType::RMSNorm => write!(f, "RMSNorm"),
            OpType::LayerNorm => write!(f, "LayerNorm"),
            OpType::Silu => write!(f, "Silu"),
            OpType::Gelu => write!(f, "Gelu"),
            OpType::Relu => write!(f, "Relu"),
            OpType::Softmax => write!(f, "Softmax"),
            OpType::Rope => write!(f, "Rope"),
            OpType::FlashAttn => write!(f, "FlashAttn"),
            OpType::MatMul => write!(f, "MatMul"),
            OpType::MatVec => write!(f, "MatVec"),
            OpType::Reshape => write!(f, "Reshape"),
            OpType::Transpose => write!(f, "Transpose"),
            OpType::Contiguous => write!(f, "Contiguous"),
            OpType::Copy => write!(f, "Copy"),
            OpType::MulMatId => write!(f, "MulMatId"),
            OpType::GetRows => write!(f, "GetRows"),
        }
    }
}

/// Additional parameters for operations
#[derive(Debug, Clone, Default)]
pub struct OpParams {
    /// For RMSNorm: epsilon
    pub eps: Option<f32>,
    /// For Scale: scale factor
    pub scale: Option<f32>,
    /// For Rope: position
    pub pos: Option<usize>,
    /// For Rope: frequency base and scale
    pub freq_base: Option<f32>,
    pub freq_scale: Option<f32>,
    /// For FlashAttn: scale, max_bias, softcap
    pub attn_scale: Option<f32>,
    pub max_bias: Option<f32>,
    pub logit_softcap: Option<f32>,
    /// For FlashAttn: attention dimensions
    pub n_heads: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub seq_len: Option<usize>,
    /// For Reshape: new shape
    pub new_shape: Option<Vec<usize>>,
    /// For Softmax: temperature
    pub temperature: Option<f32>,
}

/// A node in the compute graph
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Operation type
    pub op_type: OpType,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor name
    pub output: String,
    /// Additional parameters
    pub params: OpParams,
}

impl OpNode {
    pub fn new(op_type: OpType, inputs: Vec<String>, output: String) -> Self {
        Self { op_type, inputs, output, params: OpParams::default() }
    }

    pub fn with_params(mut self, params: OpParams) -> Self {
        self.params = params;
        self
    }
}
