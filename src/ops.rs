// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Operator definitions and type dispatch table

pub mod gemv;
pub mod quant;
pub mod vec_dot;

// Quantization type (ggml_type values)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
}

impl TryFrom<i32> for QuantType {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(QuantType::F32),
            1 => Ok(QuantType::F16),
            2 => Ok(QuantType::Q4_0),
            3 => Ok(QuantType::Q4_1),
            6 => Ok(QuantType::Q5_0),
            7 => Ok(QuantType::Q5_1),
            8 => Ok(QuantType::Q8_0),
            9 => Ok(QuantType::Q8_1),
            10 => Ok(QuantType::Q2_K),
            11 => Ok(QuantType::Q3_K),
            12 => Ok(QuantType::Q4_K),
            13 => Ok(QuantType::Q5_K),
            14 => Ok(QuantType::Q6_K),
            15 => Ok(QuantType::Q8_K),
            30 => Ok(QuantType::BF16),
            _ => Err(format!("Unknown quant type: {}", value)),
        }
    }
}

// Core operators
#[derive(Debug, Clone)]
pub enum CoreOperator {
    /// RMS normalization
    RmsNorm { hidden_size: usize, eps: f32 },

    /// Linear layer (supports quantization)
    Linear { in_features: usize, out_features: usize, quant_type: QuantType },

    /// RoPE positional encoding
    Rope { head_dim: usize, max_seq_len: usize },

    /// Flash Attention
    FlashAttention {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    },

    /// SwiGLU activation
    SwiGLU { hidden_size: usize, intermediate_size: usize },

    /// MoE router
    MoERouter { n_experts: usize, top_k: usize },

    /// MoE expert
    MoEExpert {
        expert_id: usize,
        in_features: usize,
        out_features: usize,
        quant_type: QuantType,
    },
}

// Operator execution error
#[derive(Debug)]
pub enum OpError {
    NotImplemented(String),
    InvalidInput(String),
    DeviceError(String),
}
