// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use crate::model::tensor::DataType;

/// Operation types for the executor
#[derive(Debug, Clone)]
pub enum Op {
    /// Matrix-vector multiplication (GEMV)
    /// (output_size, input_size, weight_dtype)
    Gemv { output_size: usize, input_size: usize, dtype: DataType },

    /// Transposed matrix-vector multiplication
    /// For output layer with tied embeddings
    GemvTransposed { output_size: usize, input_size: usize, dtype: DataType },

    /// RMS normalization
    RmsNorm { size: usize, eps: f32 },

    /// Softmax
    Softmax { size: usize },

    /// Rotary Position Embedding
    Rope { seq_len: usize, head_dim: usize, n_heads: usize },

    /// Flash Attention
    FlashAttention {
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },

    /// FFN with SwiGLU activation
    FfnSwiGLu { hidden_size: usize, intermediate_size: usize },

    /// Add operation
    Add { size: usize },

    /// Mul operation
    Mul { size: usize },
}

impl Op {
    /// Check if this operation is supported on GPU
    pub fn is_gpu_supported(&self) -> bool {
        match self {
            Op::Gemv { dtype, .. } | Op::GemvTransposed { dtype, .. } => {
                // Supported quantization formats on GPU
                matches!(
                    dtype,
                    DataType::Q4_0
                        | DataType::Q4_1
                        | DataType::Q5_0
                        | DataType::Q5_1
                        | DataType::Q8_0
                        | DataType::Q4_K
                        | DataType::Q5_K
                        | DataType::Q2_K
                        | DataType::Q6_K
                        | DataType::Q3_K
                )
            }
            Op::RmsNorm { .. } => true,
            Op::Softmax { .. } => true,
            Op::Rope { .. } => true,
            Op::FlashAttention { .. } => true,
            Op::FfnSwiGLu { .. } => true,
            Op::Add { .. } => true,
            Op::Mul { .. } => true,
        }
    }

    /// Get estimated compute intensity (FLOPs)
    pub fn estimated_flops(&self) -> u64 {
        match self {
            Op::Gemv { output_size, input_size, .. } => {
                2 * (*output_size as u64) * (*input_size as u64)
            }
            Op::GemvTransposed { output_size, input_size, .. } => {
                2 * (*output_size as u64) * (*input_size as u64)
            }
            Op::RmsNorm { size, .. } => 4 * (*size as u64),
            Op::Softmax { size } => 4 * (*size as u64),
            Op::Rope { seq_len, n_heads, head_dim, .. } => {
                (*seq_len as u64) * (*n_heads as u64) * (*head_dim as u64) * 10
            }
            Op::FlashAttention { seq_len, n_heads, n_kv_heads, head_dim } => {
                // Simplified: 4 * seq_len^2 * n_heads * head_dim
                let s = *seq_len as u64;
                (*n_kv_heads as u64) * (*head_dim as u64) * 4 * s * s
            }
            Op::FfnSwiGLu { hidden_size, intermediate_size } => {
                3 * (*hidden_size as u64) * (*intermediate_size as u64)
            }
            Op::Add { size } => *size as u64,
            Op::Mul { size } => *size as u64,
        }
    }
}
