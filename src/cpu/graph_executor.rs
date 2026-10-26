// SPDX-License-Identifier: AGPL-3.0-or-later
// CPU graph executor matching llama.cpp's execution model

use crate::inference::KVCache;
use crate::model::LlamaModel;

/// Pre-allocated work buffers for CPU execution
pub struct CpuWorkBuffers {
    /// Q8_K quantized input buffer
    pub q8k_input: Vec<u8>,

    /// Hidden state buffer (F32)
    pub hidden_buffer: Vec<f32>,

    /// Intermediate computation buffer
    pub temp_buffer: Vec<f32>,

    /// Attention output buffer
    pub attn_output: Vec<f32>,

    /// FFN intermediate buffer
    pub ffn_buffer: Vec<f32>,
}

impl CpuWorkBuffers {
    /// Create work buffers with specified sizes
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self {
        let qk_k = 256; // Block size for K-quant
        let nb = (hidden_size + qk_k - 1) / qk_k;

        Self {
            // Q8_K block is 264 bytes: 256 bytes for qs + 8 bytes for scales/d
            q8k_input: vec![0u8; nb * 264],
            hidden_buffer: vec![0.0f32; hidden_size],
            temp_buffer: vec![0.0f32; hidden_size.max(intermediate_size)],
            attn_output: vec![0.0f32; n_heads * head_dim],
            ffn_buffer: vec![0.0f32; intermediate_size],
        }
    }
}

/// Execute the full forward pass using the graph executor
pub fn forward_cpu(
    model: &LlamaModel,
    tokens: &[u32],
    kv_cache: &mut KVCache,
) -> Option<Vec<f32>> {
    // Delegate to the model's forward pass for now
    // The thread pool is available via parallel_for for GEMV operations
    model.forward(tokens, kv_cache)
}
