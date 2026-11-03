// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// Request-level scratch buffers for zero-allocation execution

/// Request-level scratch buffer
///
/// Pre-allocated buffers for a single request's forward pass.
/// Eliminates allocation overhead by reusing buffers across forward passes.
pub struct RequestScratch {
    /// Hidden state buffer
    pub hidden: Vec<f32>,
    /// Alternative hidden buffer (for norm output, residual computation)
    pub hidden_alt: Vec<f32>,

    /// QKV projections
    pub q_buffer: Vec<f32>,
    pub k_buffer: Vec<f32>,
    pub v_buffer: Vec<f32>,

    /// Attention output
    pub attn_out: Vec<f32>,
    pub attn_scores: Vec<f32>,

    /// FFN buffers
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,

    /// RoPE temp buffer
    pub rope_buffer: Vec<f32>,

    /// Quantization buffers
    pub q8k_input: Vec<u8>,
    pub x_padded: Vec<f32>,

    /// KV dequantization buffer
    pub kv_dequant: Vec<f32>,

    /// Logits output
    pub logits: Vec<f32>,

    /// Configuration
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

impl RequestScratch {
    /// Create a new scratch buffer for the given model dimensions
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_length: usize,
        value_length: usize,
        max_seq_len: usize,
    ) -> Self {
        // Calculate buffer sizes
        let q_size = n_heads * head_dim;
        let k_size = n_kv_heads * key_length.max(head_dim);
        let v_size = n_kv_heads * value_length.max(head_dim);
        let attn_out_size = n_heads * value_length.max(head_dim);

        // Quantization buffer size (max of hidden_size and intermediate_size)
        use crate::quant::types::QK_K;
        let max_dim = hidden_size.max(intermediate_size);
        let max_dim_padded = ((max_dim + QK_K - 1) / QK_K) * QK_K;
        let nb = max_dim_padded / QK_K;
        let q8k_size =
            nb * std::mem::size_of::<crate::quant::types::BlockQ8K>();

        Self {
            hidden: vec![0.0f32; hidden_size],
            hidden_alt: vec![0.0f32; hidden_size],
            q_buffer: vec![0.0f32; q_size],
            k_buffer: vec![0.0f32; k_size],
            v_buffer: vec![0.0f32; v_size],
            attn_out: vec![0.0f32; attn_out_size],
            attn_scores: vec![0.0f32; max_seq_len],
            ffn_gate: vec![0.0f32; intermediate_size],
            ffn_up: vec![0.0f32; intermediate_size],
            rope_buffer: vec![0.0f32; q_size.max(k_size)],
            q8k_input: vec![0u8; q8k_size],
            x_padded: vec![0.0f32; max_dim_padded],
            kv_dequant: vec![0.0f32; max_seq_len * head_dim],
            logits: vec![0.0f32; vocab_size],
            hidden_size,
            intermediate_size,
            vocab_size,
        }
    }

    /// Get the byte size of all buffers
    pub fn total_size(&self) -> usize {
        let f32_size = std::mem::size_of::<f32>();
        let u8_size = std::mem::size_of::<u8>();

        let f32_buffers = self.hidden.len()
            + self.hidden_alt.len()
            + self.q_buffer.len()
            + self.k_buffer.len()
            + self.v_buffer.len()
            + self.attn_out.len()
            + self.attn_scores.len()
            + self.ffn_gate.len()
            + self.ffn_up.len()
            + self.rope_buffer.len()
            + self.x_padded.len()
            + self.kv_dequant.len()
            + self.logits.len();

        f32_buffers * f32_size + self.q8k_input.len() * u8_size
    }
}

/// Pool of request scratch buffers
///
/// Manages a pool of pre-allocated scratch buffers for concurrent requests.
/// Buffers are allocated once and reused across requests.
pub struct RequestScratchPool {
    /// Available buffers
    free: Vec<RequestScratch>,

    /// Configuration for creating new buffers
    config: ScratchConfig,
}

/// Configuration for creating scratch buffers
struct ScratchConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    key_length: usize,
    value_length: usize,
    max_seq_len: usize,
}

impl RequestScratchPool {
    /// Create a new pool from model configuration
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_length: usize,
        value_length: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            free: Vec::new(),
            config: ScratchConfig {
                hidden_size,
                intermediate_size,
                vocab_size,
                n_heads,
                n_kv_heads,
                head_dim,
                key_length,
                value_length,
                max_seq_len,
            },
        }
    }

    /// Allocate a scratch buffer
    pub fn allocate(&mut self) -> RequestScratch {
        self.free.pop().unwrap_or_else(|| {
            RequestScratch::new(
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.vocab_size,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
                self.config.key_length,
                self.config.value_length,
                self.config.max_seq_len,
            )
        })
    }

    /// Release a scratch buffer back to the pool
    pub fn release(&mut self, scratch: RequestScratch) {
        self.free.push(scratch);
    }

    /// Get the number of available buffers
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Pre-allocate buffers
    pub fn preallocate(&mut self, count: usize) {
        for _ in 0..count {
            let scratch = RequestScratch::new(
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.vocab_size,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
                self.config.key_length,
                self.config.value_length,
                self.config.max_seq_len,
            );
            self.free.push(scratch);
        }
    }
}
