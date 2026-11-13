// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// CPU work buffers for zero-allocation inference

use super::flash_attention::FlashAttnPrefillScratch;
use crate::model::LlamaModel;

/// Pre-allocated work buffers for CPU inference
pub struct CpuWorkBuffers {
    pub hidden: Vec<f32>,
    pub hidden_alt: Vec<f32>,
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub x_q8k: Vec<u8>,
    pub x_padded: Vec<f32>,

    // Additional buffers for zero-allocation
    pub q8k_input: Vec<u8>,
    pub qkv_buffer: Vec<f32>,
    pub q_buffer: Vec<f32>,
    pub k_buffer: Vec<f32>,
    pub v_buffer: Vec<f32>,
    pub rope_buffer: Vec<f32>,
    pub attn_scores: Vec<f32>,

    // KV cache dequantization buffer (reused for each head)
    pub kv_dequant: Vec<f32>,

    // Dequantized norm weights (flat array: each norm is hidden_size elements)
    pub norm_weights: Vec<f32>,
    pub norm_weight: Vec<f32>, // Final output norm

    // Batch prefill buffers (pre-allocated to avoid per-layer allocations)
    pub batch_hidden: Vec<f32>, // Batch hidden states for prefill
    pub batch_residual: Vec<f32>,
    pub qkv_batch_output: Vec<f32>,
    pub q_batch_output: Vec<f32>,
    pub k_batch_output: Vec<f32>,
    pub v_batch_output: Vec<f32>,
    pub attn_batch_output: Vec<f32>,
    pub ffn_gate_batch: Vec<f32>,
    pub ffn_up_batch: Vec<f32>,

    // Flash attention scratch for prefill
    pub flash_attn_prefill: FlashAttnPrefillScratch,

    // Per-thread scratch for multi-threaded flash attention prefill
    // Pre-allocated to avoid per-thread Vec creation during inference
    pub flash_attn_prefill_threads: Vec<FlashAttnPrefillScratch>,

    // Per-thread padding buffers for parallel quantization
    // Pre-allocated to avoid per-thread Vec creation during prefill
    pub x_padded_threads: Vec<Vec<f32>>,

    // SSM/Mamba buffers
    pub ssm_input: Vec<f32>, // SSM input projection output
    pub ssm_conv_out: Vec<f32>, // Conv1D output
    pub ssm_state: Vec<f32>, // SSM state (persistent across tokens)
    pub ssm_y: Vec<f32>,     // SSM output before final projection
    pub ssm_delta: Vec<f32>, // Delta parameter
    pub ssm_b: Vec<f32>,     // B parameter
    pub ssm_c: Vec<f32>,     // C parameter

    // MoE FFN buffers (zero-allocation)
    pub moe_routing_scores: Vec<f32>, // n_experts routing scores
    pub moe_expert_indices: Vec<(usize, f32)>, // n_experts_used top-k experts with weights
    pub moe_routed_output: Vec<f32>,           // hidden_size routed output
    pub moe_final_output: Vec<f32>,            // hidden_size final output
    pub moe_expert_output: Vec<f32>, // intermediate_size expert output

    // MLA (Multi-head Latent Attention) buffers - zero-allocation
    pub mla_kv_a_proj: Vec<f32>, // kv_lora_rank + qk_rope_head_dim
    pub mla_kv_a_normed: Vec<f32>, // kv_lora_rank
    pub mla_kv: Vec<f32>, // n_kv_heads * (qk_nope_head_dim + value_length)

    // Config
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

impl CpuWorkBuffers {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        key_length: usize,
        value_length: usize,
        vocab_size: usize,
        qkv_out_size: usize,
        max_seq_len: usize,
        n_layers: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
    ) -> Self {
        // Use max_seq_len as max batch size for prefill
        let max_batch_size = max_seq_len;
        let q_size = n_heads * head_dim;
        // Use key_length and value_length for K/V buffer sizes (Qwen3.8-style)
        // Fall back to head_dim if key_length/value_length are 0
        let k_size = if key_length > 0 {
            n_kv_heads * key_length
        } else {
            n_kv_heads * head_dim
        };
        let v_size = if value_length > 0 {
            n_kv_heads * value_length
        } else {
            n_kv_heads * head_dim
        };
        let kv_size = k_size.max(v_size);

        // Attention output size: n_heads * v_head_dim
        let v_head_dim = if value_length > 0 { value_length } else { head_dim };
        let attn_out_size = n_heads * v_head_dim;

        Self {
            hidden: vec![0.0; hidden_size],
            hidden_alt: vec![0.0; hidden_size.max(intermediate_size)],
            normed: vec![0.0; hidden_size],
            q: vec![0.0; q_size],
            k: vec![0.0; kv_size],
            v: vec![0.0; kv_size],
            attn_out: vec![0.0; attn_out_size],
            ffn_gate: vec![0.0; intermediate_size],
            ffn_up: vec![0.0; intermediate_size],
            ffn_hidden: vec![0.0; intermediate_size],
            logits: vec![0.0; vocab_size],
            x_q8k: vec![0; 262144],
            x_padded: vec![0.0; 256 * 1024], // 256K elements for large intermediate_size
            // Buffer for batch GEMM quantization
            // Need enough for large intermediate_size models:
            // Qwen3.8-27B: intermediate_size = 3072, but some models have larger
            // 128 tokens * 4864 intermediate / 256 = 2432 blocks
            // 2432 * 292 bytes = 710KB
            // For safety, allocate 20MB to handle various model sizes
            q8k_input: vec![0; 20 * 1024 * 1024],
            qkv_buffer: vec![0.0; qkv_out_size.max(q_size + 2 * kv_size)],
            // For hybrid models, use qkv_out_size for individual buffers
            // to handle layers with different Q/K/V sizes
            q_buffer: vec![0.0; qkv_out_size.max(q_size)],
            k_buffer: vec![0.0; qkv_out_size.max(kv_size)],
            v_buffer: vec![0.0; qkv_out_size.max(kv_size)],
            rope_buffer: vec![0.0; qkv_out_size.max(q_size).max(kv_size)],
            attn_scores: vec![0.0; n_heads * 2048], // n_heads * max_seq_len for parallel execution
            kv_dequant: vec![0.0; head_dim.max(key_length).max(value_length)], // Reusable K/V dequant buffer
            norm_weights: Vec::with_capacity(n_layers * 2 * hidden_size),
            norm_weight: vec![0.0; hidden_size],
            // Batch prefill buffers - allocate for max_batch_size tokens
            batch_hidden: vec![0.0; max_batch_size * hidden_size],
            batch_residual: vec![0.0; max_batch_size * hidden_size],
            qkv_batch_output: vec![
                0.0;
                max_batch_size
                    * qkv_out_size.max(q_size + 2 * kv_size)
            ],
            // For hybrid models, use qkv_out_size for individual batch buffers
            q_batch_output: vec![
                0.0;
                max_batch_size * qkv_out_size.max(q_size)
            ],
            k_batch_output: vec![
                0.0;
                max_batch_size * qkv_out_size.max(kv_size)
            ],
            v_batch_output: vec![
                0.0;
                max_batch_size * qkv_out_size.max(kv_size)
            ],
            attn_batch_output: vec![0.0; max_batch_size * attn_out_size],
            ffn_gate_batch: vec![0.0; max_batch_size * intermediate_size],
            ffn_up_batch: vec![0.0; max_batch_size * intermediate_size],
            // Use max head dimension for scratch buffers to handle variable K/V sizes
            flash_attn_prefill: FlashAttnPrefillScratch::new(
                head_dim.max(key_length).max(value_length),
            ),
            // Pre-allocate per-thread scratch for multithreaded flash attention
            flash_attn_prefill_threads: (0..crate::cpu::default_n_threads())
                .map(|_| {
                    FlashAttnPrefillScratch::new(
                        head_dim.max(key_length).max(value_length),
                    )
                })
                .collect(),
            // Pre-allocate per-thread padding buffers for parallel quantization
            // Each buffer sized to handle padded hidden_size (QK_K aligned)
            x_padded_threads: {
                let padded_size =
                    ((hidden_size.max(intermediate_size) + 255) / 256) * 256;
                (0..crate::cpu::default_n_threads())
                    .map(|_| vec![0.0; padded_size])
                    .collect()
            },
            // SSM/Mamba buffers - allocate for max needed size
            // ssm_inner_size can be up to 4x hidden_size (e.g., Qwen3.6: 8192 for hidden=2048)
            ssm_input: vec![0.0; hidden_size * 4],
            ssm_conv_out: vec![0.0; hidden_size * 4],
            ssm_state: vec![0.0; 128 * 128 * 32], // Gated Delta Net state: head_v_dim^2 * dt_rank = 128*128*32
            ssm_y: vec![0.0; hidden_size * 4],
            ssm_delta: vec![0.0; 128], // time_step_rank (also used for gate in GDN)
            ssm_b: vec![0.0; 128],     // beta in GDN
            ssm_c: vec![0.0; hidden_size * 4], // z values for gated norm (size = d_inner)
            // MoE FFN buffers - use reasonable defaults
            // n_experts typically up to 256, n_experts_used typically 8
            moe_routing_scores: vec![0.0; 256], // n_experts max
            moe_expert_indices: vec![(0, 0.0); 16], // n_experts_used max
            moe_routed_output: vec![0.0; hidden_size],
            moe_final_output: vec![0.0; hidden_size],
            moe_expert_output: vec![
                0.0;
                intermediate_size.max(hidden_size * 2)
            ],
            // MLA buffers - zero-allocation for DeepSeek MLA attention
            // kv_a_proj: output of hidden @ attn_kv_a_mqa = kv_lora_rank + qk_rope_head_dim
            mla_kv_a_proj: vec![0.0; kv_lora_rank + qk_rope_head_dim],
            // kv_a_normed: RMSNorm output of kv_a_proj[:kv_lora_rank]
            mla_kv_a_normed: vec![0.0; kv_lora_rank],
            // kv: output of kv_a_normed @ attn_kv_b
            // size = n_kv_heads * (qk_nope_head_dim + value_length)
            // where qk_nope_head_dim = key_length - qk_rope_head_dim
            mla_kv: vec![
                0.0;
                n_kv_heads
                    * (key_length.saturating_sub(qk_rope_head_dim)
                        + value_length)
            ],
            hidden_size,
            intermediate_size,
            vocab_size,
        }
    }

    pub fn init_norm_weights(&mut self, model: &crate::model::LlamaModel) {
        let hidden_size = self.hidden_size;
        for layer in model.layers.iter() {
            // Attention norm
            if let Some(ref norm) = layer.attention_norm {
                let total_elems = norm.elem_count();
                let data = crate::inference::dequant_to_f32(
                    norm.bytes(),
                    norm.dtype,
                    total_elems,
                );
                self.norm_weights.extend_from_slice(&data);
            } else {
                self.norm_weights.extend_from_slice(&vec![0.0; hidden_size]);
            }
            // FFN norm
            if let Some(ref norm) = layer.ffn_norm {
                let total_elems = norm.elem_count();
                let data = crate::inference::dequant_to_f32(
                    norm.bytes(),
                    norm.dtype,
                    total_elems,
                );
                self.norm_weights.extend_from_slice(&data);
            } else {
                self.norm_weights.extend_from_slice(&vec![0.0; hidden_size]);
            }
        }
        // Final output norm
        if let Some(ref norm) = model.norm {
            let total_elems = norm.elem_count();
            let data = crate::inference::dequant_to_f32(
                norm.bytes(),
                norm.dtype,
                total_elems,
            );
            self.norm_weight = data;
        }
    }
}

/// Pool of CpuWorkBuffers for multi-request inference.
///
/// Pre-allocates buffers to avoid per-request allocation overhead.
pub struct CpuWorkBuffersPool {
    /// Available buffers.
    free: Vec<CpuWorkBuffers>,

    /// Configuration for creating new buffers.
    config: BufferConfig,

    /// Norm weights shared across all buffers (pre-dequantized).
    norm_weights: Vec<f32>,
    norm_weight: Vec<f32>,
}

/// Configuration for creating CpuWorkBuffers.
struct BufferConfig {
    hidden_size: usize,
    intermediate_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    key_length: usize,
    value_length: usize,
    vocab_size: usize,
    qkv_out_size: usize,
    max_seq_len: usize,
    n_layers: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
}

impl CpuWorkBuffersPool {
    /// Create a new pool from model configuration.
    pub fn new(model: &LlamaModel) -> Self {
        let config = &model.config;

        let qkv_out_size = model
            .layers
            .first()
            .and_then(|l| l.attn_qkv.as_ref())
            .map(|t| t.shape[1])
            .unwrap_or(0);

        let buffer_config = BufferConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            key_length: config.key_length,
            value_length: config.value_length,
            vocab_size: config.vocab_size,
            qkv_out_size,
            max_seq_len: 2048,
            n_layers: model.layers.len(),
            kv_lora_rank: config.kv_lora_rank,
            qk_rope_head_dim: config.qk_rope_head_dim,
        };

        // Pre-dequantize norm weights once
        let mut norm_weights = Vec::with_capacity(
            buffer_config.n_layers * 2 * buffer_config.hidden_size,
        );
        let mut norm_weight = vec![0.0; buffer_config.hidden_size];

        let hidden_size = buffer_config.hidden_size;
        for layer in model.layers.iter() {
            // Attention norm
            if let Some(ref norm) = layer.attention_norm {
                let data = crate::inference::dequant_to_f32(
                    norm.bytes(),
                    norm.dtype,
                    norm.elem_count(),
                );
                norm_weights.extend_from_slice(&data);
            } else {
                norm_weights.extend_from_slice(&vec![0.0; hidden_size]);
            }
            // FFN norm
            if let Some(ref norm) = layer.ffn_norm {
                let data = crate::inference::dequant_to_f32(
                    norm.bytes(),
                    norm.dtype,
                    norm.elem_count(),
                );
                norm_weights.extend_from_slice(&data);
            } else {
                norm_weights.extend_from_slice(&vec![0.0; hidden_size]);
            }
        }
        // Final output norm
        if let Some(ref norm) = model.norm {
            let data = crate::inference::dequant_to_f32(
                norm.bytes(),
                norm.dtype,
                norm.elem_count(),
            );
            norm_weight = data;
        }

        Self {
            free: Vec::new(),
            config: buffer_config,
            norm_weights,
            norm_weight,
        }
    }

    /// Pre-allocate a number of buffers.
    pub fn preallocate(&mut self, count: usize) {
        for _ in 0..count {
            let mut buffers = CpuWorkBuffers::new(
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
                self.config.key_length,
                self.config.value_length,
                self.config.vocab_size,
                self.config.qkv_out_size,
                self.config.max_seq_len,
                self.config.n_layers,
                self.config.kv_lora_rank,
                self.config.qk_rope_head_dim,
            );
            // Share pre-dequantized norm weights
            buffers.norm_weights = self.norm_weights.clone();
            buffers.norm_weight = self.norm_weight.clone();
            self.free.push(buffers);
        }
    }

    /// Allocate a buffer from the pool.
    pub fn allocate(&mut self) -> Option<CpuWorkBuffers> {
        if let Some(mut buffers) = self.free.pop() {
            // Reset buffers to zero (keep norm weights)
            buffers.hidden.fill(0.0);
            buffers.hidden_alt.fill(0.0);
            buffers.normed.fill(0.0);
            buffers.q.fill(0.0);
            buffers.k.fill(0.0);
            buffers.v.fill(0.0);
            buffers.attn_out.fill(0.0);
            buffers.ffn_gate.fill(0.0);
            buffers.ffn_up.fill(0.0);
            buffers.ffn_hidden.fill(0.0);
            buffers.logits.fill(0.0);
            buffers.qkv_buffer.fill(0.0);
            buffers.q_buffer.fill(0.0);
            buffers.k_buffer.fill(0.0);
            buffers.v_buffer.fill(0.0);
            buffers.rope_buffer.fill(0.0);
            buffers.attn_scores.fill(0.0);
            Some(buffers)
        } else {
            // Pool exhausted, allocate new buffer
            let mut buffers = CpuWorkBuffers::new(
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
                self.config.key_length,
                self.config.value_length,
                self.config.vocab_size,
                self.config.qkv_out_size,
                self.config.max_seq_len,
                self.config.n_layers,
                self.config.kv_lora_rank,
                self.config.qk_rope_head_dim,
            );
            buffers.norm_weights = self.norm_weights.clone();
            buffers.norm_weight = self.norm_weight.clone();
            Some(buffers)
        }
    }

    /// Release a buffer back to the pool.
    pub fn release(&mut self, buffers: CpuWorkBuffers) {
        self.free.push(buffers);
    }

    /// Get number of available buffers.
    pub fn available(&self) -> usize {
        self.free.len()
    }
}
