// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Forward executor for CPU inference

use crate::backend::arm::{
    activation::silu_f32_inplace, flash_attn::flash_attn_f32, linear, quant,
    rms_norm, rope::rope_f32_inplace,
};
use crate::inference::{
    LayerDebugState, QuantFormat, QuantizedInput, QuantizedInputBatch,
    SimpleModel, quantize_input_batch, quantize_input_to,
};
use crate::types::*;

// Timing macro for profiling (compile with --features profile)
#[cfg(feature = "profile")]
macro_rules! time_op {
    ($name:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        log::debug!(
            "[PROFILE] {}: {:.2} ms",
            $name,
            start.elapsed().as_secs_f64() * 1000.0
        );
        result
    }};
}

#[cfg(not(feature = "profile"))]
macro_rules! time_op {
    ($name:expr, $block:block) => {
        $block
    };
}

/// Pre-computed layer weight names (avoids format! in hot path)
#[derive(Clone)]
struct LayerWeightNames {
    attn_norm: String,
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_q_bias: String,
    attn_k_bias: String,
    attn_v_bias: String,
    attn_q_norm: String,
    attn_k_norm: String,
    attn_output: String,
    ffn_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

impl LayerWeightNames {
    fn new(layer_idx: usize) -> Self {
        Self {
            attn_norm: format!("blk.{}.attn_norm.weight", layer_idx),
            attn_q: format!("blk.{}.attn_q.weight", layer_idx),
            attn_k: format!("blk.{}.attn_k.weight", layer_idx),
            attn_v: format!("blk.{}.attn_v.weight", layer_idx),
            attn_q_bias: format!("blk.{}.attn_q.bias", layer_idx),
            attn_k_bias: format!("blk.{}.attn_k.bias", layer_idx),
            attn_v_bias: format!("blk.{}.attn_v.bias", layer_idx),
            attn_q_norm: format!("blk.{}.attn_q_norm.weight", layer_idx),
            attn_k_norm: format!("blk.{}.attn_k_norm.weight", layer_idx),
            attn_output: format!("blk.{}.attn_output.weight", layer_idx),
            ffn_norm: format!("blk.{}.ffn_norm.weight", layer_idx),
            ffn_gate: format!("blk.{}.ffn_gate.weight", layer_idx),
            ffn_up: format!("blk.{}.ffn_up.weight", layer_idx),
            ffn_down: format!("blk.{}.ffn_down.weight", layer_idx),
        }
    }
}

/// Forward executor for single-token generation
pub struct ForwardExecutor {
    pub model: SimpleModel,
    /// KV cache: [layer][seq_pos][head][dim] for key and value
    pub kv_cache_k: Vec<Vec<f32>>,
    pub kv_cache_v: Vec<Vec<f32>>,
    kv_capacity: usize,
    pub seq_pos: usize,
    quant_buf: QuantizedInput,
    /// Batch quantized input buffer for prefill
    quant_buf_batch: QuantizedInputBatch,
    arena: crate::memory::ComputeArena,
    hidden_buf: Vec<f32>,
    /// Batch hidden buffer for prefill: [n_tokens * hidden_size]
    hidden_buf_batch: Vec<f32>,
    pub logits_buf: Vec<f32>,
    pub n_threads: usize,
    pub n_threads_batch: usize,
    /// Pre-computed weight names for each layer
    layer_weights: Vec<LayerWeightNames>,
    /// Pre-allocated batch buffers (avoid arena allocation during forward_layer_batch)
    batch_q: Vec<f32>,
    batch_k: Vec<f32>,
    batch_v: Vec<f32>,
    batch_q_normed: Vec<f32>,
    batch_k_normed: Vec<f32>,
    batch_attn_output: Vec<f32>,
    batch_gate: Vec<f32>,
    batch_up: Vec<f32>,
    /// Debug mode: output intermediate states
    pub debug_mode: bool,
    /// Debug output for each layer
    pub debug_states: Vec<LayerDebugState>,
}

impl ForwardExecutor {
    pub fn new(model: SimpleModel) -> Self {
        Self::new_with_threads(model, 0, 0) // Use optimal_threads() for decode
    }

    pub fn new_with_threads(
        model: SimpleModel,
        n_threads: usize,
        n_threads_batch: usize,
    ) -> Self {
        let config = model.config().clone();

        // Use optimal_threads() if n_threads is 0
        let n_threads = if n_threads == 0 {
            crate::backend::thread_pool::optimal_threads()
        } else {
            n_threads
        };
        let n_threads_batch = if n_threads_batch == 0 {
            crate::backend::thread_pool::optimal_threads()
        } else {
            n_threads_batch
        };

        let work_size = crate::memory::calculate_work_size(
            config.hidden_size,
            config.intermediate_size,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim(),
            config.vocab_size,
            config.n_layers,
        );

        // Start with small capacity, expand on demand
        let initial_capacity = 512.min(config.context_length);
        let kv_cache_k =
            vec![
                vec![
                    0.0f32;
                    config.n_kv_heads * initial_capacity * config.head_dim()
                ];
                config.n_layers
            ];
        let kv_cache_v =
            vec![
                vec![
                    0.0f32;
                    config.n_kv_heads * initial_capacity * config.head_dim()
                ];
                config.n_layers
            ];

        // Pre-compute layer weight names
        let layer_weights: Vec<LayerWeightNames> =
            (0..config.n_layers).map(LayerWeightNames::new).collect();

        // Pre-allocate batch buffers for max batch size
        let max_batch = 64;
        let batch_q =
            vec![0.0f32; max_batch * config.n_heads * config.head_dim()];
        let batch_k =
            vec![0.0f32; max_batch * config.n_kv_heads * config.head_dim()];
        let batch_v =
            vec![0.0f32; max_batch * config.n_kv_heads * config.head_dim()];
        let batch_q_normed =
            vec![0.0f32; max_batch * config.n_heads * config.head_dim()];
        let batch_k_normed =
            vec![0.0f32; max_batch * config.n_kv_heads * config.head_dim()];
        let batch_attn_output =
            vec![0.0f32; max_batch * config.n_heads * config.head_dim()];
        let batch_gate = vec![0.0f32; max_batch * config.intermediate_size];
        let batch_up = vec![0.0f32; max_batch * config.intermediate_size];

        Self {
            model,
            kv_cache_k,
            kv_cache_v,
            kv_capacity: initial_capacity,
            seq_pos: 0,
            quant_buf: QuantizedInput::new(),
            quant_buf_batch: QuantizedInputBatch::new(),
            arena: crate::memory::ComputeArena::new(work_size),
            hidden_buf: vec![0.0f32; config.hidden_size],
            hidden_buf_batch: Vec::new(),
            logits_buf: vec![0.0f32; config.vocab_size],
            n_threads,
            n_threads_batch,
            layer_weights,
            batch_q,
            batch_k,
            batch_v,
            batch_q_normed,
            batch_k_normed,
            batch_attn_output,
            batch_gate,
            batch_up,
            debug_mode: false,
            debug_states: Vec::new(),
        }
    }

    fn ensure_kv_capacity(&mut self) {
        let config = self.model.config();
        if self.seq_pos >= self.kv_capacity {
            let new_capacity =
                (self.kv_capacity * 2).min(config.context_length);
            for layer in 0..config.n_layers {
                self.kv_cache_k[layer].resize(
                    config.n_kv_heads * new_capacity * config.head_dim(),
                    0.0,
                );
                self.kv_cache_v[layer].resize(
                    config.n_kv_heads * new_capacity * config.head_dim(),
                    0.0,
                );
            }
            self.kv_capacity = new_capacity;
        }
    }

    pub fn reset(&mut self) {
        self.seq_pos = 0;
        for layer in &mut self.kv_cache_k {
            layer.fill(0.0);
        }
        for layer in &mut self.kv_cache_v {
            layer.fill(0.0);
        }
    }

    /// Forward pass returning reference to logits buffer
    pub fn forward(&mut self, token_id: u32) -> Result<&[f32], String> {
        self.arena.reset();
        self.ensure_kv_capacity();

        let n_layers = self.model.config().n_layers;

        // 1. Embedding lookup
        self.embed_token(token_id)?;

        // 2. Process each layer
        for layer_idx in 0..n_layers {
            self.forward_layer(layer_idx)?;
        }

        // 3. Final norm
        self.final_norm()?;

        // 4. Output projection
        self.output_projection()?;

        self.seq_pos += 1;
        Ok(&self.logits_buf)
    }

    /// Batch forward pass for prefill (multiple tokens at once)
    /// Uses tiled GEMV for better performance when n_tokens >= 8
    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
    ) -> Result<&[f32], String> {
        let n_tokens = token_ids.len();
        if n_tokens == 0 {
            return Err("Empty token batch".to_string());
        }
        if n_tokens == 1 {
            // Fall back to single token path for efficiency
            self.forward(token_ids[0])?;
            return Ok(&self.logits_buf);
        }

        self.arena.reset();
        self.ensure_kv_capacity_batch(n_tokens);

        let config = self.model.config();
        let hidden_size = config.hidden_size;
        let n_layers = config.n_layers;

        // 1. Batch embedding lookup
        self.embed_tokens_batch(token_ids)?;

        // 2. Process each layer with batch operations
        for layer_idx in 0..n_layers {
            self.forward_layer_batch(layer_idx, n_tokens)?;
        }

        // 3. Final norm (only on last token)
        self.final_norm_batch(n_tokens)?;

        // 4. Output projection (only on last token)
        self.output_projection_batch()?;

        // Copy last token's hidden state to hidden_buf for subsequent forward calls
        let last_token_offset = (n_tokens - 1) * hidden_size;
        self.hidden_buf[..hidden_size].copy_from_slice(
            &self.hidden_buf_batch
                [last_token_offset..last_token_offset + hidden_size],
        );

        self.seq_pos += n_tokens;
        Ok(&self.logits_buf)
    }

    /// Ensure KV cache has capacity for batch
    fn ensure_kv_capacity_batch(&mut self, n_tokens: usize) {
        let config = self.model.config();
        if self.seq_pos + n_tokens > self.kv_capacity {
            let new_capacity = (self.kv_capacity * 2)
                .max(self.seq_pos + n_tokens)
                .min(config.context_length);
            for layer in 0..config.n_layers {
                self.kv_cache_k[layer].resize(
                    config.n_kv_heads * new_capacity * config.head_dim(),
                    0.0,
                );
                self.kv_cache_v[layer].resize(
                    config.n_kv_heads * new_capacity * config.head_dim(),
                    0.0,
                );
            }
            self.kv_capacity = new_capacity;
        }
    }

    /// Batch embedding lookup: fill hidden_buf_batch with [n_tokens * hidden_size]
    fn embed_tokens_batch(&mut self, token_ids: &[u32]) -> Result<(), String> {
        let n_tokens = token_ids.len();
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        // Resize batch buffer
        if self.hidden_buf_batch.len() != n_tokens * hidden_size {
            self.hidden_buf_batch.resize(n_tokens * hidden_size, 0.0);
        }

        let emb_tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or("Embedding tensor not found")?;

        let emb_data =
            self.model.get_embeddings().ok_or("No embedding weights")?;

        // Process each token's embedding
        for (token_idx, &token_id) in token_ids.iter().enumerate() {
            let out_offset = token_idx * hidden_size;
            match emb_tensor.quant_type {
                QuantType::F32 => {
                    let nb1 = emb_tensor.strides[1] as usize;
                    let col_offset = (token_id as usize) * nb1;
                    for h in 0..hidden_size {
                        self.hidden_buf_batch[out_offset + h] =
                            f32::from_le_bytes(
                                emb_data[col_offset + h * 4
                                    ..col_offset + h * 4 + 4]
                                    .try_into()
                                    .unwrap(),
                            );
                    }
                }
                QuantType::F16 => {
                    let nb1 = emb_tensor.strides[1] as usize;
                    let col_offset = (token_id as usize) * nb1;
                    for h in 0..hidden_size {
                        self.hidden_buf_batch[out_offset + h] =
                            half::f16::from_le_bytes(
                                emb_data[col_offset + h * 2
                                    ..col_offset + h * 2 + 2]
                                    .try_into()
                                    .unwrap(),
                            )
                            .to_f32();
                    }
                }
                QuantType::Q6_K => {
                    let nb = hidden_size / 256;
                    let block_size = std::mem::size_of::<BlockQ6_K>();
                    let nb1 = emb_tensor.strides[1] as usize;
                    for block_idx in 0..nb {
                        let h_base = out_offset + block_idx * 256;
                        let col_offset =
                            (token_id as usize) * nb1 + block_idx * block_size;
                        let block: &BlockQ6_K = unsafe {
                            &*(emb_data.as_ptr().add(col_offset)
                                as *const BlockQ6_K)
                        };

                        let block_vals =
                            crate::backend::arm::dequant::dequantize_q6_k_block(
                                block,
                            );
                        self.hidden_buf_batch[h_base..h_base + 256]
                            .copy_from_slice(&block_vals);
                    }
                }
                QuantType::Q5_K => {
                    let nb = hidden_size / 256;
                    let block_size = std::mem::size_of::<BlockQ5_K>();
                    let nb1 = emb_tensor.strides[1] as usize;
                    for block_idx in 0..nb {
                        let h_base = out_offset + block_idx * 256;
                        let col_offset =
                            (token_id as usize) * nb1 + block_idx * block_size;
                        let block: &BlockQ5_K = unsafe {
                            &*(emb_data.as_ptr().add(col_offset)
                                as *const BlockQ5_K)
                        };
                        let block_vals =
                            crate::backend::arm::dequant::dequant_q5_k_block(
                                block,
                            );
                        self.hidden_buf_batch[h_base..h_base + 256]
                            .copy_from_slice(&block_vals);
                    }
                }
                QuantType::Q4_K => {
                    let nb = hidden_size / 256;
                    let block_size = std::mem::size_of::<BlockQ4_K>();
                    let col_offset = (token_id as usize) * nb * block_size;
                    for block_idx in 0..nb {
                        let h_base = out_offset + block_idx * 256;
                        let block: &BlockQ4_K = unsafe {
                            &*(emb_data
                                .as_ptr()
                                .add(col_offset + block_idx * block_size)
                                as *const BlockQ4_K)
                        };
                        let block_vals =
                            crate::backend::arm::dequant::dequantize_q4_k_block(
                                block,
                            );
                        self.hidden_buf_batch[h_base..h_base + 256]
                            .copy_from_slice(&block_vals);
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported embedding type for batch: {:?}",
                        emb_tensor.quant_type
                    ));
                }
            }
        }
        Ok(())
    }

    fn embed_token(&mut self, token_id: u32) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        let emb_tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or("Embedding tensor not found")?;

        let emb_data =
            self.model.get_embeddings().ok_or("No embedding weights")?;

        match emb_tensor.quant_type {
            QuantType::F32 => {
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;
                for h in 0..hidden_size {
                    self.hidden_buf[h] = f32::from_le_bytes(
                        emb_data[col_offset + h * 4..col_offset + h * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                }
            }
            QuantType::F16 => {
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;
                for h in 0..hidden_size {
                    self.hidden_buf[h] = half::f16::from_le_bytes(
                        emb_data[col_offset + h * 2..col_offset + h * 2 + 2]
                            .try_into()
                            .unwrap(),
                    )
                    .to_f32();
                }
            }
            QuantType::Q6_K => {
                let nb = hidden_size / 256;
                let block_size = std::mem::size_of::<BlockQ6_K>();
                let nb1 = emb_tensor.strides[1] as usize; // stride between vocab entries
                for block_idx in 0..nb {
                    let h_base = block_idx * 256;
                    // For [hidden_size, vocab_size] layout, each vocab entry is stored at stride[1] intervals
                    let col_offset =
                        (token_id as usize) * nb1 + block_idx * block_size;
                    let block: &BlockQ6_K = unsafe {
                        &*(emb_data.as_ptr().add(col_offset)
                            as *const BlockQ6_K)
                    };
                    let block_vals =
                        crate::backend::arm::dequant::dequantize_q6_k_block(
                            block,
                        );
                    self.hidden_buf[h_base..h_base + 256]
                        .copy_from_slice(&block_vals);
                }
            }
            QuantType::Q5_K => {
                let nb = hidden_size / 256;
                let block_size = std::mem::size_of::<BlockQ5_K>();
                let nb1 = emb_tensor.strides[1] as usize; // stride between vocab entries
                for block_idx in 0..nb {
                    let h_base = block_idx * 256;
                    // For [hidden_size, vocab_size] layout, each vocab entry is stored at stride[1] intervals
                    // Each vocab entry has nb blocks, but they are interleaved with other vocab entries
                    let col_offset =
                        (token_id as usize) * nb1 + block_idx * block_size;
                    let block: &BlockQ5_K = unsafe {
                        &*(emb_data.as_ptr().add(col_offset)
                            as *const BlockQ5_K)
                    };
                    let block_vals =
                        crate::backend::arm::dequant::dequant_q5_k_block(block);
                    self.hidden_buf[h_base..h_base + 256]
                        .copy_from_slice(&block_vals);
                }
            }
            QuantType::Q4_K => {
                let nb = hidden_size / 256;
                let block_size = std::mem::size_of::<BlockQ4_K>();
                let col_offset = (token_id as usize) * nb * block_size;
                for block_idx in 0..nb {
                    let h_base = block_idx * 256;
                    let block: &BlockQ4_K = unsafe {
                        &*(emb_data
                            .as_ptr()
                            .add(col_offset + block_idx * block_size)
                            as *const BlockQ4_K)
                    };
                    let block_vals =
                        crate::backend::arm::dequant::dequantize_q4_k_block(
                            block,
                        );
                    self.hidden_buf[h_base..h_base + 256]
                        .copy_from_slice(&block_vals);
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported embedding type: {:?}",
                    emb_tensor.quant_type
                ));
            }
        }
        Ok(())
    }

    fn forward_layer(&mut self, layer_idx: usize) -> Result<(), String> {
        self.arena.reset();

        let config = self.model.config().clone();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();

        // Use pre-computed weight names
        let weights = &self.layer_weights[layer_idx];

        // Save hidden state before this layer (for debugging)
        let hidden_before =
            if self.debug_mode { Some(self.hidden_buf.clone()) } else { None };

        // Attention norm
        let normed_attn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.rms_norm(&self.hidden_buf, &weights.attn_norm, normed_attn)?;

        // Save normed_attn for debugging
        let normed_attn_debug =
            if self.debug_mode { Some(normed_attn.to_vec()) } else { None };

        // Quantize input for projections
        let quant_type = self.get_weight_quant_type(&weights.attn_q);
        let normed_len = normed_attn.len();
        let quant_format =
            quantize_input_to(normed_attn, quant_type, &mut self.quant_buf)?;
        let in_features = match quant_format {
            QuantFormat::Q8_0 => ((normed_len + QK8_0 - 1) / QK8_0) * QK8_0,
            QuantFormat::Q8_K => ((normed_len + QK_K - 1) / QK_K) * QK_K,
            QuantFormat::None => normed_len,
        };

        // QKV projections
        let q: &mut [f32] = self.arena.alloc_zeroed(n_heads * head_dim);
        let k: &mut [f32] = self.arena.alloc_zeroed(n_kv_heads * head_dim);
        let v: &mut [f32] = self.arena.alloc_zeroed(n_kv_heads * head_dim);

        self.linear_quantized(quant_format, in_features, &weights.attn_q, q)?;
        self.linear_quantized(quant_format, in_features, &weights.attn_k, k)?;
        self.linear_quantized(quant_format, in_features, &weights.attn_v, v)?;

        // Save Q/K/V for debugging
        let (q_debug, k_debug, v_debug) = if self.debug_mode {
            (Some(q.to_vec()), Some(k.to_vec()), Some(v.to_vec()))
        } else {
            (None, None, None)
        };

        // Add Q/K/V bias if present
        self.add_bias(q, &weights.attn_q_bias)?;
        self.add_bias(k, &weights.attn_k_bias)?;
        self.add_bias(v, &weights.attn_v_bias)?;

        // Q/K norm (Qwen3 uses per-head RMS norm on Q and K)
        // Check if Q/K norm weights exist by trying to get them
        let has_qk_norm = self.model.get_tensor(&weights.attn_q_norm).is_some()
            && self.model.get_tensor(&weights.attn_k_norm).is_some();

        if self.debug_mode && layer_idx == 0 {
            println!(
                "DEBUG: has_qk_norm={}, q_norm_weight={}, k_norm_weight={}",
                has_qk_norm, &weights.attn_q_norm, &weights.attn_k_norm
            );
        }

        if has_qk_norm {
            // Apply Q norm using temporary buffer for in-place operation
            let q_tmp: &mut [f32] = self.arena.alloc_zeroed(n_heads * head_dim);
            q_tmp.copy_from_slice(q);
            self.qk_norm(q_tmp, &weights.attn_q_norm, n_heads, head_dim, q)?;

            // Apply K norm using temporary buffer for in-place operation
            let k_tmp: &mut [f32] =
                self.arena.alloc_zeroed(n_kv_heads * head_dim);
            k_tmp.copy_from_slice(k);
            self.qk_norm(k_tmp, &weights.attn_k_norm, n_kv_heads, head_dim, k)?;
        }

        // RoPE (after Q/K norm if present)
        rope_f32_inplace(
            q,
            self.seq_pos,
            head_dim,
            head_dim,
            config.rope_freq_base,
        );
        rope_f32_inplace(
            k,
            self.seq_pos,
            head_dim,
            head_dim,
            config.rope_freq_base,
        );

        // Save Q/K normed for debugging (now it's just Q/K after RoPE)
        let (q_normed_debug, k_normed_debug) = if self.debug_mode {
            (Some(q.to_vec()), Some(k.to_vec()))
        } else {
            (None, None)
        };

        // KV cache update (k is already RoPE'd)
        let kv_head_offset = self.seq_pos * head_dim;
        for h in 0..n_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = h * self.kv_capacity * head_dim + kv_head_offset;
            self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k[src_offset..src_offset + head_dim]);
            self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }

        // Flash attention (q is already RoPE'd)
        let attn_output: &mut [f32] =
            self.arena.alloc_zeroed(n_heads * head_dim);
        for h in 0..n_heads {
            let kv_head_idx = h * n_kv_heads / n_heads;
            let k_cache = &self.kv_cache_k[layer_idx]
                [kv_head_idx * self.kv_capacity * head_dim..];
            let v_cache = &self.kv_cache_v[layer_idx]
                [kv_head_idx * self.kv_capacity * head_dim..];

            flash_attn_f32(
                &q[h * head_dim..(h + 1) * head_dim],
                &k_cache[..(self.seq_pos + 1) * head_dim],
                &v_cache[..(self.seq_pos + 1) * head_dim],
                None,
                &mut attn_output[h * head_dim..(h + 1) * head_dim],
                head_dim,
                self.seq_pos + 1,
                1.0 / (head_dim as f32).sqrt(),
            );
        }

        // Save attention output for debugging
        let attn_output_debug =
            if self.debug_mode { Some(attn_output.to_vec()) } else { None };

        // Output projection
        let attn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear(attn_output, &weights.attn_output, attn_out)?;

        // Save attn_out for debugging
        let attn_out_debug =
            if self.debug_mode { Some(attn_out.to_vec()) } else { None };

        // Residual
        for i in 0..hidden_size {
            self.hidden_buf[i] += attn_out[i];
        }

        // FFN norm
        let normed_ffn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);

        self.rms_norm(&self.hidden_buf, &weights.ffn_norm, normed_ffn)?;

        // Save FFN normed for debugging
        let normed_ffn_debug =
            if self.debug_mode { Some(normed_ffn.to_vec()) } else { None };

        // Quantize FFN input
        let ffn_normed_len = normed_ffn.len();
        let ffn_quant_format =
            quantize_input_to(normed_ffn, quant_type, &mut self.quant_buf)?;

        let ffn_in_features = match ffn_quant_format {
            QuantFormat::Q8_0 => ((ffn_normed_len + QK8_0 - 1) / QK8_0) * QK8_0,
            QuantFormat::Q8_K => ((ffn_normed_len + QK_K - 1) / QK_K) * QK_K,
            QuantFormat::None => ffn_normed_len,
        };

        // Gate/Up projections
        let gate: &mut [f32] =
            self.arena.alloc_zeroed(config.intermediate_size);
        let up: &mut [f32] = self.arena.alloc_zeroed(config.intermediate_size);

        self.linear_quantized(
            ffn_quant_format,
            ffn_in_features,
            &weights.ffn_gate,
            gate,
        )?;
        self.linear_quantized(
            ffn_quant_format,
            ffn_in_features,
            &weights.ffn_up,
            up,
        )?;

        // Save gate and up for debugging (before SiLU)
        let (gate_before_silu, up_debug) = if self.debug_mode {
            (Some(gate.to_vec()), Some(up.to_vec()))
        } else {
            (None, None)
        };

        // SwiGLU
        silu_f32_inplace(gate);

        // Save gate after SiLU for debugging
        let gate_after_silu =
            if self.debug_mode { Some(gate.to_vec()) } else { None };

        for i in 0..config.intermediate_size {
            gate[i] *= up[i];
        }

        // Save gate * up for debugging
        let gate_times_up_debug =
            if self.debug_mode { Some(gate.to_vec()) } else { None };

        // Down projection
        let ffn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear(gate, &weights.ffn_down, ffn_out)?;

        // Save FFN output for debugging
        let ffn_out_debug =
            if self.debug_mode { Some(ffn_out.to_vec()) } else { None };

        // Residual
        for i in 0..hidden_size {
            self.hidden_buf[i] += ffn_out[i];
        }

        // Save debug state
        if self.debug_mode {
            let debug_state = LayerDebugState {
                layer_idx,
                seq_pos: self.seq_pos,
                hidden_before: hidden_before.unwrap(),
                hidden_after: self.hidden_buf.clone(),
                normed_attn: normed_attn_debug.unwrap(),
                q: q_debug.unwrap(),
                k: k_debug.unwrap(),
                v: v_debug.unwrap(),
                q_normed: q_normed_debug.unwrap(),
                k_normed: k_normed_debug.unwrap(),
                attn_output: attn_output_debug.unwrap(),
                attn_out: attn_out_debug.unwrap(),
                normed_ffn: normed_ffn_debug.unwrap(),
                gate: gate_before_silu.unwrap(),
                up: up_debug.unwrap(),
                gate_times_up: gate_times_up_debug.unwrap(),
                ffn_out: ffn_out_debug.unwrap(),
            };
            self.debug_states.push(debug_state);
        }

        Ok(())
    }

    /// Batch layer forward pass using tiled GEMV
    fn forward_layer_batch(
        &mut self,
        layer_idx: usize,
        n_tokens: usize,
    ) -> Result<(), String> {
        self.arena.reset();

        let config = self.model.config().clone();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();
        let intermediate_size = config.intermediate_size;

        // Clone weight names to avoid borrow conflicts with mutable self calls
        let weights = self.layer_weights[layer_idx].clone();

        // Take ownership of batch buffers to avoid borrow conflicts
        // We'll put them back at the end
        let mut batch_q = std::mem::take(&mut self.batch_q);
        let mut batch_k = std::mem::take(&mut self.batch_k);
        let mut batch_v = std::mem::take(&mut self.batch_v);
        let mut batch_q_normed = std::mem::take(&mut self.batch_q_normed);
        let mut batch_k_normed = std::mem::take(&mut self.batch_k_normed);
        let mut batch_attn_output = std::mem::take(&mut self.batch_attn_output);
        let mut batch_gate = std::mem::take(&mut self.batch_gate);
        let mut batch_up = std::mem::take(&mut self.batch_up);

        // Resize buffers if needed and get slices
        let q_len = n_tokens * n_heads * head_dim;
        let k_len = n_tokens * n_kv_heads * head_dim;
        let v_len = n_tokens * n_kv_heads * head_dim;
        let gate_len = n_tokens * intermediate_size;

        if batch_q.len() < q_len {
            batch_q.resize(q_len, 0.0);
        }
        if batch_k.len() < k_len {
            batch_k.resize(k_len, 0.0);
        }
        if batch_v.len() < v_len {
            batch_v.resize(v_len, 0.0);
        }
        if batch_q_normed.len() < q_len {
            batch_q_normed.resize(q_len, 0.0);
        }
        if batch_k_normed.len() < k_len {
            batch_k_normed.resize(k_len, 0.0);
        }
        if batch_attn_output.len() < q_len {
            batch_attn_output.resize(q_len, 0.0);
        }
        if batch_gate.len() < gate_len {
            batch_gate.resize(gate_len, 0.0);
        }
        if batch_up.len() < gate_len {
            batch_up.resize(gate_len, 0.0);
        }

        // Zero out buffers for this batch
        batch_q[..q_len].fill(0.0);
        batch_k[..k_len].fill(0.0);
        batch_v[..v_len].fill(0.0);
        batch_q_normed[..q_len].fill(0.0);
        batch_k_normed[..k_len].fill(0.0);
        batch_attn_output[..q_len].fill(0.0);
        batch_gate[..gate_len].fill(0.0);
        batch_up[..gate_len].fill(0.0);

        // Get slices (now we own the Vecs, so no borrow conflict)
        let q = &mut batch_q[..q_len];
        let k = &mut batch_k[..k_len];
        let v = &mut batch_v[..v_len];
        let q_normed = &mut batch_q_normed[..q_len];
        let k_normed = &mut batch_k_normed[..k_len];
        let attn_output = &mut batch_attn_output[..q_len];
        let gate = &mut batch_gate[..gate_len];
        let up = &mut batch_up[..gate_len];

        // Use local Vec for transient data to avoid arena borrow conflict
        let mut normed_attn_vec = vec![0.0f32; n_tokens * hidden_size];
        let mut attn_out_vec = vec![0.0f32; n_tokens * hidden_size];
        let mut normed_ffn_vec = vec![0.0f32; n_tokens * hidden_size];
        let mut ffn_out_vec = vec![0.0f32; n_tokens * hidden_size];
        let normed_attn = &mut normed_attn_vec[..];
        let attn_out = &mut attn_out_vec[..];
        let normed_ffn = &mut normed_ffn_vec[..];
        let ffn_out = &mut ffn_out_vec[..];

        // Batch attention norm
        time_op!("attn_norm", {
            self.rms_norm_batch(
                &self.hidden_buf_batch,
                &weights.attn_norm,
                normed_attn,
                n_tokens,
            )?;
        });

        // Quantize batch input
        let quant_type = self.get_weight_quant_type(&weights.attn_q);
        time_op!("quant", {
            quantize_input_batch(
                normed_attn,
                hidden_size,
                n_tokens,
                quant_type,
                &mut self.quant_buf_batch,
            )?;
        });
        let quant_format = self.quant_buf_batch.active;
        let in_features = hidden_size;

        // Batch QKV projections using tiled GEMV
        time_op!("qkv_proj", {
            self.linear_quantized_batch(
                quant_format,
                in_features,
                n_tokens,
                &weights.attn_q,
                q,
            )?;
            self.linear_quantized_batch(
                quant_format,
                in_features,
                n_tokens,
                &weights.attn_k,
                k,
            )?;
            self.linear_quantized_batch(
                quant_format,
                in_features,
                n_tokens,
                &weights.attn_v,
                v,
            )?;
        });

        // Add Q/K/V bias if present (batch version)
        self.add_bias_batch(q, &weights.attn_q_bias, n_tokens)?;
        self.add_bias_batch(k, &weights.attn_k_bias, n_tokens)?;
        self.add_bias_batch(v, &weights.attn_v_bias, n_tokens)?;

        // Batch Q/K norm - DISABLED to match single token path
        // Note: Qwen3 uses Q/K norm, but we disable it for now to test if it's causing issues
        let has_qk_norm = self.model.get_tensor(&weights.attn_q_norm).is_some()
            && self.model.get_tensor(&weights.attn_k_norm).is_some();

        if has_qk_norm {
            time_op!("qk_norm", {
                self.qk_norm_batch(
                    q,
                    &weights.attn_q_norm,
                    n_heads,
                    head_dim,
                    n_tokens,
                    q_normed,
                )?;
                self.qk_norm_batch(
                    k,
                    &weights.attn_k_norm,
                    n_kv_heads,
                    head_dim,
                    n_tokens,
                    k_normed,
                )?;
            });
        } else {
            // Copy q and k to q_normed and k_normed without norm
            q_normed[..q.len()].copy_from_slice(q);
            k_normed[..k.len()].copy_from_slice(k);
        }

        // Batch RoPE with position offsets
        time_op!("rope", {
            for token_idx in 0..n_tokens {
                let pos = self.seq_pos + token_idx;
                rope_f32_inplace(
                    &mut q_normed[token_idx * n_heads * head_dim
                        ..(token_idx + 1) * n_heads * head_dim],
                    pos,
                    head_dim,
                    head_dim,
                    config.rope_freq_base,
                );
                rope_f32_inplace(
                    &mut k_normed[token_idx * n_kv_heads * head_dim
                        ..(token_idx + 1) * n_kv_heads * head_dim],
                    pos,
                    head_dim,
                    head_dim,
                    config.rope_freq_base,
                );
            }
        });

        // Batch KV cache update
        time_op!("kv_cache", {
            for token_idx in 0..n_tokens {
                let kv_pos = self.seq_pos + token_idx;
                let kv_offset = kv_pos * head_dim;
                for h in 0..n_kv_heads {
                    let src_offset =
                        token_idx * n_kv_heads * head_dim + h * head_dim;
                    let dst_offset =
                        h * self.kv_capacity * head_dim + kv_offset;
                    self.kv_cache_k[layer_idx]
                        [dst_offset..dst_offset + head_dim]
                        .copy_from_slice(
                            &k_normed[src_offset..src_offset + head_dim],
                        );
                    self.kv_cache_v[layer_idx]
                        [dst_offset..dst_offset + head_dim]
                        .copy_from_slice(
                            &v[token_idx * n_kv_heads * head_dim + h * head_dim
                                ..token_idx * n_kv_heads * head_dim
                                    + (h + 1) * head_dim],
                        );
                }
            }
        });

        // Batch attention (simplified: process each token)
        time_op!("attention", {
            for token_idx in 0..n_tokens {
                let cur_pos = self.seq_pos + token_idx + 1;
                for h in 0..n_heads {
                    let kv_head_idx = h * n_kv_heads / n_heads;
                    let k_cache = &self.kv_cache_k[layer_idx]
                        [kv_head_idx * self.kv_capacity * head_dim..];
                    let v_cache = &self.kv_cache_v[layer_idx]
                        [kv_head_idx * self.kv_capacity * head_dim..];
                    flash_attn_f32(
                        &q_normed[token_idx * n_heads * head_dim + h * head_dim
                            ..token_idx * n_heads * head_dim
                                + (h + 1) * head_dim],
                        &k_cache[..cur_pos * head_dim],
                        &v_cache[..cur_pos * head_dim],
                        None,
                        &mut attn_output[token_idx * n_heads * head_dim
                            + h * head_dim
                            ..token_idx * n_heads * head_dim
                                + (h + 1) * head_dim],
                        head_dim,
                        cur_pos,
                        1.0 / (head_dim as f32).sqrt(),
                    );
                }
            }
        });

        // Batch output projection
        time_op!("o_proj", {
            self.linear_batch(
                attn_output,
                n_tokens,
                &weights.attn_output,
                attn_out,
            )?;
        });

        // Batch residual
        for i in 0..n_tokens * hidden_size {
            self.hidden_buf_batch[i] += attn_out[i];
        }

        // Batch FFN norm
        time_op!("ffn_norm", {
            self.rms_norm_batch(
                &self.hidden_buf_batch,
                &weights.ffn_norm,
                normed_ffn,
                n_tokens,
            )?;
        });

        // Quantize FFN batch input
        time_op!("ffn_quant", {
            quantize_input_batch(
                normed_ffn,
                hidden_size,
                n_tokens,
                quant_type,
                &mut self.quant_buf_batch,
            )?;
        });
        let ffn_quant_format = self.quant_buf_batch.active;

        // Batch Gate/Up projections
        time_op!("gate_up_proj", {
            self.linear_quantized_batch(
                ffn_quant_format,
                hidden_size,
                n_tokens,
                &weights.ffn_gate,
                gate,
            )?;
            self.linear_quantized_batch(
                ffn_quant_format,
                hidden_size,
                n_tokens,
                &weights.ffn_up,
                up,
            )?;
        });

        // Batch SwiGLU
        time_op!("swiglu", {
            silu_f32_inplace(gate);
            for i in 0..n_tokens * intermediate_size {
                gate[i] *= up[i];
            }
        });

        // Batch Down projection
        time_op!("down_proj", {
            self.linear_batch(gate, n_tokens, &weights.ffn_down, ffn_out)?;
        });

        // Batch residual
        for i in 0..n_tokens * hidden_size {
            self.hidden_buf_batch[i] += ffn_out[i];
        }

        // Put back batch buffers
        self.batch_q = batch_q;
        self.batch_k = batch_k;
        self.batch_v = batch_v;
        self.batch_q_normed = batch_q_normed;
        self.batch_k_normed = batch_k_normed;
        self.batch_attn_output = batch_attn_output;
        self.batch_gate = batch_gate;
        self.batch_up = batch_up;

        Ok(())
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let eps = self.model.config().rms_norm_eps;
        let tensor = self.model.get_tensor_meta(weight_name);
        let weight_data = self.model.get_tensor(weight_name);

        let weight_type =
            tensor.map(|t| t.quant_type).unwrap_or(QuantType::F32);
        let weight_bytes = weight_data
            .as_deref()
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        rms_norm::rms_norm(input, weight_bytes, weight_type, output, eps)
    }

    /// Batch RMS norm: [n_tokens * hidden_size] -> [n_tokens * hidden_size]
    fn rms_norm_batch(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
        n_tokens: usize,
    ) -> Result<(), String> {
        let eps = self.model.config().rms_norm_eps;
        let tensor = self.model.get_tensor_meta(weight_name);
        let weight_data = self.model.get_tensor(weight_name);

        let weight_type =
            tensor.map(|t| t.quant_type).unwrap_or(QuantType::F32);
        let weight_bytes = weight_data
            .as_deref()
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        rms_norm::rms_norm_batch(
            input,
            weight_bytes,
            weight_type,
            output,
            n_tokens,
            eps,
        )
    }

    fn qk_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        n_heads: usize,
        head_dim: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        let eps = self.model.config().rms_norm_eps;
        let tensor = self.model.get_tensor_meta(weight_name);
        let weight_data = self.model.get_tensor(weight_name);

        let weight_type =
            tensor.map(|t| t.quant_type).unwrap_or(QuantType::F32);
        let weight_bytes = weight_data
            .as_deref()
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        rms_norm::qk_norm(
            input,
            weight_bytes,
            weight_type,
            n_heads,
            head_dim,
            output,
            eps,
        )
    }

    /// Batch QK norm: [n_tokens * n_heads * head_dim] -> [n_tokens * n_heads * head_dim]
    fn qk_norm_batch(
        &self,
        input: &[f32],
        weight_name: &str,
        n_heads: usize,
        head_dim: usize,
        n_tokens: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        let eps = self.model.config().rms_norm_eps;
        let tensor = self.model.get_tensor_meta(weight_name);
        let weight_data = self.model.get_tensor(weight_name);

        let weight_type =
            tensor.map(|t| t.quant_type).unwrap_or(QuantType::F32);
        let weight_bytes = weight_data
            .as_deref()
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        rms_norm::qk_norm_batch(
            input,
            weight_bytes,
            weight_type,
            n_heads,
            head_dim,
            n_tokens,
            output,
            eps,
        )
    }

    /// Add bias to output if bias tensor exists
    fn add_bias(
        &self,
        output: &mut [f32],
        bias_name: &str,
    ) -> Result<(), String> {
        if let Some(bias_data) = self.model.get_tensor(bias_name) {
            if bias_data.len() == output.len() * 4 {
                // F32 bias
                let bias: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        bias_data.as_ptr() as *const f32,
                        output.len(),
                    )
                };
                for i in 0..output.len() {
                    output[i] += bias[i];
                }
            } else if bias_data.len() == output.len() * 2 {
                // F16 bias
                let bias_f16: &[half::f16] = unsafe {
                    std::slice::from_raw_parts(
                        bias_data.as_ptr() as *const half::f16,
                        output.len(),
                    )
                };
                for i in 0..output.len() {
                    output[i] += bias_f16[i].to_f32();
                }
            }
        }
        Ok(())
    }

    /// Add bias to output if bias tensor exists (batch version)
    fn add_bias_batch(
        &self,
        output: &mut [f32],
        bias_name: &str,
        n_tokens: usize,
    ) -> Result<(), String> {
        if let Some(bias_data) = self.model.get_tensor(bias_name) {
            let feature_size = output.len() / n_tokens;
            if bias_data.len() == feature_size * 4 {
                // F32 bias
                let bias: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        bias_data.as_ptr() as *const f32,
                        feature_size,
                    )
                };
                for t in 0..n_tokens {
                    let offset = t * feature_size;
                    for i in 0..feature_size {
                        output[offset + i] += bias[i];
                    }
                }
            }
        }
        Ok(())
    }

    fn linear(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let tensor = self
            .model
            .get_tensor_meta(weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        let stride = tensor.strides.get(1).copied().unwrap_or(0) as usize;
        let dims = &tensor.dims;
        let in_features = dims.get(0).copied().unwrap_or(0) as usize;
        let out_features = dims.get(1).copied().unwrap_or(0) as usize;

        linear::linear_f32(
            input,
            weight_data,
            tensor.quant_type,
            stride,
            output,
            self.n_threads,
        )
    }

    fn linear_quantized(
        &self,
        quant_format: QuantFormat,
        _in_features: usize,
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let tensor = self
            .model
            .get_tensor_meta(weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        let stride = tensor.strides.get(1).copied().unwrap_or(0) as usize;

        // Note: Single-token path uses linear::linear_q8_k
        match quant_format {
            QuantFormat::Q8_0 => linear::linear_q8_0(
                &self.quant_buf.q8_0,
                self.quant_buf.q8_0.len() * QK8_0,
                weight_data,
                tensor.quant_type,
                stride,
                output,
                self.n_threads,
            ),
            QuantFormat::Q8_K => linear::linear_q8_k(
                &self.quant_buf.q8_k,
                self.quant_buf.q8_k.len() * QK_K,
                weight_data,
                tensor.quant_type,
                stride,
                output,
                self.n_threads,
            ),
            QuantFormat::None => Err("No quantized input available".into()),
        }
    }

    /// Batch linear with quantized input using tiled GEMV
    fn linear_quantized_batch(
        &self,
        quant_format: QuantFormat,
        in_features: usize,
        n_tokens: usize,
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let tensor = self
            .model
            .get_tensor_meta(weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        let stride = tensor.strides.get(1).copied().unwrap_or(0) as usize;
        let out_features = output.len() / n_tokens;

        match quant_format {
            QuantFormat::Q8_0 => linear::linear_q8_0_batch(
                &self.quant_buf_batch.q8_0,
                weight_data,
                stride,
                output,
                out_features,
                in_features,
                n_tokens,
                self.n_threads_batch,
            ),
            QuantFormat::Q8_K => {
                linear::linear_q8_k_batch(
                    &self.quant_buf_batch.q8_k,
                    weight_data,
                    tensor.quant_type,
                    stride,
                    output,
                    out_features,
                    in_features,
                    n_tokens,
                    self.n_threads_batch,
                )?;
                Ok(())
            }
            QuantFormat::None => Err("No quantized input available".into()),
        }
    }

    /// Batch linear with F32 input
    fn linear_batch(
        &mut self,
        input: &[f32],
        n_tokens: usize,
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let tensor = self
            .model
            .get_tensor_meta(weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        let stride = tensor.strides.get(1).copied().unwrap_or(0) as usize;
        let out_features = output.len() / n_tokens;
        let in_features = input.len() / n_tokens;

        linear::linear_f32_batch(
            input,
            weight_data,
            tensor.quant_type,
            stride,
            output,
            out_features,
            in_features,
            n_tokens,
            self.n_threads_batch,
            &mut self.quant_buf_batch.q8_0,
            &mut self.quant_buf_batch.q8_k,
        )
    }
}

impl ForwardExecutor {
    fn get_weight_quant_type(&self, weight_name: &str) -> QuantType {
        self.model
            .get_tensor_meta(weight_name)
            .map(|t| t.quant_type)
            .unwrap_or(QuantType::F32)
    }

    fn final_norm(&mut self) -> Result<(), String> {
        let config = self.model.config();
        let eps = config.rms_norm_eps;

        let tensor = self.model.get_tensor_meta("output_norm.weight");
        let weight_data = self.model.get_tensor("output_norm.weight");

        let weight_type =
            tensor.map(|t| t.quant_type).unwrap_or(QuantType::F32);
        let weight_bytes =
            weight_data.as_deref().ok_or("output_norm.weight not found")?;

        // Need to clone because rms_norm needs separate input/output
        let input = self.hidden_buf.clone();
        rms_norm::rms_norm(
            &input,
            weight_bytes,
            weight_type,
            &mut self.hidden_buf,
            eps,
        )
    }

    fn output_projection(&mut self) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = self.hidden_buf.len();

        // Try output.weight or lm_head.weight, fallback to token_embd.weight
        let output_tensor = self
            .model
            .get_tensor_meta("output.weight")
            .or_else(|| self.model.get_tensor_meta("lm_head.weight"))
            .or_else(|| self.model.get_tensor_meta("token_embd.weight"))
            .ok_or("No output or embedding weights found")?;

        let output_data = self
            .model
            .get_tensor(&output_tensor.name)
            .ok_or("Output weight data not found")?;

        self.logits_buf.fill(0.0);

        // For token_embd.weight (shared weights), dims are [hidden_size, vocab_size]
        // For output projection: logits[j] = dot(hidden, embd[:, j])
        let quant_type = output_tensor.quant_type;
        let vocab_size =
            output_tensor.dims.get(1).copied().unwrap_or(0) as usize;
        let actual_hidden =
            output_tensor.dims.get(0).copied().unwrap_or(0) as usize;
        let nb = (actual_hidden + QK_K - 1) / QK_K;

        if matches!(
            quant_type,
            QuantType::Q4_K | QuantType::Q5_K | QuantType::Q6_K
        ) {
            // Quantize hidden state
            self.quant_buf.resize(actual_hidden, QuantFormat::Q8_K);
            quant::quantize_q8_k(&self.hidden_buf, &mut self.quant_buf.q8_k);

            let stride_0 =
                output_tensor.strides.get(0).copied().unwrap_or(0) as usize;
            let stride_1 =
                output_tensor.strides.get(1).copied().unwrap_or(0) as usize;

            // Collect blocks for each column (token) and compute dot product
            match quant_type {
                QuantType::Q6_K => {
                    use crate::types::BlockQ6_K;
                    let mut col_blocks: Vec<BlockQ6_K> =
                        vec![unsafe { std::mem::zeroed() }; nb];
                    for j in 0..vocab_size {
                        for b in 0..nb {
                            let offset = b * stride_0 + j * stride_1;
                            col_blocks[b] = unsafe {
                                *(output_data.as_ptr().add(offset)
                                    as *const BlockQ6_K)
                            };
                        }
                        self.logits_buf[j] =
                            crate::backend::arm::vec_dot::vec_dot_q6_k_q8_k(
                                &col_blocks,
                                &self.quant_buf.q8_k,
                            );
                    }
                }
                QuantType::Q5_K => {
                    use crate::types::BlockQ5_K;
                    let mut col_blocks: Vec<BlockQ5_K> =
                        vec![unsafe { std::mem::zeroed() }; nb];
                    for j in 0..vocab_size {
                        for b in 0..nb {
                            let offset = b * stride_0 + j * stride_1;
                            col_blocks[b] = unsafe {
                                *(output_data.as_ptr().add(offset)
                                    as *const BlockQ5_K)
                            };
                        }
                        self.logits_buf[j] =
                            crate::backend::arm::vec_dot::vec_dot_q5_k_q8_k(
                                &col_blocks,
                                &self.quant_buf.q8_k,
                            );
                    }
                }
                QuantType::Q4_K => {
                    use crate::types::BlockQ4_K;
                    let mut col_blocks: Vec<BlockQ4_K> =
                        vec![unsafe { std::mem::zeroed() }; nb];
                    for j in 0..vocab_size {
                        for b in 0..nb {
                            let offset = b * stride_0 + j * stride_1;
                            col_blocks[b] = unsafe {
                                *(output_data.as_ptr().add(offset)
                                    as *const BlockQ4_K)
                            };
                        }
                        self.logits_buf[j] =
                            crate::backend::arm::vec_dot::vec_dot_q4_k_q8_k(
                                &col_blocks,
                                &self.quant_buf.q8_k,
                            );
                    }
                }
                _ => {}
            }
        } else if matches!(quant_type, QuantType::Q4_0 | QuantType::Q8_0) {
            self.quant_buf.resize(hidden_size, QuantFormat::Q8_0);
            quant::quantize_q8_0(&self.hidden_buf, &mut self.quant_buf.q8_0);
            linear::linear_q8_0(
                &self.quant_buf.q8_0,
                hidden_size,
                output_data,
                quant_type,
                0, // stride is now ignored
                &mut self.logits_buf,
                self.n_threads,
            )?;
        } else {
            linear::linear_f32(
                &self.hidden_buf,
                output_data,
                quant_type,
                0, // stride is now ignored
                &mut self.logits_buf,
                self.n_threads,
            )?;
        }

        Ok(())
    }

    /// Batch final norm - only process the last token
    fn final_norm_batch(&mut self, n_tokens: usize) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        // Only norm the last token's hidden state
        let last_token_start = (n_tokens - 1) * hidden_size;
        let last_hidden = &self.hidden_buf_batch
            [last_token_start..last_token_start + hidden_size];
        self.hidden_buf.copy_from_slice(last_hidden);

        // Use existing final_norm
        self.final_norm()?;

        // Copy back to batch buffer
        self.hidden_buf_batch[last_token_start..last_token_start + hidden_size]
            .copy_from_slice(&self.hidden_buf);
        Ok(())
    }

    /// Batch output projection - only process the last token
    fn output_projection_batch(&mut self) -> Result<(), String> {
        // Use hidden_buf which was set by final_norm_batch
        self.output_projection()
    }

    pub fn sample(&self, logits: &[f32], temperature: f32) -> u32 {
        if temperature == 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            let max_logit =
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits
                .iter()
                .map(|&l| ((l - max_logit) / temperature).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            for p in &mut probs {
                *p /= sum;
            }

            use rand::Rng;
            let mut rng = rand::thread_rng();
            let r: f32 = rng.r#gen();
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= r {
                    return i as u32;
                }
            }
            (probs.len() - 1) as u32
        }
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        let mut tokens = prompt_tokens.to_vec();

        // Use batch forward for prefill (more efficient)
        if self.forward_batch(prompt_tokens).is_err() {
            // Fallback to single token processing if batch fails
            for (i, &token) in prompt_tokens.iter().enumerate() {
                if self.forward(token).is_err() {
                    log::error!("Prefill error at token {}", i);
                    return tokens;
                }
            }
        }

        for _ in 0..max_new_tokens {
            let next_token = {
                let logits = &self.logits_buf;
                let token = self.sample(logits, temperature);
                token
            };
            tokens.push(next_token);

            if self.forward(next_token).is_err() {
                break;
            }
        }

        tokens
    }
}
