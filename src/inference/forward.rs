// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Forward executor for CPU inference

use crate::backend::arm::{
    activation::silu_f32_inplace, flash_attn::flash_attn_f32, gemv, linear,
    quant, rms_norm::rms_norm_f32, rope::rope_f32_inplace,
};
use crate::inference::SimpleModel;
use crate::types::*;

/// Quantized input storage with pre-allocated buffer
pub struct QuantizedInput {
    /// Q8_0 format blocks (for Q4_0, Q5_0, Q8_0 weights)
    pub q8_0: Vec<BlockQ8_0>,
    /// Q8_K format blocks (for Q4_K, Q5_K, Q6_K, Q8_K, IQ4_XS weights)
    pub q8_k: Vec<BlockQ8_K>,
    /// Current active format
    pub active: QuantFormat,
}

/// Batch quantized input for prefill: [n_tokens * n_blocks]
pub struct QuantizedInputBatch {
    pub q8_0: Vec<BlockQ8_0>,
    pub q8_k: Vec<BlockQ8_K>,
    pub active: QuantFormat,
    pub n_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantFormat {
    None,
    Q8_0,
    Q8_K,
}

impl QuantizedInput {
    pub fn new() -> Self {
        Self { q8_0: Vec::new(), q8_k: Vec::new(), active: QuantFormat::None }
    }

    pub fn resize(&mut self, in_features: usize, format: QuantFormat) {
        match format {
            QuantFormat::Q8_0 => {
                let nb = (in_features + QK8_0 - 1) / QK8_0;
                if self.q8_0.len() != nb {
                    self.q8_0.resize(nb, BlockQ8_0 { d: 0, qs: [0; QK8_0] });
                }
            }
            QuantFormat::Q8_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                if self.q8_k.len() != nb {
                    self.q8_k.resize(nb, unsafe { std::mem::zeroed() });
                }
            }
            QuantFormat::None => {}
        }
        self.active = format;
    }
}

impl QuantizedInputBatch {
    pub fn new() -> Self {
        Self {
            q8_0: Vec::new(),
            q8_k: Vec::new(),
            active: QuantFormat::None,
            n_tokens: 0,
        }
    }

    /// Resize for batch processing: n_tokens * in_features
    pub fn resize(
        &mut self,
        n_tokens: usize,
        in_features: usize,
        format: QuantFormat,
    ) {
        self.n_tokens = n_tokens;
        match format {
            QuantFormat::Q8_0 => {
                let nb_per_token = (in_features + QK8_0 - 1) / QK8_0;
                let total_nb = n_tokens * nb_per_token;
                if self.q8_0.len() != total_nb {
                    self.q8_0
                        .resize(total_nb, BlockQ8_0 { d: 0, qs: [0; QK8_0] });
                }
            }
            QuantFormat::Q8_K => {
                let nb_per_token = (in_features + QK_K - 1) / QK_K;
                let total_nb = n_tokens * nb_per_token;
                if self.q8_k.len() != total_nb {
                    self.q8_k.resize(total_nb, unsafe { std::mem::zeroed() });
                }
            }
            QuantFormat::None => {}
        }
        self.active = format;
    }
}

/// Pre-computed layer weight names (avoids format! in hot path)
#[derive(Clone)]
struct LayerWeightNames {
    attn_norm: String,
    attn_q: String,
    attn_k: String,
    attn_v: String,
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
    logits_buf: Vec<f32>,
    pub n_threads: usize,
    pub n_threads_batch: usize,
    /// Pre-computed weight names for each layer
    layer_weights: Vec<LayerWeightNames>,
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
                    let col_offset = (token_id as usize) * nb * block_size;
                    for block_idx in 0..nb {
                        let h_base = out_offset + block_idx * 256;
                        let block: &BlockQ6_K = unsafe {
                            &*(emb_data
                                .as_ptr()
                                .add(col_offset + block_idx * block_size)
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
                    let col_offset = (token_id as usize) * nb * block_size;
                    for block_idx in 0..nb {
                        let h_base = out_offset + block_idx * 256;
                        let block: &BlockQ5_K = unsafe {
                            &*(emb_data
                                .as_ptr()
                                .add(col_offset + block_idx * block_size)
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
                let col_offset = (token_id as usize) * nb * block_size;
                for block_idx in 0..nb {
                    let h_base = block_idx * 256;
                    let block: &BlockQ6_K = unsafe {
                        &*(emb_data
                            .as_ptr()
                            .add(col_offset + block_idx * block_size)
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
                let col_offset = (token_id as usize) * nb * block_size;
                for block_idx in 0..nb {
                    let h_base = block_idx * 256;
                    let block: &BlockQ5_K = unsafe {
                        &*(emb_data
                            .as_ptr()
                            .add(col_offset + block_idx * block_size)
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

        // Attention norm
        let normed_attn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.rms_norm(&self.hidden_buf, &weights.attn_norm, normed_attn)?;

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

        // Q/K norm
        let q_normed: &mut [f32] = self.arena.alloc_zeroed(n_heads * head_dim);
        let k_normed: &mut [f32] =
            self.arena.alloc_zeroed(n_kv_heads * head_dim);

        self.qk_norm(q, &weights.attn_q_norm, n_heads, head_dim, q_normed)?;
        self.qk_norm(k, &weights.attn_k_norm, n_kv_heads, head_dim, k_normed)?;

        // RoPE
        rope_f32_inplace(
            q_normed,
            self.seq_pos,
            head_dim,
            head_dim,
            config.rope_freq_base,
        );
        rope_f32_inplace(
            k_normed,
            self.seq_pos,
            head_dim,
            head_dim,
            config.rope_freq_base,
        );

        // KV cache update
        let kv_head_offset = self.seq_pos * head_dim;
        for h in 0..n_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = h * self.kv_capacity * head_dim + kv_head_offset;
            self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k_normed[src_offset..src_offset + head_dim]);
            self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }

        // Flash attention
        let attn_output: &mut [f32] =
            self.arena.alloc_zeroed(n_heads * head_dim);
        for h in 0..n_heads {
            let kv_head_idx = h * n_kv_heads / n_heads;
            let k_cache = &self.kv_cache_k[layer_idx]
                [kv_head_idx * self.kv_capacity * head_dim..];
            let v_cache = &self.kv_cache_v[layer_idx]
                [kv_head_idx * self.kv_capacity * head_dim..];

            flash_attn_f32(
                &q_normed[h * head_dim..(h + 1) * head_dim],
                &k_cache[..(self.seq_pos + 1) * head_dim],
                &v_cache[..(self.seq_pos + 1) * head_dim],
                None,
                &mut attn_output[h * head_dim..(h + 1) * head_dim],
                head_dim,
                self.seq_pos + 1,
                1.0 / (head_dim as f32).sqrt(),
            );
        }

        // Output projection
        let attn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear(attn_output, &weights.attn_output, attn_out)?;

        // Residual
        for i in 0..hidden_size {
            self.hidden_buf[i] += attn_out[i];
        }

        // FFN norm
        let normed_ffn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.rms_norm(&self.hidden_buf, &weights.ffn_norm, normed_ffn)?;

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

        // SwiGLU
        silu_f32_inplace(gate);
        for i in 0..config.intermediate_size {
            gate[i] *= up[i];
        }

        // Down projection
        let ffn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear(gate, &weights.ffn_down, ffn_out)?;

        // Residual
        for i in 0..hidden_size {
            self.hidden_buf[i] += ffn_out[i];
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

        // Clone weight names to avoid borrow conflicts with mutable self calls
        let weights = self.layer_weights[layer_idx].clone();

        // Batch attention norm
        let normed_attn: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * hidden_size);
        self.rms_norm_batch(
            &self.hidden_buf_batch,
            &weights.attn_norm,
            normed_attn,
            n_tokens,
        )?;

        // Quantize batch input
        let quant_type = self.get_weight_quant_type(&weights.attn_q);
        let quant_format = quantize_input_batch(
            normed_attn,
            hidden_size,
            n_tokens,
            quant_type,
            &mut self.quant_buf_batch,
        )?;
        let in_features = hidden_size;

        // Batch QKV projections using tiled GEMV
        let q: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * n_heads * head_dim);
        let k: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * n_kv_heads * head_dim);
        let v: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * n_kv_heads * head_dim);

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

        // Batch Q/K norm
        let q_normed: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * n_heads * head_dim);
        let k_normed: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * n_kv_heads * head_dim);

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

        // Batch RoPE with position offsets
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

        // Batch KV cache update
        for token_idx in 0..n_tokens {
            let kv_pos = self.seq_pos + token_idx;
            let kv_offset = kv_pos * head_dim;
            for h in 0..n_kv_heads {
                let src_offset =
                    token_idx * n_kv_heads * head_dim + h * head_dim;
                let dst_offset = h * self.kv_capacity * head_dim + kv_offset;
                self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                    .copy_from_slice(
                        &k_normed[src_offset..src_offset + head_dim],
                    );
                self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                    .copy_from_slice(
                        &v[token_idx * n_kv_heads * head_dim + h * head_dim
                            ..token_idx * n_kv_heads * head_dim
                                + (h + 1) * head_dim],
                    );
            }
        }

        // Batch attention (simplified: process each token)
        let attn_output: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * n_heads * head_dim);
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
                        ..token_idx * n_heads * head_dim + (h + 1) * head_dim],
                    &k_cache[..cur_pos * head_dim],
                    &v_cache[..cur_pos * head_dim],
                    None,
                    &mut attn_output[token_idx * n_heads * head_dim
                        + h * head_dim
                        ..token_idx * n_heads * head_dim + (h + 1) * head_dim],
                    head_dim,
                    cur_pos,
                    1.0 / (head_dim as f32).sqrt(),
                );
            }
        }

        // Batch output projection
        let attn_out: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * hidden_size);
        self.linear_batch(
            attn_output,
            n_tokens,
            &weights.attn_output,
            attn_out,
        )?;

        // Batch residual
        for i in 0..n_tokens * hidden_size {
            self.hidden_buf_batch[i] += attn_out[i];
        }

        // Batch FFN norm
        let normed_ffn: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * hidden_size);
        self.rms_norm_batch(
            &self.hidden_buf_batch,
            &weights.ffn_norm,
            normed_ffn,
            n_tokens,
        )?;

        // Quantize FFN batch input
        let ffn_quant_format = quantize_input_batch(
            normed_ffn,
            hidden_size,
            n_tokens,
            quant_type,
            &mut self.quant_buf_batch,
        )?;

        // Batch Gate/Up projections
        let gate: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * config.intermediate_size);
        let up: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * config.intermediate_size);

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

        // Batch SwiGLU
        silu_f32_inplace(gate);
        for i in 0..n_tokens * config.intermediate_size {
            gate[i] *= up[i];
        }

        // Batch Down projection
        let ffn_out: &mut [f32] =
            self.arena.alloc_zeroed(n_tokens * hidden_size);
        self.linear_batch(gate, n_tokens, &weights.ffn_down, ffn_out)?;

        // Batch residual
        for i in 0..n_tokens * hidden_size {
            self.hidden_buf_batch[i] += ffn_out[i];
        }

        Ok(())
    }

    fn rms_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let size = input.len();
        let eps = self.model.config().rms_norm_eps;
        let weight_data = self.model.get_tensor(weight_name);

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / size as f32 + eps).sqrt();

        if let Some(data) = weight_data {
            if data.len() == size * 4 {
                // F32 weight - directly cast bytes to f32 slice
                let weight: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const f32,
                        size,
                    )
                };
                rms_norm_f32(input, weight, output, eps);
            } else if data.len() == size * 2 {
                // F16 weight - need to convert (but no allocation)
                let scale = 1.0 / rms;
                for (i, chunk) in data.chunks_exact(2).enumerate() {
                    let w =
                        half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                    output[i] = input[i] * scale * w;
                }
            } else {
                for (i, inp) in input.iter().enumerate() {
                    output[i] = inp / rms;
                }
            }
        } else {
            for (i, inp) in input.iter().enumerate() {
                output[i] = inp / rms;
            }
        }
        Ok(())
    }

    /// Batch RMS norm: [n_tokens * hidden_size] -> [n_tokens * hidden_size]
    fn rms_norm_batch(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
        n_tokens: usize,
    ) -> Result<(), String> {
        let hidden_size = input.len() / n_tokens;
        let eps = self.model.config().rms_norm_eps;
        let weight_data = self.model.get_tensor(weight_name);

        // Get weight
        let weight: Vec<f32> = if let Some(data) = weight_data {
            if data.len() == hidden_size * 4 {
                unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const f32,
                        hidden_size,
                    )
                    .to_vec()
                }
            } else if data.len() == hidden_size * 2 {
                data.chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                    })
                    .collect()
            } else {
                vec![1.0; hidden_size]
            }
        } else {
            vec![1.0; hidden_size]
        };

        // Process each token
        for token_idx in 0..n_tokens {
            let in_offset = token_idx * hidden_size;
            let out_offset = token_idx * hidden_size;

            let sum_sq: f32 = input[in_offset..in_offset + hidden_size]
                .iter()
                .map(|x| x * x)
                .sum();
            let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
            let scale = 1.0 / rms;

            for h in 0..hidden_size {
                output[out_offset + h] =
                    input[in_offset + h] * scale * weight[h];
            }
        }
        Ok(())
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
        let weight_data = self.model.get_tensor(weight_name);

        if let Some(data) = weight_data {
            if data.len() == head_dim * 4 {
                // F32 weight - directly cast bytes to f32 slice
                let shared_weight: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const f32,
                        head_dim,
                    )
                };

                for h in 0..n_heads {
                    let head_start = h * head_dim;
                    let head_input = &input[head_start..head_start + head_dim];
                    let ss: f32 = head_input.iter().map(|x| x * x).sum::<f32>()
                        / head_dim as f32;
                    let scale = 1.0f32 / (ss + eps).sqrt();
                    for d in 0..head_dim {
                        output[head_start + d] =
                            head_input[d] * scale * shared_weight[d];
                    }
                }
            } else if data.len() == head_dim * 2 {
                // F16 weight
                let shared_weight_f16: &[half::f16] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const half::f16,
                        head_dim,
                    )
                };

                for h in 0..n_heads {
                    let head_start = h * head_dim;
                    let head_input = &input[head_start..head_start + head_dim];
                    let ss: f32 = head_input.iter().map(|x| x * x).sum::<f32>()
                        / head_dim as f32;
                    let scale = 1.0f32 / (ss + eps).sqrt();
                    for d in 0..head_dim {
                        output[head_start + d] = head_input[d]
                            * scale
                            * shared_weight_f16[d].to_f32();
                    }
                }
            } else {
                output.copy_from_slice(input);
            }
        } else {
            output.copy_from_slice(input);
        }
        Ok(())
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
        let weight_data = self.model.get_tensor(weight_name);
        let eps = 1e-5f32;

        let weight: Vec<f32> = if let Some(data) = weight_data {
            if data.len() == head_dim * 4 {
                unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const f32,
                        head_dim,
                    )
                    .to_vec()
                }
            } else if data.len() == head_dim * 2 {
                data.chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                    })
                    .collect()
            } else {
                vec![1.0; head_dim]
            }
        } else {
            vec![1.0; head_dim]
        };

        for token_idx in 0..n_tokens {
            for h in 0..n_heads {
                let head_start = token_idx * n_heads * head_dim + h * head_dim;
                let head_input = &input[head_start..head_start + head_dim];
                let ss: f32 = head_input.iter().map(|x| x * x).sum::<f32>()
                    / head_dim as f32;
                let scale = 1.0f32 / (ss + eps).sqrt();
                for d in 0..head_dim {
                    output[head_start + d] = head_input[d] * scale * weight[d];
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
            QuantFormat::Q8_0 => {
                // Use batched GEMV
                let nb_per_token = (in_features + QK8_0 - 1) / QK8_0;
                gemv::mul_mat_q4_0(
                    weight_data,
                    stride,
                    &self.quant_buf_batch.q8_0,
                    output,
                    out_features,
                    in_features,
                    n_tokens,
                    self.n_threads_batch,
                );
                Ok(())
            }
            QuantFormat::Q8_K => {
                // Use batched GEMV with tiled path for n_tokens >= 8
                let nb_per_token = (in_features + QK_K - 1) / QK_K;
                match tensor.quant_type {
                    QuantType::Q5_K => {
                        gemv::mul_mat_q5_k(
                            weight_data,
                            stride,
                            &self.quant_buf_batch.q8_k,
                            output,
                            out_features,
                            in_features,
                            n_tokens,
                            self.n_threads_batch,
                        );
                        Ok(())
                    }
                    QuantType::Q6_K => {
                        gemv::mul_mat_q6_k(
                            weight_data,
                            stride,
                            &self.quant_buf_batch.q8_k,
                            output,
                            out_features,
                            in_features,
                            n_tokens,
                            self.n_threads_batch,
                        );
                        Ok(())
                    }
                    QuantType::Q4_K => {
                        gemv::mul_mat_q4_k(
                            weight_data,
                            stride,
                            &self.quant_buf_batch.q8_k,
                            output,
                            out_features,
                            in_features,
                            n_tokens,
                            self.n_threads_batch,
                        );
                        Ok(())
                    }
                    _ => Err(format!(
                        "Unsupported weight type for Q8_K batch: {:?}",
                        tensor.quant_type
                    )),
                }
            }
            QuantFormat::None => Err("No quantized input available".into()),
        }
    }

    /// Batch linear with F32 input
    /// Handles quantized weights by quantizing input into a local buffer
    fn linear_batch(
        &self,
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

        match tensor.quant_type {
            QuantType::F16 => {
                let stride_f16 = stride / 2;
                for token_idx in 0..n_tokens {
                    let in_start = token_idx * in_features;
                    let out_start = token_idx * out_features;
                    for i in 0..out_features {
                        let col_offset = i * stride_f16;
                        let mut sum = 0.0f32;
                        for j in 0..in_features {
                            let f16_val = half::f16::from_le_bytes([
                                weight_data[(col_offset + j) * 2],
                                weight_data[(col_offset + j) * 2 + 1],
                            ]);
                            sum += f16_val.to_f32() * input[in_start + j];
                        }
                        output[out_start + i] = sum;
                    }
                }
                Ok(())
            }
            QuantType::Q5_K | QuantType::Q6_K | QuantType::Q4_K => {
                // Quantize input into local buffer
                let nb_per_token = (in_features + QK_K - 1) / QK_K;
                let mut q8_k_local = vec![
                    unsafe { std::mem::zeroed() };
                    n_tokens * nb_per_token
                ];
                for token_idx in 0..n_tokens {
                    let in_start = token_idx * in_features;
                    let out_start = token_idx * nb_per_token;
                    quant::quantize_q8_k(
                        &input[in_start..in_start + in_features],
                        &mut q8_k_local[out_start..out_start + nb_per_token],
                    );
                }
                match tensor.quant_type {
                    QuantType::Q5_K => {
                        gemv::mul_mat_q5_k(
                            weight_data,
                            stride,
                            &q8_k_local,
                            output,
                            out_features,
                            in_features,
                            n_tokens,
                            self.n_threads_batch,
                        );
                        Ok(())
                    }
                    QuantType::Q6_K => {
                        gemv::mul_mat_q6_k(
                            weight_data,
                            stride,
                            &q8_k_local,
                            output,
                            out_features,
                            in_features,
                            n_tokens,
                            self.n_threads_batch,
                        );
                        Ok(())
                    }
                    QuantType::Q4_K => {
                        gemv::mul_mat_q4_k(
                            weight_data,
                            stride,
                            &q8_k_local,
                            output,
                            out_features,
                            in_features,
                            n_tokens,
                            self.n_threads_batch,
                        );
                        Ok(())
                    }
                    _ => unreachable!(),
                }
            }
            QuantType::Q4_0 | QuantType::Q8_0 => {
                let nb_per_token = (in_features + QK8_0 - 1) / QK8_0;
                let mut q8_0_local = vec![
                    BlockQ8_0 { d: 0, qs: [0; QK8_0] };
                    n_tokens * nb_per_token
                ];
                for token_idx in 0..n_tokens {
                    let in_start = token_idx * in_features;
                    let out_start = token_idx * nb_per_token;
                    quant::quantize_q8_0(
                        &input[in_start..in_start + in_features],
                        &mut q8_0_local[out_start..out_start + nb_per_token],
                    );
                }
                gemv::mul_mat_q4_0(
                    weight_data,
                    stride,
                    &q8_0_local,
                    output,
                    out_features,
                    in_features,
                    n_tokens,
                    self.n_threads_batch,
                );
                Ok(())
            }
            _ => Err(format!(
                "Unsupported weight type for linear_batch: {:?}",
                tensor.quant_type
            )),
        }
    }
}

/// Quantize input into the provided buffer
fn quantize_input_to(
    input: &[f32],
    quant_type: QuantType,
    quant_buf: &mut QuantizedInput,
) -> Result<QuantFormat, String> {
    match quant_type {
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            quant_buf.resize(input.len(), QuantFormat::Q8_0);
            quant::quantize_q8_0(input, &mut quant_buf.q8_0);
            Ok(QuantFormat::Q8_0)
        }
        QuantType::Q4_K
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::Q8_K
        | QuantType::IQ4_XS => {
            quant_buf.resize(input.len(), QuantFormat::Q8_K);
            quant::quantize_q8_k(input, &mut quant_buf.q8_k);
            Ok(QuantFormat::Q8_K)
        }
        _ => Err(format!("Unsupported quant type: {:?}", quant_type)),
    }
}

/// Batch quantize input: [n_tokens * hidden_size] -> QuantizedInputBatch
fn quantize_input_batch(
    input: &[f32],
    hidden_size: usize,
    n_tokens: usize,
    quant_type: QuantType,
    quant_buf: &mut QuantizedInputBatch,
) -> Result<QuantFormat, String> {
    match quant_type {
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            quant_buf.resize(n_tokens, hidden_size, QuantFormat::Q8_0);
            // Quantize each token's input
            let nb_per_token = (hidden_size + QK8_0 - 1) / QK8_0;
            for token_idx in 0..n_tokens {
                let in_start = token_idx * hidden_size;
                let out_start = token_idx * nb_per_token;
                quant::quantize_q8_0(
                    &input[in_start..in_start + hidden_size],
                    &mut quant_buf.q8_0[out_start..out_start + nb_per_token],
                );
            }
            Ok(QuantFormat::Q8_0)
        }
        QuantType::Q4_K
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::Q8_K
        | QuantType::IQ4_XS => {
            quant_buf.resize(n_tokens, hidden_size, QuantFormat::Q8_K);
            // Quantize each token's input
            let nb_per_token = (hidden_size + QK_K - 1) / QK_K;
            for token_idx in 0..n_tokens {
                let in_start = token_idx * hidden_size;
                let out_start = token_idx * nb_per_token;
                quant::quantize_q8_k(
                    &input[in_start..in_start + hidden_size],
                    &mut quant_buf.q8_k[out_start..out_start + nb_per_token],
                );
            }
            Ok(QuantFormat::Q8_K)
        }
        _ => Err(format!("Unsupported quant type for batch: {:?}", quant_type)),
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
        let hidden_size = config.hidden_size;

        let norm_weight = self
            .model
            .get_tensor_meta("output_norm.weight")
            .ok_or("output_norm.weight not found")?;

        let weight_data = self
            .model
            .get_tensor("output_norm.weight")
            .ok_or("output_norm weight data not found")?;

        match norm_weight.quant_type {
            QuantType::F32 => {
                // F32 weight - directly cast bytes to f32 slice
                let weight: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr() as *const f32,
                        hidden_size,
                    )
                };
                rms_norm_f32(
                    &self.hidden_buf.clone(),
                    weight,
                    &mut self.hidden_buf,
                    config.rms_norm_eps,
                );
            }
            QuantType::F16 => {
                // F16 weight - process without allocation
                let eps = config.rms_norm_eps;
                let sum_sq: f32 = self.hidden_buf.iter().map(|x| x * x).sum();
                let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
                let scale = 1.0 / rms;
                for (i, chunk) in weight_data.chunks_exact(2).enumerate() {
                    let w =
                        half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                    self.hidden_buf[i] *= scale * w;
                }
            }
            _ => return Err("Unsupported norm quant type".into()),
        }
        Ok(())
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
        let stride = output_tensor.strides[1] as usize;

        // Quantize hidden_buf
        let quant_type = output_tensor.quant_type;
        if matches!(
            quant_type,
            QuantType::Q4_K | QuantType::Q5_K | QuantType::Q6_K
        ) {
            self.quant_buf.resize(hidden_size, QuantFormat::Q8_K);
            quant::quantize_q8_k(&self.hidden_buf, &mut self.quant_buf.q8_k);
            linear::linear_q8_k(
                &self.quant_buf.q8_k,
                hidden_size,
                output_data,
                quant_type,
                stride,
                &mut self.logits_buf,
                self.n_threads,
            )?;
        } else if matches!(quant_type, QuantType::Q4_0 | QuantType::Q8_0) {
            self.quant_buf.resize(hidden_size, QuantFormat::Q8_0);
            quant::quantize_q8_0(&self.hidden_buf, &mut self.quant_buf.q8_0);
            linear::linear_q8_0(
                &self.quant_buf.q8_0,
                hidden_size,
                output_data,
                quant_type,
                stride,
                &mut self.logits_buf,
                self.n_threads,
            )?;
        } else {
            linear::linear_f32(
                &self.hidden_buf,
                output_data,
                quant_type,
                stride,
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
                    eprintln!("Prefill error at token {}", i);
                    return tokens;
                }
            }
        }

        for _ in 0..max_new_tokens {
            let next_token = {
                let logits = &self.logits_buf;
                self.sample(logits, temperature)
            };
            tokens.push(next_token);

            if self.forward(next_token).is_err() {
                break;
            }
        }

        tokens
    }
}
