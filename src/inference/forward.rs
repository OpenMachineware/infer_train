// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Forward executor for CPU inference

use crate::backend::arm::{
    flash_attn::flash_attn_f32,
    quant,
    rms_norm::rms_norm_f32,
    rope::{rope_f32, rope_f32_inplace},
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

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    None,
    Q8_0,
    Q8_K,
}

impl QuantizedInput {
    pub fn new() -> Self {
        Self { q8_0: Vec::new(), q8_k: Vec::new(), active: QuantFormat::None }
    }

    /// Resize buffers for given input size
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

/// Forward executor for single-token generation
pub struct ForwardExecutor {
    pub model: SimpleModel,
    /// KV cache: [layer][seq_pos][head][dim] for key and value
    pub kv_cache_k: Vec<Vec<f32>>,
    pub kv_cache_v: Vec<Vec<f32>>,
    /// Current KV cache capacity (in tokens)
    kv_capacity: usize,
    /// Current sequence position
    pub seq_pos: usize,
    /// Pre-allocated quantized input buffer
    quant_buf: QuantizedInput,
    /// Compute arena for zero-allocation temporary buffers
    arena: crate::memory::ComputeArena,
    /// Pre-allocated hidden buffer for layer-to-layer passing
    hidden_buf: Vec<f32>,
    /// Pre-allocated logits buffer for output
    logits_buf: Vec<f32>,
    /// Thread count for decode (n_tokens=1), 0 = auto
    pub n_threads: usize,
    /// Thread count for prefill (n_tokens>1), 0 = auto
    pub n_threads_batch: usize,
}

/// Layer trace for debugging
#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub layer_idx: usize,
    pub hidden_before: Vec<f32>,
    pub hidden_after: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub ffn_output: Vec<f32>,
    // FFN intermediate outputs for debugging
    pub ffn_normed: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_hidden: Vec<f32>,
}

/// Full forward trace for debugging
#[derive(Debug, Clone)]
pub struct ForwardTrace {
    pub token_id: u32,
    pub embedding: Vec<f32>,
    pub layer_traces: Vec<LayerTrace>,
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

impl ForwardExecutor {
    /// Create with default thread settings (auto)
    pub fn new(model: SimpleModel) -> Self {
        Self::new_with_threads(model, 0, 0)
    }

    /// Create with specified thread counts
    ///
    /// # Arguments
    /// * `n_threads` - Thread count for decode (n_tokens=1), 0 = auto
    /// * `n_threads_batch` - Thread count for prefill (n_tokens>1), 0 = auto
    pub fn new_with_threads(
        model: SimpleModel,
        n_threads: usize,
        n_threads_batch: usize,
    ) -> Self {
        let config = model.config().clone();
        let n_layers = config.n_layers;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();
        let _context_length = config.context_length;

        // Initialize KV cache with small initial capacity (512 tokens)
        // Will grow dynamically as needed
        const INITIAL_CAPACITY: usize = 512;
        let kv_size = n_kv_heads * head_dim * INITIAL_CAPACITY;

        // Calculate work buffer size for arena
        let work_size = crate::memory::calculate_work_size(
            config.hidden_size,
            config.intermediate_size,
            config.n_heads,
            config.n_kv_heads,
            head_dim,
            config.vocab_size,
            n_layers,
        );
        let arena = crate::memory::ComputeArena::new(work_size);

        // Pre-allocate output buffers
        let hidden_buf = vec![0.0f32; config.hidden_size];
        let logits_buf = vec![0.0f32; config.vocab_size];

        Self {
            model,
            kv_cache_k: vec![vec![0.0f32; kv_size]; n_layers],
            kv_cache_v: vec![vec![0.0f32; kv_size]; n_layers],
            kv_capacity: INITIAL_CAPACITY,
            seq_pos: 0,
            quant_buf: QuantizedInput::new(),
            arena,
            hidden_buf,
            logits_buf,
            n_threads,
            n_threads_batch,
        }
    }

    /// Ensure KV cache has enough capacity for the current sequence position
    fn ensure_kv_capacity(&mut self) {
        let config = self.model.config();
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();
        let context_length = config.context_length;

        // Check if we need to grow
        if self.seq_pos >= self.kv_capacity {
            // Double capacity, but don't exceed context_length
            let new_capacity = (self.kv_capacity * 2).min(context_length);
            let new_kv_size = n_kv_heads * head_dim * new_capacity;

            // Resize each layer's cache
            for layer in 0..self.kv_cache_k.len() {
                self.kv_cache_k[layer].resize(new_kv_size, 0.0);
                self.kv_cache_v[layer].resize(new_kv_size, 0.0);
            }

            self.kv_capacity = new_capacity;
        }
    }

    /// Get thread count for current operation
    /// - n_tokens = 1: decode, use n_threads
    /// - n_tokens > 1: prefill, use n_threads_batch
    pub fn get_n_threads(&self, n_tokens: usize) -> usize {
        if n_tokens <= 1 { self.n_threads } else { self.n_threads_batch }
    }

    /// Reset executor state for a new inference run
    /// Clears KV cache and resets sequence position
    pub fn reset(&mut self) {
        // Clear only used portion of KV cache
        let config = self.model.config();
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();
        let used_size =
            n_kv_heads * head_dim * self.seq_pos.min(self.kv_capacity);

        for layer in 0..self.kv_cache_k.len() {
            for i in 0..used_size {
                self.kv_cache_k[layer][i] = 0.0;
                self.kv_cache_v[layer][i] = 0.0;
            }
        }
        self.seq_pos = 0;
    }

    /// Forward pass for a single token
    /// Forward pass for a single token
    ///
    /// Returns a reference to the logits buffer (zero-allocation).
    /// The reference is valid until the next forward call.
    pub fn forward(&mut self, token_id: u32) -> Result<&[f32], String> {
        // Reset arena for new forward pass
        self.arena.reset();

        // Ensure KV cache has enough capacity
        self.ensure_kv_capacity();

        let config = self.model.config();

        // 1. Embedding lookup into hidden_buf
        {
            let emb_tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .ok_or("Embedding tensor not found")?;

            let emb_data =
                self.model.get_embeddings().ok_or("No embedding weights")?;

            let hidden_size = config.hidden_size;

            match emb_tensor.quant_type {
                QuantType::F32 => {
                    let nb1 = emb_tensor.strides[1] as usize;
                    let col_offset = (token_id as usize) * nb1;

                    for h in 0..hidden_size {
                        let offset = col_offset + h * 4;
                        if offset + 4 > emb_data.len() {
                            return Err(format!(
                                "Token {} out of range",
                                token_id
                            ));
                        }
                        self.hidden_buf[h] = f32::from_le_bytes(
                            emb_data[offset..offset + 4].try_into().unwrap(),
                        );
                    }
                }
                QuantType::F16 => {
                    let nb1 = emb_tensor.strides[1] as usize;
                    let col_offset = (token_id as usize) * nb1;

                    for h in 0..hidden_size {
                        let offset = col_offset + h * 2;
                        if offset + 2 > emb_data.len() {
                            return Err(format!(
                                "Token {} out of range",
                                token_id
                            ));
                        }
                        self.hidden_buf[h] = half::f16::from_le_bytes(
                            emb_data[offset..offset + 2].try_into().unwrap(),
                        )
                        .to_f32();
                    }
                }
                QuantType::Q6_K => {
                    // Q6_K embedding: each column is hidden_size elements
                    // hidden_size = 1024 = 4 * QK_K (256)
                    let nb = hidden_size / 256; // number of blocks per column
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
                        for h in 0..256 {
                            self.hidden_buf[h_base + h] = block_vals[h];
                        }
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported embedding quantization: {:?}",
                        emb_tensor.quant_type
                    ));
                }
            }
        }

        // 2. Process each layer in-place
        for layer_idx in 0..config.n_layers {
            self.forward_layer_inplace(layer_idx)?;
        }

        // 3. Final norm in-place
        self.final_norm_inplace()?;

        // 4. Output projection into logits_buf
        self.output_projection_inplace()?;

        // Increment sequence position
        self.seq_pos += 1;

        Ok(&self.logits_buf)
    }

    /// Public method for getting embedding (for debugging)
    pub fn get_embedding(&self, token_id: u32) -> Result<Vec<f32>, String> {
        self.embed_token(token_id)
    }

    /// Embedding lookup for a single token
    fn embed_token(&self, token_id: u32) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        // Find embedding tensor info
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
                // Column-major: token_embd.weight has dims [hidden_size, vocab_size]
                // Each column is one token's embedding
                let nb1 = emb_tensor.strides[1] as usize; // column stride in bytes
                let col_offset = (token_id as usize) * nb1;

                let mut embedding = vec![0.0f32; hidden_size];
                for h in 0..hidden_size {
                    let offset = col_offset + h * 4;
                    if offset + 4 > emb_data.len() {
                        return Err(format!("Token {} out of range", token_id));
                    }
                    embedding[h] = f32::from_le_bytes(
                        emb_data[offset..offset + 4].try_into().unwrap(),
                    );
                }
                Ok(embedding)
            }
            QuantType::F16 => {
                // Column-major: token_embd.weight has dims [hidden_size, vocab_size]
                // Each column is one token's embedding
                // Column stride (nb1) = hidden_size * 2 (size of F16)
                let nb1 = emb_tensor.strides[1] as usize; // column stride in bytes
                let col_offset = (token_id as usize) * nb1;

                let mut embedding = vec![0.0f32; hidden_size];
                for h in 0..hidden_size {
                    let offset = col_offset + h * 2; // within the column
                    if offset + 2 > emb_data.len() {
                        return Err(format!(
                            "Token {} out of range at h={}",
                            token_id, h
                        ));
                    }
                    let bytes = &emb_data[offset..offset + 2];
                    embedding[h] =
                        half::f16::from_le_bytes(bytes.try_into().unwrap())
                            .to_f32();
                }
                Ok(embedding)
            }
            QuantType::Q4_K => {
                // Column-major: token_embd.weight has dims [hidden_size, vocab_size]
                // Each column is one token's embedding
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let nb1 = emb_tensor.strides[1] as usize; // column stride in bytes
                let col_offset = (token_id as usize) * nb1;

                if col_offset + nb * std::mem::size_of::<BlockQ4_K>()
                    > emb_data.len()
                {
                    return Err(format!("Token {} out of range", token_id));
                }

                let mut embedding = vec![0.0f32; hidden_size];
                let col_data = &emb_data[col_offset..];

                crate::backend::arm::dequant::dequant_q4_k(
                    col_data,
                    &mut embedding,
                    nb * QK_K,
                );

                Ok(embedding)
            }
            QuantType::Q6_K => {
                // Column-major: token_embd.weight has dims [hidden_size, vocab_size]
                // Each column is one token's embedding
                // Use strides to access the correct column
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let nb1 = emb_tensor.strides[1] as usize; // column stride in bytes
                let col_offset = (token_id as usize) * nb1;

                if col_offset + nb * std::mem::size_of::<BlockQ6_K>()
                    > emb_data.len()
                {
                    return Err(format!("Token {} out of range", token_id));
                }

                let mut embedding = vec![0.0f32; hidden_size];
                let col_data = &emb_data[col_offset..];

                for i in 0..nb {
                    let block_offset = i * std::mem::size_of::<BlockQ6_K>();
                    let block_bytes = &col_data[block_offset
                        ..block_offset + std::mem::size_of::<BlockQ6_K>()];
                    let _dequant = crate::backend::arm::dequant::dequant_q6_k(
                        block_bytes,
                        &mut embedding[i * QK_K..],
                        QK_K.min(hidden_size - i * QK_K),
                    );
                }

                Ok(embedding)
            }
            _ => Err(format!(
                "Unsupported embedding quantization: {:?}",
                emb_tensor.quant_type
            )),
        }
    }

    /// Embedding lookup into provided buffer (zero-allocation version)
    fn embed_token_into(
        &self,
        token_id: u32,
        output: &mut [f32],
    ) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        if output.len() != hidden_size {
            return Err(format!(
                "Output buffer size {} != hidden_size {}",
                output.len(),
                hidden_size
            ));
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

        match emb_tensor.quant_type {
            QuantType::F32 => {
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;

                for h in 0..hidden_size {
                    let offset = col_offset + h * 4;
                    if offset + 4 > emb_data.len() {
                        return Err(format!("Token {} out of range", token_id));
                    }
                    output[h] = f32::from_le_bytes(
                        emb_data[offset..offset + 4].try_into().unwrap(),
                    );
                }
            }
            QuantType::F16 => {
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;

                for h in 0..hidden_size {
                    let offset = col_offset + h * 2;
                    if offset + 2 > emb_data.len() {
                        return Err(format!(
                            "Token {} out of range at h={}",
                            token_id, h
                        ));
                    }
                    let bytes = &emb_data[offset..offset + 2];
                    output[h] =
                        half::f16::from_le_bytes(bytes.try_into().unwrap())
                            .to_f32();
                }
            }
            QuantType::Q4_K => {
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;

                if col_offset + nb * std::mem::size_of::<BlockQ4_K>()
                    > emb_data.len()
                {
                    return Err(format!("Token {} out of range", token_id));
                }

                let col_data = &emb_data[col_offset..];
                crate::backend::arm::dequant::dequant_q4_k(
                    col_data,
                    output,
                    nb * QK_K,
                );
            }
            QuantType::Q6_K => {
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let nb1 = emb_tensor.strides[1] as usize;
                let col_offset = (token_id as usize) * nb1;

                if col_offset + nb * std::mem::size_of::<BlockQ6_K>()
                    > emb_data.len()
                {
                    return Err(format!("Token {} out of range", token_id));
                }

                let col_data = &emb_data[col_offset..];
                crate::backend::arm::dequant::dequant_q6_k(
                    col_data,
                    output,
                    nb * QK_K,
                );
            }
            _ => {
                return Err(format!(
                    "Unsupported embedding quantization: {:?}",
                    emb_tensor.quant_type
                ));
            }
        }

        Ok(())
    }

    /// Forward pass for a single layer
    pub fn forward_layer(
        &mut self,
        input: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>, String> {
        let config = self.model.config().clone();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();

        // === Attention Block ===
        // 1. RMS Norm before attention
        let normed_attn = self.rms_norm(
            input,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            hidden_size,
        )?;

        // 2. Pre-quantize normed_attn for QKV projections (shared input)
        // Use pre-allocated buffer to avoid repeated Vec allocation
        let quant_type = self
            .get_weight_quant_type(&format!("blk.{}.attn_q.weight", layer_idx));
        let (qkv_q8_0, qkv_q8_k, in_features) = match quant_type {
            QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
                let nb = (normed_attn.len() + QK8_0 - 1) / QK8_0;
                self.quant_buf.resize(normed_attn.len(), QuantFormat::Q8_0);
                crate::backend::arm::quant::quantize_q8_0(
                    &normed_attn,
                    &mut self.quant_buf.q8_0,
                );
                (Some(&self.quant_buf.q8_0 as &[BlockQ8_0]), None, nb * QK8_0)
            }
            QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_K
            | QuantType::IQ4_XS => {
                let nb = (normed_attn.len() + QK_K - 1) / QK_K;
                self.quant_buf.resize(normed_attn.len(), QuantFormat::Q8_K);
                crate::backend::arm::quant::quantize_q8_k(
                    &normed_attn,
                    &mut self.quant_buf.q8_k,
                );
                (None, Some(&self.quant_buf.q8_k as &[BlockQ8_K]), nb * QK_K)
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type: {:?}",
                    quant_type
                ));
            }
        };

        // 3. QKV projections using pre-quantized input
        let q = self.linear_quantized_inner(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_q.weight", layer_idx),
            n_heads * head_dim,
        )?;
        let k = self.linear_quantized_inner(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_k.weight", layer_idx),
            n_kv_heads * head_dim,
        )?;
        let v = self.linear_quantized_inner(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_v.weight", layer_idx),
            n_kv_heads * head_dim,
        )?;

        // 4. Q/K norm (RMS norm per head, then multiply by shared weight)
        // llama.cpp: ggml_rms_norm per row (each head) then ggml_mul
        let q_normed = self.qk_norm(
            &q,
            &format!("blk.{}.attn_q_norm.weight", layer_idx),
            n_heads,
            head_dim,
        )?;
        let k_normed = self.qk_norm(
            &k,
            &format!("blk.{}.attn_k_norm.weight", layer_idx),
            n_kv_heads,
            head_dim,
        )?;

        // 5. Apply RoPE to Q and K
        let freq_base = config.rope_freq_base;
        let mut q_rope = q_normed.clone();
        let mut k_rope = k_normed.clone();
        for h in 0..n_heads {
            rope_f32(
                &q_normed[h * head_dim..(h + 1) * head_dim],
                &mut q_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                freq_base,
            );
        }
        for h in 0..n_kv_heads {
            rope_f32(
                &k_normed[h * head_dim..(h + 1) * head_dim],
                &mut k_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                freq_base,
            );
        }

        // 6. Update KV cache
        let kv_head_offset = self.seq_pos * head_dim;
        let kv_capacity = self.kv_capacity; // Use actual capacity, not max context_length
        for h in 0..n_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = h * kv_capacity * head_dim + kv_head_offset;
            self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k_rope[src_offset..src_offset + head_dim]);
            self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }

        // 7. Flash Attention
        let mut attn_output = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv_head_idx = h * n_kv_heads / n_heads;
            let k_cache = &self.kv_cache_k[layer_idx]
                [kv_head_idx * kv_capacity * head_dim..];
            let v_cache = &self.kv_cache_v[layer_idx]
                [kv_head_idx * kv_capacity * head_dim..];

            let k_slice = &k_cache[..(self.seq_pos + 1) * head_dim];
            let v_slice = &v_cache[..(self.seq_pos + 1) * head_dim];

            flash_attn_f32(
                &q_rope[h * head_dim..(h + 1) * head_dim],
                k_slice,
                v_slice,
                None,
                &mut attn_output[h * head_dim..(h + 1) * head_dim],
                head_dim,
                self.seq_pos + 1,
                1.0 / (head_dim as f32).sqrt(),
            );
        }

        // 8. Output projection
        let attn_out = self.linear(
            &attn_output,
            &format!("blk.{}.attn_output.weight", layer_idx),
            hidden_size,
        )?;

        // Residual connection
        let hidden: Vec<f32> =
            input.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // === FFN Block ===
        // 1. RMS Norm before FFN
        let normed_ffn = self.rms_norm(
            &hidden,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            hidden_size,
        )?;

        // 2. Pre-quantize normed_ffn for gate/up projections (shared input)
        let quant_type = self.get_weight_quant_type(&format!(
            "blk.{}.ffn_gate.weight",
            layer_idx
        ));
        let (gate_up_q8_0, gate_up_q8_k, ffn_in_features) = match quant_type {
            QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
                let nb = (normed_ffn.len() + QK8_0 - 1) / QK8_0;
                self.quant_buf.resize(normed_ffn.len(), QuantFormat::Q8_0);
                crate::backend::arm::quant::quantize_q8_0(
                    &normed_ffn,
                    &mut self.quant_buf.q8_0,
                );
                (Some(&self.quant_buf.q8_0 as &[BlockQ8_0]), None, nb * QK8_0)
            }
            QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_K
            | QuantType::IQ4_XS => {
                let nb = (normed_ffn.len() + QK_K - 1) / QK_K;
                self.quant_buf.resize(normed_ffn.len(), QuantFormat::Q8_K);
                crate::backend::arm::quant::quantize_q8_k(
                    &normed_ffn,
                    &mut self.quant_buf.q8_k,
                );
                (None, Some(&self.quant_buf.q8_k as &[BlockQ8_K]), nb * QK_K)
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type: {:?}",
                    quant_type
                ));
            }
        };

        // 3. Gate and Up projections using pre-quantized input
        let gate = self.linear_quantized_inner(
            gate_up_q8_0,
            gate_up_q8_k,
            ffn_in_features,
            &format!("blk.{}.ffn_gate.weight", layer_idx),
            config.intermediate_size,
        )?;
        let up = self.linear_quantized_inner(
            gate_up_q8_0,
            gate_up_q8_k,
            ffn_in_features,
            &format!("blk.{}.ffn_up.weight", layer_idx),
            config.intermediate_size,
        )?;

        // 4. SwiGLU activation
        let mut ffn_hidden = vec![0.0f32; config.intermediate_size];
        crate::backend::arm::activation::silu_f32(&gate, &mut ffn_hidden);
        for i in 0..config.intermediate_size {
            ffn_hidden[i] *= up[i];
        }

        // 5. Down projection
        let ffn_out = self.linear(
            &ffn_hidden,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            hidden_size,
        )?;

        // Add residual
        let output: Vec<f32> =
            hidden.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        Ok(output)
    }

    /// Forward pass for a single layer in-place (zero-allocation version)
    fn forward_layer_inplace(
        &mut self,
        layer_idx: usize,
    ) -> Result<(), String> {
        // Reset arena at the start of each layer
        // This allows reuse of memory across layers
        self.arena.reset();

        let config = self.model.config().clone();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();

        // === Attention Block ===
        // 1. RMS Norm before attention - use arena for temp buffer
        let normed_attn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.rms_norm_into(
            &self.hidden_buf,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            normed_attn,
        )?;

        // 2. Pre-quantize normed_attn for QKV projections (shared input)
        let quant_type = self
            .get_weight_quant_type(&format!("blk.{}.attn_q.weight", layer_idx));
        let (qkv_q8_0, qkv_q8_k, in_features) = match quant_type {
            QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
                let nb = (normed_attn.len() + QK8_0 - 1) / QK8_0;
                self.quant_buf.resize(normed_attn.len(), QuantFormat::Q8_0);
                crate::backend::arm::quant::quantize_q8_0(
                    normed_attn,
                    &mut self.quant_buf.q8_0,
                );
                (Some(&self.quant_buf.q8_0 as &[BlockQ8_0]), None, nb * QK8_0)
            }
            QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_K
            | QuantType::IQ4_XS => {
                let nb = (normed_attn.len() + QK_K - 1) / QK_K;
                self.quant_buf.resize(normed_attn.len(), QuantFormat::Q8_K);
                crate::backend::arm::quant::quantize_q8_k(
                    normed_attn,
                    &mut self.quant_buf.q8_k,
                );
                (None, Some(&self.quant_buf.q8_k as &[BlockQ8_K]), nb * QK_K)
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type: {:?}",
                    quant_type
                ));
            }
        };

        // 3. QKV projections - use arena for temp buffers
        let q: &mut [f32] = self.arena.alloc_zeroed(n_heads * head_dim);
        let k: &mut [f32] = self.arena.alloc_zeroed(n_kv_heads * head_dim);
        let v: &mut [f32] = self.arena.alloc_zeroed(n_kv_heads * head_dim);

        self.linear_quantized_into(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_q.weight", layer_idx),
            q,
        )?;
        self.linear_quantized_into(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_k.weight", layer_idx),
            k,
        )?;
        self.linear_quantized_into(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_v.weight", layer_idx),
            v,
        )?;

        // 4. Q/K norm (RMS norm per head)
        let q_normed: &mut [f32] = self.arena.alloc_zeroed(n_heads * head_dim);
        let k_normed: &mut [f32] =
            self.arena.alloc_zeroed(n_kv_heads * head_dim);

        self.qk_norm_into(
            q,
            &format!("blk.{}.attn_q_norm.weight", layer_idx),
            n_heads,
            head_dim,
            q_normed,
        )?;
        self.qk_norm_into(
            k,
            &format!("blk.{}.attn_k_norm.weight", layer_idx),
            n_kv_heads,
            head_dim,
            k_normed,
        )?;

        // 5. Apply RoPE to Q and K (in-place on normed buffers)
        let freq_base = config.rope_freq_base;
        rope_f32_inplace(q_normed, self.seq_pos, head_dim, head_dim, freq_base);
        rope_f32_inplace(k_normed, self.seq_pos, head_dim, head_dim, freq_base);

        // 6. Update KV cache
        let kv_head_offset = self.seq_pos * head_dim;
        let kv_capacity = self.kv_capacity;
        for h in 0..n_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = h * kv_capacity * head_dim + kv_head_offset;
            self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k_normed[src_offset..src_offset + head_dim]);
            self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }

        // 7. Flash Attention - use arena for output
        let attn_output: &mut [f32] =
            self.arena.alloc_zeroed(n_heads * head_dim);
        for h in 0..n_heads {
            let kv_head_idx = h * n_kv_heads / n_heads;
            let k_cache = &self.kv_cache_k[layer_idx]
                [kv_head_idx * kv_capacity * head_dim..];
            let v_cache = &self.kv_cache_v[layer_idx]
                [kv_head_idx * kv_capacity * head_dim..];

            let k_slice = &k_cache[..(self.seq_pos + 1) * head_dim];
            let v_slice = &v_cache[..(self.seq_pos + 1) * head_dim];

            flash_attn_f32(
                &q_normed[h * head_dim..(h + 1) * head_dim],
                k_slice,
                v_slice,
                None,
                &mut attn_output[h * head_dim..(h + 1) * head_dim],
                head_dim,
                self.seq_pos + 1,
                1.0 / (head_dim as f32).sqrt(),
            );
        }

        // 8. Output projection - write to arena temp, then add residual to hidden_buf
        let attn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear_into(
            attn_output,
            &format!("blk.{}.attn_output.weight", layer_idx),
            attn_out,
        )?;

        // Residual connection: hidden_buf = hidden_buf + attn_out
        for i in 0..hidden_size {
            self.hidden_buf[i] += attn_out[i];
        }

        // === FFN Block ===
        // 1. RMS Norm before FFN - use arena temp
        let normed_ffn: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.rms_norm_into(
            &self.hidden_buf,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            normed_ffn,
        )?;

        // 2. Pre-quantize normed_ffn for gate/up projections
        let (gate_up_q8_0, gate_up_q8_k, ffn_in_features) = match quant_type {
            QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
                let nb = (normed_ffn.len() + QK8_0 - 1) / QK8_0;
                self.quant_buf.resize(normed_ffn.len(), QuantFormat::Q8_0);
                crate::backend::arm::quant::quantize_q8_0(
                    normed_ffn,
                    &mut self.quant_buf.q8_0,
                );
                (Some(&self.quant_buf.q8_0 as &[BlockQ8_0]), None, nb * QK8_0)
            }
            QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_K
            | QuantType::IQ4_XS => {
                let nb = (normed_ffn.len() + QK_K - 1) / QK_K;
                self.quant_buf.resize(normed_ffn.len(), QuantFormat::Q8_K);
                crate::backend::arm::quant::quantize_q8_k(
                    normed_ffn,
                    &mut self.quant_buf.q8_k,
                );
                (None, Some(&self.quant_buf.q8_k as &[BlockQ8_K]), nb * QK_K)
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type: {:?}",
                    quant_type
                ));
            }
        };

        // 3. Gate and Up projections - use arena
        let gate: &mut [f32] =
            self.arena.alloc_zeroed(config.intermediate_size);
        let up: &mut [f32] = self.arena.alloc_zeroed(config.intermediate_size);

        self.linear_quantized_into(
            gate_up_q8_0,
            gate_up_q8_k,
            ffn_in_features,
            &format!("blk.{}.ffn_gate.weight", layer_idx),
            gate,
        )?;
        self.linear_quantized_into(
            gate_up_q8_0,
            gate_up_q8_k,
            ffn_in_features,
            &format!("blk.{}.ffn_up.weight", layer_idx),
            up,
        )?;

        // 4. SwiGLU activation - in-place on gate buffer
        crate::backend::arm::activation::silu_f32_inplace(gate);
        for i in 0..config.intermediate_size {
            gate[i] *= up[i];
        }

        // 5. Down projection - use arena temp
        let ffn_out: &mut [f32] = self.arena.alloc_zeroed(hidden_size);
        self.linear_into(
            gate,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            ffn_out,
        )?;

        // Add residual: hidden_buf = hidden_buf + ffn_out
        for i in 0..hidden_size {
            self.hidden_buf[i] += ffn_out[i];
        }

        Ok(())
    }

    /// Get the quantization type of a weight tensor
    fn get_weight_quant_type(&self, weight_name: &str) -> QuantType {
        self.model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == weight_name)
            .map(|t| t.quant_type)
            .unwrap_or(QuantType::F32)
    }

    /// Inner implementation for linear with quantized input
    fn linear_quantized_inner(
        &self,
        q8_0: Option<&[BlockQ8_0]>,
        q8_k: Option<&[BlockQ8_K]>,
        in_features: usize,
        weight_name: &str,
        out_features: usize,
    ) -> Result<Vec<f32>, String> {
        let tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        let nb1 = tensor.strides.get(1).copied().unwrap_or(0) as usize;

        let mut output = vec![0.0f32; out_features];

        match tensor.quant_type {
            QuantType::Q6_K => {
                if let Some(input_q8k) = q8_k {
                    crate::backend::arm::gemv::gemv_q6_k_transposed(
                        weight_data,
                        nb1,
                        input_q8k,
                        &mut output,
                        out_features,
                        in_features,
                        self.n_threads,
                    );
                }
            }
            QuantType::Q5_K => {
                if let Some(input_q8k) = q8_k {
                    crate::backend::arm::gemv::gemv_q5_k_transposed(
                        weight_data,
                        nb1,
                        input_q8k,
                        &mut output,
                        out_features,
                        in_features,
                        self.n_threads,
                    );
                }
            }
            QuantType::Q4_K => {
                if let Some(input_q8k) = q8_k {
                    crate::backend::arm::gemv::gemv_q4_k_transposed(
                        weight_data,
                        nb1,
                        input_q8k,
                        &mut output,
                        out_features,
                        in_features,
                        self.n_threads,
                    );
                }
            }
            QuantType::Q4_0 => {
                if let Some(input_q80) = q8_0 {
                    crate::backend::arm::gemv::gemv_q4_0_transposed(
                        weight_data,
                        nb1,
                        input_q80,
                        &mut output,
                        out_features,
                        in_features,
                        self.n_threads,
                    );
                }
            }
            QuantType::IQ4_XS => {
                if let Some(input_q8k) = q8_k {
                    let nb = (in_features + QK_K - 1) / QK_K;
                    for i in 0..out_features {
                        let col_offset = i * nb1;
                        let col_blocks: &[BlockIQ4XS] = unsafe {
                            std::slice::from_raw_parts(
                                weight_data.as_ptr().add(col_offset)
                                    as *const BlockIQ4XS,
                                nb,
                            )
                        };
                        output[i] =
                            crate::backend::arm::vec_dot::vec_dot_iq4_xs_q8_k(
                                col_blocks, input_q8k,
                            );
                    }
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type for linear_quantized: {:?}",
                    tensor.quant_type
                ));
            }
        }

        Ok(output)
    }

    /// RMS normalization
    fn rms_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        size: usize,
    ) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let eps = config.rms_norm_eps;
        let weight_data = self.model.get_tensor(weight_name);
        let mut output = vec![0.0f32; size];

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / size as f32 + eps).sqrt();

        if let Some(data) = weight_data {
            if data.len() == size * 4 {
                let weight: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                rms_norm_f32(input, &weight, &mut output, eps);
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

        Ok(output)
    }

    /// RMS norm writing to provided output buffer (zero-allocation version)
    fn rms_norm_into(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let size = input.len();
        let config = self.model.config();
        let eps = config.rms_norm_eps;
        let weight_data = self.model.get_tensor(weight_name);

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / size as f32 + eps).sqrt();

        if let Some(data) = weight_data {
            if data.len() == size * 4 {
                let weight: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                rms_norm_f32(input, &weight, output, eps);
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

    /// Q/K norm: RMS norm per head, then multiply by shared weight
    /// llama.cpp: ggml_rms_norm computes RMS per row (each head separately)
    /// Input shape: [n_heads, head_dim], weight shape: [head_dim] (broadcast)
    pub fn qk_norm(
        &self,
        input: &[f32],
        weight_name: &str,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let total_dim = n_heads * head_dim;
        let mut output = vec![0.0f32; total_dim];

        // Weight is shared across heads: [head_dim]
        let weight_data = self.model.get_tensor(weight_name);
        let shared_weight: Vec<f32> = if let Some(data) = weight_data {
            if data.len() == head_dim * 4 {
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            } else if data.len() == head_dim * 2 {
                data.chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect()
            } else {
                output.copy_from_slice(input);
                return Ok(output);
            }
        } else {
            output.copy_from_slice(input);
            return Ok(output);
        };

        let eps = self.model.config.rms_norm_eps;

        // llama.cpp reshapes Qcur to [head_dim, n_heads, n_tokens]
        // Then ggml_rms_norm normalizes each row (each head's head_dim elements)
        // This is equivalent to per-head RMS norm
        for h in 0..n_heads {
            let head_start = h * head_dim;
            let head_input = &input[head_start..head_start + head_dim];

            // Compute RMS for this head only
            let ss: f32 =
                head_input.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let scale = 1.0f32 / (ss + eps).sqrt();

            // Normalize and multiply by weight
            for d in 0..head_dim {
                output[head_start + d] =
                    head_input[d] * scale * shared_weight[d];
            }
        }

        Ok(output)
    }

    /// Q/K norm writing to provided output buffer (zero-allocation version)
    fn qk_norm_into(
        &self,
        input: &[f32],
        weight_name: &str,
        n_heads: usize,
        head_dim: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        let eps = self.model.config.rms_norm_eps;

        // Weight is shared across heads: [head_dim]
        let weight_data = self.model.get_tensor(weight_name);
        let shared_weight: Vec<f32> = if let Some(data) = weight_data {
            if data.len() == head_dim * 4 {
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            } else if data.len() == head_dim * 2 {
                data.chunks_exact(2)
                    .map(|chunk| {
                        half::f16::from_le_bytes(chunk.try_into().unwrap())
                            .to_f32()
                    })
                    .collect()
            } else {
                output.copy_from_slice(input);
                return Ok(());
            }
        } else {
            output.copy_from_slice(input);
            return Ok(());
        };

        // Per-head RMS norm
        for h in 0..n_heads {
            let head_start = h * head_dim;
            let head_input = &input[head_start..head_start + head_dim];

            let ss: f32 =
                head_input.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let scale = 1.0f32 / (ss + eps).sqrt();

            for d in 0..head_dim {
                output[head_start + d] =
                    head_input[d] * scale * shared_weight[d];
            }
        }

        Ok(())
    }

    /// Linear layer (GEMV) with quantized weights
    /// GGUF stores weights in column-major: tensor dims [ne[0], ne[1]] = [in_features, out_features]
    /// Each column is one output feature, so we use strides to access columns.
    pub fn linear(
        &self,
        input: &[f32],
        weight_name: &str,
        out_features: usize,
    ) -> Result<Vec<f32>, String> {
        let in_features = input.len();

        let tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        // Get strides from tensor info
        let nb1 = tensor.strides.get(1).copied().unwrap_or(0) as usize;

        let mut output = vec![0.0f32; out_features];

        match tensor.quant_type {
            QuantType::F16 => {
                // F16 is not quantized, stride is in elements
                // Column i starts at offset i * nb1 / 2 (since F16 is 2 bytes)
                let stride_f16 = nb1 / 2; // elements per column
                for i in 0..out_features {
                    let col_offset = i * stride_f16;
                    let mut sum = 0.0f32;
                    for j in 0..in_features {
                        let f16_val = half::f16::from_le_bytes([
                            weight_data[(col_offset + j) * 2],
                            weight_data[(col_offset + j) * 2 + 1],
                        ]);
                        sum += f16_val.to_f32() * input[j];
                    }
                    output[i] = sum;
                }
            }
            QuantType::Q6_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                // Use multi-threaded transposed GEMV
                crate::backend::arm::gemv::gemv_q6_k_transposed(
                    weight_data,
                    nb1,
                    &input_q8k,
                    &mut output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q5_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                crate::backend::arm::gemv::gemv_q5_k_transposed(
                    weight_data,
                    nb1,
                    &input_q8k,
                    &mut output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q4_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                crate::backend::arm::gemv::gemv_q4_k_transposed(
                    weight_data,
                    nb1,
                    &input_q8k,
                    &mut output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::IQ4_XS => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                for i in 0..out_features {
                    let col_offset = i * nb1;
                    let col_blocks: &[BlockIQ4XS] = unsafe {
                        std::slice::from_raw_parts(
                            weight_data.as_ptr().add(col_offset)
                                as *const BlockIQ4XS,
                            nb,
                        )
                    };
                    output[i] =
                        crate::backend::arm::vec_dot::vec_dot_iq4_xs_q8_k(
                            col_blocks, &input_q8k,
                        );
                }
            }
            QuantType::Q4_0 => {
                let nb = (in_features + QK4_0 - 1) / QK4_0;
                let mut input_q80: Vec<BlockQ8_0> =
                    vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
                crate::backend::arm::quant::quantize_q8_0(
                    input,
                    &mut input_q80,
                );

                crate::backend::arm::gemv::gemv_q4_0_transposed(
                    weight_data,
                    nb1,
                    &input_q80,
                    &mut output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q8_0 => {
                let nb = (in_features + QK8_0 - 1) / QK8_0;
                let mut input_q80: Vec<BlockQ8_0> =
                    vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
                crate::backend::arm::quant::quantize_q8_0(
                    input,
                    &mut input_q80,
                );

                let nb1 = tensor.strides[1] as usize;

                for i in 0..out_features {
                    let col_offset = i * nb1;
                    let col_blocks: &[BlockQ8_0] = unsafe {
                        std::slice::from_raw_parts(
                            weight_data.as_ptr().add(col_offset)
                                as *const BlockQ8_0,
                            nb,
                        )
                    };
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q8_0(
                        col_blocks, &input_q80,
                    );
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type for linear: {:?}",
                    tensor.quant_type
                ));
            }
        }

        Ok(output)
    }

    /// Linear layer writing to provided output buffer (zero-allocation version)
    /// Quantizes input internally
    fn linear_into(
        &self,
        input: &[f32],
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let in_features = input.len();
        let out_features = output.len();

        let tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        let nb1 = tensor.strides.get(1).copied().unwrap_or(0) as usize;

        match tensor.quant_type {
            QuantType::F16 => {
                let stride_f16 = nb1 / 2;
                for i in 0..out_features {
                    let col_offset = i * stride_f16;
                    let mut sum = 0.0f32;
                    for j in 0..in_features {
                        let f16_val = half::f16::from_le_bytes([
                            weight_data[(col_offset + j) * 2],
                            weight_data[(col_offset + j) * 2 + 1],
                        ]);
                        sum += f16_val.to_f32() * input[j];
                    }
                    output[i] = sum;
                }
            }
            QuantType::Q6_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                crate::backend::arm::gemv::gemv_q6_k_transposed(
                    weight_data,
                    nb1,
                    &input_q8k,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q5_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                crate::backend::arm::gemv::gemv_q5_k_transposed(
                    weight_data,
                    nb1,
                    &input_q8k,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q4_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                let mut input_q8k: Vec<BlockQ8_K> =
                    vec![unsafe { std::mem::zeroed() }; nb];
                crate::backend::arm::quant::quantize_q8_k(
                    input,
                    &mut input_q8k,
                );

                crate::backend::arm::gemv::gemv_q4_k_transposed(
                    weight_data,
                    nb1,
                    &input_q8k,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q4_0 => {
                let nb = (in_features + QK4_0 - 1) / QK4_0;
                let mut input_q80: Vec<BlockQ8_0> =
                    vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
                crate::backend::arm::quant::quantize_q8_0(
                    input,
                    &mut input_q80,
                );

                crate::backend::arm::gemv::gemv_q4_0_transposed(
                    weight_data,
                    nb1,
                    &input_q80,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q8_0 => {
                let nb = (in_features + QK8_0 - 1) / QK8_0;
                let mut input_q80: Vec<BlockQ8_0> =
                    vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
                crate::backend::arm::quant::quantize_q8_0(
                    input,
                    &mut input_q80,
                );

                for i in 0..out_features {
                    let col_offset = i * nb1;
                    let col_blocks: &[BlockQ8_0] = unsafe {
                        std::slice::from_raw_parts(
                            weight_data.as_ptr().add(col_offset)
                                as *const BlockQ8_0,
                            nb,
                        )
                    };
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q8_0(
                        col_blocks, &input_q80,
                    );
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type for linear_into: {:?}",
                    tensor.quant_type
                ));
            }
        }

        Ok(())
    }

    /// Linear layer with pre-quantized input (zero-allocation version)
    /// Used when input is already quantized (shared across multiple projections)
    fn linear_quantized_into(
        &self,
        input_q8_0: Option<&[BlockQ8_0]>,
        input_q8_k: Option<&[BlockQ8_K]>,
        in_features: usize,
        weight_name: &str,
        output: &mut [f32],
    ) -> Result<(), String> {
        let out_features = output.len();

        let tensor = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == weight_name)
            .ok_or(format!("Weight '{}' not found", weight_name))?;

        let weight_data = self
            .model
            .get_tensor(weight_name)
            .ok_or(format!("Weight data '{}' not found", weight_name))?;

        let nb1 = tensor.strides.get(1).copied().unwrap_or(0) as usize;

        match tensor.quant_type {
            QuantType::Q6_K => {
                let input_q8k = input_q8_k.ok_or("Expected Q8_K input")?;
                crate::backend::arm::gemv::gemv_q6_k_transposed(
                    weight_data,
                    nb1,
                    input_q8k,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q5_K => {
                let input_q8k = input_q8_k.ok_or("Expected Q8_K input")?;
                crate::backend::arm::gemv::gemv_q5_k_transposed(
                    weight_data,
                    nb1,
                    input_q8k,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q4_K => {
                let input_q8k = input_q8_k.ok_or("Expected Q8_K input")?;
                crate::backend::arm::gemv::gemv_q4_k_transposed(
                    weight_data,
                    nb1,
                    input_q8k,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::IQ4_XS => {
                let input_q8k = input_q8_k.ok_or("Expected Q8_K input")?;
                let nb = (in_features + QK_K - 1) / QK_K;
                for i in 0..out_features {
                    let col_offset = i * nb1;
                    let col_blocks: &[BlockIQ4XS] = unsafe {
                        std::slice::from_raw_parts(
                            weight_data.as_ptr().add(col_offset)
                                as *const BlockIQ4XS,
                            nb,
                        )
                    };
                    output[i] =
                        crate::backend::arm::vec_dot::vec_dot_iq4_xs_q8_k(
                            col_blocks, input_q8k,
                        );
                }
            }
            QuantType::Q4_0 => {
                let input_q80 = input_q8_0.ok_or("Expected Q8_0 input")?;
                crate::backend::arm::gemv::gemv_q4_0_transposed(
                    weight_data,
                    nb1,
                    input_q80,
                    output,
                    out_features,
                    in_features,
                    self.n_threads,
                );
            }
            QuantType::Q8_0 => {
                let input_q80 = input_q8_0.ok_or("Expected Q8_0 input")?;
                let nb = (in_features + QK8_0 - 1) / QK8_0;
                for i in 0..out_features {
                    let col_offset = i * nb1;
                    let col_blocks: &[BlockQ8_0] = unsafe {
                        std::slice::from_raw_parts(
                            weight_data.as_ptr().add(col_offset)
                                as *const BlockQ8_0,
                            nb,
                        )
                    };
                    output[i] = crate::backend::arm::vec_dot::vec_dot_q8_0(
                        col_blocks, input_q80,
                    );
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type for linear_quantized_into: {:?}",
                    tensor.quant_type
                ));
            }
        }

        Ok(())
    }

    /// Final layer normalization
    pub fn final_norm(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        self.rms_norm(input, "output_norm.weight", config.hidden_size)
    }

    /// Final normalization in-place (zero-allocation version)
    fn final_norm_inplace(&mut self) -> Result<(), String> {
        let config = self.model.config();
        let hidden_size = config.hidden_size;

        // Get norm weight
        let norm_weight = self
            .model
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == "output_norm.weight")
            .ok_or("output_norm.weight not found")?;

        let weight_data = self
            .model
            .get_tensor("output_norm.weight")
            .ok_or("output_norm weight data not found")?;

        // Dequantize weight if needed
        let weight: Vec<f32> = match norm_weight.quant_type {
            QuantType::F32 => weight_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .take(hidden_size)
                .collect(),
            QuantType::F16 => weight_data
                .chunks_exact(2)
                .map(|chunk| {
                    half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f32()
                })
                .take(hidden_size)
                .collect(),
            _ => {
                return Err(format!(
                    "Unsupported norm weight quant: {:?}",
                    norm_weight.quant_type
                ));
            }
        };

        // Compute RMS norm in-place
        let epsilon = 1e-6f32;
        let ss: f32 = self.hidden_buf.iter().map(|x| x * x).sum::<f32>()
            / hidden_size as f32;
        let inv_rms = 1.0 / (ss + epsilon).sqrt();

        for i in 0..hidden_size {
            self.hidden_buf[i] = self.hidden_buf[i] * inv_rms * weight[i];
        }

        Ok(())
    }

    /// Output projection to logits
    pub fn output_projection(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<f32>, String> {
        let config = self.model.config();
        let vocab_size = config.vocab_size;
        let hidden_size = hidden.len();

        let (output_tensor, output_data) = if let Some(tensor) =
            self.model.gguf.tensors.iter().find(|t| {
                t.name == "output.weight" || t.name == "lm_head.weight"
            }) {
            let data = self
                .model
                .get_tensor(&tensor.name)
                .ok_or("Output weight data not found")?;
            (tensor, data)
        } else {
            let tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .ok_or("No output or embedding weights found")?;
            let data = self
                .model
                .get_tensor(&tensor.name)
                .ok_or("Embedding weight data not found")?;
            (tensor, data)
        };

        let in_features = hidden.len();
        let mut logits = vec![0.0f32; vocab_size];

        match output_tensor.quant_type {
            QuantType::Q6_K => {
                // GGUF stores weights column-major: dims [hidden_size, vocab_size]
                // Each column is one token's embedding for output projection
                // Use batched GEMV instead of per-column vec_dot for efficiency
                let nb = (in_features + QK_K - 1) / QK_K;
                let nb1 = output_tensor.strides[1] as usize; // column stride in bytes

                let mut input_q8k = vec![unsafe { std::mem::zeroed() }; nb];
                quant::quantize_row_q8_k(hidden, &mut input_q8k);

                crate::backend::arm::gemv::gemv_q6_k_transposed(
                    output_data,
                    nb1,
                    &input_q8k,
                    &mut logits,
                    vocab_size,
                    in_features,
                    0, // auto-detect thread count
                );
            }
            QuantType::Q4_K => {
                // GGUF stores weights column-major: dims [hidden_size, vocab_size]
                // Each column is one token's embedding for output projection
                // Use batched GEMV instead of per-column vec_dot for efficiency
                let nb = (in_features + QK_K - 1) / QK_K;
                let nb1 = output_tensor.strides[1] as usize; // column stride in bytes

                let mut input_q8k = vec![unsafe { std::mem::zeroed() }; nb];
                quant::quantize_row_q8_k(hidden, &mut input_q8k);

                crate::backend::arm::gemv::gemv_q4_k_transposed(
                    output_data,
                    nb1,
                    &input_q8k,
                    &mut logits,
                    vocab_size,
                    in_features,
                    0, // auto-detect thread count
                );
            }
            QuantType::Q5_K => {
                // GGUF stores weights column-major: dims [hidden_size, vocab_size]
                // Each column is one token's embedding for output projection
                // Use batched GEMV instead of per-column vec_dot for efficiency
                let nb = (in_features + QK_K - 1) / QK_K;
                let nb1 = output_tensor.strides[1] as usize; // column stride in bytes

                let mut input_q8k = vec![unsafe { std::mem::zeroed() }; nb];
                quant::quantize_row_q8_k(hidden, &mut input_q8k);

                crate::backend::arm::gemv::gemv_q5_k_transposed(
                    output_data,
                    nb1,
                    &input_q8k,
                    &mut logits,
                    vocab_size,
                    in_features,
                    0, // auto-detect thread count
                );
            }
            QuantType::F16 => {
                // GGUF stores weights column-major: dims [hidden_size, vocab_size]
                // Use strides to access columns
                let nb1 = output_tensor.strides[1] as usize / 2; // column stride in F16 elements

                for v in 0..vocab_size {
                    let mut sum = 0.0f32;
                    for h in 0..hidden_size {
                        let f16_val = half::f16::from_le_bytes([
                            output_data[(v * nb1 + h) * 2],
                            output_data[(v * nb1 + h) * 2 + 1],
                        ]);
                        sum += hidden[h] * f16_val.to_f32();
                    }
                    logits[v] = sum;
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported output quantization: {:?}",
                    output_tensor.quant_type
                ));
            }
        }

        Ok(logits)
    }

    /// Output projection in-place (zero-allocation version)
    fn output_projection_inplace(&mut self) -> Result<(), String> {
        let config = self.model.config();
        let vocab_size = config.vocab_size;
        let hidden_size = self.hidden_buf.len();

        let (output_tensor, output_data) = if let Some(tensor) =
            self.model.gguf.tensors.iter().find(|t| {
                t.name == "output.weight" || t.name == "lm_head.weight"
            }) {
            let data = self
                .model
                .get_tensor(&tensor.name)
                .ok_or("Output weight data not found")?;
            (tensor, data)
        } else {
            let tensor = self
                .model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .ok_or("No output or embedding weights found")?;
            let data = self
                .model
                .get_tensor(&tensor.name)
                .ok_or("Embedding weight data not found")?;
            (tensor, data)
        };

        // Zero logits buffer
        self.logits_buf.fill(0.0);

        match output_tensor.quant_type {
            QuantType::Q6_K => {
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let nb1 = output_tensor.strides[1] as usize;

                // Quantize input to Q8_K using pre-allocated buffer
                self.quant_buf.resize(hidden_size, QuantFormat::Q8_K);
                quant::quantize_row_q8_k(
                    &self.hidden_buf,
                    &mut self.quant_buf.q8_k,
                );

                crate::backend::arm::gemv::gemv_q6_k_transposed(
                    output_data,
                    nb1,
                    &self.quant_buf.q8_k,
                    &mut self.logits_buf,
                    vocab_size,
                    hidden_size,
                    self.n_threads,
                );
            }
            QuantType::Q4_K => {
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let nb1 = output_tensor.strides[1] as usize;

                self.quant_buf.resize(hidden_size, QuantFormat::Q8_K);
                quant::quantize_row_q8_k(
                    &self.hidden_buf,
                    &mut self.quant_buf.q8_k,
                );

                crate::backend::arm::gemv::gemv_q4_k_transposed(
                    output_data,
                    nb1,
                    &self.quant_buf.q8_k,
                    &mut self.logits_buf,
                    vocab_size,
                    hidden_size,
                    self.n_threads,
                );
            }
            QuantType::Q5_K => {
                let nb = (hidden_size + QK_K - 1) / QK_K;
                let nb1 = output_tensor.strides[1] as usize;

                self.quant_buf.resize(hidden_size, QuantFormat::Q8_K);
                quant::quantize_row_q8_k(
                    &self.hidden_buf,
                    &mut self.quant_buf.q8_k,
                );

                crate::backend::arm::gemv::gemv_q5_k_transposed(
                    output_data,
                    nb1,
                    &self.quant_buf.q8_k,
                    &mut self.logits_buf,
                    vocab_size,
                    hidden_size,
                    self.n_threads,
                );
            }
            QuantType::F16 => {
                let nb1 = output_tensor.strides[1] as usize / 2;

                for v in 0..vocab_size {
                    let mut sum = 0.0f32;
                    for h in 0..hidden_size {
                        let f16_val = half::f16::from_le_bytes([
                            output_data[(v * nb1 + h) * 2],
                            output_data[(v * nb1 + h) * 2 + 1],
                        ]);
                        sum += self.hidden_buf[h] * f16_val.to_f32();
                    }
                    self.logits_buf[v] = sum;
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported output quantization: {:?}",
                    output_tensor.quant_type
                ));
            }
        }

        Ok(())
    }

    /// Sample next token from logits
    pub fn sample(&self, logits: &[f32], temperature: f32) -> u32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();

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
            let exp_sum: f32 = logits
                .iter()
                .map(|&l| ((l - max_logit) / temperature).exp())
                .sum();

            let r = rng.r#gen::<f32>();
            let mut cumsum = 0.0;
            for (i, &logit) in logits.iter().enumerate() {
                cumsum += ((logit - max_logit) / temperature).exp() / exp_sum;
                if cumsum >= r {
                    return i as u32;
                }
            }
            (logits.len() - 1) as u32
        }
    }

    /// Generate tokens autoregressively
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        let mut tokens = prompt_tokens.to_vec();

        println!("Generating {} new tokens...", max_new_tokens);

        // Prefill: process all prompt tokens
        // The last prompt token's logits will be used for the first generation
        println!("Prefilling {} tokens...", tokens.len());
        let mut logits: Option<Vec<f32>> = None;
        for (i, &token) in tokens.iter().enumerate() {
            match self.forward(token) {
                Ok(l) => {
                    // Copy logits to break the borrow
                    logits = Some(l.to_vec());
                }
                Err(e) => {
                    eprintln!("Prefill error at token {}: {}", i, e);
                    return tokens;
                }
            }
        }
        let seq_pos_after_prefill = self.seq_pos;
        println!("Prefill complete. seq_pos={}", seq_pos_after_prefill);

        // Generation loop
        for i in 0..max_new_tokens {
            let logits = match logits.take() {
                Some(l) => l,
                None => {
                    // After first iteration, we need to forward the last token
                    match self.forward(tokens.last().copied().unwrap_or(0)) {
                        Ok(l) => l.to_vec(),
                        Err(e) => {
                            eprintln!("Forward error at step {}: {}", i, e);
                            break;
                        }
                    }
                }
            };

            let next_token = self.sample(&logits, temperature);
            println!("  Token {}: {}", i, next_token);
            tokens.push(next_token);
        }

        tokens
    }

    /// Forward pass with detailed tracing for debugging
    pub fn forward_with_trace(
        &mut self,
        token_id: u32,
    ) -> Result<ForwardTrace, String> {
        let config = self.model.config();
        let mut layer_traces = Vec::new();

        // 1. Embedding lookup
        let mut hidden = self.embed_token(token_id)?;
        let embedding = hidden.clone();

        // 2. Process each layer with tracing
        for layer_idx in 0..config.n_layers {
            let (hidden_after, trace) =
                self.forward_layer_with_trace(&hidden, layer_idx)?;
            layer_traces.push(trace);
            hidden = hidden_after;
        }

        // 3. Final norm
        let final_hidden = self.final_norm(&hidden)?;
        hidden = final_hidden.clone();

        // 4. Output projection
        let logits = self.output_projection(&hidden)?;

        // Increment sequence position
        self.seq_pos += 1;

        Ok(ForwardTrace {
            token_id,
            embedding,
            layer_traces,
            final_hidden,
            logits,
        })
    }

    /// Forward pass for a single layer with tracing
    fn forward_layer_with_trace(
        &mut self,
        input: &[f32],
        layer_idx: usize,
    ) -> Result<(Vec<f32>, LayerTrace), String> {
        let config = self.model.config().clone();
        let hidden_size = config.hidden_size;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim();

        let hidden_before = input.to_vec();

        // === Attention Block ===
        // 1. RMS Norm before attention
        let normed_attn = self.rms_norm(
            input,
            &format!("blk.{}.attn_norm.weight", layer_idx),
            hidden_size,
        )?;

        // 2. Pre-quantize normed_attn for QKV projections (shared input)
        // Use pre-allocated buffer to avoid repeated Vec allocation
        let quant_type = self
            .get_weight_quant_type(&format!("blk.{}.attn_q.weight", layer_idx));
        let (qkv_q8_0, qkv_q8_k, in_features) = match quant_type {
            QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
                let nb = (normed_attn.len() + QK8_0 - 1) / QK8_0;
                self.quant_buf.resize(normed_attn.len(), QuantFormat::Q8_0);
                crate::backend::arm::quant::quantize_q8_0(
                    &normed_attn,
                    &mut self.quant_buf.q8_0,
                );
                (Some(&self.quant_buf.q8_0 as &[BlockQ8_0]), None, nb * QK8_0)
            }
            QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_K
            | QuantType::IQ4_XS => {
                let nb = (normed_attn.len() + QK_K - 1) / QK_K;
                self.quant_buf.resize(normed_attn.len(), QuantFormat::Q8_K);
                crate::backend::arm::quant::quantize_q8_k(
                    &normed_attn,
                    &mut self.quant_buf.q8_k,
                );
                (None, Some(&self.quant_buf.q8_k as &[BlockQ8_K]), nb * QK_K)
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type: {:?}",
                    quant_type
                ));
            }
        };

        // 3. QKV projections using pre-quantized input
        let q = self.linear_quantized_inner(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_q.weight", layer_idx),
            n_heads * head_dim,
        )?;
        let k = self.linear_quantized_inner(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_k.weight", layer_idx),
            n_kv_heads * head_dim,
        )?;
        let v = self.linear_quantized_inner(
            qkv_q8_0,
            qkv_q8_k,
            in_features,
            &format!("blk.{}.attn_v.weight", layer_idx),
            n_kv_heads * head_dim,
        )?;

        // 4. Q/K norm
        let q_normed = self.qk_norm(
            &q,
            &format!("blk.{}.attn_q_norm.weight", layer_idx),
            n_heads,
            head_dim,
        )?;
        let k_normed = self.qk_norm(
            &k,
            &format!("blk.{}.attn_k_norm.weight", layer_idx),
            n_kv_heads,
            head_dim,
        )?;

        // 5. Apply RoPE to Q and K
        let freq_base = config.rope_freq_base;
        let mut q_rope = q_normed.clone();
        let mut k_rope = k_normed.clone();
        for h in 0..n_heads {
            rope_f32(
                &q_normed[h * head_dim..(h + 1) * head_dim],
                &mut q_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                freq_base,
            );
        }
        for h in 0..n_kv_heads {
            rope_f32(
                &k_normed[h * head_dim..(h + 1) * head_dim],
                &mut k_rope[h * head_dim..(h + 1) * head_dim],
                self.seq_pos,
                head_dim,
                head_dim,
                freq_base,
            );
        }

        // 6. Update KV cache
        let kv_head_offset = self.seq_pos * head_dim;
        let kv_capacity = self.kv_capacity; // Use actual capacity, not max context_length
        for h in 0..n_kv_heads {
            let src_offset = h * head_dim;
            let dst_offset = h * kv_capacity * head_dim + kv_head_offset;
            self.kv_cache_k[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&k_rope[src_offset..src_offset + head_dim]);
            self.kv_cache_v[layer_idx][dst_offset..dst_offset + head_dim]
                .copy_from_slice(&v[src_offset..src_offset + head_dim]);
        }

        // 7. Flash Attention
        let mut attn_output = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv_head_idx = h * n_kv_heads / n_heads;
            let k_cache = &self.kv_cache_k[layer_idx]
                [kv_head_idx * kv_capacity * head_dim..];
            let v_cache = &self.kv_cache_v[layer_idx]
                [kv_head_idx * kv_capacity * head_dim..];

            let k_slice = &k_cache[..(self.seq_pos + 1) * head_dim];
            let v_slice = &v_cache[..(self.seq_pos + 1) * head_dim];

            flash_attn_f32(
                &q_rope[h * head_dim..(h + 1) * head_dim],
                k_slice,
                v_slice,
                None,
                &mut attn_output[h * head_dim..(h + 1) * head_dim],
                head_dim,
                self.seq_pos + 1,
                1.0 / (head_dim as f32).sqrt(),
            );
        }

        // 8. Output projection
        let attn_out = self.linear(
            &attn_output,
            &format!("blk.{}.attn_output.weight", layer_idx),
            hidden_size,
        )?;

        // Residual connection
        let hidden: Vec<f32> =
            input.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // === FFN Block ===
        // 1. RMS Norm before FFN
        let normed_ffn = self.rms_norm(
            &hidden,
            &format!("blk.{}.ffn_norm.weight", layer_idx),
            hidden_size,
        )?;

        // 2. Pre-quantize normed_ffn for gate/up projections (shared input)
        // Use pre-allocated buffer to avoid repeated Vec allocation
        let quant_type = self.get_weight_quant_type(&format!(
            "blk.{}.ffn_gate.weight",
            layer_idx
        ));
        let (gate_up_q8_0, gate_up_q8_k, ffn_in_features) = match quant_type {
            QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
                let nb = (normed_ffn.len() + QK8_0 - 1) / QK8_0;
                self.quant_buf.resize(normed_ffn.len(), QuantFormat::Q8_0);
                crate::backend::arm::quant::quantize_q8_0(
                    &normed_ffn,
                    &mut self.quant_buf.q8_0,
                );
                (Some(&self.quant_buf.q8_0 as &[BlockQ8_0]), None, nb * QK8_0)
            }
            QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_K
            | QuantType::IQ4_XS => {
                let nb = (normed_ffn.len() + QK_K - 1) / QK_K;
                self.quant_buf.resize(normed_ffn.len(), QuantFormat::Q8_K);
                crate::backend::arm::quant::quantize_q8_k(
                    &normed_ffn,
                    &mut self.quant_buf.q8_k,
                );
                (None, Some(&self.quant_buf.q8_k as &[BlockQ8_K]), nb * QK_K)
            }
            _ => {
                return Err(format!(
                    "Unsupported quantization type: {:?}",
                    quant_type
                ));
            }
        };

        // 3. Gate and Up projections using pre-quantized input
        let gate = self.linear_quantized_inner(
            gate_up_q8_0,
            gate_up_q8_k,
            ffn_in_features,
            &format!("blk.{}.ffn_gate.weight", layer_idx),
            config.intermediate_size,
        )?;
        let up = self.linear_quantized_inner(
            gate_up_q8_0,
            gate_up_q8_k,
            ffn_in_features,
            &format!("blk.{}.ffn_up.weight", layer_idx),
            config.intermediate_size,
        )?;

        // 4. SwiGLU activation
        let mut ffn_hidden = vec![0.0f32; config.intermediate_size];
        crate::backend::arm::activation::silu_f32(&gate, &mut ffn_hidden);
        for i in 0..config.intermediate_size {
            ffn_hidden[i] *= up[i];
        }

        // 5. Down projection
        let ffn_out = self.linear(
            &ffn_hidden,
            &format!("blk.{}.ffn_down.weight", layer_idx),
            hidden_size,
        )?;

        // Residual connection
        let output: Vec<f32> =
            hidden.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

        Ok((
            output.clone(),
            LayerTrace {
                layer_idx,
                hidden_before,
                hidden_after: output,
                q_normed,
                k_normed,
                attn_output,
                ffn_output: ffn_out,
                ffn_normed: normed_ffn,
                ffn_gate: gate,
                ffn_up: up,
                ffn_hidden,
            },
        ))
    }
}
