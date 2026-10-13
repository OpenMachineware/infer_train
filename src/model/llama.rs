// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Llama model architecture definition

use crate::inference::{dequant_to_f32, KVCacheBase};
use crate::model::config::ModelConfig;
use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;
use std::collections::HashMap;

/// Llama model layers
pub struct LlamaModel {
    pub config: ModelConfig,

    // Embedding
    pub tok_embeddings: Option<Tensor>,

    // Transformer blocks
    pub layers: Vec<LlamaLayer>,

    // Output
    pub norm: Option<Tensor>,
    pub output: Option<Tensor>,
}

pub struct LlamaLayer {
    pub layer_idx: usize,

    // Attention
    pub attention_norm: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub wo: Option<Tensor>,

    // FFN
    pub ffn_norm: Option<Tensor>,
    pub w1: Option<Tensor>, // gate
    pub w2: Option<Tensor>, // down
    pub w3: Option<Tensor>, // up
}

impl LlamaModel {
    pub fn from_weights(store: &WeightStore, config: ModelConfig) -> Self {
        let mut model = LlamaModel {
            config,
            tok_embeddings: None,
            layers: Vec::new(),
            norm: None,
            output: None,
        };

        // Load embeddings - try multiple naming conventions
        model.tok_embeddings = store
            .get("tok_embeddings.weight")
            .or_else(|| store.get("token_embd.weight"))
            .cloned();

        // Load layers
        for i in 0..model.config.n_layers {
            let layer = LlamaLayer::from_weights(store, i);
            model.layers.push(layer);
        }

        // Load output norm - try multiple naming conventions
        model.norm = store
            .get("norm.weight")
            .or_else(|| store.get("output_norm.weight"))
            .cloned();

        // Load output projection - try multiple naming conventions
        model.output = store
            .get("output.weight")
            .or_else(|| store.get("lm_head.weight"))
            .cloned();

        model
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn param_count(&self) -> usize {
        let mut count = 0usize;

        if let Some(ref t) = self.tok_embeddings {
            count += t.elem_count();
        }
        for layer in &self.layers {
            count += layer.param_count();
        }
        if let Some(ref t) = self.norm {
            count += t.elem_count();
        }
        if let Some(ref t) = self.output {
            count += t.elem_count();
        }

        count
    }

    /// Public test method for embedding lookup
    pub fn test_embedding_lookup(&self, token: u32) -> Option<Vec<f32>> {
        self.embedding_lookup(token)
    }

    /// Forward pass with GPU context
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCache,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        gpu_weight_buffers: &mut std::collections::HashMap<
            (usize, &'static str),
            metal::Buffer,
        >,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let hidden_size = self.config.hidden_size;
        let n_heads = self.config.n_heads;
        let head_dim = hidden_size / n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let vocab_size = self.config.vocab_size;

        // Process each token
        let mut hidden = Vec::with_capacity(hidden_size);
        for &token in tokens.iter() {
            // 1. Embedding lookup (CPU)
            hidden = self.embedding_lookup(token)?;

            // 2. Process through layers (GPU)
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = self.forward_layer_gpu(
                    layer,
                    layer_idx,
                    &hidden,
                    kv_cache,
                    n_heads,
                    head_dim,
                    n_kv_heads,
                    gpu_ctx,
                    gpu_weight_buffers,
                )?;
            }

            // Advance KV cache position
            kv_cache.advance();
        }

        // 3. Final norm and output projection (GPU)
        let final_hidden = self.apply_output_norm_gpu(&hidden, gpu_ctx)?;
        let logits = self.output_projection_gpu(
            &final_hidden,
            vocab_size,
            gpu_ctx,
            gpu_weight_buffers,
        )?;

        Some(logits)
    }

    /// Forward pass: tokens → logits (CPU version)
    /// Returns logits for the last token position
    pub fn forward(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCache,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let hidden_size = self.config.hidden_size;
        let n_heads = self.config.n_heads;
        let head_dim = hidden_size / n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let vocab_size = self.config.vocab_size;

        // Process each token
        let mut hidden = Vec::with_capacity(hidden_size);
        for &token in tokens.iter() {
            // 1. Embedding lookup
            hidden = self.embedding_lookup(token)?;

            // 2. Process through layers
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = self.forward_layer(
                    layer, layer_idx, &hidden, kv_cache, n_heads, head_dim,
                    n_kv_heads,
                )?;
            }

            // Advance KV cache position
            kv_cache.advance();
        }

        // 3. Final norm and output projection
        let final_hidden = self.apply_output_norm(&hidden)?;
        let logits = self.output_projection(&final_hidden, vocab_size)?;

        Some(logits)
    }

    fn embedding_lookup(&self, token: u32) -> Option<Vec<f32>> {
        let embed = self.tok_embeddings.as_ref()?;
        let hidden_size = self.config.hidden_size;
        let dtype = embed.dtype;

        // GGUF stores embeddings transposed: shape is [hidden_size, vocab_size]
        // So for token N, we need to read column N (stride by hidden_size)
        let bytes = embed.bytes();
        let bytes_per_row = dtype.bytes_for_elems(hidden_size);
        let token = token as usize;

        // For quantized formats with block-based storage, we need to handle
        // the strided access differently. Each row is hidden_size elements.
        // Token N's embedding starts at byte offset: N * bytes_per_row
        let offset = token * bytes_per_row;

        if offset + bytes_per_row > bytes.len() {
            return None;
        }

        let data = &bytes[offset..offset + bytes_per_row];
        Some(dequant_to_f32(data, dtype, hidden_size))
    }

    fn forward_layer(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden: &[f32],
        kv_cache: &mut crate::inference::KVCache,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Option<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let pos = kv_cache.seq_len();

        // 1. Attention norm
        let normed = self.apply_rms_norm(
            hidden,
            layer.attention_norm.as_ref()?,
            hidden_size,
        )?;

        // 2. QKV projections
        let q = self.gemv(&normed, layer.wq.as_ref()?, hidden_size)?;
        let k =
            self.gemv(&normed, layer.wk.as_ref()?, n_kv_heads * head_dim)?;
        let v =
            self.gemv(&normed, layer.wv.as_ref()?, n_kv_heads * head_dim)?;

        // 3. RoPE
        let q_rope = self.apply_rope(&q, pos, n_heads, head_dim)?;
        let k_rope = self.apply_rope(&k, pos, n_kv_heads, head_dim)?;

        // 4. Attention (updates KV cache)
        let attn_out = self.attention(
            &q_rope, &k_rope, &v, layer_idx, kv_cache, n_heads, head_dim,
            n_kv_heads,
        )?;

        // 5. Output projection
        let attn_proj =
            self.gemv(&attn_out, layer.wo.as_ref()?, hidden_size)?;

        // 6. Residual
        let residual: Vec<f32> =
            hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();

        // 7. FFN norm
        let ffn_normed = self.apply_rms_norm(
            &residual,
            layer.ffn_norm.as_ref()?,
            hidden_size,
        )?;

        // 8. FFN (SwiGLU)
        let ffn_out = self.ffn_swiglu(&ffn_normed, layer, hidden_size)?;

        // 9. Residual
        Some(residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect())
    }

    fn apply_rms_norm(
        &self,
        x: &[f32],
        weight: &Tensor,
        size: usize,
    ) -> Option<Vec<f32>> {
        let w = dequant_to_f32(weight.bytes(), weight.dtype, size);
        let eps = 1e-5f32;

        let mut sum_sq = 0.0f32;
        for i in 0..size {
            sum_sq += x[i] * x[i];
        }
        let rms = (sum_sq / size as f32 + eps).sqrt();

        let mut out = vec![0.0f32; size];
        for i in 0..size {
            out[i] = (x[i] / rms) * w[i];
        }
        Some(out)
    }

    fn apply_rope(
        &self,
        x: &[f32],
        pos: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Option<Vec<f32>> {
        let mut out = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            for d in (0..head_dim).step_by(2) {
                let idx = h * head_dim + d;
                let theta =
                    pos as f32 / 10000.0f32.powf(d as f32 / head_dim as f32);
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                if d + 1 < head_dim {
                    out[idx] = x[idx] * cos_theta - x[idx + 1] * sin_theta;
                    out[idx + 1] = x[idx] * sin_theta + x[idx + 1] * cos_theta;
                }
            }
        }
        Some(out)
    }

    fn gemv(&self, x: &[f32], w: &Tensor, out_size: usize) -> Option<Vec<f32>> {
        let in_size = x.len();
        let total_elems = in_size * out_size;
        let w_data = dequant_to_f32(w.bytes(), w.dtype, total_elems);

        let mut out = vec![0.0f32; out_size];
        for i in 0..out_size {
            let mut sum = 0.0f32;
            for j in 0..in_size {
                sum += w_data[i * in_size + j] * x[j];
            }
            out[i] = sum;
        }
        Some(out)
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        layer_idx: usize,
        kv_cache: &mut crate::inference::KVCache,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Option<Vec<f32>> {
        // Store K, V in cache (will be stored at current seq_len position)
        kv_cache.set_k(layer_idx, k, n_kv_heads, head_dim);
        kv_cache.set_v(layer_idx, v, n_kv_heads, head_dim);

        let seq_len = kv_cache.seq_len() + 1; // +1 because we just added
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut out = vec![0.0f32; n_heads * head_dim];

        // For each query head
        for h in 0..n_heads {
            // GQA mapping: multiple Q heads share one KV head
            let kv_head = h * n_kv_heads / n_heads;

            // Compute attention scores for all positions
            let mut scores = vec![0.0f32; seq_len];
            for s in 0..seq_len {
                let k_s = kv_cache.get_k(layer_idx, kv_head, s);
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[h * head_dim + d] * k_s[d];
                }
                scores[s] = score * scale;
            }

            // Softmax
            let max_score =
                scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in 0..seq_len {
                scores[s] = (scores[s] - max_score).exp();
                sum_exp += scores[s];
            }
            for s in 0..seq_len {
                scores[s] /= sum_exp;
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    let v_s = kv_cache.get_v(layer_idx, kv_head, s);
                    sum += scores[s] * v_s[d];
                }
                out[h * head_dim + d] = sum;
            }
        }

        Some(out)
    }

    fn ffn_swiglu(
        &self,
        x: &[f32],
        layer: &LlamaLayer,
        hidden_size: usize,
    ) -> Option<Vec<f32>> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let intermediate = hidden_size * 4; // Standard FFN expansion

        let gate = self.gemv(x, layer.w1.as_ref()?, intermediate)?;
        let up = self.gemv(x, layer.w3.as_ref()?, intermediate)?;

        // SiLU activation and element-wise multiply
        let hidden: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u) // silu(g) * u
            .collect();

        // Down projection
        self.gemv(&hidden, layer.w2.as_ref()?, hidden_size)
    }

    fn apply_output_norm(&self, hidden: &[f32]) -> Option<Vec<f32>> {
        self.apply_rms_norm(
            hidden,
            self.norm.as_ref()?,
            self.config.hidden_size,
        )
    }

    fn output_projection(
        &self,
        hidden: &[f32],
        vocab_size: usize,
    ) -> Option<Vec<f32>> {
        self.gemv(hidden, self.output.as_ref()?, vocab_size)
    }

    // GPU versions of operators

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn forward_layer_gpu(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden: &[f32],
        kv_cache: &mut crate::inference::KVCache,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        gpu_weight_buffers: &mut std::collections::HashMap<
            (usize, &'static str),
            metal::Buffer,
        >,
    ) -> Option<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let pos = kv_cache.seq_len();

        // 1. Attention norm (GPU)
        let normed = self.apply_rms_norm_gpu(
            hidden,
            layer.attention_norm.as_ref()?,
            hidden_size,
            gpu_ctx,
        )?;

        // 2. QKV projections (GPU)
        let q = self.gemv_gpu(
            &normed,
            layer.wq.as_ref()?,
            hidden_size,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "wq",
        )?;
        let k = self.gemv_gpu(
            &normed,
            layer.wk.as_ref()?,
            n_kv_heads * head_dim,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "wk",
        )?;
        let v = self.gemv_gpu(
            &normed,
            layer.wv.as_ref()?,
            n_kv_heads * head_dim,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "wv",
        )?;

        // 3. RoPE (GPU)
        let q_rope =
            self.apply_rope_gpu(&q, pos, n_heads, head_dim, gpu_ctx)?;
        let k_rope =
            self.apply_rope_gpu(&k, pos, n_kv_heads, head_dim, gpu_ctx)?;

        // 4. Attention (GPU - updates KV cache)
        let attn_out = self.attention_gpu(
            &q_rope, &k_rope, &v, layer_idx, kv_cache, n_heads, head_dim,
            n_kv_heads, gpu_ctx,
        )?;

        // 5. Output projection (GPU)
        let attn_proj = self.gemv_gpu(
            &attn_out,
            layer.wo.as_ref()?,
            hidden_size,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "wo",
        )?;

        // 6. Residual
        let residual: Vec<f32> =
            hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();

        // 7. FFN norm (GPU)
        let ffn_normed = self.apply_rms_norm_gpu(
            &residual,
            layer.ffn_norm.as_ref()?,
            hidden_size,
            gpu_ctx,
        )?;

        // 8. FFN (SwiGLU) (GPU)
        let ffn_out = self.ffn_swiglu_gpu(
            &ffn_normed,
            layer,
            hidden_size,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
        )?;

        // 9. Residual
        Some(residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect())
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn apply_rms_norm_gpu(
        &self,
        x: &[f32],
        weight: &Tensor,
        size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        let w = dequant_to_f32(weight.bytes(), weight.dtype, size);
        gpu_ctx.rms_norm_f32(1, size, x, &w, 1e-5).ok()
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn gemv_gpu(
        &self,
        x: &[f32],
        w: &Tensor,
        out_size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        gpu_weight_buffers: &mut std::collections::HashMap<
            (usize, &'static str),
            metal::Buffer,
        >,
        layer_idx: usize,
        weight_name: &'static str,
    ) -> Option<Vec<f32>> {
        let in_size = x.len();
        let w_bytes = w.bytes();

        // Try to use pre-uploaded buffer
        let weight_key = (layer_idx, weight_name);
        let use_cached = gpu_weight_buffers.contains_key(&weight_key);

        // Select appropriate MV kernel based on weight dtype
        match w.dtype {
            crate::model::tensor::DataType::Q4_0 => {
                // Reinterpret bytes as BlockQ4_0 slice
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ4_0>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ4_0] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q4_0_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q4_1 => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ4_1>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ4_1] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q4_1_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q5_0 => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ5_0>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ5_0] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q5_0_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q5_1 => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ5_1>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ5_1] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q5_1_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q8_0 => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ8_0>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ8_0] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q8_0_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q2_K => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ2K>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ2K] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q2_k_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q3_K => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ3K>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ3K] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q3_k_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q4_K => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ4K>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ4K] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q4_k_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q5_K => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ5K>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ5K] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q5_k_f32(out_size, in_size, weights, x).ok()
            }
            crate::model::tensor::DataType::Q6_K => {
                let block_size =
                    std::mem::size_of::<crate::quant::types::BlockQ6K>();
                let n_blocks = w_bytes.len() / block_size;
                let weights: &[crate::quant::types::BlockQ6K] = unsafe {
                    std::slice::from_raw_parts(
                        w_bytes.as_ptr() as *const _,
                        n_blocks,
                    )
                };
                gpu_ctx.mv_q6_k_f32(out_size, in_size, weights, x).ok()
            }
            _ => {
                // Fallback to CPU for unsupported formats
                self.gemv(x, w, out_size)
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn apply_rope_gpu(
        &self,
        x: &[f32],
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        // For now, use CPU RoPE (RoPE is memory-bound, GPU advantage is small)
        // TODO: Implement GPU RoPE when needed
        self.apply_rope(x, pos, n_heads, head_dim)
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        layer_idx: usize,
        kv_cache: &mut crate::inference::KVCache,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        _gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        // For now, use CPU attention
        // TODO: Implement GPU Flash Attention with KV cache integration
        self.attention(
            q, k, v, layer_idx, kv_cache, n_heads, head_dim, n_kv_heads,
        )
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn ffn_swiglu_gpu(
        &self,
        x: &[f32],
        layer: &LlamaLayer,
        hidden_size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        gpu_weight_buffers: &mut std::collections::HashMap<
            (usize, &'static str),
            metal::Buffer,
        >,
        layer_idx: usize,
    ) -> Option<Vec<f32>> {
        let intermediate = hidden_size * 4;

        // Gate and Up projections (GPU)
        let gate = self.gemv_gpu(
            x,
            layer.w1.as_ref()?,
            intermediate,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "w1",
        )?;
        let up = self.gemv_gpu(
            x,
            layer.w3.as_ref()?,
            intermediate,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "w3",
        )?;

        // SiLU activation (GPU)
        let silu_gate = gpu_ctx.silu_f32(intermediate, &gate).ok()?;

        // Element-wise multiply (CPU - small operation)
        let hidden: Vec<f32> =
            silu_gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();

        // Down projection (GPU)
        self.gemv_gpu(
            &hidden,
            layer.w2.as_ref()?,
            hidden_size,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "w2",
        )
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn apply_output_norm_gpu(
        &self,
        hidden: &[f32],
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        self.apply_rms_norm_gpu(
            hidden,
            self.norm.as_ref()?,
            self.config.hidden_size,
            gpu_ctx,
        )
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn output_projection_gpu(
        &self,
        hidden: &[f32],
        vocab_size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        gpu_weight_buffers: &mut std::collections::HashMap<
            (usize, &'static str),
            metal::Buffer,
        >,
    ) -> Option<Vec<f32>> {
        // Output weights are stored at layer index = num_layers
        let layer_idx = self.layers.len();
        self.gemv_gpu(
            hidden,
            self.output.as_ref()?,
            vocab_size,
            gpu_ctx,
            gpu_weight_buffers,
            layer_idx,
            "output",
        )
    }
}

impl LlamaLayer {
    pub fn from_weights(store: &WeightStore, layer_idx: usize) -> Self {
        // Try both Llama-style and Qwen-style naming conventions
        let llama_prefix = format!("layers.{}", layer_idx);
        let qwen_prefix = format!("blk.{}", layer_idx);

        LlamaLayer {
            layer_idx,

            // Attention
            attention_norm: store
                .get(&format!("{}.attention_norm.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_norm.weight", qwen_prefix))
                })
                .cloned(),
            wq: store
                .get(&format!("{}.attention.wq.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_q.weight", qwen_prefix))
                })
                .cloned(),
            wk: store
                .get(&format!("{}.attention.wk.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_k.weight", qwen_prefix))
                })
                .cloned(),
            wv: store
                .get(&format!("{}.attention.wv.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_v.weight", qwen_prefix))
                })
                .cloned(),
            wo: store
                .get(&format!("{}.attention.wo.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_output.weight", qwen_prefix))
                })
                .cloned(),

            // FFN
            ffn_norm: store
                .get(&format!("{}.ffn_norm.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_norm.weight", qwen_prefix))
                })
                .cloned(),
            w1: store
                .get(&format!("{}.feed_forward.w1.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_gate.weight", qwen_prefix))
                })
                .cloned(),
            w2: store
                .get(&format!("{}.feed_forward.w2.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_down.weight", qwen_prefix))
                })
                .cloned(),
            w3: store
                .get(&format!("{}.feed_forward.w3.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.ffn_up.weight", qwen_prefix))
                })
                .cloned(),
        }
    }

    pub fn param_count(&self) -> usize {
        let mut count = 0usize;

        let tensors: Vec<Option<&Tensor>> = vec![
            self.attention_norm.as_ref(),
            self.wq.as_ref(),
            self.wk.as_ref(),
            self.wv.as_ref(),
            self.wo.as_ref(),
            self.ffn_norm.as_ref(),
            self.w1.as_ref(),
            self.w2.as_ref(),
            self.w3.as_ref(),
        ];

        for tensor in tensors.iter().flatten() {
            count += tensor.elem_count();
        }

        count
    }

    pub fn is_complete(&self) -> bool {
        self.attention_norm.is_some()
            && self.wq.is_some()
            && self.wk.is_some()
            && self.wv.is_some()
            && self.wo.is_some()
            && self.ffn_norm.is_some()
            && self.w1.is_some()
            && self.w2.is_some()
            && self.w3.is_some()
    }
}

/// Alternative tensor name patterns (different GGUF naming conventions)
pub fn find_tensor_name(
    store: &WeightStore,
    patterns: &[&str],
) -> Option<String> {
    for pattern in patterns {
        if store.get(pattern).is_some() {
            return Some(pattern.to_string());
        }
    }
    None
}

/// Build tensor name mapping from GGUF file
pub fn build_tensor_mapping(store: &WeightStore) -> HashMap<String, String> {
    let mut mapping = HashMap::new();

    for name in store.tensor_names() {
        // Map common variations
        if name.contains("tok_embeddings") || name.contains("embed_tokens") {
            mapping.insert("tok_embeddings".to_string(), name.clone());
        }
        if name.starts_with("layers.") && name.contains("attention_norm") {
            if let Some(layer_idx) = extract_layer_index(name) {
                mapping.insert(
                    format!("layers.{}.attention_norm", layer_idx),
                    name.clone(),
                );
            }
        }
    }

    mapping
}

fn extract_layer_index(name: &str) -> Option<usize> {
    if let Some(start) = name.find("layers.") {
        let rest = &name[start + 7..];
        if let Some(end) = rest.find('.') {
            if let Ok(idx) = rest[..end].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}
