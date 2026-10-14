// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Llama model architecture definition

use crate::inference::{dequant_to_f32, KVCacheBase};
use crate::model::config::ModelConfig;
use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;
use std::collections::HashMap;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::backend::Backend;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::graph::ComputeGraph;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::metal_backend::MetalBackend;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::op::{OpNode, OpParams, OpType};

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
    pub q_norm: Option<Tensor>, // Qwen-style Q norm
    pub k_norm: Option<Tensor>, // Qwen-style K norm

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

        // Fix config n_heads/n_kv_heads from actual weight shapes
        // GGUF metadata may be incorrect for some models
        if let Some(first_layer) = model.layers.first() {
            if let (Some(wq), Some(wk)) = (&first_layer.wq, &first_layer.wk) {
                // GGUF uses column-major: shape = [K, M], where K=input, M=output
                let q_out_size = wq.shape[1];
                let k_out_size = wk.shape[1];
                let hidden_size = model.config.hidden_size;

                // Determine head_dim by finding a divisor that works for both Q and K
                // and matches the pattern: n_heads * head_dim == hidden_size
                // Common head_dim values: 32, 64, 128
                let mut head_dim = 0;
                for &candidate in &[32, 64, 128, 256] {
                    if q_out_size % candidate == 0
                        && k_out_size % candidate == 0
                        && hidden_size % candidate == 0
                    {
                        let n_heads = q_out_size / candidate;
                        if n_heads * candidate == hidden_size
                            || n_heads == hidden_size / candidate
                        {
                            head_dim = candidate;
                            break;
                        }
                    }
                }

                if head_dim == 0 {
                    // Fall back to config's head_dim
                    head_dim = hidden_size / model.config.n_heads;
                }

                let actual_n_heads = q_out_size / head_dim;
                let actual_n_kv_heads = k_out_size / head_dim;

                if actual_n_heads != model.config.n_heads
                    || actual_n_kv_heads != model.config.n_kv_heads
                    || head_dim != model.config.head_dim
                {
                    model.config.n_heads = actual_n_heads;
                    model.config.n_kv_heads = actual_n_kv_heads;
                    model.config.head_dim = head_dim;
                    model.config.rope_dim = head_dim; // Also fix rope_dim
                }
            }
        }

        // Load output norm - try multiple naming conventions
        model.norm = store
            .get("norm.weight")
            .or_else(|| store.get("output_norm.weight"))
            .cloned();

        // Load output projection - try multiple naming conventions
        // For tied embeddings, reuse token embeddings as output projection
        model.output = store
            .get("output.weight")
            .or_else(|| store.get("lm_head.weight"))
            .cloned()
            .or_else(|| model.tok_embeddings.clone()); // Tied embeddings

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
    pub fn embedding_lookup(&self, token: u32) -> Option<Vec<f32>> {
        self.embedding_lookup_impl(token)
    }

    /// Apply RMS norm (public for debugging)
    pub fn apply_rms_norm(
        &self,
        x: &[f32],
        weight: &Tensor,
        size: usize,
    ) -> Option<Vec<f32>> {
        self.apply_rms_norm_impl(x, weight, size)
    }

    /// GEMV operation (public for debugging)
    pub fn gemv(
        &self,
        x: &[f32],
        w: &Tensor,
        out_size: usize,
    ) -> Option<Vec<f32>> {
        self.gemv_impl(x, w, out_size)
    }

    /// Forward pass with GPU context
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCache,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let hidden_size = self.config.hidden_size;
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let n_kv_heads = self.config.n_kv_heads;
        let vocab_size = self.config.vocab_size;

        // Process each token
        let mut hidden = Vec::with_capacity(hidden_size);
        for &token in tokens.iter() {
            // 1. Embedding lookup (CPU)
            hidden = self.embedding_lookup_impl(token)?;

            // 2. Process through layers (GPU)
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = self.forward_layer_gpu(
                    layer, layer_idx, &hidden, kv_cache, n_heads, head_dim,
                    n_kv_heads, gpu_ctx,
                )?;
            }

            // Advance KV cache position
            kv_cache.advance();
        }

        // 3. Final norm and output projection (GPU)
        let final_hidden = self.apply_output_norm_gpu(&hidden, gpu_ctx)?;
        let logits =
            self.output_projection_gpu(&final_hidden, vocab_size, gpu_ctx)?;

        Some(logits)
    }

    /// Forward pass with GPU KV cache (zero-copy Flash Attention)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu_zerocopy(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let hidden_size = self.config.hidden_size;
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let n_kv_heads = self.config.n_kv_heads;
        let vocab_size = self.config.vocab_size;

        // Process each token
        let mut hidden = Vec::with_capacity(hidden_size);
        for &token in tokens.iter() {
            // 1. Embedding lookup (CPU)
            hidden = self.embedding_lookup_impl(token)?;

            // 2. Process through layers (GPU with zero-copy)
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = self.forward_layer_gpu_zerocopy(
                    layer, layer_idx, &hidden, kv_cache, n_heads, head_dim,
                    n_kv_heads, gpu_ctx,
                )?;
            }

            // Advance KV cache position
            kv_cache.advance();
        }

        // 3. Final norm and output projection (GPU)
        let final_hidden = self.apply_output_norm_gpu(&hidden, gpu_ctx)?;
        let logits =
            self.output_projection_gpu(&final_hidden, vocab_size, gpu_ctx)?;

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
        let head_dim = self.config.head_dim;
        let n_kv_heads = self.config.n_kv_heads;
        let vocab_size = self.config.vocab_size;

        // Process each token
        let mut hidden = Vec::with_capacity(hidden_size);
        for &token in tokens.iter() {
            // 1. Embedding lookup
            hidden = self.embedding_lookup_impl(token)?;

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

    fn embedding_lookup_impl(&self, token: u32) -> Option<Vec<f32>> {
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
        let normed = self.apply_rms_norm_impl(
            hidden,
            layer.attention_norm.as_ref()?,
            hidden_size,
        )?;

        // 2. QKV projections (use actual weight shapes, not config)
        let wq = layer.wq.as_ref()?;
        let wk = layer.wk.as_ref()?;
        let wv = layer.wv.as_ref()?;

        // GGUF uses column-major: shape = [K, M], where K=input, M=output
        let q_out_size = wq.shape[1]; // Second dimension is output
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];

        let q = self.gemv_impl(&normed, wq, q_out_size)?;
        let k = self.gemv_impl(&normed, wk, k_out_size)?;
        let v = self.gemv_impl(&normed, wv, v_out_size)?;

        // Update n_heads and n_kv_heads based on actual weight shapes
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // 2.5 Apply Q/K norms (Qwen-style)
        // Qwen uses separate norms for Q and K after projection
        if let Some(ref q_norm) = layer.q_norm {
            // q_norm shape is [head_dim], apply per-head
            for h in 0..actual_n_heads {
                let start = h * head_dim;
                let end = start + head_dim;
                if end > q.len() {
                    break;
                }
                let q_head = &q[start..end];
                let normed =
                    self.apply_rms_norm_impl(q_head, q_norm, head_dim)?;
                // Copy back - need mutable slice
                let q_mut = unsafe {
                    std::slice::from_raw_parts_mut(
                        q.as_ptr() as *mut f32,
                        q.len(),
                    )
                };
                q_mut[start..end].copy_from_slice(&normed);
            }
        }
        if let Some(ref k_norm) = layer.k_norm {
            // k_norm shape is [head_dim], apply per-head
            for h in 0..actual_n_kv_heads {
                let start = h * head_dim;
                let end = start + head_dim;
                if end > k.len() {
                    break;
                }
                let k_head = &k[start..end];
                let normed =
                    self.apply_rms_norm_impl(k_head, k_norm, head_dim)?;
                let k_mut = unsafe {
                    std::slice::from_raw_parts_mut(
                        k.as_ptr() as *mut f32,
                        k.len(),
                    )
                };
                k_mut[start..end].copy_from_slice(&normed);
            }
        }

        // 3. RoPE (use actual n_heads)
        let q_rope = self.apply_rope(&q, pos, actual_n_heads, head_dim)?;
        let k_rope = self.apply_rope(&k, pos, actual_n_kv_heads, head_dim)?;

        // 4. Attention (use actual n_heads)
        let attn_out = self.attention(
            &q_rope,
            &k_rope,
            &v,
            layer_idx,
            kv_cache,
            actual_n_heads,
            head_dim,
            actual_n_kv_heads,
        )?;

        // 5. Output projection
        let attn_proj =
            self.gemv_impl(&attn_out, layer.wo.as_ref()?, hidden_size)?;

        // 6. Residual
        let residual: Vec<f32> =
            hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();

        // 7. FFN norm
        let ffn_normed = self.apply_rms_norm_impl(
            &residual,
            layer.ffn_norm.as_ref()?,
            hidden_size,
        )?;

        // 8. FFN (SwiGLU)
        let ffn_out = self.ffn_swiglu(&ffn_normed, layer, hidden_size)?;

        // 9. Residual
        Some(residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect())
    }

    fn apply_rms_norm_impl(
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

    fn gemv_impl(
        &self,
        x: &[f32],
        w: &Tensor,
        out_size: usize,
    ) -> Option<Vec<f32>> {
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

    /// Transposed gemv: weight is stored as [in_size, out_size] instead of [out_size, in_size]
    /// This is needed for tied embeddings where weight shape is [hidden_size, vocab_size]
    fn gemv_transposed(
        &self,
        x: &[f32],
        w: &Tensor,
        out_size: usize,
    ) -> Option<Vec<f32>> {
        let in_size = x.len();
        let total_elems = w.elem_count();
        let w_data = dequant_to_f32(w.bytes(), w.dtype, total_elems);

        // Weight shape is [in_size, out_size], so row stride is out_size
        let row_stride = w.shape[1]; // vocab_size for embeddings

        let mut out = vec![0.0f32; out_size];
        for i in 0..out_size {
            let mut sum = 0.0f32;
            for j in 0..in_size {
                // Weight[j, i] is at position j * row_stride + i
                sum += w_data[j * row_stride + i] * x[j];
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
        self.gemv_impl(&hidden, layer.w2.as_ref()?, hidden_size)
    }

    fn apply_output_norm(&self, hidden: &[f32]) -> Option<Vec<f32>> {
        self.apply_rms_norm_impl(
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
        let output = self.output.as_ref()?;
        // Detect weight format and use appropriate gemv
        if output.shape.len() == 2 && output.shape[0] == hidden.len() {
            // Weight is [hidden_size, vocab_size] - transposed
            self.gemv_transposed(hidden, output, output.shape[1])
        } else {
            // Weight is [vocab_size, hidden_size] - normal
            self.gemv(hidden, output, vocab_size)
        }
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
        // Use actual weight shapes, not config (config may be wrong)
        let q_out_size = layer.wq.as_ref()?.shape[1]; // Column-major: shape = [K, M]
        let k_out_size = layer.wk.as_ref()?.shape[1];
        let v_out_size = layer.wv.as_ref()?.shape[1];

        let q =
            self.gemv_gpu(&normed, layer.wq.as_ref()?, q_out_size, gpu_ctx)?;
        let k =
            self.gemv_gpu(&normed, layer.wk.as_ref()?, k_out_size, gpu_ctx)?;
        let v =
            self.gemv_gpu(&normed, layer.wv.as_ref()?, v_out_size, gpu_ctx)?;

        // Derive actual n_heads from weight shapes
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // 3. RoPE (GPU) - use actual n_heads from weights
        let q_rope =
            self.apply_rope_gpu(&q, pos, actual_n_heads, head_dim, gpu_ctx)?;
        let k_rope =
            self.apply_rope_gpu(&k, pos, actual_n_kv_heads, head_dim, gpu_ctx)?;

        // 4. Attention (GPU - updates KV cache) - use actual n_heads from weights
        let attn_out = self.attention_gpu(
            &q_rope,
            &k_rope,
            &v,
            layer_idx,
            kv_cache,
            actual_n_heads,
            head_dim,
            actual_n_kv_heads,
            gpu_ctx,
        )?;

        // 5. Output projection (GPU)
        let attn_proj =
            self.gemv_gpu(&attn_out, layer.wo.as_ref()?, hidden_size, gpu_ctx)?;

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
        let ffn_out =
            self.ffn_swiglu_gpu(&ffn_normed, layer, hidden_size, gpu_ctx)?;

        // 9. Residual
        Some(residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect())
    }

    /// Forward layer with GPU KV cache (zero-copy Flash Attention)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn forward_layer_gpu_zerocopy(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden: &[f32],
        kv_cache: &mut crate::inference::KVCacheGPU,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
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
        let q_out_size = layer.wq.as_ref()?.shape[1];
        let k_out_size = layer.wk.as_ref()?.shape[1];
        let v_out_size = layer.wv.as_ref()?.shape[1];

        let q =
            self.gemv_gpu(&normed, layer.wq.as_ref()?, q_out_size, gpu_ctx)?;
        let k =
            self.gemv_gpu(&normed, layer.wk.as_ref()?, k_out_size, gpu_ctx)?;
        let v =
            self.gemv_gpu(&normed, layer.wv.as_ref()?, v_out_size, gpu_ctx)?;

        // Derive actual n_heads from weight shapes
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // 3. RoPE (use CPU for now, RoPE is memory-bound)
        let q_rope = self.apply_rope(&q, pos, actual_n_heads, head_dim)?;
        let k_rope = self.apply_rope(&k, pos, actual_n_kv_heads, head_dim)?;

        // 4. Update KV cache on GPU
        kv_cache.set_k(layer_idx, &k_rope);
        kv_cache.set_v(layer_idx, &v);

        // 5. Flash Attention with GPU KV cache
        let attn_out = self.flash_attention_gpu(
            &q_rope,
            kv_cache,
            layer_idx,
            actual_n_heads,
            head_dim,
            actual_n_kv_heads,
            pos + 1, // seq_len after this token
            gpu_ctx,
        )?;

        // 6. Output projection (GPU)
        let attn_proj =
            self.gemv_gpu(&attn_out, layer.wo.as_ref()?, hidden_size, gpu_ctx)?;

        // 7. Residual
        let residual: Vec<f32> =
            hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();

        // 8. FFN norm (GPU)
        let ffn_normed = self.apply_rms_norm_gpu(
            &residual,
            layer.ffn_norm.as_ref()?,
            hidden_size,
            gpu_ctx,
        )?;

        // 9. FFN (SwiGLU) (GPU)
        let ffn_out =
            self.ffn_swiglu_gpu(&ffn_normed, layer, hidden_size, gpu_ctx)?;

        // 10. Residual
        Some(residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect())
    }

    /// Flash Attention with GPU KV cache (zero-copy)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn flash_attention_gpu(
        &self,
        q: &[f32],
        kv_cache: &crate::inference::KVCacheGPU,
        layer_idx: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        seq_len: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        // Get GPU buffers for K and V
        let k_buffer = kv_cache.k_buffer(layer_idx);
        let v_buffer = kv_cache.v_buffer(layer_idx);

        // Create GPU buffer for Q
        let q_buffer = std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
            q.as_ptr() as *const std::ffi::c_void,
            (q.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Run Flash Attention on GPU
        let output_buffer = gpu_ctx
            .flash_attn_f16_gpu(
                1, // n_seqs (single batch)
                n_heads, n_kv_heads, head_dim,
                seq_len, // current sequence length
                &q_buffer, k_buffer, v_buffer,
            )
            .ok()?;

        // Read output back to CPU
        let output_len = n_heads * head_dim;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                output_len,
            );
        }

        Some(output)
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
    ) -> Option<Vec<f32>> {
        let in_size = x.len();
        let w_bytes = w.bytes();

        // Select appropriate MV kernel based on weight dtype
        match w.dtype {
            crate::model::tensor::DataType::F32 => {
                // Fallback to CPU for F32
                self.gemv(x, w, out_size)
            }
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
    ) -> Option<Vec<f32>> {
        let intermediate = hidden_size * 4;

        // Gate and Up projections (GPU)
        let gate =
            self.gemv_gpu(x, layer.w1.as_ref()?, intermediate, gpu_ctx)?;
        let up = self.gemv_gpu(x, layer.w3.as_ref()?, intermediate, gpu_ctx)?;

        // SiLU activation (GPU)
        let silu_gate = gpu_ctx.silu_f32(intermediate, &gate).ok()?;

        // Element-wise multiply (CPU - small operation)
        let hidden: Vec<f32> =
            silu_gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();

        // Down projection (GPU)
        self.gemv_gpu(&hidden, layer.w2.as_ref()?, hidden_size, gpu_ctx)
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
    ) -> Option<Vec<f32>> {
        let output = self.output.as_ref()?;

        // Detect weight format
        if output.shape.len() == 2 && output.shape[0] == hidden.len() {
            // Weight is [hidden_size, vocab_size] - transposed
            // Use CPU transposed gemv for now
            // TODO: Implement GPU transposed MV kernel
            self.gemv_transposed(hidden, output, output.shape[1])
        } else {
            // Weight is [vocab_size, hidden_size] - normal
            self.gemv_gpu(hidden, output, vocab_size, gpu_ctx)
        }
    }

    /// Forward pass using ComputeGraph (zero-copy GPU inference)
    /// All tensors stay on GPU, no CPU round-trips during computation
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_graph(
        &self,
        tokens: &[u32],
        backend: &MetalBackend,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // For now, process tokens one at a time
        // TODO: Batch processing
        let token = tokens[0];
        let pos = 0; // Position for first token

        // 1. Embedding lookup - for now use CPU, TODO: GPU embedding
        let hidden = self.embedding_lookup_impl(token)?;

        // 2. Build compute graph for layer processing
        let mut hidden_tensor = backend
            .alloc_tensor("hidden", vec![hidden_size], DataType::F32)
            .ok()?;
        backend
            .copy_to_tensor(&mut hidden_tensor, bytemuck::cast_slice(&hidden))
            .ok()?;

        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_tensor = self.forward_layer_graph(
                layer,
                layer_idx,
                hidden_tensor,
                pos,
                backend,
            )?;
        }

        // 3. Final norm and output projection
        let final_hidden =
            self.apply_output_norm_graph(&hidden_tensor, backend)?;
        let logits =
            self.output_projection_graph(&final_hidden, vocab_size, backend)?;

        // 4. Copy result back to CPU
        let mut logits_data = vec![0.0f32; vocab_size];
        backend
            .copy_from_tensor(
                &logits,
                bytemuck::cast_slice_mut(&mut logits_data),
            )
            .ok()?;

        Some(logits_data)
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn forward_layer_graph(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden: crate::compute::tensor::Tensor,
        pos: usize,
        backend: &MetalBackend,
    ) -> Option<crate::compute::tensor::Tensor> {
        let hidden_size = self.config.hidden_size;
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let n_kv_heads = self.config.n_kv_heads;

        // === Step 1: Attention Norm ===
        let mut graph = ComputeGraph::new();
        graph.add_tensor(hidden.clone());
        graph.add_input("hidden");

        let attn_norm_w = self
            .tensor_to_compute(layer.attention_norm.as_ref()?, "attn_norm_w")?;
        graph.add_tensor(attn_norm_w);
        graph.add_input("attn_norm_w");

        graph.add_op(OpNode::new(
            OpType::RMSNorm,
            vec!["hidden".into(), "attn_norm_w".into()],
            "normed".to_string(),
        ));

        backend.execute(&mut graph).ok()?;
        let normed = graph.get_tensor("normed")?.clone();

        // === Step 2: QKV Projections ===
        let mut graph = ComputeGraph::new();
        graph.add_tensor(normed.clone());
        graph.add_input("normed");

        // Q projection
        let wq = self.tensor_to_compute(layer.wq.as_ref()?, "wq")?;
        let q_out_size = wq.shape[1];
        graph.add_tensor(wq);
        graph.add_input("wq");
        graph.add_op(OpNode::new(
            OpType::MatVec,
            vec!["wq".into(), "normed".into()],
            "q".to_string(),
        ));

        // K projection
        let wk = self.tensor_to_compute(layer.wk.as_ref()?, "wk")?;
        let k_out_size = wk.shape[1];
        graph.add_tensor(wk);
        graph.add_input("wk");
        graph.add_op(OpNode::new(
            OpType::MatVec,
            vec!["wk".into(), "normed".into()],
            "k".to_string(),
        ));

        // V projection
        let wv = self.tensor_to_compute(layer.wv.as_ref()?, "wv")?;
        let v_out_size = wv.shape[1];
        graph.add_tensor(wv);
        graph.add_input("wv");
        graph.add_op(OpNode::new(
            OpType::MatVec,
            vec!["wv".into(), "normed".into()],
            "v".to_string(),
        ));

        backend.execute(&mut graph).ok()?;
        let q = graph.get_tensor("q")?.clone();
        let k = graph.get_tensor("k")?.clone();
        let v = graph.get_tensor("v")?.clone();

        // Derive actual n_heads from weight shapes
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // === Step 3: RoPE ===
        // Apply RoPE to Q and K
        let mut graph = ComputeGraph::new();
        graph.add_tensor(q.clone());
        graph.add_input("q");

        // RoPE expects shape [n_dims, n_heads, n_seqs]
        // Q shape is [n_heads * head_dim] - reshape to [head_dim, n_heads, 1]
        let mut rope_params = OpParams::default();
        rope_params.pos = Some(pos);
        rope_params.freq_base = Some(self.config.rope_freq_base);

        graph.add_op(
            OpNode::new(OpType::Rope, vec!["q".into()], "q_rope".to_string())
                .with_params(rope_params.clone()),
        );
        backend.execute(&mut graph).ok()?;
        let q_rope = graph.get_tensor("q_rope")?.clone();

        let mut graph = ComputeGraph::new();
        graph.add_tensor(k.clone());
        graph.add_input("k");
        graph.add_op(
            OpNode::new(OpType::Rope, vec!["k".into()], "k_rope".to_string())
                .with_params(rope_params),
        );
        backend.execute(&mut graph).ok()?;
        let k_rope = graph.get_tensor("k_rope")?.clone();

        // === Step 4: Flash Attention ===
        // TODO: Need KV cache integration
        // For now, use simplified attention

        // === Step 5: Output Projection ===
        let mut graph = ComputeGraph::new();
        graph.add_tensor(q_rope.clone()); // Using Q as placeholder for attention output
        graph.add_input("attn_out");

        let wo = self.tensor_to_compute(layer.wo.as_ref()?, "wo")?;
        graph.add_tensor(wo);
        graph.add_input("wo");
        graph.add_op(OpNode::new(
            OpType::MatVec,
            vec!["wo".into(), "attn_out".into()],
            "attn_proj".to_string(),
        ));

        backend.execute(&mut graph).ok()?;
        let attn_proj = graph.get_tensor("attn_proj")?.clone();

        // === Step 6: Residual (GPU) ===
        let mut graph = ComputeGraph::new();
        graph.add_tensor(hidden.clone());
        graph.add_input("hidden");
        graph.add_tensor(attn_proj.clone());
        graph.add_input("attn_proj");
        graph.add_op(OpNode::new(
            OpType::Add,
            vec!["hidden".into(), "attn_proj".into()],
            "residual".to_string(),
        ));
        backend.execute(&mut graph).ok()?;
        let residual = graph.get_tensor("residual")?.clone();

        // === Step 7: FFN Norm ===
        let mut graph = ComputeGraph::new();
        graph.add_tensor(residual.clone());
        graph.add_input("residual");

        let ffn_norm_w =
            self.tensor_to_compute(layer.ffn_norm.as_ref()?, "ffn_norm_w")?;
        graph.add_tensor(ffn_norm_w);
        graph.add_input("ffn_norm_w");

        graph.add_op(OpNode::new(
            OpType::RMSNorm,
            vec!["residual".into(), "ffn_norm_w".into()],
            "ffn_normed".to_string(),
        ));

        backend.execute(&mut graph).ok()?;
        let ffn_normed = graph.get_tensor("ffn_normed")?.clone();

        // === Step 8: FFN (SwiGLU) ===
        let intermediate = hidden_size * 4;

        let mut graph = ComputeGraph::new();
        graph.add_tensor(ffn_normed.clone());
        graph.add_input("ffn_normed");

        // Gate projection
        let w1 = self.tensor_to_compute(layer.w1.as_ref()?, "w1")?;
        graph.add_tensor(w1);
        graph.add_input("w1");
        graph.add_op(OpNode::new(
            OpType::MatVec,
            vec!["w1".into(), "ffn_normed".into()],
            "gate".to_string(),
        ));

        // Up projection
        let w3 = self.tensor_to_compute(layer.w3.as_ref()?, "w3")?;
        graph.add_tensor(w3);
        graph.add_input("w3");
        graph.add_op(OpNode::new(
            OpType::MatVec,
            vec!["w3".into(), "ffn_normed".into()],
            "up".to_string(),
        ));

        backend.execute(&mut graph).ok()?;
        let gate = graph.get_tensor("gate")?.clone();
        let up = graph.get_tensor("up")?.clone();

        // SiLU activation
        let mut graph = ComputeGraph::new();
        graph.add_tensor(gate.clone());
        graph.add_input("gate");
        graph.add_op(OpNode::new(
            OpType::Silu,
            vec!["gate".into()],
            "silu_gate".to_string(),
        ));
        backend.execute(&mut graph).ok()?;
        let silu_gate = graph.get_tensor("silu_gate")?.clone();

        // Element-wise multiply (GPU)
        let mut graph = ComputeGraph::new();
        graph.add_tensor(silu_gate.clone());
        graph.add_input("silu_gate");
        graph.add_tensor(up.clone());
        graph.add_input("up");
        graph.add_op(OpNode::new(
            OpType::Mul,
            vec!["silu_gate".into(), "up".into()],
            "hidden_out".to_string(),
        ));
        backend.execute(&mut graph).ok()?;
        let hidden_out = graph.get_tensor("hidden_out")?.clone();

        // Down projection
        let mut graph = ComputeGraph::new();
        graph.add_tensor(hidden_out.clone());
        graph.add_input("hidden_out");

        let w2 = self.tensor_to_compute(layer.w2.as_ref()?, "w2")?;
        graph.add_tensor(w2);
        graph.add_input("w2");
        graph.add_op(OpNode::new(
            OpType::MatVec,
            vec!["w2".into(), "hidden_out".into()],
            "ffn_out".to_string(),
        ));

        backend.execute(&mut graph).ok()?;
        let ffn_out = graph.get_tensor("ffn_out")?.clone();

        // === Step 9: Final Residual (GPU) ===
        let mut graph = ComputeGraph::new();
        graph.add_tensor(residual.clone());
        graph.add_input("residual");
        graph.add_tensor(ffn_out.clone());
        graph.add_input("ffn_out");
        graph.add_op(OpNode::new(
            OpType::Add,
            vec!["residual".into(), "ffn_out".into()],
            "output".to_string(),
        ));
        backend.execute(&mut graph).ok()?;
        graph.get_tensor("output").cloned()
    }

    /// Helper to read tensor data as f32
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn read_tensor_f32(
        &self,
        tensor: &crate::compute::tensor::Tensor,
        backend: &MetalBackend,
    ) -> Option<Vec<f32>> {
        let n = tensor.elem_count();
        let mut data = vec![0.0f32; n];
        backend
            .copy_from_tensor(tensor, bytemuck::cast_slice_mut(&mut data))
            .ok()?;
        Some(data)
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn apply_output_norm_graph(
        &self,
        hidden: &crate::compute::tensor::Tensor,
        backend: &MetalBackend,
    ) -> Option<crate::compute::tensor::Tensor> {
        let hidden_size = self.config.hidden_size;

        let mut graph = ComputeGraph::new();
        graph.add_tensor(hidden.clone());
        graph.add_input("hidden");

        let weight =
            self.tensor_to_compute(self.norm.as_ref()?, "norm_weight")?;
        graph.add_tensor(weight);
        graph.add_input("norm_weight");

        let norm_op = OpNode::new(
            OpType::RMSNorm,
            vec!["hidden".into(), "norm_weight".into()],
            "normed".to_string(),
        );
        graph.add_op(norm_op);

        backend.execute(&mut graph).ok()?;
        graph.get_tensor("normed").cloned()
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn output_projection_graph(
        &self,
        hidden: &crate::compute::tensor::Tensor,
        vocab_size: usize,
        backend: &MetalBackend,
    ) -> Option<crate::compute::tensor::Tensor> {
        let mut graph = ComputeGraph::new();
        graph.add_tensor(hidden.clone());
        graph.add_input("hidden");

        let weight =
            self.tensor_to_compute(self.output.as_ref()?, "output_weight")?;
        graph.add_tensor(weight);
        graph.add_input("output_weight");

        let matvec_op = OpNode::new(
            OpType::MatVec,
            vec!["output_weight".into(), "hidden".into()],
            "logits".to_string(),
        );
        graph.add_op(matvec_op);

        backend.execute(&mut graph).ok()?;
        let logits = graph.get_tensor("logits")?.clone();

        // Reshape to [vocab_size]
        Some(crate::compute::tensor::Tensor::new(
            "logits",
            vec![vocab_size],
            DataType::F32,
            logits.storage.clone(),
        ))
    }

    /// Convert model Tensor (with optional GPU buffer) to compute Tensor
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn tensor_to_compute(
        &self,
        tensor: &Tensor,
        name: &str,
    ) -> Option<crate::compute::tensor::Tensor> {
        // Check if GPU buffer exists
        if let Some(gpu_buffer) = tensor.gpu_buffer.borrow().as_ref() {
            // Use existing GPU buffer
            return Some(crate::compute::tensor::Tensor::from_metal_buffer(
                name,
                tensor.shape.clone(),
                tensor.dtype,
                gpu_buffer.clone(),
            ));
        }

        // Upload to GPU via backend
        let backend = MetalBackend::new().ok()?;
        let mut compute_tensor = backend
            .alloc_tensor(name, tensor.shape.clone(), tensor.dtype)
            .ok()?;
        backend.copy_to_tensor(&mut compute_tensor, tensor.bytes()).ok()?;

        // Cache the GPU buffer in the model tensor
        if let Some(buffer) = compute_tensor.as_metal_buffer() {
            *tensor.gpu_buffer.borrow_mut() = Some(buffer);
        }

        Some(compute_tensor)
    }
}

impl LlamaLayer {
    pub fn from_weights(store: &WeightStore, layer_idx: usize) -> Self {
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
            q_norm: store
                .get(&format!("{}.attention.q_norm.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_q_norm.weight", qwen_prefix))
                })
                .cloned(),
            k_norm: store
                .get(&format!("{}.attention.k_norm.weight", llama_prefix))
                .or_else(|| {
                    store.get(&format!("{}.attn_k_norm.weight", qwen_prefix))
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
