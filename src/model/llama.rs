// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Llama model architecture definition

#![allow(dead_code)]

use crate::inference::{KVCacheBase, dequant_to_f32};
use crate::model::config::ModelConfig;
use crate::model::tensor::{DataType, Tensor};
use crate::model::weights::WeightStore;
use std::collections::HashMap;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::MetalBackend;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::backend::Backend;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::graph::ComputeGraph;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::compute::{OpNode, OpType};
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;

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

                // Expected GQA ratio from config
                let config_gqa_ratio =
                    model.config.n_heads / model.config.n_kv_heads.max(1);

                // Find all valid head_dim candidates and pick the one closest to config
                let mut best_head_dim = 0;
                let mut best_n_heads_diff = usize::MAX;

                for &candidate in &[32, 64, 80, 128, 160, 256] {
                    if q_out_size % candidate == 0
                        && k_out_size % candidate == 0
                    {
                        let n_heads = q_out_size / candidate;
                        let n_kv_heads = k_out_size / candidate;
                        // Validate: n_heads should be a multiple of n_kv_heads (GQA ratio)
                        let gqa_ratio = n_heads / n_kv_heads;
                        if gqa_ratio == config_gqa_ratio {
                            // Prefer candidate that gives n_heads closest to config
                            let n_heads_diff = (n_heads as i32
                                - model.config.n_heads as i32)
                                .abs()
                                as usize;
                            if n_heads_diff < best_n_heads_diff {
                                best_head_dim = candidate;
                                best_n_heads_diff = n_heads_diff;
                            }
                        }
                    }
                }

                if best_head_dim == 0 {
                    // Fall back to config's head_dim
                    best_head_dim = model.config.head_dim;
                }

                let actual_n_heads = q_out_size / best_head_dim;
                let actual_n_kv_heads = k_out_size / best_head_dim;

                if actual_n_heads != model.config.n_heads
                    || actual_n_kv_heads != model.config.n_kv_heads
                    || best_head_dim != model.config.head_dim
                {
                    model.config.n_heads = actual_n_heads;
                    model.config.n_kv_heads = actual_n_kv_heads;
                    model.config.head_dim = best_head_dim;
                    model.config.rope_dim = best_head_dim; // Also fix rope_dim
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

    /// Fully GPU-native forward pass (no CPU-GPU transfers for intermediate tensors)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu_native(
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
        let _vocab_size = self.config.vocab_size;

        // Process each token, keeping track of the last hidden state
        let mut last_hidden_buffer: Option<std::rc::Rc<metal::Buffer>> = None;

        for &token in tokens.iter() {
            // 1. Embedding lookup (CPU - unavoidable for token lookup)
            let hidden = self.embedding_lookup_impl(token)?;

            // Upload hidden state to GPU
            let hidden_buffer =
                std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
                    hidden.as_ptr() as *const std::ffi::c_void,
                    (hidden_size * std::mem::size_of::<f32>()) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                ));

            // 2. Process through layers (fully on GPU)
            let mut hidden_buffer = hidden_buffer;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden_buffer = self.forward_layer_gpu_buffer(
                    layer,
                    layer_idx,
                    &hidden_buffer,
                    kv_cache,
                    n_heads,
                    head_dim,
                    n_kv_heads,
                    gpu_ctx,
                )?;
            }

            // Save the last hidden buffer
            last_hidden_buffer = Some(hidden_buffer);

            // Advance KV cache position
            kv_cache.advance();
        }

        // 3. Final norm and output projection (GPU)
        let hidden_buffer = last_hidden_buffer?;
        let final_norm = self.norm.as_ref()?;
        let final_norm_buffer = gpu_ctx.get_or_create_buffer(final_norm);
        let final_hidden_buffer = gpu_ctx
            .rms_norm_f32_gpu(
                &hidden_buffer,
                &final_norm_buffer,
                1,
                hidden_size,
                1e-5,
            )
            .ok()?;

        // Output projection
        let output = self.output.as_ref()?;

        // Check if weights are transposed
        let is_transposed =
            output.shape.len() == 2 && output.shape[0] == hidden_size;

        let actual_vocab_size = if is_transposed {
            // Weight is [hidden_size, vocab_size] - transposed
            output.shape[1]
        } else {
            // Weight is [vocab_size, hidden_size] - normal
            output.shape[0]
        };

        let logits = if is_transposed {
            // Transposed weights: fallback to CPU for now
            // TODO: Implement GPU transposed GEMV
            let hidden_data: Vec<f32> = unsafe {
                let ptr = final_hidden_buffer.contents() as *const f32;
                let len = hidden_size;
                let mut data = vec![0.0f32; len];
                std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), len);
                data
            };
            self.gemv_transposed(&hidden_data, output, actual_vocab_size)?
        } else {
            // Normal weights: use GPU
            let output_buffer = gpu_ctx.get_or_create_buffer(output);
            let logits_buffer = self.gemv_gpu_buffer(
                &final_hidden_buffer,
                &output_buffer,
                actual_vocab_size,
                output.dtype,
                gpu_ctx,
            )?;

            // Download logits to CPU
            let mut logits = vec![0.0f32; actual_vocab_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    logits_buffer.contents() as *const f32,
                    logits.as_mut_ptr(),
                    actual_vocab_size,
                );
            }
            logits
        };

        Some(logits)
    }

    /// Batch prefill using GEMM (more efficient for multiple tokens)
    /// Process all tokens in parallel using batch matrix operations
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_prefill_batch(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let batch_size = tokens.len();
        let hidden_size = self.config.hidden_size;
        let _n_heads = self.config.n_heads;
        let _head_dim = self.config.head_dim;
        let _n_kv_heads = self.config.n_kv_heads;

        // 1. Batch embedding lookup: [batch_size, hidden_size]
        let hidden = self.embedding_lookup_batch(tokens)?;

        // For batch prefill, we use GEMM with N = batch_size
        // W @ X^T where X is [batch_size, hidden_size]
        // Result is [batch_size, hidden_size] for each weight

        // Process each layer with batch operations
        let mut layer_hidden = hidden;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 1. Attention norm (broadcast across batch)
            let attn_norm = layer.attention_norm.as_ref()?;
            let mut normed = Vec::with_capacity(batch_size * hidden_size);
            for i in 0..batch_size {
                let token_hidden =
                    &layer_hidden[i * hidden_size..(i + 1) * hidden_size];
                let token_normed =
                    self.apply_rms_norm(token_hidden, attn_norm, hidden_size)?;
                normed.extend(token_normed);
            }

            // 2. QKV projections using GEMM
            // GGUF weights are column-major: shape = [K, M] where K=input, M=output
            let wq = layer.wq.as_ref()?;
            let wk = layer.wk.as_ref()?;
            let wv = layer.wv.as_ref()?;

            let q_out_size = wq.shape[1]; // n_heads * head_dim
            let k_out_size = wk.shape[1]; // n_kv_heads * head_dim
            let v_out_size = wv.shape[1]; // n_kv_heads * head_dim

            // Derive actual head dimensions from weight shapes
            let actual_head_dim = k_out_size / self.config.n_kv_heads;
            let actual_n_heads = q_out_size / actual_head_dim;
            let actual_n_kv_heads = k_out_size / actual_head_dim;

            let q =
                self.gemm_batch(&normed, wq, q_out_size, batch_size, gpu_ctx)?;
            let k =
                self.gemm_batch(&normed, wk, k_out_size, batch_size, gpu_ctx)?;
            let v =
                self.gemm_batch(&normed, wv, v_out_size, batch_size, gpu_ctx)?;

            // 3. RoPE (apply to each token)
            let mut q_rope = Vec::with_capacity(batch_size * q_out_size);
            let mut k_rope = Vec::with_capacity(batch_size * k_out_size);

            for i in 0..batch_size {
                let pos = i; // Position in sequence
                let q_token = &q[i * q_out_size..(i + 1) * q_out_size];
                let k_token = &k[i * k_out_size..(i + 1) * k_out_size];

                let q_r = self.apply_rope(
                    q_token,
                    pos,
                    actual_n_heads,
                    actual_head_dim,
                )?;
                let k_r = self.apply_rope(
                    k_token,
                    pos,
                    actual_n_kv_heads,
                    actual_head_dim,
                )?;

                // Update KV cache for each token (before extending)
                let v_token = &v[i * v_out_size..(i + 1) * v_out_size];
                kv_cache.set_k(layer_idx, &k_r);
                kv_cache.set_v(layer_idx, v_token);
                kv_cache.advance();

                q_rope.extend(q_r);
                k_rope.extend(k_r);
            }

            // 4. Flash attention (process each query against all K,V up to that position)
            let mut attn_out = Vec::with_capacity(batch_size * hidden_size);
            for i in 0..batch_size {
                let q_token = &q_rope[i * q_out_size..(i + 1) * q_out_size];
                let attn = self.flash_attention_prefill(
                    q_token,
                    kv_cache,
                    layer_idx,
                    actual_n_heads,
                    actual_head_dim,
                    actual_n_kv_heads,
                    i + 1, // seq_len: positions 0 through i are available
                )?;
                attn_out.extend(attn);
            }

            // 5. Output projection using GEMM
            let wo = layer.wo.as_ref()?;
            let attn_proj = self.gemm_batch(
                &attn_out,
                wo,
                hidden_size,
                batch_size,
                gpu_ctx,
            )?;

            // 6. Residual add
            let residual: Vec<f32> = layer_hidden
                .iter()
                .zip(attn_proj.iter())
                .map(|(h, a)| h + a)
                .collect();

            // 7. FFN norm (broadcast across batch)
            let ffn_norm = layer.ffn_norm.as_ref()?;
            let mut ffn_normed = Vec::with_capacity(batch_size * hidden_size);
            for i in 0..batch_size {
                let token_hidden =
                    &residual[i * hidden_size..(i + 1) * hidden_size];
                let token_normed =
                    self.apply_rms_norm(token_hidden, ffn_norm, hidden_size)?;
                ffn_normed.extend(token_normed);
            }

            // 8. FFN (SwiGLU) using GEMM
            let w1 = layer.w1.as_ref()?;
            let w2 = layer.w2.as_ref()?;
            let w3 = layer.w3.as_ref()?;

            // FFN weight shapes:
            // w1 (gate): [hidden_size, intermediate_size]
            // w3 (up): [hidden_size, intermediate_size]
            // w2 (down): [intermediate_size, hidden_size]
            let gate = self.gemm_batch(
                &ffn_normed,
                w1,
                w1.shape[1],
                batch_size,
                gpu_ctx,
            )?;
            let up = self.gemm_batch(
                &ffn_normed,
                w3,
                w3.shape[1],
                batch_size,
                gpu_ctx,
            )?;

            // SiLU activation
            let swiglu: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(g, u)| {
                    let silu = g / (1.0 + (-g).exp());
                    silu * u
                })
                .collect();

            let ffn_out =
                self.gemm_batch(&swiglu, w2, hidden_size, batch_size, gpu_ctx)?;

            // 9. Final residual
            layer_hidden = residual
                .iter()
                .zip(ffn_out.iter())
                .map(|(r, f)| r + f)
                .collect();
        }

        // Final norm and output (only for last token)
        let final_hidden = &layer_hidden[(batch_size - 1) * hidden_size..];
        let final_norm = self.norm.as_ref()?;
        let final_normed =
            self.apply_rms_norm(final_hidden, final_norm, hidden_size)?;

        let output = self.output.as_ref()?;
        let logits =
            self.gemv(&final_normed, output, self.config.vocab_size)?;

        Some(logits)
    }

    /// Debug version of forward_prefill_batch that checks intermediate values
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_prefill_batch_debug(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let batch_size = tokens.len();
        let hidden_size = self.config.hidden_size;

        fn check_stats(name: &str, data: &[f32]) {
            let num_nan = data.iter().filter(|&&v| v.is_nan()).count();
            let num_inf = data.iter().filter(|&&v| v.is_infinite()).count();
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            println!(
                "  {}: len={}, nan={}, inf={}, range=[{:.4}, {:.4}]",
                name,
                data.len(),
                num_nan,
                num_inf,
                min,
                max
            );
        }

        // 1. Batch embedding lookup
        let hidden = self.embedding_lookup_batch(tokens)?;
        check_stats("embeddings", &hidden);

        let mut layer_hidden = hidden;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            println!("\nLayer {}:", layer_idx);

            // 1. Attention norm
            let attn_norm = layer.attention_norm.as_ref()?;
            let mut normed = Vec::with_capacity(batch_size * hidden_size);
            for i in 0..batch_size {
                let token_hidden =
                    &layer_hidden[i * hidden_size..(i + 1) * hidden_size];
                let token_normed =
                    self.apply_rms_norm(token_hidden, attn_norm, hidden_size)?;
                normed.extend(token_normed);
            }
            check_stats("attn_normed", &normed);

            // 2. QKV projections
            let wq = layer.wq.as_ref()?;
            let wk = layer.wk.as_ref()?;
            let wv = layer.wv.as_ref()?;

            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let v_out_size = wv.shape[1];

            let actual_head_dim = k_out_size / self.config.n_kv_heads;
            let actual_n_heads = q_out_size / actual_head_dim;
            let actual_n_kv_heads = k_out_size / actual_head_dim;

            let q =
                self.gemm_batch(&normed, wq, q_out_size, batch_size, gpu_ctx)?;
            check_stats("q", &q);
            let k =
                self.gemm_batch(&normed, wk, k_out_size, batch_size, gpu_ctx)?;
            check_stats("k", &k);
            let v =
                self.gemm_batch(&normed, wv, v_out_size, batch_size, gpu_ctx)?;
            check_stats("v", &v);

            // 3. RoPE
            let mut q_rope = Vec::with_capacity(batch_size * q_out_size);
            let mut k_rope = Vec::with_capacity(batch_size * k_out_size);

            for i in 0..batch_size {
                let pos = i;
                let q_token = &q[i * q_out_size..(i + 1) * q_out_size];
                let k_token = &k[i * k_out_size..(i + 1) * k_out_size];

                let q_r = self.apply_rope(
                    q_token,
                    pos,
                    actual_n_heads,
                    actual_head_dim,
                )?;
                let k_r = self.apply_rope(
                    k_token,
                    pos,
                    actual_n_kv_heads,
                    actual_head_dim,
                )?;

                let v_token = &v[i * v_out_size..(i + 1) * v_out_size];
                kv_cache.set_k(layer_idx, &k_r);
                kv_cache.set_v(layer_idx, v_token);
                kv_cache.advance();

                q_rope.extend(q_r);
                k_rope.extend(k_r);
            }
            check_stats("q_rope", &q_rope);
            check_stats("k_rope", &k_rope);

            // 4. Flash attention
            let mut attn_out = Vec::with_capacity(batch_size * hidden_size);
            for i in 0..batch_size {
                let q_token = &q_rope[i * q_out_size..(i + 1) * q_out_size];
                let attn = self.flash_attention_prefill(
                    q_token,
                    kv_cache,
                    layer_idx,
                    actual_n_heads,
                    actual_head_dim,
                    actual_n_kv_heads,
                    i + 1,
                )?;
                attn_out.extend(attn);
            }
            check_stats("attn_out", &attn_out);

            // 5. Output projection
            let wo = layer.wo.as_ref()?;
            let attn_proj = self.gemm_batch(
                &attn_out,
                wo,
                hidden_size,
                batch_size,
                gpu_ctx,
            )?;
            check_stats("attn_proj", &attn_proj);

            // 6. Residual
            let residual: Vec<f32> = layer_hidden
                .iter()
                .zip(attn_proj.iter())
                .map(|(h, a)| h + a)
                .collect();
            check_stats("residual", &residual);

            // 7. FFN norm
            let ffn_norm = layer.ffn_norm.as_ref()?;
            let mut ffn_normed = Vec::with_capacity(batch_size * hidden_size);
            for i in 0..batch_size {
                let token_hidden =
                    &residual[i * hidden_size..(i + 1) * hidden_size];
                let token_normed =
                    self.apply_rms_norm(token_hidden, ffn_norm, hidden_size)?;
                ffn_normed.extend(token_normed);
            }
            check_stats("ffn_normed", &ffn_normed);

            // 8. FFN
            let w1 = layer.w1.as_ref()?;
            let w2 = layer.w2.as_ref()?;
            let w3 = layer.w3.as_ref()?;

            // FFN weight shapes:
            // w1 (gate): [hidden_size, intermediate_size]
            // w3 (up): [hidden_size, intermediate_size]
            // w2 (down): [intermediate_size, hidden_size]
            let gate = self.gemm_batch(
                &ffn_normed,
                w1,
                w1.shape[1],
                batch_size,
                gpu_ctx,
            )?;
            check_stats("gate", &gate);
            let up = self.gemm_batch(
                &ffn_normed,
                w3,
                w3.shape[1],
                batch_size,
                gpu_ctx,
            )?;
            check_stats("up", &up);

            let swiglu: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(g, u)| {
                    let silu = g / (1.0 + (-g).exp());
                    silu * u
                })
                .collect();
            check_stats("swiglu", &swiglu);

            let ffn_out =
                self.gemm_batch(&swiglu, w2, hidden_size, batch_size, gpu_ctx)?;
            check_stats("ffn_out", &ffn_out);

            // 9. Final residual
            layer_hidden = residual
                .iter()
                .zip(ffn_out.iter())
                .map(|(r, f)| r + f)
                .collect();

            // Stop after first layer if NaN detected
            if layer_hidden.iter().any(|&v| v.is_nan()) {
                println!("  NaN detected after layer {}, stopping", layer_idx);
                break;
            }
        }

        // Final norm and output
        let final_hidden = &layer_hidden[(batch_size - 1) * hidden_size..];
        let final_norm = self.norm.as_ref()?;
        let final_normed =
            self.apply_rms_norm(final_hidden, final_norm, hidden_size)?;
        check_stats("final_normed", &final_normed);

        let output = self.output.as_ref()?;
        let logits =
            self.gemv(&final_normed, output, self.config.vocab_size)?;
        check_stats("logits", &logits);

        Some(logits)
    }

    /// GEMM for batch processing: W @ X^T
    /// x: [batch, k] (input batch)
    /// w: [m, k] quantized weights
    /// Returns: [batch, m]
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn gemm_batch(
        &self,
        x: &[f32],
        w: &Tensor,
        m: usize,
        batch: usize,
        _gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        // Derive K from the weight matrix shape
        // GGUF stores weights in column-major: shape = [K, M]
        let k = w.shape[0];

        // For now, use sequential GEMV since GEMM template has issues with batch > 1
        // TODO: Fix GEMM template for batch processing
        let mut result = Vec::with_capacity(batch * m);
        for i in 0..batch {
            let x_token = &x[i * k..(i + 1) * k];
            let out = self.gemv(x_token, w, m)?;
            result.extend(out);
        }
        Some(result)

        /* Original GEMM code (not working for batch > 1)
        match w.dtype {
            crate::model::tensor::DataType::Q4_K => {
                let weights: &[BlockQ4K] = bytemuck::cast_slice(w.bytes());
                gpu_ctx.gemm_template_q4_k_f32(m, batch, k, weights, x).ok()
            }
            crate::model::tensor::DataType::Q5_K => {
                let weights: &[BlockQ5K] = bytemuck::cast_slice(w.bytes());
                gpu_ctx.gemm_template_q5_k_f32(m, batch, k, weights, x).ok()
            }
            crate::model::tensor::DataType::Q2_K => {
                let weights: &[BlockQ2K] = bytemuck::cast_slice(w.bytes());
                gpu_ctx.gemm_template_q2_k_f32(m, batch, k, weights, x).ok()
            }
            crate::model::tensor::DataType::Q3_K => {
                let weights: &[BlockQ3K] = bytemuck::cast_slice(w.bytes());
                gpu_ctx.gemm_template_q3_k_f32(m, batch, k, weights, x).ok()
            }
            crate::model::tensor::DataType::Q6_K => {
                let weights: &[BlockQ6K] = bytemuck::cast_slice(w.bytes());
                gpu_ctx.gemm_template_q6_k_f32(m, batch, k, weights, x).ok()
            }
            _ => {
                // Fallback to sequential GEMV for unsupported formats
                let mut result = Vec::with_capacity(batch * m);
                for i in 0..batch {
                    let x_token = &x[i * k..(i + 1) * k];
                    let out = self.gemv(x_token, w, m)?;
                    result.extend(out);
                }
                Some(result)
            }
        }
        */
    }

    /// Flash attention for prefill: query against all KV positions
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn flash_attention_prefill(
        &self,
        q: &[f32],
        kv_cache: &crate::inference::KVCacheGPU,
        layer_idx: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        seq_len: usize, // Number of KV positions available
    ) -> Option<Vec<f32>> {
        // For non-standard architectures where hidden_size != n_heads * head_dim,
        // we still need to output the correct size for the attention output
        let q_out_size = n_heads * head_dim;
        let mut output = vec![0.0f32; q_out_size];
        let n_rep = n_heads / n_kv_heads; // GQA

        for h in 0..n_heads {
            let kv_head = h / n_rep;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Get Q for this head
            let q_start = h * head_dim;
            let q_head = &q[q_start..q_start + head_dim];

            // Compute attention scores for all KV positions
            let mut scores: Vec<f32> = Vec::with_capacity(seq_len);

            for pos in 0..seq_len {
                let k_vals = kv_cache.get_k(layer_idx, pos);
                let k_start = kv_head * head_dim;
                let k_head = &k_vals[k_start..k_start + head_dim];

                // QK^T
                let score: f32 = q_head
                    .iter()
                    .zip(k_head.iter())
                    .map(|(qi, ki)| qi * ki)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }

            // Softmax
            let max_score =
                scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 =
                scores.iter().map(|s| (s - max_score).exp()).sum();
            let softmax: Vec<f32> = scores
                .iter()
                .map(|s| (s - max_score).exp() / exp_sum)
                .collect();

            // Weighted sum of V
            for pos in 0..seq_len {
                let v_vals = kv_cache.get_v(layer_idx, pos);
                let v_start = kv_head * head_dim;
                let v_head = &v_vals[v_start..v_start + head_dim];
                let weight = softmax[pos];

                for (i, &v) in v_head.iter().enumerate() {
                    output[h * head_dim + i] += weight * v;
                }
            }
        }

        Some(output)
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

    /// Batch embedding lookup for prefill
    /// Returns [batch_size, hidden_size] in row-major order
    fn embedding_lookup_batch(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let mut result = Vec::with_capacity(tokens.len() * hidden_size);

        for &token in tokens {
            let embedding = self.embedding_lookup_impl(token)?;
            result.extend(embedding);
        }

        Some(result)
    }

    fn forward_layer(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden: &[f32],
        kv_cache: &mut crate::inference::KVCache,
        _n_heads: usize,
        head_dim: usize,
        _n_kv_heads: usize,
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

        // Optimize by processing in blocks for better cache behavior
        // For Q6_K: block_size = 256
        let block_size = 256;

        match w.dtype {
            crate::model::tensor::DataType::Q6_K => {
                // Process blocks without full dequantization
                let bytes = w.bytes();
                let block_size_bytes =
                    std::mem::size_of::<crate::quant::types::BlockQ6K>();
                let blocks_per_row = out_size / block_size;

                let mut out = vec![0.0f32; out_size];

                // For each input element
                for j in 0..in_size {
                    let x_val = x[j];

                    // For each block in this row
                    for block_idx in 0..blocks_per_row {
                        let byte_offset =
                            (j * blocks_per_row + block_idx) * block_size_bytes;
                        let block_bytes =
                            &bytes[byte_offset..byte_offset + block_size_bytes];
                        let block: crate::quant::types::BlockQ6K = unsafe {
                            std::ptr::read(block_bytes.as_ptr() as *const _)
                        };

                        // Dequantize the block
                        let block_vals =
                            crate::inference::dequant::dequantize_q6_k_block(
                                &block,
                            );

                        // Add contribution to 256 outputs
                        let output_start = block_idx * block_size;
                        for k in 0..block_size {
                            out[output_start + k] += block_vals[k] * x_val;
                        }
                    }
                }

                Some(out)
            }
            _ => {
                // Fallback to full dequantization for other formats
                let total_elems = w.elem_count();
                let w_data = crate::inference::dequant_to_f32(
                    w.bytes(),
                    w.dtype,
                    total_elems,
                );

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
        }
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
        _n_heads: usize,
        head_dim: usize,
        _n_kv_heads: usize,
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
        _n_heads: usize,
        head_dim: usize,
        _n_kv_heads: usize,
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

    /// Forward layer with fully GPU-native operations (no CPU-GPU transfers)
    /// Input and output are GPU buffers
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn forward_layer_gpu_buffer(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden_buffer: &std::rc::Rc<Buffer>,
        kv_cache: &mut crate::inference::KVCacheGPU,
        _n_heads: usize,
        head_dim: usize,
        _n_kv_heads: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<std::rc::Rc<Buffer>> {
        let hidden_size = self.config.hidden_size;
        let pos = kv_cache.seq_len();

        // 1. Attention norm (GPU, buffer in/out)
        let attention_norm = layer.attention_norm.as_ref()?;
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
        let normed_buffer = gpu_ctx
            .rms_norm_f32_gpu(
                hidden_buffer,
                &norm_weight_buffer,
                1,
                hidden_size,
                1e-5,
            )
            .ok()?;

        // 2. QKV projections (GPU, buffer in/out)
        let q_out_size = layer.wq.as_ref()?.shape[1];
        let k_out_size = layer.wk.as_ref()?.shape[1];
        let v_out_size = layer.wv.as_ref()?.shape[1];

        let wq = layer.wq.as_ref()?;
        let wk = layer.wk.as_ref()?;
        let wv = layer.wv.as_ref()?;

        // Get or create weight buffers (cached)
        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        let wk_buffer = gpu_ctx.get_or_create_buffer(wk);
        let wv_buffer = gpu_ctx.get_or_create_buffer(wv);

        // GEMV with GPU buffers
        let q_buffer = self.gemv_gpu_buffer(
            &normed_buffer,
            &wq_buffer,
            q_out_size,
            wq.dtype,
            gpu_ctx,
        )?;
        let k_buffer = self.gemv_gpu_buffer(
            &normed_buffer,
            &wk_buffer,
            k_out_size,
            wk.dtype,
            gpu_ctx,
        )?;
        let v_buffer = self.gemv_gpu_buffer(
            &normed_buffer,
            &wv_buffer,
            v_out_size,
            wv.dtype,
            gpu_ctx,
        )?;

        // Derive actual n_heads from weight shapes
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // 3. RoPE on GPU
        let q_rope_buffer = gpu_ctx
            .rope_neox_f32_gpu(
                &q_buffer,
                &[pos as i32],
                1,
                actual_n_heads,
                head_dim,
                10000.0,
                1.0,
            )
            .ok()?;
        let k_rope_buffer = gpu_ctx
            .rope_neox_f32_gpu(
                &k_buffer,
                &[pos as i32],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
            )
            .ok()?;

        // 4. Update KV cache on GPU (direct buffer copy)
        kv_cache.set_k_buffer(layer_idx, &k_rope_buffer);
        kv_cache.set_v_buffer(layer_idx, &v_buffer);

        // 5. Flash Attention with GPU buffers
        let attn_out_buffer = self.flash_attention_gpu_buffer(
            &q_rope_buffer,
            kv_cache,
            layer_idx,
            actual_n_heads,
            head_dim,
            actual_n_kv_heads,
            pos + 1,
            gpu_ctx,
        )?;

        // 6. Output projection (GPU)
        let wo = layer.wo.as_ref()?;
        let wo_buffer = gpu_ctx.get_or_create_buffer(wo);
        let attn_proj_buffer = self.gemv_gpu_buffer(
            &attn_out_buffer,
            &wo_buffer,
            hidden_size,
            wo.dtype,
            gpu_ctx,
        )?;

        // 7. Residual add on GPU
        let residual_buffer = gpu_ctx
            .add_f32_gpu(hidden_buffer, &attn_proj_buffer, hidden_size)
            .ok()?;

        // 8. FFN norm (GPU)
        let ffn_norm = layer.ffn_norm.as_ref()?;
        let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
        let ffn_normed_buffer = gpu_ctx
            .rms_norm_f32_gpu(
                &residual_buffer,
                &ffn_norm_buffer,
                1,
                hidden_size,
                1e-5,
            )
            .ok()?;

        // 9. FFN (SwiGLU) on GPU
        let ffn_out_buffer = self.ffn_swiglu_gpu_buffer(
            &ffn_normed_buffer,
            layer,
            hidden_size,
            gpu_ctx,
        )?;

        // 10. Final residual on GPU
        gpu_ctx.add_f32_gpu(&residual_buffer, &ffn_out_buffer, hidden_size).ok()
    }

    /// GEMV with GPU buffers (no CPU transfer for supported formats)
    /// For unsupported formats, falls back to CPU (with GPU-CPU transfer)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn gemv_gpu_buffer(
        &self,
        input_buffer: &std::rc::Rc<Buffer>,
        weights_buffer: &std::rc::Rc<Buffer>,
        out_size: usize,
        dtype: crate::model::tensor::DataType,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<std::rc::Rc<Buffer>> {
        // Get input size from buffer length
        let in_size =
            input_buffer.length() as usize / std::mem::size_of::<f32>();

        match dtype {
            crate::model::tensor::DataType::Q4_0 => gpu_ctx
                .mv_q4_0_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q4_1 => gpu_ctx
                .mv_q4_1_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q5_0 => gpu_ctx
                .mv_q5_0_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q5_1 => gpu_ctx
                .mv_q5_1_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q8_0 => gpu_ctx
                .mv_q8_0_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // K-quant formats
            crate::model::tensor::DataType::Q2_K => gpu_ctx
                .mv_q2_k_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q3_K => gpu_ctx
                .mv_q3_k_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q4_K => gpu_ctx
                .mv_q4_k_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q5_K => gpu_ctx
                .mv_q5_k_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            crate::model::tensor::DataType::Q6_K => gpu_ctx
                .mv_q6_k_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ4_NL - supported
            crate::model::tensor::DataType::IQ4_NL => gpu_ctx
                .mv_iq4_nl_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ4_XS - supported
            crate::model::tensor::DataType::IQ4_XS => gpu_ctx
                .mv_iq4_xs_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ1_S - supported
            crate::model::tensor::DataType::IQ1_S => gpu_ctx
                .mv_iq1_s_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ2_XXS - supported
            crate::model::tensor::DataType::IQ2_XXS => gpu_ctx
                .mv_iq2_xxs_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ2_XS - supported
            crate::model::tensor::DataType::IQ2_XS => gpu_ctx
                .mv_iq2_xs_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ2_S - supported
            crate::model::tensor::DataType::IQ2_S => gpu_ctx
                .mv_iq2_s_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ3_XXS - supported
            crate::model::tensor::DataType::IQ3_XXS => gpu_ctx
                .mv_iq3_xxs_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // IQ3_S - supported
            crate::model::tensor::DataType::IQ3_S => gpu_ctx
                .mv_iq3_s_f32_gpu_weights(
                    out_size,
                    in_size,
                    weights_buffer,
                    input_buffer,
                )
                .ok(),
            // Unsupported formats - use CPU fallback
            _ => None,
        }
    }

    /// Flash Attention with GPU buffers (no CPU transfer)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn flash_attention_gpu_buffer(
        &self,
        q_buffer: &std::rc::Rc<Buffer>,
        kv_cache: &crate::inference::KVCacheGPU,
        layer_idx: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        seq_len: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<std::rc::Rc<Buffer>> {
        // Get K and V buffers from GPU KV cache
        let k_buffer = kv_cache.k_buffer(layer_idx);
        let v_buffer = kv_cache.v_buffer(layer_idx);

        // Run flash attention on GPU
        gpu_ctx
            .flash_attn_f16_gpu(
                1, // n_seqs
                n_heads, n_kv_heads, head_dim, seq_len, q_buffer, &k_buffer,
                &v_buffer,
            )
            .ok()
    }

    /// FFN SwiGLU with GPU buffers (no CPU transfer)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn ffn_swiglu_gpu_buffer(
        &self,
        input_buffer: &std::rc::Rc<Buffer>,
        layer: &LlamaLayer,
        hidden_size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<std::rc::Rc<Buffer>> {
        let w1 = layer.w1.as_ref()?;
        let w2 = layer.w2.as_ref()?;
        let w3 = layer.w3.as_ref()?;

        let intermediate_size = w2.shape[1];

        // Get or create weight buffers
        let w1_buffer = gpu_ctx.get_or_create_buffer(w1);
        let w2_buffer = gpu_ctx.get_or_create_buffer(w2);
        let w3_buffer = gpu_ctx.get_or_create_buffer(w3);

        // Gate projection: gate = input @ w1
        let gate_buffer = self.gemv_gpu_buffer(
            input_buffer,
            &w1_buffer,
            intermediate_size,
            w1.dtype,
            gpu_ctx,
        )?;

        // Up projection: up = input @ w3
        let up_buffer = self.gemv_gpu_buffer(
            input_buffer,
            &w3_buffer,
            intermediate_size,
            w3.dtype,
            gpu_ctx,
        )?;

        // SiLU(gate) * up on GPU
        let silu_gate_buffer =
            gpu_ctx.silu_f32_gpu(&gate_buffer, intermediate_size).ok()?;
        let swiglu_buffer = gpu_ctx
            .mul_f32_gpu(&silu_gate_buffer, &up_buffer, intermediate_size)
            .ok()?;

        // Down projection: output = swiglu @ w2
        self.gemv_gpu_buffer(
            &swiglu_buffer,
            &w2_buffer,
            hidden_size,
            w2.dtype,
            gpu_ctx,
        )
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

    /// Simple CPU attention for fallback (when GPU Flash Attention not supported)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    fn attention_cpu_simple(
        &self,
        q: &[f32],
        kv_cache: &crate::inference::KVCacheGPU,
        layer_idx: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_groups = n_heads / n_kv_heads;

        let mut output = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_head = h / n_groups;

            // Compute attention scores for this head
            let mut scores = vec![0.0f32; seq_len];
            for pos in 0..seq_len {
                // Get K for this position
                let k_pos = kv_cache.get_k(layer_idx, pos);
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = h * head_dim + d;
                    let k_idx = kv_head * head_dim + d;
                    score += q[q_idx] * k_pos[k_idx];
                }
                scores[pos] = score * scale;
            }

            // Softmax
            let max_score =
                scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
            for s in &mut scores {
                *s = (*s - max_score).exp() / sum;
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for pos in 0..seq_len {
                    let v_pos = kv_cache.get_v(layer_idx, pos);
                    let v_idx = kv_head * head_dim + d;
                    val += scores[pos] * v_pos[v_idx];
                }
                output[h * head_dim + d] = val;
            }
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
        _gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
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

    /// Forward pass using ComputeGraph architecture (zero-copy GPU inference)
    /// This is the new architecture that minimizes kernel dispatch overhead
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_graph(
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

        // Create MetalBackend
        let backend = MetalBackend::new(gpu_ctx);
        let ctx = crate::compute::backend::ExecutionContext {
            device: crate::compute::storage::Device::Metal,
            seq_len: kv_cache.seq_len(),
            batch_size: 1,
        };

        // Process each token
        for &token in tokens.iter() {
            // 1. Embedding lookup (CPU for now)
            let hidden = self.embedding_lookup_impl(token)?;
            let hidden_bytes = unsafe {
                std::slice::from_raw_parts(
                    hidden.as_ptr() as *const u8,
                    hidden.len() * std::mem::size_of::<f32>(),
                )
            };

            // Upload to GPU
            let hidden_storage = backend.alloc_tensor_with_data(
                &[hidden_size],
                DataType::F32,
                hidden_bytes,
            );

            // Build compute graph for this token
            let mut graph = ComputeGraph::new();

            // Add input hidden state
            graph.add_tensor(
                "hidden",
                vec![hidden_size],
                DataType::F32,
                Some(hidden_storage),
            );
            graph.add_input("hidden");

            // Process through layers
            let mut current_hidden = "hidden".to_string();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // Add layer weights to graph
                let norm_name = format!("layer{}_attn_norm", layer_idx);
                let norm_w = layer.attention_norm.as_ref()?;
                let norm_storage = gpu_ctx.get_or_create_buffer(norm_w);
                graph.add_tensor(
                    &norm_name,
                    norm_w.shape.clone(),
                    norm_w.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(norm_storage.as_ref().clone()),
                    )),
                );

                // RMS Norm
                let normed_name = format!("layer{}_normed", layer_idx);
                graph.add_op(
                    OpType::RmsNorm { eps: 1e-5 },
                    vec![current_hidden.clone(), norm_name.clone()],
                    normed_name.clone(),
                );

                // QKV projections
                let wq = layer.wq.as_ref()?;
                let wk = layer.wk.as_ref()?;
                let wv = layer.wv.as_ref()?;

                let wq_name = format!("layer{}_wq", layer_idx);
                let wq_storage = gpu_ctx.get_or_create_buffer(wq);
                graph.add_tensor(
                    &wq_name,
                    wq.shape.clone(),
                    wq.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(wq_storage.as_ref().clone()),
                    )),
                );

                let wk_name = format!("layer{}_wk", layer_idx);
                let wk_storage = gpu_ctx.get_or_create_buffer(wk);
                graph.add_tensor(
                    &wk_name,
                    wk.shape.clone(),
                    wk.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(wk_storage.as_ref().clone()),
                    )),
                );

                let wv_name = format!("layer{}_wv", layer_idx);
                let wv_storage = gpu_ctx.get_or_create_buffer(wv);
                graph.add_tensor(
                    &wv_name,
                    wv.shape.clone(),
                    wv.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(wv_storage.as_ref().clone()),
                    )),
                );

                // MatMul for Q, K, V
                let q_name = format!("layer{}_q", layer_idx);
                let k_name = format!("layer{}_k", layer_idx);
                let v_name = format!("layer{}_v", layer_idx);
                graph.add_op(
                    OpType::MatMul,
                    vec![normed_name.clone(), wq_name.clone()],
                    q_name.clone(),
                );
                graph.add_op(
                    OpType::MatMul,
                    vec![normed_name.clone(), wk_name.clone()],
                    k_name.clone(),
                );
                graph.add_op(
                    OpType::MatMul,
                    vec![normed_name.clone(), wv_name.clone()],
                    v_name.clone(),
                );

                // RoPE
                let q_rope_name = format!("layer{}_q_rope", layer_idx);
                let k_rope_name = format!("layer{}_k_rope", layer_idx);
                graph.add_op(
                    OpType::RopeNeox {
                        n_heads,
                        head_dim,
                        rope_freq_base: self.config.rope_freq_base,
                        rope_freq_scale: 1.0,
                    },
                    vec![q_name.clone()],
                    q_rope_name.clone(),
                );
                graph.add_op(
                    OpType::RopeNeox {
                        n_heads: n_kv_heads,
                        head_dim,
                        rope_freq_base: self.config.rope_freq_base,
                        rope_freq_scale: 1.0,
                    },
                    vec![k_name.clone()],
                    k_rope_name.clone(),
                );

                // TODO: KV cache update and Flash Attention

                // Output projection
                let wo = layer.wo.as_ref()?;
                let wo_name = format!("layer{}_wo", layer_idx);
                let wo_storage = gpu_ctx.get_or_create_buffer(wo);
                graph.add_tensor(
                    &wo_name,
                    wo.shape.clone(),
                    wo.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(wo_storage.as_ref().clone()),
                    )),
                );

                // For now, skip attention and just use Q as placeholder
                let attn_out_name = format!("layer{}_attn_out", layer_idx);
                graph.add_op(
                    OpType::MatMul,
                    vec![q_rope_name.clone(), wo_name.clone()],
                    attn_out_name.clone(),
                );

                // Residual add
                let residual_name = format!("layer{}_residual", layer_idx);
                graph.add_op(
                    OpType::Add,
                    vec![current_hidden.clone(), attn_out_name.clone()],
                    residual_name.clone(),
                );

                // FFN Norm
                let ffn_norm_name = format!("layer{}_ffn_norm", layer_idx);
                let ffn_norm = layer.ffn_norm.as_ref()?;
                let ffn_norm_storage = gpu_ctx.get_or_create_buffer(ffn_norm);
                graph.add_tensor(
                    &ffn_norm_name,
                    ffn_norm.shape.clone(),
                    ffn_norm.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(ffn_norm_storage.as_ref().clone()),
                    )),
                );

                let ffn_normed_name = format!("layer{}_ffn_normed", layer_idx);
                graph.add_op(
                    OpType::RmsNorm { eps: 1e-5 },
                    vec![residual_name.clone(), ffn_norm_name.clone()],
                    ffn_normed_name.clone(),
                );

                // FFN (SwiGLU)
                let w1 = layer.w1.as_ref()?;
                let w2 = layer.w2.as_ref()?;
                let w3 = layer.w3.as_ref()?;

                let w1_name = format!("layer{}_w1", layer_idx);
                let w1_storage = gpu_ctx.get_or_create_buffer(w1);
                graph.add_tensor(
                    &w1_name,
                    w1.shape.clone(),
                    w1.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(w1_storage.as_ref().clone()),
                    )),
                );

                let w2_name = format!("layer{}_w2", layer_idx);
                let w2_storage = gpu_ctx.get_or_create_buffer(w2);
                graph.add_tensor(
                    &w2_name,
                    w2.shape.clone(),
                    w2.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(w2_storage.as_ref().clone()),
                    )),
                );

                let w3_name = format!("layer{}_w3", layer_idx);
                let w3_storage = gpu_ctx.get_or_create_buffer(w3);
                graph.add_tensor(
                    &w3_name,
                    w3.shape.clone(),
                    w3.dtype,
                    Some(crate::compute::storage::Storage::Metal(
                        std::sync::Arc::new(w3_storage.as_ref().clone()),
                    )),
                );

                // Gate and up projections
                let gate_name = format!("layer{}_gate", layer_idx);
                let up_name = format!("layer{}_up", layer_idx);
                graph.add_op(
                    OpType::MatMul,
                    vec![ffn_normed_name.clone(), w1_name.clone()],
                    gate_name.clone(),
                );
                graph.add_op(
                    OpType::MatMul,
                    vec![ffn_normed_name.clone(), w3_name.clone()],
                    up_name.clone(),
                );

                // SiLU activation
                let silu_name = format!("layer{}_silu", layer_idx);
                graph.add_op(
                    OpType::Silu,
                    vec![gate_name.clone()],
                    silu_name.clone(),
                );

                // Element-wise mul
                let hidden_out_name = format!("layer{}_hidden_out", layer_idx);
                graph.add_op(
                    OpType::Mul,
                    vec![silu_name.clone(), up_name.clone()],
                    hidden_out_name.clone(),
                );

                // Down projection
                let ffn_out_name = format!("layer{}_ffn_out", layer_idx);
                graph.add_op(
                    OpType::MatMul,
                    vec![hidden_out_name.clone(), w2_name.clone()],
                    ffn_out_name.clone(),
                );

                // Final residual
                let output_name = format!("layer{}_output", layer_idx);
                graph.add_op(
                    OpType::Add,
                    vec![residual_name.clone(), ffn_out_name.clone()],
                    output_name.clone(),
                );

                current_hidden = output_name;
            }

            // Final norm and output projection
            let norm_weight = self.norm.as_ref()?;
            let norm_weight_storage = gpu_ctx.get_or_create_buffer(norm_weight);
            graph.add_tensor(
                "final_norm",
                norm_weight.shape.clone(),
                norm_weight.dtype,
                Some(crate::compute::storage::Storage::Metal(
                    std::sync::Arc::new(norm_weight_storage.as_ref().clone()),
                )),
            );

            let final_normed_name = "final_normed".to_string();
            graph.add_op(
                OpType::RmsNorm { eps: 1e-5 },
                vec![current_hidden.clone(), "final_norm".to_string()],
                final_normed_name.clone(),
            );

            // Output projection
            let output_weight = self.output.as_ref()?;
            let output_storage = gpu_ctx.get_or_create_buffer(output_weight);
            graph.add_tensor(
                "output_weight",
                output_weight.shape.clone(),
                output_weight.dtype,
                Some(crate::compute::storage::Storage::Metal(
                    std::sync::Arc::new(output_storage.as_ref().clone()),
                )),
            );

            let logits_name = "logits".to_string();
            graph.add_op(
                OpType::MatMul,
                vec![final_normed_name.clone(), "output_weight".to_string()],
                logits_name.clone(),
            );

            // Mark output
            graph.add_output(&logits_name);

            // Execute graph
            let outputs = backend.execute(&graph, &ctx).ok()?;

            // Copy result back to CPU
            let logits_storage = &outputs[0];
            let logits_bytes = backend.copy_to_cpu(logits_storage).ok()?;
            let logits: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    logits_bytes.as_ptr() as *const f32,
                    logits_bytes.len() / 4,
                )
                .to_vec()
            };

            // Advance KV cache
            kv_cache.advance();

            return Some(logits);
        }

        None
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
