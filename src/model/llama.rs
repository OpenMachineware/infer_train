// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Llama model architecture definition

#![allow(dead_code)]

use crate::dag::OptimizedDag;
use crate::inference::{KVCacheBase, dequant_to_f32};
use crate::model::config::ModelConfig;
use crate::model::tensor::{DataType, Tensor, TensorData};
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

    /// Cached optimized DAG for forward pass
    /// Built once at model load, reused for all forward passes
    pub optimized_dag: Option<OptimizedDag>,
}

impl LlamaModel {
    /// Check if model uses MQA/MLA architecture (not yet supported on GPU)
    pub fn uses_mqa(&self) -> bool {
        self.layers.iter().any(|layer| {
            layer.attn_kv_a_mqa.is_some() || layer.attn_kv_b.is_some()
        })
    }
}

pub struct LlamaLayer {
    pub layer_idx: usize,

    // Attention
    pub attention_norm: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub wo: Option<Tensor>,
    pub bq: Option<Tensor>,     // Q bias
    pub bk: Option<Tensor>,     // K bias
    pub bv: Option<Tensor>,     // V bias
    pub q_norm: Option<Tensor>, // Qwen-style Q norm
    pub k_norm: Option<Tensor>, // Qwen-style K norm

    // Fused QKV (for models like Qwen3.8)
    pub attn_qkv: Option<Tensor>,

    // MQA attention (for models like DeepSeek-Coder-V2)
    pub attn_kv_a_mqa: Option<Tensor>,
    pub attn_kv_b: Option<Tensor>,
    pub attn_kv_a_norm: Option<Tensor>,

    // FFN
    pub ffn_norm: Option<Tensor>,
    pub w1: Option<Tensor>, // gate
    pub w2: Option<Tensor>, // down
    pub w3: Option<Tensor>, // up

    // MoE FFN (for models like DeepSeek-Coder-V2)
    pub ffn_gate_inp: Option<Tensor>, // Expert routing
    pub ffn_gate_exps: Option<Tensor>, // Expert gate weights
    pub ffn_up_exps: Option<Tensor>,  // Expert up weights
    pub ffn_down_exps: Option<Tensor>, // Expert down weights
    pub ffn_gate_shexp: Option<Tensor>, // Shared expert gate
    pub ffn_up_shexp: Option<Tensor>, // Shared expert up
    pub ffn_down_shexp: Option<Tensor>, // Shared expert down
}

impl LlamaModel {
    pub fn from_weights(store: &WeightStore, config: ModelConfig) -> Self {
        let mut model = LlamaModel {
            config,
            tok_embeddings: None,
            layers: Vec::new(),
            norm: None,
            output: None,
            optimized_dag: None,
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

        // Transpose output layer if needed (GGUF stores output weights transposed)
        // This enables using standard GEMV kernel instead of transposed kernel
        model.transpose_output_layer();

        // Build and optimize DAG for GPU inference
        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        {
            use crate::dag::builder::DagBuilder;
            use crate::dag::optimizer::DagOptimizer;

            let mut builder = DagBuilder::new();
            let mut dag = builder.build_model_dag(&model);

            // Apply fusion optimizations
            let fused_count = DagOptimizer::optimize(&mut dag);
            if fused_count > 0 {
                println!("DAG optimization: {} operations fused", fused_count);
            }

            // Convert to optimized DAG with pre-computed execution order
            model.optimized_dag = Some(dag.into_optimized());
        }

        model
    }

    /// Transpose output layer weights if stored in transposed format
    /// GGUF stores output weights as [hidden_size, vocab_size] (column-major)
    /// We transpose to [vocab_size, hidden_size] for standard GEMV
    fn transpose_output_layer(&mut self) {
        let output = match &mut self.output {
            Some(t) => t,
            None => return,
        };

        // Check if weights are transposed: shape = [hidden_size, vocab_size]
        let hidden_size = self.config.hidden_size;
        if output.shape.len() != 2 || output.shape[0] != hidden_size {
            // Not transposed or wrong shape, nothing to do
            return;
        }

        let vocab_size = output.shape[1];
        println!(
            "Transposing output layer: [{}, {}] -> [{}, {}]",
            hidden_size, vocab_size, vocab_size, hidden_size
        );

        // Dequantize to F32
        let total_elems = hidden_size * vocab_size;
        let data_f32 =
            dequant_to_f32(output.bytes(), output.dtype, total_elems);

        // Transpose: column-major [H, V] -> row-major [V, H]
        let mut transposed = vec![0.0f32; total_elems];
        for v in 0..vocab_size {
            for h in 0..hidden_size {
                // Original: column-major [H, V] -> index = h + v * H
                // Transposed: row-major [V, H] -> index = v * H + h
                transposed[v * hidden_size + h] = data_f32[h + v * hidden_size];
            }
        }

        // Convert to F16 and store
        let data_f16: Vec<u16> = transposed
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();

        // Update tensor
        output.shape = vec![vocab_size, hidden_size];
        output.dtype = DataType::F16;
        output.data = TensorData::Owned(
            unsafe {
                std::slice::from_raw_parts(
                    data_f16.as_ptr() as *const u8,
                    data_f16.len() * 2,
                )
            }
            .to_vec(),
        );

        // Clear any cached GPU buffer
        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        {
            *output.gpu_buffer.borrow_mut() = None;
        }

        println!("Output layer transposed successfully");
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

    /// Get Q, K, V projections from layer
    /// Supports both standard (wq/wk/wv) and fused QKV (attn_qkv) architectures
    /// Returns (q, k, v, q_out_size, k_out_size, v_out_size)
    pub fn get_qkv_proj(
        &self,
        layer: &LlamaLayer,
        hidden: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize, usize)> {
        // Try fused QKV first (Qwen3.8-style)
        if let Some(ref qkv) = layer.attn_qkv {
            // Fused QKV: shape = [hidden_size, q_size + k_size + v_size]
            let qkv_out_size = qkv.shape[1];
            let q_out_size = n_heads * head_dim;
            let k_out_size = n_kv_heads * head_dim;
            let v_out_size = n_kv_heads * head_dim;

            // Compute fused projection
            let qkv_proj = self.gemv_impl(hidden, qkv, qkv_out_size)?;

            // Split into Q, K, V
            let q = qkv_proj[0..q_out_size].to_vec();
            let k = qkv_proj[q_out_size..q_out_size + k_out_size].to_vec();
            let v = qkv_proj
                [q_out_size + k_out_size..q_out_size + k_out_size + v_out_size]
                .to_vec();

            return Some((q, k, v, q_out_size, k_out_size, v_out_size));
        }

        // Try standard separate Q, K, V
        if let (Some(wq), Some(wk), Some(wv)) =
            (&layer.wq, &layer.wk, &layer.wv)
        {
            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let v_out_size = wv.shape[1];

            let mut q = self.gemv_impl(hidden, wq, q_out_size)?;
            let mut k = self.gemv_impl(hidden, wk, k_out_size)?;
            let mut v = self.gemv_impl(hidden, wv, v_out_size)?;

            // Add biases if present (biases are stored as F32)
            if let Some(bq) = &layer.bq {
                let bias_data: &[f32] = bytemuck::cast_slice(bq.bytes());
                for i in 0..q_out_size.min(bias_data.len()) {
                    q[i] += bias_data[i];
                }
            }
            if let Some(bk) = &layer.bk {
                let bias_data: &[f32] = bytemuck::cast_slice(bk.bytes());
                for i in 0..k_out_size.min(bias_data.len()) {
                    k[i] += bias_data[i];
                }
            }
            if let Some(bv) = &layer.bv {
                let bias_data: &[f32] = bytemuck::cast_slice(bv.bytes());
                for i in 0..v_out_size.min(bias_data.len()) {
                    v[i] += bias_data[i];
                }
            }

            return Some((q, k, v, q_out_size, k_out_size, v_out_size));
        }

        // Try MQA (DeepSeek-Coder-V2-style MLA)
        // Multi-head Latent Attention uses low-rank compression for KV
        // Flow:
        // 1. Q = hidden @ attn_q
        // 2. KV_A = hidden @ attn_kv_a_mqa → [kv_lora_rank + qk_rope_head_dim]
        //    - KV_A[:kv_lora_rank] = compressed KV latent
        //    - KV_A[kv_lora_rank:] = key RoPE latent
        // 3. KV_A_normed = RMSNorm(KV_A[:kv_lora_rank], attn_kv_a_norm)
        // 4. KV = KV_A_normed @ attn_kv_b → [n_kv_heads * (qk_nope_head_dim + value_length)]
        // 5. K = concat(KV_A[kv_lora_rank:], KV[:n_kv_heads * qk_nope_head_dim]) per head
        // 6. V = KV[n_kv_heads * qk_nope_head_dim:]
        if let (Some(wq), Some(kv_a), Some(kv_b), Some(kv_a_norm)) = (
            &layer.wq,
            &layer.attn_kv_a_mqa,
            &layer.attn_kv_b,
            &layer.attn_kv_a_norm,
        ) {
            let q_out_size = wq.shape[1];
            let kv_a_out_size = kv_a.shape[1]; // kv_lora_rank + qk_rope_head_dim
            let kv_b_out_size = kv_b.shape[1]; // n_kv_heads * (qk_nope_head_dim + value_length)

            // 1. Q projection
            let q = self.gemv_impl(hidden, wq, q_out_size)?;

            // 2. KV_A projection (compressed)
            let kv_a_proj = self.gemv_impl(hidden, kv_a, kv_a_out_size)?;

            // 3. KV_A normalization (RMSNorm on the first kv_lora_rank elements)
            let kv_lora_rank = self.config.kv_lora_rank;
            let qk_rope_head_dim = self.config.qk_rope_head_dim;
            if kv_lora_rank == 0
                || kv_lora_rank + qk_rope_head_dim > kv_a_out_size
            {
                return None;
            }

            // RMSNorm on kv_a[:kv_lora_rank]
            let norm_data: &[f32] = bytemuck::cast_slice(kv_a_norm.bytes());
            let eps = self.config.rms_norm_eps;

            // Compute RMS
            let mut ss = 0.0f32;
            for i in 0..kv_lora_rank {
                ss += kv_a_proj[i] * kv_a_proj[i];
            }
            ss /= kv_lora_rank as f32;
            let rms = 1.0 / (ss + eps).sqrt();

            // Normalize and scale
            let kv_a_normed: Vec<f32> = kv_a_proj[0..kv_lora_rank]
                .iter()
                .zip(norm_data.iter())
                .map(|(x, w)| x * rms * w)
                .collect();

            // 4. KV_B projection (decompress to full KV)
            // kv_b shape: [kv_lora_rank, n_kv_heads * (qk_nope_head_dim + value_length)]
            let kv = self.gemv_impl(&kv_a_normed, kv_b, kv_b_out_size)?;

            // 5. Construct K and V
            // qk_nope_head_dim = key_length - qk_rope_head_dim
            let key_length = self.config.key_length;
            let value_length = self.config.value_length;
            let qk_nope_head_dim = key_length - qk_rope_head_dim;

            // The RoPE part of the key is shared across all heads
            let k_rope =
                &kv_a_proj[kv_lora_rank..kv_lora_rank + qk_rope_head_dim];

            // The non-RoPE part of the key and value come from kv
            // kv layout: [k_nope_head_0, ..., k_nope_head_n, v_head_0, ..., v_head_n]
            let k_nope_size = n_kv_heads * qk_nope_head_dim;
            let v_size = n_kv_heads * value_length;

            // Construct K: for each head, concat(k_rope, k_nope[head])
            let mut k = Vec::with_capacity(n_kv_heads * key_length);
            for head in 0..n_kv_heads {
                // Add RoPE part (shared)
                k.extend_from_slice(k_rope);
                // Add non-RoPE part (per-head)
                let nope_start = head * qk_nope_head_dim;
                let nope_end = (head + 1) * qk_nope_head_dim;
                k.extend_from_slice(&kv[nope_start..nope_end]);
            }

            // V is the remaining part
            let v = kv[k_nope_size..k_nope_size + v_size].to_vec();

            return Some((
                q,
                k,
                v,
                q_out_size,
                n_kv_heads * key_length,
                v_size,
            ));
        }

        None
    }

    /// Compute FFN output from layer
    /// Supports both standard FFN and MoE architectures
    pub fn compute_ffn(
        &self,
        layer: &LlamaLayer,
        hidden: &[f32],
        hidden_size: usize,
    ) -> Option<Vec<f32>> {
        // Try MoE FFN first (DeepSeek-Coder-V2-style)
        if layer.ffn_gate_exps.is_some() || layer.ffn_gate_shexp.is_some() {
            return self.compute_moe_ffn(layer, hidden, hidden_size);
        }

        // Standard FFN (SwiGLU)
        if let (Some(w1), Some(w2), Some(w3)) =
            (&layer.w1, &layer.w2, &layer.w3)
        {
            let intermediate_size = w1.shape[1];

            // gate = silu(hidden @ w1)
            let gate = self.gemv_impl(hidden, w1, intermediate_size)?;
            // up = hidden @ w3
            let up = self.gemv_impl(hidden, w3, intermediate_size)?;

            // hidden_ffn = gate * silu(up)
            let mut hidden_ffn = vec![0.0f32; intermediate_size];
            for i in 0..intermediate_size {
                let silu = up[i] / (1.0 + (-up[i]).exp());
                hidden_ffn[i] = gate[i] * silu;
            }

            // output = hidden_ffn @ w2
            let output = self.gemv_impl(&hidden_ffn, w2, hidden_size);
            return output;
        }

        None
    }

    /// Compute MoE FFN output
    /// DeepSeek-Coder-V2 uses routed experts + shared experts
    fn compute_moe_ffn(
        &self,
        layer: &LlamaLayer,
        hidden: &[f32],
        hidden_size: usize,
    ) -> Option<Vec<f32>> {
        // 1. Compute routing scores
        let routing_scores = if let Some(gate_inp) = &layer.ffn_gate_inp {
            // gate_inp: [hidden_size, n_experts]
            let n_experts = gate_inp.shape[1];
            let mut scores = vec![0.0f32; n_experts];

            // Compute routing scores: hidden @ gate_inp
            // gate_inp is [hidden_size, n_experts], so hidden @ gate_inp gives [n_experts]
            let weight: &[f32] = bytemuck::cast_slice(gate_inp.bytes());
            for e in 0..n_experts {
                for i in 0..hidden_size {
                    scores[e] += hidden[i] * weight[i * n_experts + e];
                }
            }

            // Softmax
            let max_score =
                scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
            for s in &mut scores {
                *s = ((*s - max_score).exp()) / sum;
            }

            scores
        } else {
            // No routing, just use shared expert
            return self.compute_shared_expert(layer, hidden, hidden_size);
        };

        // 2. Select top-k experts
        let n_experts_used = self.config.n_experts_used;
        let mut expert_indices: Vec<(usize, f32)> =
            routing_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        expert_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k: Vec<(usize, f32)> =
            expert_indices.into_iter().take(n_experts_used).collect();

        // Renormalize top-k routing scores
        let sum_weights: f32 = top_k.iter().map(|(_, w)| w).sum();
        let top_k: Vec<(usize, f32)> = if sum_weights > 0.0 {
            top_k.into_iter().map(|(i, w)| (i, w / sum_weights)).collect()
        } else {
            top_k
        };

        // 3. Compute routed expert outputs
        let mut routed_output = vec![0.0f32; hidden_size];

        if let (Some(gate_exps), Some(up_exps), Some(down_exps)) =
            (&layer.ffn_gate_exps, &layer.ffn_up_exps, &layer.ffn_down_exps)
        {
            let intermediate_size = gate_exps.shape[1];
            let n_experts = gate_exps.shape[2];

            // Dequantize the 3D tensors ONCE (this is the key optimization)
            // Shape: [out_features, in_features, n_experts]
            // For gate_exps: [intermediate_size, hidden_size, n_experts]
            let gate_dequant = dequant_to_f32(
                gate_exps.bytes(),
                gate_exps.dtype,
                intermediate_size * hidden_size * n_experts,
            );
            let up_dequant = dequant_to_f32(
                up_exps.bytes(),
                up_exps.dtype,
                intermediate_size * hidden_size * n_experts,
            );
            let down_dequant = dequant_to_f32(
                down_exps.bytes(),
                down_exps.dtype,
                hidden_size * intermediate_size * n_experts,
            );

            for (expert_idx, routing_weight) in &top_k {
                // Compute expert output using pre-dequantized weights
                let expert_gate = self.gemv_dequant_expert(
                    hidden,
                    &gate_dequant,
                    *expert_idx,
                    intermediate_size,
                    hidden_size,
                    n_experts,
                )?;
                let expert_up = self.gemv_dequant_expert(
                    hidden,
                    &up_dequant,
                    *expert_idx,
                    intermediate_size,
                    hidden_size,
                    n_experts,
                )?;

                // SwiGLU
                let mut hidden_ffn = vec![0.0f32; intermediate_size];
                for i in 0..intermediate_size {
                    let silu = expert_up[i] / (1.0 + (-expert_up[i]).exp());
                    hidden_ffn[i] = expert_gate[i] * silu;
                }

                // Down projection
                let expert_down = self.gemv_dequant_expert(
                    &hidden_ffn,
                    &down_dequant,
                    *expert_idx,
                    hidden_size,
                    intermediate_size,
                    n_experts,
                )?;

                // Add weighted output
                for i in 0..hidden_size {
                    routed_output[i] += routing_weight * expert_down[i];
                }
            }
        }

        // 4. Compute shared expert output
        let shared_output =
            self.compute_shared_expert(layer, hidden, hidden_size)?;

        // 5. Combine routed + shared outputs
        let mut final_output = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            final_output[i] = routed_output[i] + shared_output[i];
        }

        Some(final_output)
    }

    /// GEMV with pre-dequantized expert weights (interleaved format)
    fn gemv_dequant_expert(
        &self,
        x: &[f32],
        dequant_weights: &[f32],
        expert_idx: usize,
        out_size: usize,
        in_size: usize,
        n_experts: usize,
    ) -> Option<Vec<f32>> {
        let mut out = vec![0.0f32; out_size];

        for i in 0..out_size {
            let mut sum = 0.0f32;
            for j in 0..in_size {
                // Index for W[i, j, expert_idx]: i * in_size * n_experts + j * n_experts + expert_idx
                let idx = i * in_size * n_experts + j * n_experts + expert_idx;
                sum += dequant_weights[idx] * x[j];
            }
            out[i] = sum;
        }

        Some(out)
    }

    /// Compute shared expert FFN (always used)
    fn compute_shared_expert(
        &self,
        layer: &LlamaLayer,
        hidden: &[f32],
        hidden_size: usize,
    ) -> Option<Vec<f32>> {
        if let (Some(gate), Some(up), Some(down)) =
            (&layer.ffn_gate_shexp, &layer.ffn_up_shexp, &layer.ffn_down_shexp)
        {
            let intermediate_size = gate.shape[1];

            // Shared expert FFN (SwiGLU)
            let gate_out = self.gemv_impl(hidden, gate, intermediate_size)?;
            let up_out = self.gemv_impl(hidden, up, intermediate_size)?;

            let mut hidden_ffn = vec![0.0f32; intermediate_size];
            for i in 0..intermediate_size {
                let silu = up_out[i] / (1.0 + (-up_out[i]).exp());
                hidden_ffn[i] = gate_out[i] * silu;
            }

            self.gemv_impl(&hidden_ffn, down, hidden_size)
        } else {
            Some(vec![0.0f32; hidden_size])
        }
    }

    /// Compute expert gate + up projection
    fn compute_expert_gate(
        &self,
        hidden: &[f32],
        gate_exps: &Tensor,
        up_exps: &Tensor,
        expert_idx: usize,
        intermediate_size: usize,
    ) -> Option<Vec<f32>> {
        // 3D tensor shape: [out_features, in_features, n_experts]
        // For MoE, experts are interleaved, so we need to extract the correct slices
        //
        // For quantized formats, the data is stored per-row with interleaved experts:
        // Row 0: [expert0_block0, expert1_block0, ..., expert63_block0, expert0_block1, ...]
        // Row 1: ...
        //
        // We compute GEMV directly by iterating over rows and extracting expert weights

        let hidden_size = hidden.len();
        let n_experts = gate_exps.shape[2];

        // Compute gate and up using interleaved expert weights
        let gate_out = self.gemv_expert(
            hidden,
            gate_exps,
            expert_idx,
            intermediate_size,
            n_experts,
        )?;
        let up_out = self.gemv_expert(
            hidden,
            up_exps,
            expert_idx,
            intermediate_size,
            n_experts,
        )?;

        // SwiGLU
        let mut hidden_ffn = vec![0.0f32; intermediate_size];
        for i in 0..intermediate_size {
            let silu = up_out[i] / (1.0 + (-up_out[i]).exp());
            hidden_ffn[i] = gate_out[i] * silu;
        }

        Some(hidden_ffn)
    }

    /// Compute expert down projection
    fn compute_expert_down(
        &self,
        hidden_ffn: &[f32],
        down_exps: &Tensor,
        expert_idx: usize,
        hidden_size: usize,
    ) -> Option<Vec<f32>> {
        let intermediate_size = hidden_ffn.len();
        let n_experts = down_exps.shape[2];
        self.gemv_expert(
            hidden_ffn,
            down_exps,
            expert_idx,
            hidden_size,
            n_experts,
        )
    }

    /// GEMV for interleaved expert weights
    /// Expert weights tensor shape: [out_features, in_features, n_experts]
    /// Each row has weights for all experts, interleaved
    fn gemv_expert(
        &self,
        x: &[f32],
        weights: &Tensor,
        expert_idx: usize,
        out_size: usize,
        n_experts: usize,
    ) -> Option<Vec<f32>> {
        let in_size = x.len();
        let mut out = vec![0.0f32; out_size];

        // For each output row, we need to compute: out[i] = sum_j W[i,j,expert_idx] * x[j]
        // The weights for row i, expert expert_idx are stored in the interleaved format

        match weights.dtype {
            DataType::F32 => {
                // F32: each element is 4 bytes
                let data: &[f32] = bytemuck::cast_slice(weights.bytes());
                for i in 0..out_size {
                    let mut sum = 0.0f32;
                    for j in 0..in_size {
                        // Index for W[i, j, expert_idx]: i * in_size * n_experts + j * n_experts + expert_idx
                        let idx = i * in_size * n_experts
                            + j * n_experts
                            + expert_idx;
                        sum += data[idx] * x[j];
                    }
                    out[i] = sum;
                }
            }
            DataType::F16 => {
                let data: &[u16] = bytemuck::cast_slice(weights.bytes());
                for i in 0..out_size {
                    let mut sum = 0.0f32;
                    for j in 0..in_size {
                        let idx = i * in_size * n_experts
                            + j * n_experts
                            + expert_idx;
                        let f16_val = half::f16::from_bits(data[idx]);
                        sum += f16_val.to_f32() * x[j];
                    }
                    out[i] = sum;
                }
            }
            // For quantized formats, we need a different approach
            // The quantization is per-block within each expert's weights
            // For now, use a simplified approach that dequantizes the whole tensor
            _ => {
                // Dequantize the entire 3D tensor and then extract expert weights
                // This is inefficient but correct
                let total_elems = out_size * in_size * n_experts;
                let dequant =
                    dequant_to_f32(weights.bytes(), weights.dtype, total_elems);

                for i in 0..out_size {
                    let mut sum = 0.0f32;
                    for j in 0..in_size {
                        let idx = i * in_size * n_experts
                            + j * n_experts
                            + expert_idx;
                        if idx < dequant.len() {
                            sum += dequant[idx] * x[j];
                        }
                    }
                    out[i] = sum;
                }
            }
        }

        Some(out)
    }

    /// Extract expert weights from a 3D tensor
    fn get_expert_weights(
        &self,
        weights: &Tensor,
        expert_idx: usize,
    ) -> Option<Tensor> {
        // The expert weights tensor has shape [out_features, in_features, n_experts]
        // We need to extract a 2D slice for the specific expert

        let shape = &weights.shape;
        let out_features = shape[0];
        let in_features = shape[1];
        let n_experts = shape.get(2).copied().unwrap_or(1);

        if expert_idx >= n_experts {
            return None;
        }

        // Calculate the byte offset for this expert
        let bytes_per_expert =
            weights.dtype.bytes_for_elems(out_features * in_features);
        let offset = expert_idx * bytes_per_expert;

        // Create a new tensor view for this expert
        Some(Tensor {
            name: format!("{}_expert_{}", weights.name, expert_idx),
            shape: vec![out_features, in_features],
            dtype: weights.dtype,
            data: crate::model::tensor::TensorData::Owned(
                weights
                    .bytes()
                    .get(offset..offset + bytes_per_expert)?
                    .to_vec(),
            ),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            gpu_buffer: std::cell::RefCell::new(None),
        })
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
            // Transposed weights: use GPU transposed GEMV
            // Download hidden state to CPU (needed for transposed kernel)
            let hidden_data: Vec<f32> = unsafe {
                let ptr = final_hidden_buffer.contents() as *const f32;
                let len = hidden_size;
                let mut data = vec![0.0f32; len];
                std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), len);
                data
            };

            // Get weights as BlockQ6K slice
            let weights: &[crate::quant::types::BlockQ6K] = unsafe {
                let ptr = output.bytes().as_ptr()
                    as *const crate::quant::types::BlockQ6K;
                let len = output.bytes().len()
                    / std::mem::size_of::<crate::quant::types::BlockQ6K>();
                std::slice::from_raw_parts(ptr, len)
            };

            gpu_ctx
                .mv_q6_k_f32_transposed(
                    actual_vocab_size,
                    hidden_size,
                    weights,
                    &hidden_data,
                )
                .ok()?
        } else {
            // Normal weights: use GPU with CPU fallback for unsupported formats (like F16)
            let logits_buffer = self.gemv_gpu_buffer_with_fallback(
                &final_hidden_buffer,
                output,
                actual_vocab_size,
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
            // Reset KV cache position for this layer
            kv_cache.reset_seq_len();

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

            let mut q =
                self.gemm_batch(&normed, wq, q_out_size, batch_size, gpu_ctx)?;
            let mut k =
                self.gemm_batch(&normed, wk, k_out_size, batch_size, gpu_ctx)?;
            let mut v =
                self.gemm_batch(&normed, wv, v_out_size, batch_size, gpu_ctx)?;

            // Add biases if present (for each token in batch)
            if let Some(bq) = &layer.bq {
                let bias_data: &[f32] = bytemuck::cast_slice(bq.bytes());
                for t in 0..batch_size {
                    for i in 0..q_out_size.min(bias_data.len()) {
                        q[t * q_out_size + i] += bias_data[i];
                    }
                }
            }
            if let Some(bk) = &layer.bk {
                let bias_data: &[f32] = bytemuck::cast_slice(bk.bytes());
                for t in 0..batch_size {
                    for i in 0..k_out_size.min(bias_data.len()) {
                        k[t * k_out_size + i] += bias_data[i];
                    }
                }
            }
            if let Some(bv) = &layer.bv {
                let bias_data: &[f32] = bytemuck::cast_slice(bv.bytes());
                for t in 0..batch_size {
                    for i in 0..v_out_size.min(bias_data.len()) {
                        v[t * v_out_size + i] += bias_data[i];
                    }
                }
            }

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

        // Use output_projection which handles transposed weights correctly
        let logits =
            self.output_projection(&final_normed, self.config.vocab_size)?;

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

            // Reset KV cache position for this layer
            kv_cache.reset_seq_len();

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
        let vocab_size = output.shape[1]; // Use actual tensor dimension
        let logits = self.gemv(&final_normed, output, vocab_size)?;
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
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        // Derive K from the weight matrix shape
        // GGUF stores weights in column-major: shape = [K, M]
        let k = w.shape[0];

        // Use matmul_q*_k_f32 which dispatches to MV for N=1, GEMM for N>=32
        let gpu_result = match w.dtype {
            crate::model::tensor::DataType::Q4_K => {
                let weights: &[crate::quant::types::BlockQ4K] =
                    bytemuck::cast_slice(w.bytes());
                gpu_ctx.matmul_q4_k_f32(m, batch, k, weights, x).ok()
            }
            crate::model::tensor::DataType::Q5_K => {
                let weights: &[crate::quant::types::BlockQ5K] =
                    bytemuck::cast_slice(w.bytes());
                gpu_ctx.matmul_q5_k_f32(m, batch, k, weights, x).ok()
            }
            crate::model::tensor::DataType::Q6_K => {
                let weights: &[crate::quant::types::BlockQ6K] =
                    bytemuck::cast_slice(w.bytes());
                gpu_ctx.matmul_q6_k_f32(m, batch, k, weights, x).ok()
            }
            _ => None,
        };

        // Check if GPU result is valid (no NaN)
        if let Some(ref output) = gpu_result {
            let has_nan = output.iter().any(|&v| v.is_nan());
            if !has_nan {
                return gpu_result;
            }
            eprintln!(
                "[GEMM] GPU returned NaN, falling back to CPU (M={}, N={}, K={}, dtype={:?})",
                m, batch, k, w.dtype
            );
        }

        // Fallback to sequential CPU GEMV
        let mut result = Vec::with_capacity(batch * m);
        for i in 0..batch {
            let x_token = &x[i * k..(i + 1) * k];
            let out = self.gemv(x_token, w, m)?;
            result.extend(out);
        }
        Some(result)
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

    pub fn embedding_lookup_impl(&self, token: u32) -> Option<Vec<f32>> {
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
            eprintln!(
                "Embedding lookup failed: token {} (offset {}) exceeds embedding matrix (len {})",
                token,
                offset + bytes_per_row,
                bytes.len()
            );
            return None;
        }

        let data = &bytes[offset..offset + bytes_per_row];

        let result = dequant_to_f32(data, dtype, hidden_size);
        Some(result)
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

    pub fn forward_layer(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden: &[f32],
        kv_cache: &mut crate::inference::KVCache,
        _n_heads: usize,
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

        // 2. QKV projections (support standard and MQA architectures)
        let (q, k, v, q_out_size, k_out_size, v_out_size) = self.get_qkv_proj(
            layer,
            &normed,
            self.config.n_heads,
            n_kv_heads,
            head_dim,
        )?;

        // For MLA, the head_dim for K/V is different from Q
        let k_head_dim = if layer.attn_kv_a_mqa.is_some() {
            self.config.key_length
        } else {
            head_dim
        };

        // For MLA, Q's head_dim is also different from config.head_dim
        let q_head_dim = q_out_size / self.config.n_heads;

        // Update n_heads and n_kv_heads based on actual weight shapes
        let actual_n_heads = self.config.n_heads;
        let actual_n_kv_heads = k_out_size / k_head_dim;

        // For standard attention, apply Q/K norms (Qwen-style)
        // For MQA, norms are already applied in compute_qkv
        let mut q = q;
        let mut k = k;
        if layer.wk.is_some() {
            // Standard attention - apply Q/K norms if present
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
        }

        // 3. RoPE
        // For standard attention: apply RoPE to full Q and K
        // For MLA: Q gets full RoPE, K only applies RoPE to the rope part
        let q_rope = if layer.attn_kv_a_mqa.is_some() {
            // MLA: Q's head_dim is q_head_dim (192 for DeepSeek)
            self.apply_rope(&q, pos, actual_n_heads, q_head_dim)?
        } else {
            self.apply_rope(&q, pos, actual_n_heads, q_head_dim)?
        };

        let k_rope = if layer.attn_kv_a_mqa.is_some() {
            // MLA: Only apply RoPE to the rope part of K
            // K structure per head: [qk_rope_head_dim (rope), qk_nope_head_dim (non-rope)]
            let qk_rope_head_dim = self.config.qk_rope_head_dim;
            let key_length = self.config.key_length;

            // Apply RoPE only to the rope part of each head
            let mut k_with_rope = Vec::with_capacity(k.len());
            for h in 0..actual_n_kv_heads {
                let head_start = h * key_length;
                let rope_end = head_start + qk_rope_head_dim;
                let nope_end = head_start + key_length;

                // Apply RoPE to rope part
                let k_rope_head = &k[head_start..rope_end];
                let rope_applied =
                    self.apply_rope(k_rope_head, pos, 1, qk_rope_head_dim)?;
                k_with_rope.extend(rope_applied);

                // Copy non-rope part unchanged
                k_with_rope.extend_from_slice(&k[rope_end..nope_end]);
            }
            k_with_rope
        } else {
            self.apply_rope(&k, pos, actual_n_kv_heads, head_dim)?
        };

        // For MLA, use key_length as head_dim for both Q and K
        // In MLA, Q is projected to qk_head_dim (key_length), not standard head_dim
        let actual_head_dim = if layer.attn_kv_a_mqa.is_some() {
            self.config.key_length
        } else {
            q_head_dim
        };

        // 4. Attention (use actual n_heads and head_dim)
        let attn_out = self.attention(
            &q_rope,
            &k_rope,
            &v,
            layer_idx,
            kv_cache,
            actual_n_heads,
            q_head_dim,
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

        // 8. FFN (SwiGLU or MoE)
        let ffn_out = self.compute_ffn(layer, &ffn_normed, hidden_size)?;

        // 9. Residual
        Some(residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect())
    }

    pub fn apply_rms_norm_impl(
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

    pub fn apply_rope(
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

    pub fn gemv_impl(
        &self,
        x: &[f32],
        w: &Tensor,
        out_size: usize,
    ) -> Option<Vec<f32>> {
        let in_size = x.len();
        let total_elems = in_size * out_size;
        let w_data = dequant_to_f32(w.bytes(), w.dtype, total_elems);

        let mut out = vec![0.0f32; out_size];
        // GGUF stores tensors in column-major: shape = [K, M] where K=input, M=output
        // Column-major: element (row, col) = (i, j) is at index i + j * K = i + j * in_size
        // For GEMV y = W @ x: y[i] = sum_j W[i,j] * x[j]
        // Need to access W[i,j] which is at j + i * K = j + i * in_size (transpose!)
        for i in 0..out_size {
            let mut sum = 0.0f32;
            for j in 0..in_size {
                // W[i,j] in column-major shape [K, M] is at j + i * K
                sum += w_data[j + i * in_size] * x[j];
            }
            out[i] = sum;
        }
        Some(out)
    }

    /// Transposed gemv: weight is stored as [in_size, out_size] instead of [out_size, in_size]
    /// This is needed for tied embeddings where weight shape is [hidden_size, vocab_size]
    pub fn gemv_transposed(
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
                // Fallback to full dequantization for Q6_K
                // The block layout for transposed matrices is complex and needs careful handling
                // of partial blocks (when out_size is not a multiple of 256)
                let total_elems = w.elem_count();
                let w_data = crate::inference::dequant_to_f32(
                    w.bytes(),
                    w.dtype,
                    total_elems,
                );

                // Weight shape is [in_size, out_size], so row stride is out_size
                let row_stride = w.shape[1];

                let mut out = vec![0.0f32; out_size];
                for i in 0..out_size {
                    let mut sum = 0.0f32;
                    for j in 0..in_size {
                        sum += w_data[j * row_stride + i] * x[j];
                    }
                    out[i] = sum;
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

    pub fn attention(
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
        // For MLA, K and V have different dimensions
        // k_head_dim = key_length, v_head_dim = value_length
        // For standard attention, both use head_dim
        let k_head_dim = k.len() / n_kv_heads;
        let v_head_dim = v.len() / n_kv_heads;

        // Store K, V in cache (will be stored at current seq_len position)
        kv_cache.set_k(layer_idx, k, n_kv_heads, k_head_dim);
        kv_cache.set_v(layer_idx, v, n_kv_heads, v_head_dim);

        let seq_len = kv_cache.seq_len() + 1; // +1 because we just added
        // Use v_head_dim for output dimension, not k_head_dim
        let v_head_dim = v.len() / n_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut out = vec![0.0f32; n_heads * v_head_dim];

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
            for d in 0..v_head_dim {
                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    let v_s = kv_cache.get_v(layer_idx, kv_head, s);
                    sum += scores[s] * v_s[d];
                }
                out[h * v_head_dim + d] = sum;
            }
        }

        Some(out)
    }

    pub fn ffn_swiglu(
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
        if output.shape.len() == 2 && output.shape[1] == hidden.len() {
            // Weight is [vocab_size, hidden_size] - transposed
            self.gemv_transposed(hidden, output, output.shape[0])
        } else {
            // Weight is [hidden_size, vocab_size] - normal
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
    pub fn forward_layer_gpu_buffer(
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

        // GEMV with GPU buffers, with CPU fallback for unsupported formats
        let q_buffer = self.gemv_gpu_buffer_with_fallback(
            &normed_buffer,
            wq,
            q_out_size,
            gpu_ctx,
        )?;
        let k_buffer = self.gemv_gpu_buffer_with_fallback(
            &normed_buffer,
            wk,
            k_out_size,
            gpu_ctx,
        )?;
        let v_buffer = self.gemv_gpu_buffer_with_fallback(
            &normed_buffer,
            wv,
            v_out_size,
            gpu_ctx,
        )?;

        // Derive actual n_heads from weight shapes
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // 2.5 Apply Q/K norms (Qwen-style) on GPU
        // Qwen uses separate norms for Q and K after projection
        let q_buffer = if let Some(ref q_norm) = layer.q_norm {
            let q_norm_buffer = gpu_ctx.get_or_create_buffer(q_norm);
            gpu_ctx
                .rms_norm_f32_gpu(
                    &q_buffer,
                    &q_norm_buffer,
                    actual_n_heads, // n_rows = n_heads (normalize each head)
                    head_dim,       // n_cols = head_dim
                    1e-5,
                )
                .ok()?
        } else {
            q_buffer
        };

        let k_buffer = if let Some(ref k_norm) = layer.k_norm {
            let k_norm_buffer = gpu_ctx.get_or_create_buffer(k_norm);
            gpu_ctx
                .rms_norm_f32_gpu(
                    &k_buffer,
                    &k_norm_buffer,
                    actual_n_kv_heads, // n_rows = n_kv_heads
                    head_dim,          // n_cols = head_dim
                    1e-5,
                )
                .ok()?
        } else {
            k_buffer
        };

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

        // 6. Output projection (GPU with CPU fallback)
        let wo = layer.wo.as_ref()?;
        let attn_proj_buffer = self.gemv_gpu_buffer_with_fallback(
            &attn_out_buffer,
            wo,
            hidden_size,
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

    /// Forward pass with batched GPU encoding
    /// Creates separate command buffers per layer to avoid Metal operation limits
    /// Following llama.cpp's approach: split compute graph into chunks per command buffer
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_gpu_native_batched(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        // DAG executor has KV cache update bug - disabled for now
        // TODO: Fix KV cache update in DAG executor
        /*
        // For decode (single token), use DAG executor with kernel fusion
        if tokens.len() == 1 {
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            {
                use crate::dag::executor::DagExecutor;
                return DagExecutor::forward_dag(self, tokens, kv_cache, gpu_ctx);
            }
        }
        */

        // For prefill (batch > 1), use optimized batched encoding
        // Timing
        let enable_timing = std::env::var("INFER_TIMING").is_ok();
        let mut total_embed_time = std::time::Duration::ZERO;
        let mut total_upload_time = std::time::Duration::ZERO;
        let mut total_layer_time = std::time::Duration::ZERO;
        let mut total_output_time = std::time::Duration::ZERO;

        let hidden_size = self.config.hidden_size;
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Create shared reuse pool for buffers (persisted across forward passes)
        use std::cell::RefCell;
        use std::rc::Rc;
        let reuse_pool: Rc<RefCell<Vec<std::rc::Rc<Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));

        // Create ONE forward pool for the entire forward pass
        // This ensures buffers stay alive until we've read the final output
        let mut forward_pool = crate::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool.clone(),
        );

        // Process each token
        let mut last_hidden_buffer: Option<std::rc::Rc<Buffer>> = None;
        let mut logits = Vec::new();

        for &token in tokens.iter() {
            // 1. Embedding lookup (CPU)
            let t0 = std::time::Instant::now();
            let hidden = self.embedding_lookup_impl(token)?;
            total_embed_time += t0.elapsed();

            // Upload to GPU
            let t1 = std::time::Instant::now();
            let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            total_upload_time += t1.elapsed();

            // 2. Process through layers + final operations in ONE command buffer
            // Encode ALL operations into a single command buffer for best performance
            let mut layer_input = hidden_buffer.clone();
            let n_layers = self.layers.len();

            let t_encode_start = std::time::Instant::now();

            // Create ONE command buffer for all operations
            // Use unretained references for better performance (we manage lifetimes via ForwardBufferPool)
            let command_buffer =
                queue.new_command_buffer_with_unretained_references();
            let encoder = command_buffer.new_compute_command_encoder();

            // Encode all layers
            for layer_idx in 0..n_layers {
                let layer_output = self.forward_layer_gpu_encode(
                    &self.layers[layer_idx],
                    layer_idx,
                    &layer_input,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                )?;

                // Update for next layer
                layer_input = layer_output;
            }

            // 3. Final norm + Output projection (in same command buffer)
            let final_norm = self.norm.as_ref()?;
            let final_norm_buffer = gpu_ctx.get_or_create_buffer(final_norm);
            let final_hidden_buffer = forward_pool.get(hidden_size * 4);
            let output = self.output.as_ref()?;
            let vocab_size =
                if output.shape.len() == 2 && output.shape[1] == hidden_size {
                    output.shape[0]
                } else {
                    output.shape[0]
                };

            // Final norm
            gpu_ctx.rms_norm_encode(
                encoder,
                &layer_input,
                &final_norm_buffer,
                &final_hidden_buffer,
                1,
                hidden_size,
                1e-5,
            );

            // Output projection
            let output_weights = gpu_ctx.get_or_create_buffer(output);
            let logits_buffer = forward_pool.get(vocab_size * 4);

            if !gpu_ctx.mv_encode(
                encoder,
                &output_weights,
                &final_hidden_buffer,
                &logits_buffer,
                vocab_size,
                hidden_size,
                output.dtype,
            ) {
                return None;
            }

            encoder.end_encoding();
            let encode_time = t_encode_start.elapsed();
            total_layer_time += encode_time;

            // 4. GPU execution
            let t_gpu_start = std::time::Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_time = t_gpu_start.elapsed();
            total_output_time += gpu_time;

            // 5. Download logits
            let t_download_start = std::time::Instant::now();
            logits = vec![0.0f32; vocab_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    logits_buffer.contents() as *const f32,
                    logits.as_mut_ptr(),
                    vocab_size,
                );
            }
            let download_time = t_download_start.elapsed();

            if enable_timing {
                eprintln!(
                    "[TIMING] Encode: {:?}, GPU: {:?}, Download: {:?}",
                    encode_time, gpu_time, download_time
                );
            }

            last_hidden_buffer = Some(layer_input);
            kv_cache.advance();
        }

        // Print timing if enabled
        if enable_timing {
            eprintln!("[TIMING] Embedding: {:?}", total_embed_time);
            eprintln!("[TIMING] Upload: {:?}", total_upload_time);
            eprintln!("[TIMING] Layers: {:?}", total_layer_time);
            eprintln!("[TIMING] Output: {:?}", total_output_time);
            eprintln!(
                "[TIMING] Total: {:?}",
                total_embed_time
                    + total_upload_time
                    + total_layer_time
                    + total_output_time
            );
        }

        Some(logits)
    }

    /// Batched embedding lookup - returns [batch, hidden_size] on GPU
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn embedding_lookup_batched_gpu(
        &self,
        tokens: &[u32],
        device: &metal::DeviceRef,
    ) -> Option<(std::rc::Rc<Buffer>, usize)> {
        if tokens.is_empty() || self.tok_embeddings.is_none() {
            return None;
        }

        let embed = self.tok_embeddings.as_ref()?;
        let hidden_size = self.config.hidden_size;
        let dtype = embed.dtype;
        let batch = tokens.len();

        // Dequantize all token embeddings to F32
        let mut all_embeddings = vec![0.0f32; batch * hidden_size];

        for (i, &token) in tokens.iter().enumerate() {
            let embedding = self.embedding_lookup_impl(token)?;
            all_embeddings[i * hidden_size..(i + 1) * hidden_size]
                .copy_from_slice(&embedding);
        }

        // Create GPU buffer
        let buffer = std::rc::Rc::new(device.new_buffer_with_data(
            all_embeddings.as_ptr() as *const std::ffi::c_void,
            (batch * hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        Some((buffer, batch))
    }

    /// Forward one layer with batched encoding
    /// All operations are encoded into the shared encoder for batching
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_layer_gpu_encode(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        hidden_buffer: &std::rc::Rc<Buffer>,
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        forward_pool: &mut crate::buffer::ForwardBufferPool,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Option<std::rc::Rc<Buffer>> {
        let hidden_size = self.config.hidden_size;
        let pos = kv_cache.seq_len();
        let head_dim = self.config.head_dim;

        // Register the input buffer to keep it alive during GPU execution
        let hidden_buffer = forward_pool.register(hidden_buffer.clone());

        // Get layer weights
        let attention_norm = layer.attention_norm.as_ref()?;
        let wq = layer.wq.as_ref()?;
        let wk = layer.wk.as_ref()?;
        let wv = layer.wv.as_ref()?;
        let wo = layer.wo.as_ref()?;
        let ffn_norm = layer.ffn_norm.as_ref()?;

        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // ====================================================================
        // Create all output buffers using forward pool (keeps them alive!)
        // Note: We use get() instead of get_zeroed() because all kernels
        // write to all elements of their output buffers
        // ====================================================================

        // 1. Attention norm output
        let normed_buffer = forward_pool.get(hidden_size * 4);

        // 2. QKV outputs
        let q_buffer = forward_pool.get(q_out_size * 4);
        let k_buffer = forward_pool.get(k_out_size * 4);
        let v_buffer = forward_pool.get(v_out_size * 4);

        // 3. RoPE outputs
        let q_rope_buffer = forward_pool.get(q_out_size * 4);
        let k_rope_buffer = forward_pool.get(k_out_size * 4);

        // 4. Flash attention output
        let attn_out_buffer = forward_pool.get(q_out_size * 4);

        // 5. Output projection
        let attn_proj_buffer = forward_pool.get(hidden_size * 4);

        // 6. Residual add
        let residual_buffer = forward_pool.get(hidden_size * 4);

        // 7. FFN norm output
        let ffn_normed_buffer = forward_pool.get(hidden_size * 4);

        // 8. FFN intermediate outputs (for SwiGLU)
        let intermediate_size =
            layer.w1.as_ref().map(|w| w.shape[1]).unwrap_or(0);
        let gate_buffer = forward_pool.get(intermediate_size * 4);
        let up_buffer = forward_pool.get(intermediate_size * 4);
        let silu_buffer = forward_pool.get(intermediate_size * 4);
        let hidden_ffn_buffer = forward_pool.get(intermediate_size * 4);
        let ffn_out_buffer = forward_pool.get(hidden_size * 4);

        // 9. Final output
        let final_buffer = forward_pool.get(hidden_size * 4);

        // ====================================================================
        // Encode all operations into shared encoder (no commit!)
        // ====================================================================

        // 1. Attention norm
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
        gpu_ctx.rms_norm_encode(
            encoder,
            &hidden_buffer,
            &norm_weight_buffer,
            &normed_buffer,
            1,
            hidden_size,
            1e-5,
        );

        // 2. QKV projections
        if self
            .gemv_encode(
                encoder,
                &normed_buffer,
                wq,
                q_out_size,
                gpu_ctx,
                &q_buffer,
            )
            .is_none()
        {
            return None;
        }
        if self
            .gemv_encode(
                encoder,
                &normed_buffer,
                wk,
                k_out_size,
                gpu_ctx,
                &k_buffer,
            )
            .is_none()
        {
            return None;
        }
        if self
            .gemv_encode(
                encoder,
                &normed_buffer,
                wv,
                v_out_size,
                gpu_ctx,
                &v_buffer,
            )
            .is_none()
        {
            return None;
        }

        // 3. RoPE
        gpu_ctx.rope_neox_encode(
            encoder,
            &q_buffer,
            &[pos as i32],
            1,
            actual_n_heads,
            head_dim,
            10000.0,
            1.0,
            &q_rope_buffer,
        );
        gpu_ctx.rope_neox_encode(
            encoder,
            &k_buffer,
            &[pos as i32],
            1,
            actual_n_kv_heads,
            head_dim,
            10000.0,
            1.0,
            &k_rope_buffer,
        );

        // 4. Update KV cache using GPU kernel
        let k_cache = kv_cache.k_buffer(layer_idx);
        let v_cache = kv_cache.v_buffer(layer_idx);
        gpu_ctx.kv_update_k_encode(
            encoder,
            &k_rope_buffer,
            k_cache,
            pos,
            kv_cache.max_seq(),
            actual_n_kv_heads,
            head_dim,
        );
        gpu_ctx.kv_update_v_encode(
            encoder,
            &v_buffer,
            v_cache,
            pos,
            kv_cache.max_seq(),
            actual_n_kv_heads,
            head_dim,
        );

        // Memory barrier to ensure KV cache writes are visible to Flash Attention
        use metal::ResourceRef;
        encoder.memory_barrier_with_resources(&[
            k_cache.as_ref() as &ResourceRef,
            v_cache.as_ref() as &ResourceRef,
        ]);

        // 5. Flash attention
        gpu_ctx.flash_attn_f16_encode(
            encoder,
            1,
            actual_n_heads,
            actual_n_kv_heads,
            head_dim,
            pos + 1,
            kv_cache.max_seq(),
            &q_rope_buffer,
            k_cache,
            v_cache,
            &attn_out_buffer,
        );

        // 6. Output projection
        self.gemv_encode(
            encoder,
            &attn_out_buffer,
            wo,
            hidden_size,
            gpu_ctx,
            &attn_proj_buffer,
        )?;

        // 7. Residual add
        gpu_ctx.add_encode(
            encoder,
            &hidden_buffer,
            &attn_proj_buffer,
            &residual_buffer,
            hidden_size,
        );

        // 8. FFN norm
        let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
        gpu_ctx.rms_norm_encode(
            encoder,
            &residual_buffer,
            &ffn_norm_buffer,
            &ffn_normed_buffer,
            1,
            hidden_size,
            1e-5,
        );

        // 9. FFN (gate, up, SiLU, mul, down)
        let w1 = layer.w1.as_ref()?;
        let w2 = layer.w2.as_ref()?;
        let w3 = layer.w3.as_ref()?;

        self.gemv_encode(
            encoder,
            &ffn_normed_buffer,
            w1,
            intermediate_size,
            gpu_ctx,
            &gate_buffer,
        )?;
        self.gemv_encode(
            encoder,
            &ffn_normed_buffer,
            w3,
            intermediate_size,
            gpu_ctx,
            &up_buffer,
        )?;

        // Fused SiLU(gate) * up
        gpu_ctx.fused_swi_glu_encode(
            encoder,
            &gate_buffer,
            &up_buffer,
            &hidden_ffn_buffer,
            intermediate_size,
        );

        // Down projection
        self.gemv_encode(
            encoder,
            &hidden_ffn_buffer,
            w2,
            hidden_size,
            gpu_ctx,
            &ffn_out_buffer,
        )?;

        // 10. Final residual add
        gpu_ctx.add_encode(
            encoder,
            &residual_buffer,
            &ffn_out_buffer,
            &final_buffer,
            hidden_size,
        );

        // Memory barrier to ensure ALL outputs are visible to next layer
        // Note: We need to barrier ALL intermediate buffers, not just final output
        // Metal can reorder operations on different buffers
        encoder.memory_barrier_with_resources(&[
            final_buffer.as_ref() as &ResourceRef,
            residual_buffer.as_ref() as &ResourceRef,
            ffn_out_buffer.as_ref() as &ResourceRef,
        ]);

        // No commit here - caller handles single commit for all layers
        // WARNING: Multiple layers in same command buffer can cause NaN issues
        // Workaround: Use separate command buffers for each layer
        Some(final_buffer)
    }

    /// Forward one layer with batched prefill
    /// Processes batch tokens in parallel using GEMM instead of GEMV
    /// input: [batch, hidden_size] F32 on GPU
    /// Returns: [batch, hidden_size] F32 on GPU
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_layer_prefill_batched(
        &self,
        layer: &LlamaLayer,
        layer_idx: usize,
        input_buffer: &std::rc::Rc<Buffer>,
        batch: usize,
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        forward_pool: &mut crate::buffer::ForwardBufferPool,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Option<std::rc::Rc<Buffer>> {
        let hidden_size = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let start_pos = kv_cache.seq_len();

        // Register input buffer
        let input_buffer = forward_pool.register(input_buffer.clone());

        // Get layer weights
        let attention_norm = layer.attention_norm.as_ref()?;
        let wq = layer.wq.as_ref()?;
        let wk = layer.wk.as_ref()?;
        let wv = layer.wv.as_ref()?;
        let wo = layer.wo.as_ref()?;
        let ffn_norm = layer.ffn_norm.as_ref()?;
        let w1 = layer.w1.as_ref()?;
        let w2 = layer.w2.as_ref()?;
        let w3 = layer.w3.as_ref()?;

        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];
        let n_heads = q_out_size / head_dim;
        let n_kv_heads = k_out_size / head_dim;
        let intermediate_size = w1.shape[1];

        // ================================================================
        // Allocate output buffers
        // ================================================================

        // 1. Attention norm output [batch, hidden_size]
        let normed_buffer = forward_pool.get(batch * hidden_size * 4);

        // 2. F16 version for GEMM [batch, hidden_size]
        let normed_f16_buffer = forward_pool.get(batch * hidden_size * 2);

        // 3. QKV outputs [batch, q/k/v_out_size]
        let q_buffer = forward_pool.get(batch * q_out_size * 4);
        let k_buffer = forward_pool.get(batch * k_out_size * 4);
        let v_buffer = forward_pool.get(batch * v_out_size * 4);

        // 4. F16 versions for KV cache
        let k_f16_buffer = forward_pool.get(batch * k_out_size * 2);
        let v_f16_buffer = forward_pool.get(batch * v_out_size * 2);

        // 5. RoPE outputs [batch, q/k_out_size]
        let q_rope_buffer = forward_pool.get(batch * q_out_size * 4);
        let k_rope_buffer = forward_pool.get(batch * k_out_size * 4);

        // 6. Flash prefill output [batch, q_out_size]
        let attn_out_buffer = forward_pool.get(batch * q_out_size * 4);

        // 7. F16 for output projection
        let attn_out_f16_buffer = forward_pool.get(batch * q_out_size * 2);

        // 8. Output projection [batch, hidden_size]
        let attn_proj_buffer = forward_pool.get(batch * hidden_size * 4);

        // 9. Residual add [batch, hidden_size]
        let residual_buffer = forward_pool.get(batch * hidden_size * 4);

        // 10. FFN norm [batch, hidden_size]
        let ffn_normed_buffer = forward_pool.get(batch * hidden_size * 4);
        let ffn_normed_f16_buffer = forward_pool.get(batch * hidden_size * 2);

        // 11. FFN intermediate [batch, intermediate_size]
        let gate_buffer = forward_pool.get(batch * intermediate_size * 4);
        let up_buffer = forward_pool.get(batch * intermediate_size * 4);
        let gate_f16_buffer = forward_pool.get(batch * intermediate_size * 2);
        let up_f16_buffer = forward_pool.get(batch * intermediate_size * 2);

        // 12. Fused SwiGLU output
        let hidden_ffn_buffer = forward_pool.get(batch * intermediate_size * 4);

        // 13. FFN output [batch, hidden_size]
        let ffn_out_f16_buffer = forward_pool.get(batch * hidden_size * 2);
        let ffn_out_buffer = forward_pool.get(batch * hidden_size * 4);

        // 14. Final output [batch, hidden_size]
        let final_buffer = forward_pool.get(batch * hidden_size * 4);

        // ================================================================
        // Encode operations
        // ================================================================

        // 1. Batched RMSNorm
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
        gpu_ctx.rms_norm_encode(
            encoder,
            &input_buffer,
            &norm_weight_buffer,
            &normed_buffer,
            batch,
            hidden_size,
            1e-5,
        );

        // 2. F32->F16 conversion for GEMM
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &normed_buffer,
            &normed_f16_buffer,
            batch * hidden_size,
        );

        // 3. QKV projections using GEMM
        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        let wk_buffer = gpu_ctx.get_or_create_buffer(wk);
        let wv_buffer = gpu_ctx.get_or_create_buffer(wv);

        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &wq_buffer,
            &normed_f16_buffer,
            &q_buffer,
            q_out_size,
            batch,
            hidden_size,
        );
        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &wk_buffer,
            &normed_f16_buffer,
            &k_buffer,
            k_out_size,
            batch,
            hidden_size,
        );
        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &wv_buffer,
            &normed_f16_buffer,
            &v_buffer,
            v_out_size,
            batch,
            hidden_size,
        );

        // 4. F32->F16 for KV cache
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &k_buffer,
            &k_f16_buffer,
            batch * k_out_size,
        );
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &v_buffer,
            &v_f16_buffer,
            batch * v_out_size,
        );

        // 5. RoPE for each position
        let positions: Vec<i32> =
            (start_pos as i32..(start_pos + batch) as i32).collect();
        gpu_ctx.rope_neox_encode(
            encoder,
            &q_buffer,
            &positions,
            batch,
            n_heads,
            head_dim,
            10000.0,
            1.0,
            &q_rope_buffer,
        );
        gpu_ctx.rope_neox_encode(
            encoder,
            &k_f16_buffer,
            &positions,
            batch,
            n_kv_heads,
            head_dim,
            10000.0,
            1.0,
            &k_rope_buffer,
        );

        // 6. Update KV cache (batched)
        let k_cache = kv_cache.k_buffer(layer_idx);
        let v_cache = kv_cache.v_buffer(layer_idx);
        let max_seq = kv_cache.max_seq();

        gpu_ctx.kv_update_batched_f16_encode(
            encoder,
            &k_rope_buffer,
            k_cache,
            start_pos,
            max_seq,
            n_kv_heads,
            head_dim,
            batch,
        );
        gpu_ctx.kv_update_batched_f16_encode(
            encoder,
            &v_f16_buffer,
            v_cache,
            start_pos,
            max_seq,
            n_kv_heads,
            head_dim,
            batch,
        );

        // 7. Flash Prefill attention
        // Q: [batch, n_heads, head_dim], F32
        // K, V: from KV cache [n_kv_heads, max_seq, head_dim], F16
        let seq_len = start_pos + batch;

        // Q layout: [batch, n_heads, head_dim]
        // Strides: batch stride = n_heads * head_dim, head stride = head_dim, seq stride = 1
        let q_stride_b = n_heads * head_dim;
        let q_stride_h = head_dim;
        let q_stride_s = 1;

        // K/V cache layout: [n_kv_heads, max_seq, head_dim]
        // Strides: head stride = max_seq * head_dim, seq stride = head_dim
        // No batch stride (shared across batch)
        let kv_stride_h = max_seq * head_dim;
        let kv_stride_s = head_dim;

        // For single-sequence prefill: batch=1, seq_q=number of tokens
        // The kernel expects batch to be the number of independent sequences,
        // not the number of tokens in a single sequence.
        gpu_ctx.flash_prefill_batch_encode(
            encoder,
            1, // batch = 1 (single sequence)
            n_heads,
            batch,   // seq_q = number of tokens to process
            seq_len, // seq_k = total sequence length
            head_dim,
            &q_rope_buffer,
            k_cache,
            v_cache,
            &attn_out_buffer,
            q_stride_b,
            q_stride_h,
            q_stride_s,
            0, // k_stride_b (shared KV cache, no batch dimension)
            kv_stride_h,
            kv_stride_s,
            0, // v_stride_b (shared KV cache)
            kv_stride_h,
            kv_stride_s,
            q_stride_b, // o_stride_b (same as Q)
            q_stride_h, // o_stride_h
            q_stride_s, // o_stride_s
            start_pos,  // start_pos for causal mask
        );

        // 8. F32->F16 for output projection
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &attn_out_buffer,
            &attn_out_f16_buffer,
            batch * q_out_size,
        );

        // 9. Output projection (GEMM)
        let wo_buffer = gpu_ctx.get_or_create_buffer(wo);
        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &wo_buffer,
            &attn_out_f16_buffer,
            &attn_proj_buffer,
            hidden_size,
            batch,
            q_out_size,
        );

        // 10. Residual add (batched)
        gpu_ctx.add_encode(
            encoder,
            &input_buffer,
            &attn_proj_buffer,
            &residual_buffer,
            batch * hidden_size,
        );

        // 11. FFN norm
        let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
        gpu_ctx.rms_norm_encode(
            encoder,
            &residual_buffer,
            &ffn_norm_buffer,
            &ffn_normed_buffer,
            batch,
            hidden_size,
            1e-5,
        );

        // 12. F32->F16 for FFN GEMM
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &ffn_normed_buffer,
            &ffn_normed_f16_buffer,
            batch * hidden_size,
        );

        // 13. FFN projections (Gate + Up)
        let w1_buffer = gpu_ctx.get_or_create_buffer(w1);
        let w3_buffer = gpu_ctx.get_or_create_buffer(w3);

        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &w1_buffer,
            &ffn_normed_f16_buffer,
            &gate_buffer,
            intermediate_size,
            batch,
            hidden_size,
        );
        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &w3_buffer,
            &ffn_normed_f16_buffer,
            &up_buffer,
            intermediate_size,
            batch,
            hidden_size,
        );

        // 14. Fused SwiGLU: SiLU(gate) * up
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &gate_buffer,
            &gate_f16_buffer,
            batch * intermediate_size,
        );
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &up_buffer,
            &up_f16_buffer,
            batch * intermediate_size,
        );
        gpu_ctx.fused_swiglu_encode(
            encoder,
            &gate_f16_buffer,
            &up_f16_buffer,
            &hidden_ffn_buffer,
            batch * intermediate_size,
        );

        // 15. F32->F16 for down projection
        gpu_ctx.f32_to_f16_encode(
            encoder,
            &hidden_ffn_buffer,
            &ffn_out_f16_buffer,
            batch * intermediate_size,
        );

        // 16. Down projection (GEMM)
        let w2_buffer = gpu_ctx.get_or_create_buffer(w2);
        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &w2_buffer,
            &ffn_out_f16_buffer,
            &ffn_out_buffer,
            hidden_size,
            batch,
            intermediate_size,
        );

        // 17. Final residual add
        gpu_ctx.add_encode(
            encoder,
            &residual_buffer,
            &ffn_out_buffer,
            &final_buffer,
            batch * hidden_size,
        );

        Some(final_buffer)
    }

    /// GEMV with encode into shared encoder (for batched forward pass)
    /// Returns the output buffer, or None if format not supported
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn gemv_encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input_buffer: &std::rc::Rc<Buffer>,
        weight: &crate::model::tensor::Tensor,
        out_size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
        output_buffer: &std::rc::Rc<Buffer>,
    ) -> Option<()> {
        let in_size =
            input_buffer.length() as usize / std::mem::size_of::<f32>();

        // Use GPU kernel for encoding into shared command buffer
        let weight_buffer = gpu_ctx.get_or_create_buffer(weight);

        if !gpu_ctx.mv_encode(
            encoder,
            &weight_buffer,
            input_buffer,
            output_buffer,
            out_size,
            in_size,
            weight.dtype,
        ) {
            eprintln!(
                "[WARN] GEMV dtype {:?} not supported for GPU encode",
                weight.dtype
            );
            return None;
        }

        Some(())
    }

    /// GEMV with GPU buffers (no CPU transfer for supported formats)
    /// For unsupported formats, falls back to CPU (with GPU-CPU transfer)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn gemv_gpu_buffer(
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
            // F16 - supported (for transposed output layer)
            crate::model::tensor::DataType::F16 => gpu_ctx
                .mv_f16_f32_gpu_weights(
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

    /// GEMV with GPU buffers, with CPU fallback for unsupported formats
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn gemv_gpu_buffer_with_fallback(
        &self,
        input_buffer: &std::rc::Rc<Buffer>,
        weight: &Tensor,
        out_size: usize,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<std::rc::Rc<Buffer>> {
        let in_size =
            input_buffer.length() as usize / std::mem::size_of::<f32>();
        let weights_buffer = gpu_ctx.get_or_create_buffer(weight);

        // Try GPU first
        if let Some(output) = self.gemv_gpu_buffer(
            input_buffer,
            &weights_buffer,
            out_size,
            weight.dtype,
            gpu_ctx,
        ) {
            return Some(output);
        }

        // CPU fallback: download input, compute on CPU, upload output
        let mut input_data = vec![0.0f32; in_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_buffer.contents() as *const f32,
                input_data.as_mut_ptr(),
                in_size,
            );
        }

        // CPU GEMV
        let output = self.gemv(&input_data, weight, out_size)?;

        // Upload to GPU
        let output_buffer =
            std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
                output.as_ptr() as *const std::ffi::c_void,
                (out_size * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

        Some(output_buffer)
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
        let kv_stride = kv_cache.max_seq();

        // Run flash attention on GPU
        gpu_ctx
            .flash_attn_f16_gpu(
                1, // n_seqs
                n_heads, n_kv_heads, head_dim, seq_len, kv_stride, q_buffer,
                &k_buffer, &v_buffer,
            )
            .ok()
    }

    /// FFN SwiGLU with GPU buffers (with CPU fallback for unsupported formats)
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

        // Gate projection: gate = input @ w1 (with CPU fallback)
        let gate_buffer = self.gemv_gpu_buffer_with_fallback(
            input_buffer,
            w1,
            intermediate_size,
            gpu_ctx,
        )?;

        // Up projection: up = input @ w3 (with CPU fallback)
        let up_buffer = self.gemv_gpu_buffer_with_fallback(
            input_buffer,
            w3,
            intermediate_size,
            gpu_ctx,
        )?;

        // SiLU(gate) * up on GPU
        let silu_gate_buffer =
            gpu_ctx.silu_f32_gpu(&gate_buffer, intermediate_size).ok()?;
        let swiglu_buffer = gpu_ctx
            .mul_f32_gpu(&silu_gate_buffer, &up_buffer, intermediate_size)
            .ok()?;

        // Down projection: output = swiglu @ w2 (with CPU fallback)
        self.gemv_gpu_buffer_with_fallback(
            &swiglu_buffer,
            w2,
            hidden_size,
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
        let kv_stride = kv_cache.max_seq();

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
                seq_len,   // current sequence length
                kv_stride, // max_seq for KV cache layout
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
    ///
    /// Note: Currently falls back to forward_gpu_native for correctness.
    /// The ComputeGraph infrastructure is in place for future optimizations
    /// like kernel fusion and batched dispatch.
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn forward_graph(
        &self,
        tokens: &[u32],
        kv_cache: &mut crate::inference::KVCacheGPU,
        gpu_ctx: &crate::quant::vec_dot::gpu::metal::MetalContext,
    ) -> Option<Vec<f32>> {
        // For now, use the existing GPU-native implementation
        // The ComputeGraph infrastructure is ready for future optimizations
        self.forward_gpu_native(tokens, kv_cache, gpu_ctx)
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
            bq: store
                .get(&format!("{}.attention.wq.bias", llama_prefix))
                .or_else(|| store.get(&format!("{}.attn_q.bias", qwen_prefix)))
                .cloned(),
            bk: store
                .get(&format!("{}.attention.wk.bias", llama_prefix))
                .or_else(|| store.get(&format!("{}.attn_k.bias", qwen_prefix)))
                .cloned(),
            bv: store
                .get(&format!("{}.attention.wv.bias", llama_prefix))
                .or_else(|| store.get(&format!("{}.attn_v.bias", qwen_prefix)))
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

            // Fused QKV (for Qwen3.8-style models)
            attn_qkv: store
                .get(&format!("{}.attn_qkv.weight", qwen_prefix))
                .cloned(),

            // MQA attention (for DeepSeek-Coder-V2-style models)
            attn_kv_a_mqa: store
                .get(&format!("{}.attn_kv_a_mqa.weight", qwen_prefix))
                .cloned(),
            attn_kv_b: store
                .get(&format!("{}.attn_kv_b.weight", qwen_prefix))
                .cloned(),
            attn_kv_a_norm: store
                .get(&format!("{}.attn_kv_a_norm.weight", qwen_prefix))
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

            // MoE FFN (for DeepSeek-Coder-V2-style models)
            ffn_gate_inp: store
                .get(&format!("{}.ffn_gate_inp.weight", qwen_prefix))
                .cloned(),
            ffn_gate_exps: store
                .get(&format!("{}.ffn_gate_exps.weight", qwen_prefix))
                .cloned(),
            ffn_up_exps: store
                .get(&format!("{}.ffn_up_exps.weight", qwen_prefix))
                .cloned(),
            ffn_down_exps: store
                .get(&format!("{}.ffn_down_exps.weight", qwen_prefix))
                .cloned(),
            ffn_gate_shexp: store
                .get(&format!("{}.ffn_gate_shexp.weight", qwen_prefix))
                .cloned(),
            ffn_up_shexp: store
                .get(&format!("{}.ffn_up_shexp.weight", qwen_prefix))
                .cloned(),
            ffn_down_shexp: store
                .get(&format!("{}.ffn_down_shexp.weight", qwen_prefix))
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
