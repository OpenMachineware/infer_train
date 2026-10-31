// SPDX-License-Identifier: AGPL-3.0-or-later
//
// CPU Executor - Zero-allocation CPU inference execution
//
// This module provides an executor that pre-allocates all buffers needed
// for CPU inference, eliminating per-forward allocation overhead.

use crate::inference::KVCache;
use crate::inference::kv_cache::KVCacheBase;
use crate::model::llama::{LlamaLayer, LlamaModel};

/// CPU Executor with pre-allocated work buffers
///
/// This executor eliminates allocation overhead by pre-allocating all
/// intermediate buffers needed for forward pass.
#[allow(dead_code)]
pub struct CpuExecutor {
    // Pre-allocated work buffers
    hidden: Vec<f32>,
    hidden_alt: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_hidden: Vec<f32>,
    logits: Vec<f32>,

    // Attention scores buffer (variable size)
    attn_scores: Vec<f32>,

    // MLA buffers (Multi-head Latent Attention)
    kv_a_proj: Vec<f32>,   // kv_lora_rank + qk_rope_head_dim
    kv_a_normed: Vec<f32>, // kv_lora_rank
    kv: Vec<f32>,          // n_kv_heads * (qk_nope_head_dim + value_length)

    // Configuration
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    q_size: usize,
    k_size: usize,
    v_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    v_head_dim: usize,

    // MLA configuration
    uses_mla: bool,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    key_length: usize,
    value_length: usize,
    q_head_dim: usize, // Q's head_dim (different from K/V for MLA)
}

impl CpuExecutor {
    /// Create a new CPU executor with pre-allocated buffers
    pub fn new(model: &LlamaModel) -> Self {
        let hidden_size = model.config.hidden_size;
        let intermediate_size = model.config.intermediate_size;
        let vocab_size = model.config.vocab_size;
        let n_heads = model.config.n_heads;
        let n_kv_heads = model.config.n_kv_heads;
        let head_dim = model.config.head_dim;

        // Check if model uses MLA (Multi-head Latent Attention)
        let uses_mla = model.uses_mqa();
        let kv_lora_rank = model.config.kv_lora_rank;
        let qk_rope_head_dim = model.config.qk_rope_head_dim;
        let key_length = model.config.key_length;
        let value_length = model.config.value_length;

        // For MLA, Q's head_dim is different from K/V
        // Q uses q_head_dim (from wq shape), K uses key_length, V uses value_length
        let q_head_dim = if uses_mla && model.layers.first().is_some() {
            let layer = model.layers.first().unwrap();
            if let Some(wq) = &layer.wq {
                wq.shape[1] / n_heads
            } else {
                head_dim
            }
        } else {
            head_dim
        };

        // Calculate buffer sizes based on architecture
        let q_size = n_heads * q_head_dim;
        let (k_size, v_size, v_head_dim) = if uses_mla {
            // MLA: K and V have different dimensions
            let k_size = n_kv_heads * key_length;
            let v_size = n_kv_heads * value_length;
            let v_head_dim = value_length;
            (k_size, v_size, v_head_dim)
        } else {
            let k_size = n_kv_heads * head_dim;
            let v_size = n_kv_heads * head_dim;
            let v_head_dim = v_size / n_kv_heads;
            (k_size, v_size, v_head_dim)
        };

        // Pre-allocate all buffers
        let hidden = vec![0.0f32; hidden_size];
        let hidden_alt = vec![0.0f32; hidden_size];
        let normed = vec![0.0f32; hidden_size];
        let q = vec![0.0f32; q_size];
        let k = vec![0.0f32; k_size];
        let v = vec![0.0f32; v_size];
        let attn_out = vec![0.0f32; n_heads * v_head_dim];
        let ffn_gate = vec![0.0f32; intermediate_size];
        let ffn_up = vec![0.0f32; intermediate_size];
        let ffn_hidden = vec![0.0f32; intermediate_size];
        let logits = vec![0.0f32; vocab_size];
        let attn_scores = vec![0.0f32; 2048]; // max_seq_len

        // MLA buffers
        let kv_a_proj = if uses_mla {
            vec![0.0f32; kv_lora_rank + qk_rope_head_dim]
        } else {
            vec![]
        };
        let kv_a_normed =
            if uses_mla { vec![0.0f32; kv_lora_rank] } else { vec![] };
        let kv = if uses_mla {
            let qk_nope_head_dim = key_length - qk_rope_head_dim;
            vec![0.0f32; n_kv_heads * (qk_nope_head_dim + value_length)]
        } else {
            vec![]
        };

        CpuExecutor {
            hidden,
            hidden_alt,
            normed,
            q,
            k,
            v,
            attn_out,
            ffn_gate,
            ffn_up,
            ffn_hidden,
            logits,
            attn_scores,
            kv_a_proj,
            kv_a_normed,
            kv,
            hidden_size,
            intermediate_size,
            vocab_size,
            q_size,
            k_size,
            v_size,
            n_heads,
            n_kv_heads,
            head_dim,
            v_head_dim,
            uses_mla,
            kv_lora_rank,
            qk_rope_head_dim,
            key_length,
            value_length,
            q_head_dim,
        }
    }

    /// Get total memory used by pre-allocated buffers
    pub fn memory_usage(&self) -> usize {
        let total_elements = self.hidden.len() * 2 +  // hidden + hidden_alt
            self.normed.len() +
            self.q.len() +
            self.k.len() +
            self.v.len() +
            self.attn_out.len() +
            self.ffn_gate.len() * 3 +  // gate + up + hidden
            self.logits.len() +
            self.attn_scores.len();

        total_elements * std::mem::size_of::<f32>()
    }

    /// Execute forward pass with zero allocation
    pub fn forward(
        &mut self,
        model: &LlamaModel,
        tokens: &[u32],
        kv_cache: &mut KVCache,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() || model.tok_embeddings.is_none() {
            return None;
        }

        // Process each token
        for &token in tokens.iter() {
            // 1. Embedding lookup
            model.embedding_lookup_into(token, &mut self.hidden)?;

            // 2. Process through layers
            for (layer_idx, layer) in model.layers.iter().enumerate() {
                self.forward_layer(model, layer, layer_idx, kv_cache)?;
            }

            // Advance KV cache
            kv_cache.advance();
        }

        // 3. Final norm and output projection
        model.apply_rms_norm_into(
            &self.hidden,
            model.norm.as_ref()?,
            &mut self.normed,
        )?;
        model.gemv_into(&self.normed, model.output.as_ref()?, &mut self.logits);

        // Return logits (copy, since buffers are reused)
        Some(self.logits.clone())
    }

    /// Forward pass through a single layer
    fn forward_layer(
        &mut self,
        model: &LlamaModel,
        layer: &LlamaLayer,
        layer_idx: usize,
        kv_cache: &mut KVCache,
    ) -> Option<()> {
        // 1. Attention norm
        model.apply_rms_norm_into(
            &self.hidden,
            layer.attention_norm.as_ref()?,
            &mut self.normed,
        )?;

        // 2. QKV projections
        self.forward_qkv(model, layer)?;

        // 3. RoPE
        let pos = kv_cache.seq_len();

        if self.uses_mla {
            // MLA: Q gets full RoPE, K only applies RoPE to the rope part
            let q_copy = self.q.clone();

            // Apply RoPE to Q with q_head_dim
            model.apply_rope_into(
                &q_copy,
                pos,
                self.n_heads,
                self.q_head_dim,
                &mut self.q,
            )?;

            // For K, apply RoPE only to the rope part of each head
            // K structure per head: [qk_rope_head_dim (rope), qk_nope_head_dim (non-rope)]
            for head in 0..self.n_kv_heads {
                let k_head_start = head * self.key_length;

                // Extract rope part of this head
                let mut k_rope: Vec<f32> = self.k
                    [k_head_start..k_head_start + self.qk_rope_head_dim]
                    .to_vec();

                // Apply RoPE to rope part
                model.apply_rope_into(
                    &k_rope,
                    pos,
                    1, // single head
                    self.qk_rope_head_dim,
                    &mut self.k
                        [k_head_start..k_head_start + self.qk_rope_head_dim],
                )?;
            }
        } else {
            // Standard attention: apply RoPE to full Q and K
            let q_copy = self.q.clone();
            let k_copy = self.k.clone();
            model.apply_rope_into(
                &q_copy,
                pos,
                self.n_heads,
                self.head_dim,
                &mut self.q,
            )?;
            model.apply_rope_into(
                &k_copy,
                pos,
                self.n_kv_heads,
                self.head_dim,
                &mut self.k,
            )?;
        }

        // 4. Attention
        self.forward_attention(layer_idx, kv_cache)?;

        // 5. Output projection
        model.gemv_into(
            &self.attn_out,
            layer.wo.as_ref()?,
            &mut self.hidden_alt,
        );

        // 6. Residual
        for i in 0..self.hidden_size {
            self.hidden[i] += self.hidden_alt[i];
        }

        // 7. FFN norm
        model.apply_rms_norm_into(
            &self.hidden,
            layer.ffn_norm.as_ref()?,
            &mut self.normed,
        )?;

        // 8. FFN
        self.forward_ffn(model, layer)?;

        // 9. Residual
        for i in 0..self.hidden_size {
            self.hidden[i] += self.hidden_alt[i];
        }

        Some(())
    }

    /// QKV projections with pre-allocated buffers
    fn forward_qkv(
        &mut self,
        model: &LlamaModel,
        layer: &LlamaLayer,
    ) -> Option<()> {
        if self.uses_mla {
            // MLA (Multi-head Latent Attention) - DeepSeek-Coder-V2 style
            // Flow:
            // 1. Q = hidden @ attn_q
            // 2. KV_A = hidden @ attn_kv_a_mqa → [kv_lora_rank + qk_rope_head_dim]
            // 3. KV_A_normed = RMSNorm(KV_A[:kv_lora_rank], attn_kv_a_norm)
            // 4. KV = KV_A_normed @ attn_kv_b
            // 5. Construct K, V

            // 1. Q projection
            model.gemv_into(&self.normed, layer.wq.as_ref()?, &mut self.q);

            // 2. KV_A projection (compressed)
            model.gemv_into(
                &self.normed,
                layer.attn_kv_a_mqa.as_ref()?,
                &mut self.kv_a_proj,
            );

            // 3. KV_A normalization (RMSNorm on the first kv_lora_rank elements)
            let kv_a_norm = layer.attn_kv_a_norm.as_ref()?;
            let norm_data: &[f32] = bytemuck::cast_slice(kv_a_norm.bytes());
            let eps = model.config.rms_norm_eps;

            // Compute RMS
            let mut ss = 0.0f32;
            for i in 0..self.kv_lora_rank {
                ss += self.kv_a_proj[i] * self.kv_a_proj[i];
            }
            ss /= self.kv_lora_rank as f32;
            let rms = 1.0 / (ss + eps).sqrt();

            // Normalize and scale
            for i in 0..self.kv_lora_rank {
                self.kv_a_normed[i] = self.kv_a_proj[i] * rms * norm_data[i];
            }

            // 4. KV_B projection (decompress to full KV)
            model.gemv_into(
                &self.kv_a_normed,
                layer.attn_kv_b.as_ref()?,
                &mut self.kv,
            );

            // 5. Construct K and V
            // qk_nope_head_dim = key_length - qk_rope_head_dim
            let qk_nope_head_dim = self.key_length - self.qk_rope_head_dim;

            // The RoPE part of the key is shared across all heads
            let k_rope_start = self.kv_lora_rank;
            let k_rope_end = k_rope_start + self.qk_rope_head_dim;

            // The non-RoPE part of the key and value come from kv
            // kv layout: [k_nope_head_0, ..., k_nope_head_n, v_head_0, ..., v_head_n]
            let k_nope_size = self.n_kv_heads * qk_nope_head_dim;

            // Construct K: for each head, concat(k_rope, k_nope[head])
            for head in 0..self.n_kv_heads {
                let k_head_start = head * self.key_length;

                // Add RoPE part (shared)
                for d in 0..self.qk_rope_head_dim {
                    self.k[k_head_start + d] = self.kv_a_proj[k_rope_start + d];
                }

                // Add non-RoPE part (per-head)
                let nope_start = head * qk_nope_head_dim;
                for d in 0..qk_nope_head_dim {
                    self.k[k_head_start + self.qk_rope_head_dim + d] =
                        self.kv[nope_start + d];
                }
            }

            // V is the remaining part of kv
            for i in 0..self.v_size {
                self.v[i] = self.kv[k_nope_size + i];
            }
        } else {
            // Standard attention
            model.gemv_into(&self.normed, layer.wq.as_ref()?, &mut self.q);
            model.gemv_into(&self.normed, layer.wk.as_ref()?, &mut self.k);
            model.gemv_into(&self.normed, layer.wv.as_ref()?, &mut self.v);
        }
        Some(())
    }

    /// Attention computation with pre-allocated buffers
    fn forward_attention(
        &mut self,
        layer_idx: usize,
        kv_cache: &mut KVCache,
    ) -> Option<()> {
        // Determine dimensions based on architecture
        let (k_head_dim, k_dim_for_q) = if self.uses_mla {
            // MLA: K uses key_length, Q uses q_head_dim for K matching
            (self.key_length, self.key_length)
        } else {
            (self.head_dim, self.head_dim)
        };

        // Store K, V in cache with correct dimensions
        kv_cache.set_k(layer_idx, &self.k, self.n_kv_heads, k_head_dim);
        kv_cache.set_v(layer_idx, &self.v, self.n_kv_heads, self.v_head_dim);

        let seq_len = kv_cache.seq_len() + 1;
        let scale = 1.0f32 / (k_dim_for_q as f32).sqrt();

        // Clear output
        for elem in self.attn_out.iter_mut() {
            *elem = 0.0;
        }

        // For each query head
        for h in 0..self.n_heads {
            let kv_head = h * self.n_kv_heads / self.n_heads;

            // Compute attention scores
            let scores_start = h * seq_len;
            if scores_start + seq_len > self.attn_scores.len() {
                return None;
            }

            let scores =
                &mut self.attn_scores[scores_start..scores_start + seq_len];

            // Compute scores
            // Q uses q_head_dim (different from K for MLA)
            let q_dim =
                if self.uses_mla { self.q_head_dim } else { self.head_dim };

            for s in 0..seq_len {
                let k_s = kv_cache.get_k(layer_idx, kv_head, s);
                let mut score = 0.0f32;
                for d in 0..q_dim.min(k_head_dim) {
                    score += self.q[h * q_dim + d] * k_s[d];
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
            for d in 0..self.v_head_dim {
                let mut sum = 0.0f32;
                for s in 0..seq_len {
                    let v_s = kv_cache.get_v(layer_idx, kv_head, s);
                    sum += scores[s] * v_s[d];
                }
                self.attn_out[h * self.v_head_dim + d] = sum;
            }
        }

        Some(())
    }

    /// FFN computation with pre-allocated buffers
    fn forward_ffn(
        &mut self,
        model: &LlamaModel,
        layer: &LlamaLayer,
    ) -> Option<()> {
        // Use model's compute_ffn which handles both standard FFN and MoE
        // (not zero-allocation for MoE but functional)
        let output =
            model.compute_ffn(layer, &self.normed, self.hidden_size)?;
        self.hidden_alt.copy_from_slice(&output);
        Some(())
    }
}
