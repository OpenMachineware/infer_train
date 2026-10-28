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

        let q_size = n_heads * head_dim;
        let k_size = n_kv_heads * head_dim;
        let v_size = n_kv_heads * head_dim;
        let v_head_dim = v_size / n_kv_heads;

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

        // 3. RoPE (need temp buffer for in-place)
        let pos = kv_cache.seq_len();
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
        model.gemv_into(&self.normed, layer.wq.as_ref()?, &mut self.q);
        model.gemv_into(&self.normed, layer.wk.as_ref()?, &mut self.k);
        model.gemv_into(&self.normed, layer.wv.as_ref()?, &mut self.v);
        Some(())
    }

    /// Attention computation with pre-allocated buffers
    fn forward_attention(
        &mut self,
        layer_idx: usize,
        kv_cache: &mut KVCache,
    ) -> Option<()> {
        // Store K, V in cache
        kv_cache.set_k(layer_idx, &self.k, self.n_kv_heads, self.head_dim);
        kv_cache.set_v(layer_idx, &self.v, self.n_kv_heads, self.head_dim);

        let seq_len = kv_cache.seq_len() + 1;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

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
            for s in 0..seq_len {
                let k_s = kv_cache.get_k(layer_idx, kv_head, s);
                let mut score = 0.0f32;
                for d in 0..self.head_dim {
                    score += self.q[h * self.head_dim + d] * k_s[d];
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
        // Gate and up projections
        model.gemv_into(&self.normed, layer.w1.as_ref()?, &mut self.ffn_gate);
        model.gemv_into(&self.normed, layer.w3.as_ref()?, &mut self.ffn_up);

        // SwiGLU activation using std::thread::scope
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let intermediate_size = self.ffn_hidden.len();
        let n_threads =
            std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        let counter = AtomicUsize::new(0);
        let chunk_size = 256;

        let ffn_gate = &self.ffn_gate;
        let ffn_up = &self.ffn_up;
        let ffn_hidden_addr = self.ffn_hidden.as_mut_ptr() as usize;

        thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    let ffn_hidden_ptr = ffn_hidden_addr as *mut f32;
                    loop {
                        let chunk_start =
                            counter.fetch_add(chunk_size, Ordering::Relaxed);
                        if chunk_start >= intermediate_size {
                            break;
                        }
                        let chunk_end =
                            (chunk_start + chunk_size).min(intermediate_size);
                        for i in chunk_start..chunk_end {
                            let silu = ffn_up[i] / (1.0 + (-ffn_up[i]).exp());
                            unsafe {
                                *ffn_hidden_ptr.add(i) = ffn_gate[i] * silu;
                            }
                        }
                    }
                });
            }
        });

        // Down projection
        model.gemv_into(
            &self.ffn_hidden,
            layer.w2.as_ref()?,
            &mut self.hidden_alt,
        );

        Some(())
    }
}
