// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Unified model loader with architecture detection

use crate::model::config::ModelConfig;
use crate::model::gguf::GgufFile;
use crate::model::hybrid::HybridMoESsmLayer;
use crate::model::llama::LlamaLayer;
use crate::model::mamba::MambaLayer;
use crate::model::moe::MoeLayer;
use crate::model::tensor::Tensor;
use crate::model::weights::WeightStore;

/// Model architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    MoE,
    Mamba,
    Hybrid,
    Unknown,
}

/// Unified model with multiple architecture support
pub struct Model {
    pub config: ModelConfig,
    pub architecture: Architecture,

    // Common components
    pub tok_embeddings: Option<Tensor>,
    pub norm: Option<Tensor>,
    pub output: Option<Tensor>,

    // Architecture-specific layers
    pub llama_layers: Vec<LlamaLayer>,
    pub moe_layers: Vec<MoeLayer>,
    pub mamba_layers: Vec<MambaLayer>,
    pub hybrid_layers: Vec<HybridMoESsmLayer>,
}

impl Model {
    pub fn from_gguf(path: &str) -> Result<Self, String> {
        let gguf = GgufFile::open(path).map_err(|e| e.to_string())?;
        let config = ModelConfig::from_gguf(&gguf);
        let store = WeightStore::from_gguf(path).map_err(|e| e.to_string())?;

        Ok(Self::from_weights(&store, config))
    }

    pub fn from_weights(store: &WeightStore, config: ModelConfig) -> Self {
        let architecture = Self::detect_architecture(store, &config);

        // Load embeddings
        let tok_embeddings = store
            .get("tok_embeddings.weight")
            .or_else(|| store.get("token_embd.weight"))
            .cloned();

        // Load output norm
        let norm = store
            .get("norm.weight")
            .or_else(|| store.get("output_norm.weight"))
            .cloned();

        // Load output projection
        let output = store
            .get("output.weight")
            .or_else(|| store.get("lm_head.weight"))
            .cloned();

        // Load layers based on architecture
        let (llama_layers, moe_layers, mamba_layers, hybrid_layers) =
            match architecture {
                Architecture::Llama => {
                    let layers: Vec<LlamaLayer> = (0..config.n_layers)
                        .map(|i| LlamaLayer::from_weights(store, i))
                        .collect();
                    (layers, vec![], vec![], vec![])
                }
                Architecture::MoE => {
                    let layers: Vec<MoeLayer> = (0..config.n_layers)
                        .map(|i| MoeLayer::from_weights(store, i))
                        .collect();
                    (vec![], layers, vec![], vec![])
                }
                Architecture::Mamba => {
                    let layers: Vec<MambaLayer> = (0..config.n_layers)
                        .map(|i| MambaLayer::from_weights(store, i))
                        .collect();
                    (vec![], vec![], layers, vec![])
                }
                Architecture::Hybrid => {
                    let layers: Vec<HybridMoESsmLayer> = (0..config.n_layers)
                        .map(|i| HybridMoESsmLayer::from_weights(store, i))
                        .collect();
                    (vec![], vec![], vec![], layers)
                }
                Architecture::Unknown => {
                    // Fallback: try Llama
                    let layers: Vec<LlamaLayer> = (0..config.n_layers)
                        .map(|i| LlamaLayer::from_weights(store, i))
                        .collect();
                    (layers, vec![], vec![], vec![])
                }
            };

        Model {
            config,
            architecture,
            tok_embeddings,
            norm,
            output,
            llama_layers,
            moe_layers,
            mamba_layers,
            hybrid_layers,
        }
    }

    fn detect_architecture(
        store: &WeightStore,
        config: &ModelConfig,
    ) -> Architecture {
        // Check first few layers for architecture hints
        for i in 0..std::cmp::min(5, config.n_layers) {
            let prefix = format!("blk.{}.", i);

            // Check for MoE + SSM hybrid (has both ffn_down_exps and ssm_in)
            let has_moe =
                store.get(&format!("{}ffn_down_exps.weight", prefix)).is_some()
                    || store
                        .get(&format!("{}ffn_gate_inp.weight", prefix))
                        .is_some();
            let has_ssm = store
                .get(&format!("{}ssm_in.weight", prefix))
                .is_some()
                || store.get(&format!("{}ssm_out.weight", prefix)).is_some();

            if has_moe && has_ssm {
                return Architecture::Hybrid;
            }

            // Check for pure MoE
            if has_moe {
                return Architecture::MoE;
            }

            // Check for pure Mamba/SSM
            if has_ssm {
                return Architecture::Mamba;
            }
        }

        // Default: Llama-style
        Architecture::Llama
    }

    pub fn layer_count(&self) -> usize {
        self.llama_layers.len()
            + self.moe_layers.len()
            + self.mamba_layers.len()
            + self.hybrid_layers.len()
    }

    pub fn complete_layers(&self) -> usize {
        let llama =
            self.llama_layers.iter().filter(|l| l.is_complete()).count();
        let moe = self.moe_layers.iter().filter(|l| l.is_complete()).count();
        let mamba =
            self.mamba_layers.iter().filter(|l| l.is_complete()).count();
        let hybrid =
            self.hybrid_layers.iter().filter(|l| l.is_complete()).count();
        llama + moe + mamba + hybrid
    }

    pub fn param_count(&self) -> usize {
        let mut count = 0usize;

        if let Some(ref t) = self.tok_embeddings {
            count += t.elem_count();
        }
        if let Some(ref t) = self.norm {
            count += t.elem_count();
        }
        if let Some(ref t) = self.output {
            count += t.elem_count();
        }

        for layer in &self.llama_layers {
            count += layer.param_count();
        }
        for layer in &self.moe_layers {
            count += layer.param_count();
        }
        for layer in &self.mamba_layers {
            count += layer.param_count();
        }
        for layer in &self.hybrid_layers {
            count += layer.param_count();
        }

        count
    }

    /// Forward pass for a single layer
    pub fn forward_layer(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        kv_cache: &mut crate::inference::KVCache,
    ) -> Option<Vec<f32>> {
        match self.architecture {
            Architecture::Llama => {
                if layer_idx < self.llama_layers.len() {
                    // Use LlamaModel's forward_layer
                    // For now, return None - this will be handled by LlamaModel
                    None
                } else {
                    None
                }
            }
            Architecture::MoE => {
                if layer_idx < self.moe_layers.len() {
                    // MoE forward
                    None
                } else {
                    None
                }
            }
            Architecture::Mamba => {
                if layer_idx < self.mamba_layers.len() {
                    self.forward_mamba_layer(
                        hidden,
                        &self.mamba_layers[layer_idx],
                    )
                } else {
                    None
                }
            }
            Architecture::Hybrid => {
                if layer_idx < self.hybrid_layers.len() {
                    self.forward_hybrid_layer(
                        hidden,
                        &self.hybrid_layers[layer_idx],
                    )
                } else {
                    None
                }
            }
            Architecture::Unknown => None,
        }
    }

    /// Forward pass for Mamba/SSM layer (Qwen3.5 architecture)
    fn forward_mamba_layer(
        &self,
        hidden: &[f32],
        layer: &MambaLayer,
    ) -> Option<Vec<f32>> {
        let hidden_size = self.config.hidden_size;

        // 1. Attention norm
        let normed = self.apply_rms_norm(
            hidden,
            layer.attention_norm.as_ref()?,
            hidden_size,
        )?;

        // 2. QKV projection (fused or separate)
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        // For Qwen3.5 full attention: head_dim from key_length, for SSM: use config.head_dim
        let head_dim = if self.config.key_length > 0 {
            self.config.key_length
        } else {
            self.config.head_dim
        };

        let (q, k, v) = if let Some(attn_qkv) = &layer.attn_qkv {
            // Fused QKV projection
            let qkv_out_size = attn_qkv.shape[1];
            let qkv = self.gemv(&normed, attn_qkv, qkv_out_size)?;

            let q_size = n_heads * head_dim;
            let k_size = n_kv_heads * head_dim;
            let v_size = n_kv_heads * self.config.value_length;

            (
                qkv[0..q_size].to_vec(),
                qkv[q_size..q_size + k_size].to_vec(),
                qkv[q_size + k_size..q_size + k_size + v_size].to_vec(),
            )
        } else if let (Some(wq), Some(wk), Some(wv)) =
            (&layer.wq, &layer.wk, &layer.wv)
        {
            // Separate Q, K, V projections
            let q = self.gemv(&normed, wq, wq.shape[1])?;
            let k = self.gemv(&normed, wk, wk.shape[1])?;
            let v = self.gemv(&normed, wv, wv.shape[1])?;
            (q, k, v)
        } else {
            return None;
        };

        // 4. Apply RoPE to Q and K
        let pos = 0; // TODO: use kv_cache position
        let q_rope = self.apply_rope(&q, pos, n_heads, head_dim)?;
        let k_rope = self.apply_rope(&k, pos, n_kv_heads, head_dim)?;

        // 5. Attention computation
        let attn_out = self
            .attention(&q_rope, &k_rope, &v, n_heads, head_dim, n_kv_heads)?;

        // 6. Output projection (wo or attn_output for full attention, or use attn_gate for SSM)
        let attn_proj = if let Some(wo) = &layer.wo {
            // Full attention layer: has wo output projection
            self.gemv(&attn_out, wo, hidden_size)?
        } else if let Some(attn_gate) = &layer.attn_gate {
            // SSM layer: attn_gate is the combined attention + gate mechanism
            // attn_gate shape: [hidden_size, intermediate_size]
            // For Gated DeltaNet, this gates the attention output
            let gate_out = self.gemv(&normed, attn_gate, attn_gate.shape[1])?;

            // Apply attention output with gating
            // The attention output (n_heads * head_dim) needs to match gate output
            let attn_size = attn_out.len();
            let gate_size = gate_out.len();

            if attn_size == gate_size {
                // Direct gating
                gate_out
                    .iter()
                    .zip(attn_out.iter())
                    .map(|(g, a)| {
                        let gate = 1.0 / (1.0 + (-g).exp());
                        gate * a
                    })
                    .collect()
            } else {
                // SSM layer: gate projects to intermediate, use ssm_out to project back
                // For now, use a simplified approach
                gate_out
            }
        } else {
            // No output projection available
            attn_out
        };

        // Residual
        let residual: Vec<f32> = if attn_proj.len() == hidden_size {
            hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect()
        } else {
            // Intermediate size, need to project back via ssm_out
            hidden.to_vec()
        };

        // Post-attention norm
        let post_normed = if let Some(post_norm) = &layer.post_attention_norm {
            self.apply_rms_norm(&residual, post_norm, hidden_size)?
        } else {
            residual
        };

        // 10. SSM computation (if present)
        if layer.is_ssm() {
            let ssm_out = self.compute_ssm(&post_normed, layer, hidden_size)?;

            // 11. Combine SSM and FFN (if present)
            if let (Some(ffn_gate), Some(ffn_up), Some(ffn_down)) =
                (&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down)
            {
                let ffn_normed = if let Some(ffn_norm) = &layer.ffn_norm {
                    self.apply_rms_norm(&post_normed, ffn_norm, hidden_size)?
                } else {
                    post_normed.clone()
                };

                let intermediate_size = ffn_gate.shape[1];
                let gate_out =
                    self.gemv(&ffn_normed, ffn_gate, intermediate_size)?;
                let up_out =
                    self.gemv(&ffn_normed, ffn_up, intermediate_size)?;

                let mut ffn_hidden = vec![0.0f32; intermediate_size];
                for i in 0..intermediate_size {
                    let silu = up_out[i] / (1.0 + (-up_out[i]).exp());
                    ffn_hidden[i] = gate_out[i] * silu;
                }

                let ffn_out = self.gemv(&ffn_hidden, ffn_down, hidden_size)?;

                // Combine SSM and FFN
                let combined: Vec<f32> = ssm_out
                    .iter()
                    .zip(ffn_out.iter())
                    .map(|(s, f)| s + f)
                    .collect();

                // Final residual
                Some(
                    post_normed
                        .iter()
                        .zip(combined.iter())
                        .map(|(r, c)| r + c)
                        .collect(),
                )
            } else {
                // Only SSM
                Some(
                    post_normed
                        .iter()
                        .zip(ssm_out.iter())
                        .map(|(r, s)| r + s)
                        .collect(),
                )
            }
        } else {
            // No SSM, just FFN
            if let (Some(ffn_gate), Some(ffn_up), Some(ffn_down)) =
                (&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down)
            {
                let intermediate_size = ffn_gate.shape[1];
                let gate_out =
                    self.gemv(&post_normed, ffn_gate, intermediate_size)?;
                let up_out =
                    self.gemv(&post_normed, ffn_up, intermediate_size)?;

                let mut ffn_hidden = vec![0.0f32; intermediate_size];
                for i in 0..intermediate_size {
                    let silu = up_out[i] / (1.0 + (-up_out[i]).exp());
                    ffn_hidden[i] = gate_out[i] * silu;
                }

                let ffn_out = self.gemv(&ffn_hidden, ffn_down, hidden_size)?;
                Some(
                    post_normed
                        .iter()
                        .zip(ffn_out.iter())
                        .map(|(r, f)| r + f)
                        .collect(),
                )
            } else {
                Some(post_normed)
            }
        }
    }

    /// Compute SSM (State Space Model) forward pass
    fn compute_ssm(
        &self,
        hidden: &[f32],
        layer: &MambaLayer,
        hidden_size: usize,
    ) -> Option<Vec<f32>> {
        // SSM output projection
        if let Some(ssm_out_weight) = &layer.ssm_out {
            // For Qwen3.5: ssm_out shape=[6144, 5120]
            // This projects from intermediate to hidden_size
            let ssm_intermediate = ssm_out_weight.shape[0];

            // Simple projection (actual SSM is more complex)
            // For now, just use the output projection as a placeholder
            // TODO: Implement full SSM computation with state
            self.gemv(hidden, ssm_out_weight, hidden_size)
        } else {
            Some(vec![0.0f32; hidden_size])
        }
    }

    /// Forward pass for Hybrid MoE + SSM layer
    fn forward_hybrid_layer(
        &self,
        hidden: &[f32],
        layer: &HybridMoESsmLayer,
    ) -> Option<Vec<f32>> {
        // Similar to Mamba layer but with MoE FFN
        // TODO: Implement
        None
    }

    // Helper methods
    fn apply_rms_norm(
        &self,
        x: &[f32],
        weight: &Tensor,
        size: usize,
    ) -> Option<Vec<f32>> {
        let w = crate::inference::dequant_to_f32(
            weight.bytes(),
            weight.dtype,
            size,
        );
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
        let w_data =
            crate::inference::dequant_to_f32(w.bytes(), w.dtype, total_elems);

        let mut out = vec![0.0f32; out_size];
        for i in 0..out_size {
            let mut sum = 0.0f32;
            for j in 0..in_size {
                sum += w_data[j + i * in_size] * x[j];
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
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Option<Vec<f32>> {
        // Simplified attention (no KV cache, single position)
        let mut out = vec![0.0f32; n_heads * head_dim];

        // Grouped Query Attention
        let n_rep = n_heads / n_kv_heads;

        for h in 0..n_heads {
            let kv_h = h / n_rep; // Map to KV head
            let q_offset = h * head_dim;
            let k_offset = kv_h * head_dim;
            let v_offset = kv_h * (v.len() / n_kv_heads);

            // Q @ K^T (single position, so just dot product)
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[q_offset + d] * k[k_offset + d];
            }
            score /= (head_dim as f32).sqrt();

            // Softmax (single value, so it's just the value itself)
            let attn_weight = 1.0 / (1.0 + (-score).exp()); // sigmoid as placeholder

            // Weighted V
            let v_head_dim = v.len() / n_kv_heads;
            for d in 0..v_head_dim.min(head_dim) {
                out[q_offset + d] = attn_weight * v[v_offset + d];
            }
        }

        Some(out)
    }
}
