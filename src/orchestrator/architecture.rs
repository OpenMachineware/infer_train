// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Architecture detection for different model types.
//!
//! Supports:
//! - Standard: Regular LLaMA attention (wq, wk, wv, wo)
//! - MQA: Multi-Query Attention (attn_kv_a_mqa, attn_kv_b)
//! - MLA: Multi-Latent Attention (DeepSeek-V2/V3)
//! - MoE: Mixture of Experts (DeepSeek, Mixtral)

use crate::model::llama::LlamaModel;

/// Attention architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard attention with separate Q, K, V projections
    Standard,
    /// Multi-Query Attention (shared K, V across heads)
    Mqa,
    /// Multi-Latent Attention (DeepSeek-V2/V3)
    Mla,
    /// Fused QKV projection
    FusedQkv,
}

/// FFN architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnType {
    /// Standard FFN with gate and up projections
    SwiGLU,
    /// Mixture of Experts
    MoE { num_experts: usize, num_experts_per_tok: usize },
    /// Standard MLP (no gate)
    Mlp,
}

/// Model architecture configuration
#[derive(Debug, Clone)]
pub struct Architecture {
    /// Attention architecture type
    pub attention: AttentionType,

    /// FFN architecture type
    pub ffn: FfnType,

    /// Hidden size
    pub hidden_size: usize,

    /// Number of attention heads
    pub n_heads: usize,

    /// Number of KV heads (for GQA/MQA)
    pub n_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Number of layers
    pub n_layers: usize,

    /// Intermediate size for FFN
    pub intermediate_size: usize,

    /// KV cache dtype
    pub kv_cache_dtype: crate::model::tensor::DataType,
}

impl Architecture {
    /// Detect architecture from a LlamaModel
    pub fn detect(model: &LlamaModel) -> Self {
        let attention = Self::detect_attention_type(model);
        let ffn = Self::detect_ffn_type(model);

        Self {
            attention,
            ffn,
            hidden_size: model.config.hidden_size,
            n_heads: model.config.n_heads,
            n_kv_heads: model.config.n_kv_heads,
            head_dim: model.config.head_dim,
            n_layers: model.config.n_layers,
            intermediate_size: model.config.intermediate_size,
            kv_cache_dtype: crate::model::tensor::DataType::F16,
        }
    }

    /// Detect attention architecture type
    fn detect_attention_type(model: &LlamaModel) -> AttentionType {
        // Check first layer for attention pattern
        if let Some(layer) = model.layers.first() {
            // MLA: has attn_kv_a_mqa
            if layer.attn_kv_a_mqa.is_some() {
                return AttentionType::Mla;
            }

            // Fused QKV: check if there's a fused tensor
            if layer.attn_qkv.is_some() {
                return AttentionType::FusedQkv;
            }
        }

        // Check if uses MQA (different n_heads vs n_kv_heads)
        let n_heads = model.config.n_heads;
        let n_kv_heads = model.config.n_kv_heads;

        if n_kv_heads < n_heads {
            return AttentionType::Mqa;
        }

        AttentionType::Standard
    }

    /// Detect FFN architecture type
    fn detect_ffn_type(model: &LlamaModel) -> FfnType {
        // Check first layer for FFN pattern
        if let Some(layer) = model.layers.first() {
            // MoE: has ffn_gate_exps (expert weights)
            if layer.ffn_gate_exps.is_some() {
                let num_experts = model.config.n_experts;
                let num_experts_per_tok = model.config.n_experts_used;
                return FfnType::MoE { num_experts, num_experts_per_tok };
            }
        }

        // Default to SwiGLU (gate_proj, up_proj, down_proj)
        FfnType::SwiGLU
    }

    /// Check if GPU supports this architecture
    pub fn is_gpu_supported(&self) -> bool {
        match self.attention {
            AttentionType::Standard => true,
            AttentionType::Mqa => false, // CPU fallback for now
            AttentionType::Mla => false, // CPU fallback for now
            AttentionType::FusedQkv => true,
        }
    }

    /// Get GPU support status description
    pub fn gpu_support_status(&self) -> &'static str {
        match self.attention {
            AttentionType::Standard => "Fully supported",
            AttentionType::Mqa => "CPU fallback (MQA not yet on GPU)",
            AttentionType::Mla => "CPU fallback (MLA not yet on GPU)",
            AttentionType::FusedQkv => "Fully supported",
        }
    }
}
