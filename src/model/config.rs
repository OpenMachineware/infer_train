// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Model configuration from GGUF metadata

use crate::model::gguf::{GgufFile, MetadataValue};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    // Architecture
    pub architecture: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize, // Explicit head_dim (derived from weights, not hidden_size/n_heads)
    pub vocab_size: usize,

    // MoE (Mixture of Experts)
    pub n_experts: usize,        // Total number of experts
    pub n_experts_used: usize,   // Number of experts used per token (top-k)
    pub n_experts_shared: usize, // Number of shared experts
    pub expert_intermediate_size: usize, // Intermediate size for each expert

    // DeepSeek MQA (Multi-Query Attention)
    pub kv_lora_rank: usize, // KV LoRA rank (compressed KV dimension)
    pub qk_rope_head_dim: usize, // QK RoPE dimension per head (stored in kv_a_mqa)
    pub key_length: usize, // Key length after decompression (total qk_head_dim)
    pub value_length: usize, // Value length after decompression

    // SSM/Mamba (State Space Model)
    pub ssm_conv_kernel: usize, // Conv1D kernel size (usually 4)
    pub ssm_state_size: usize,  // SSM state dimension (usually 128)
    pub ssm_group_count: usize, // Number of SSM groups (usually 32)
    pub ssm_time_step_rank: usize, // Time step rank (usually 32)
    pub ssm_inner_size: usize,  // SSM inner dimension

    // RoPE
    pub rope_dim: usize,
    pub rope_freq_base: f32,

    // Normalization
    pub rms_norm_eps: f32,

    // Context
    pub max_position_embeddings: usize,

    // Tokenizer
    pub bos_token_id: u32,
    pub eos_token_id: u32,

    // Raw metadata for extensions
    pub metadata: HashMap<String, MetadataValue>,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let meta = &gguf.metadata;

        let architecture =
            get_string(meta, "general.architecture").unwrap_or_default();

        // Architecture-specific prefix (e.g., "qwen3." for keys like "qwen3.block_count")
        let arch_prefix = format!("{}.", architecture);
        let arch_meta: HashMap<String, MetadataValue> = meta
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(&arch_prefix)
                    .map(|stripped| (stripped.to_string(), v.clone()))
            })
            .collect();

        let hidden_size = get_u64(meta, "general.hidden_size")
            .or_else(|| get_u64(&arch_meta, "embedding_length"))
            .unwrap_or(4096) as usize;

        let intermediate_size = get_u64(meta, "general.intermediate_size")
            .or_else(|| get_u64(&arch_meta, "feed_forward_length"))
            .unwrap_or(11008) as usize;

        let n_layers = get_u64(meta, "general.block_count")
            .or_else(|| get_u64(&arch_meta, "block_count"))
            .unwrap_or(32) as usize;

        let n_heads = get_u64(meta, "general.attention.head_count")
            .or_else(|| get_u64(&arch_meta, "attention.head_count"))
            .unwrap_or(32) as usize;

        let n_kv_heads = get_u64(meta, "general.attention.head_count_kv")
            .or_else(|| get_u64(&arch_meta, "attention.head_count_kv"))
            .unwrap_or(n_heads as u64) as usize;

        let vocab_size = get_u64(meta, "general.vocabulary_size")
            .or_else(|| get_u64(&arch_meta, "vocabulary_size"))
            .unwrap_or(32000) as usize;

        let rope_dim = get_u64(meta, "general.rope.dimension_count")
            .or_else(|| get_u64(&arch_meta, "rope.dimension_count"))
            .unwrap_or(128) as usize;

        let rope_freq_base = get_f32(meta, "general.rope.freq_base")
            .or_else(|| get_f32(&arch_meta, "rope.freq_base"))
            .or_else(|| get_f32(&arch_meta, "rope_dimension_count"))
            .unwrap_or(10000.0);

        let rms_norm_eps = get_f32(meta, "general.layer_norm_rms_epsilon")
            .or_else(|| get_f32(&arch_meta, "layer_norm_rms_epsilon"))
            .or_else(|| get_f32(&arch_meta, "attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let max_position_embeddings = get_u64(meta, "general.context_length")
            .or_else(|| get_u64(&arch_meta, "context_length"))
            .unwrap_or(2048) as usize;

        let bos_token_id =
            get_u64(meta, "general.tokenizer.bos_token_id").unwrap_or(1) as u32;

        let eos_token_id =
            get_u64(meta, "general.tokenizer.eos_token_id").unwrap_or(2) as u32;

        // Attention head dimensions
        // For models with explicit key_length (like Qwen3.5/Qwen3.8), use it as head_dim
        // For MLA (kv_lora_rank > 0), use key_length/value_length from metadata
        // For standard attention, also try to get key_length from metadata (some models have it)
        // If not available, will be inferred from weight shapes later
        let key_length =
            get_u64(&arch_meta, "attention.key_length").unwrap_or(0) as usize;
        let value_length =
            get_u64(&arch_meta, "attention.value_length").unwrap_or(0) as usize;

        // Initial head_dim: use key_length if available, otherwise derive from hidden_size/n_heads
        let head_dim =
            if key_length > 0 { key_length } else { hidden_size / n_heads };

        // MoE parameters (DeepSeek, Qwen-MoE, etc.)
        let n_experts =
            get_u64(&arch_meta, "expert_count").unwrap_or(0) as usize;
        let n_experts_used =
            get_u64(&arch_meta, "expert_used_count").unwrap_or(0) as usize;
        let n_experts_shared =
            get_u64(&arch_meta, "expert_shared_count").unwrap_or(0) as usize;
        let expert_intermediate_size =
            get_u64(&arch_meta, "expert_feed_forward_length")
                .unwrap_or(intermediate_size as u64) as usize;

        // DeepSeek MLA parameters
        let kv_lora_rank =
            get_u64(&arch_meta, "attention.kv_lora_rank").unwrap_or(0) as usize;

        // qk_rope_head_dim is derived from rope_dim for DeepSeek models
        // For MLA, kv_a_mqa output = kv_lora_rank + qk_rope_head_dim
        let qk_rope_head_dim = if kv_lora_rank > 0 {
            rope_dim // DeepSeek uses rope_dim for the RoPE part of QK
        } else {
            0
        };

        // SSM/Mamba parameters (Qwen3.5-MoE, Qwen3.8, etc.)
        let ssm_conv_kernel =
            get_u64(&arch_meta, "ssm.conv_kernel").unwrap_or(4) as usize;
        let ssm_state_size =
            get_u64(&arch_meta, "ssm.state_size").unwrap_or(128) as usize;
        let ssm_group_count =
            get_u64(&arch_meta, "ssm.group_count").unwrap_or(32) as usize;
        let ssm_time_step_rank =
            get_u64(&arch_meta, "ssm.time_step_rank").unwrap_or(32) as usize;
        let ssm_inner_size = get_u64(&arch_meta, "ssm.inner_size")
            .unwrap_or(hidden_size as u64)
            as usize;

        ModelConfig {
            architecture,
            hidden_size,
            intermediate_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            n_experts,
            n_experts_used,
            n_experts_shared,
            expert_intermediate_size,
            kv_lora_rank,
            qk_rope_head_dim,
            key_length,
            value_length,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
            rope_dim,
            rope_freq_base,
            rms_norm_eps,
            max_position_embeddings,
            bos_token_id,
            eos_token_id,
            metadata: meta.clone(),
        }
    }

    pub fn n_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

fn get_string(
    meta: &HashMap<String, MetadataValue>,
    key: &str,
) -> Option<String> {
    match meta.get(key) {
        Some(MetadataValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn get_u64(meta: &HashMap<String, MetadataValue>, key: &str) -> Option<u64> {
    match meta.get(key) {
        Some(MetadataValue::Uint64(v)) => Some(*v),
        Some(MetadataValue::Int64(v)) => Some(*v as u64),
        Some(MetadataValue::Uint32(v)) => Some(*v as u64),
        Some(MetadataValue::Int32(v)) => Some(*v as u64),
        _ => None,
    }
}

fn get_f32(meta: &HashMap<String, MetadataValue>, key: &str) -> Option<f32> {
    match meta.get(key) {
        Some(MetadataValue::Float32(v)) => Some(*v),
        Some(MetadataValue::Float64(v)) => Some(*v as f32),
        _ => None,
    }
}
