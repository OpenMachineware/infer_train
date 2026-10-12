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
    pub vocab_size: usize,

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

        ModelConfig {
            architecture,
            hidden_size,
            intermediate_size,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            rope_dim,
            rope_freq_base,
            rms_norm_eps,
            max_position_embeddings,
            bos_token_id,
            eos_token_id,
            metadata: meta.clone(),
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.n_heads
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
