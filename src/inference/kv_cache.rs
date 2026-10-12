// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// KV Cache: multi-format key-value storage for attention
// Reference: llama.cpp llama_kv_cache

use crate::model::config::ModelConfig;
use crate::model::tensor::DataType;

/// KV Cache format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheFormat {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
}

impl KVCacheFormat {
    pub fn bytes_per_elem(&self) -> usize {
        match self {
            KVCacheFormat::F32 => 4,
            KVCacheFormat::F16 => 2,
            KVCacheFormat::BF16 => 2,
            KVCacheFormat::Q8_0 => 34, // 32 bytes + 2 byte scale
            KVCacheFormat::Q4_0 => 18, // 16 bytes + 2 byte scale
            KVCacheFormat::Q4_1 => 20, // 16 bytes + 4 bytes scale+min
            KVCacheFormat::Q5_0 => 22, // 16 bytes + 4 bytes qh + 2 bytes scale
            KVCacheFormat::Q5_1 => 24, // 16 bytes + 4 bytes qh + 4 bytes scale+min
        }
    }
}

/// KV Cache trait: unified interface for different formats
pub trait KVCacheBase {
    /// Get current sequence length
    fn seq_len(&self) -> usize;

    /// Clear cache
    fn clear(&mut self);

    /// Get storage format
    fn format(&self) -> KVCacheFormat;

    /// Get K cache for a specific layer and head (FP32 for compatibility)
    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32>;

    /// Get V cache for a specific layer and head (FP32 for compatibility)
    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32>;

    /// Set K cache for a specific layer (quantizes internally)
    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    );

    /// Set V cache for a specific layer (quantizes internally)
    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    );
}

/// FP32 KV Cache (simplest implementation)
pub struct KVCacheF32 {
    k_cache: Vec<Vec<Vec<Vec<f32>>>>, // [n_layers, n_kv_heads, max_seq, head_dim]
    v_cache: Vec<Vec<Vec<Vec<f32>>>>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
}

impl KVCacheF32 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let head_dim = config.hidden_size / config.n_heads;

        KVCacheF32 {
            k_cache: vec![
                vec![
                    vec![vec![0.0; head_dim]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![0.0; head_dim]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            max_seq,
        }
    }
}

impl KVCacheBase for KVCacheF32 {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                for pos in 0..self.max_seq {
                    self.k_cache[layer][head][pos].fill(0.0);
                    self.v_cache[layer][head][pos].fill(0.0);
                }
            }
        }
    }

    fn format(&self) -> KVCacheFormat {
        KVCacheFormat::F32
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        self.k_cache[layer][head][pos].clone()
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        self.v_cache[layer][head][pos].clone()
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        // K is [n_kv_heads * head_dim], store at position seq_len
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            self.k_cache[layer][head][self.seq_len]
                .copy_from_slice(&k[src_start..src_end]);
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        // V is [n_kv_heads * head_dim], store at position seq_len
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            self.v_cache[layer][head][self.seq_len]
                .copy_from_slice(&v[src_start..src_end]);
        }
    }
}

/// Generic KV Cache (enum over different formats)
pub enum KVCache {
    F32(KVCacheF32),
    // TODO: add Q8_0, Q4_0, F16 variants
}

impl KVCache {
    pub fn new(
        config: &ModelConfig,
        max_seq: usize,
        format: KVCacheFormat,
    ) -> Self {
        match format {
            KVCacheFormat::F32 => {
                KVCache::F32(KVCacheF32::new(config, max_seq))
            }
            _ => {
                // Default to F32 for unsupported formats
                eprintln!(
                    "Warning: KV cache format {:?} not implemented, using F32",
                    format
                );
                KVCache::F32(KVCacheF32::new(config, max_seq))
            }
        }
    }

    pub fn advance(&mut self) {
        match self {
            KVCache::F32(cache) => cache.seq_len += 1,
        }
    }
}

// Implement KVCacheBase for KVCache enum
impl KVCacheBase for KVCache {
    fn seq_len(&self) -> usize {
        match self {
            KVCache::F32(cache) => cache.seq_len(),
        }
    }

    fn clear(&mut self) {
        match self {
            KVCache::F32(cache) => cache.clear(),
        }
    }

    fn format(&self) -> KVCacheFormat {
        match self {
            KVCache::F32(cache) => cache.format(),
        }
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        match self {
            KVCache::F32(cache) => cache.get_k(layer, head, pos),
        }
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        match self {
            KVCache::F32(cache) => cache.get_v(layer, head, pos),
        }
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        match self {
            KVCache::F32(cache) => cache.set_k(layer, k, n_kv_heads, head_dim),
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        match self {
            KVCache::F32(cache) => cache.set_v(layer, v, n_kv_heads, head_dim),
        }
    }
}
