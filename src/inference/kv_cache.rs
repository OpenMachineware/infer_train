// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// KV Cache: multi-format key-value storage for attention

#![allow(dead_code)]

use crate::model::config::ModelConfig;

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

use crate::quant::kv_cache::{
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, QK,
    dequantize_row_f16, dequantize_row_q4_0, dequantize_row_q4_1,
    dequantize_row_q5_0, dequantize_row_q5_1, dequantize_row_q8_0,
    quantize_row_f16, quantize_row_q4_0, quantize_row_q4_1, quantize_row_q5_0,
    quantize_row_q5_1, quantize_row_q8_0,
};

/// FP32 KV Cache (simplest implementation)
pub struct KVCacheF32 {
    k_cache: Vec<Vec<Vec<Vec<f32>>>>, // [n_layers, n_kv_heads, max_seq, k_head_dim]
    v_cache: Vec<Vec<Vec<Vec<f32>>>>, // [n_layers, n_kv_heads, max_seq, v_head_dim]
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    k_head_dim: usize, // Key dimension per head (key_length for MLA)
    v_head_dim: usize, // Value dimension per head (value_length for MLA)
    max_seq: usize,
}

/// F16 KV Cache
pub struct KVCacheF16 {
    k_cache: Vec<Vec<Vec<Vec<u16>>>>, // [n_layers, n_kv_heads, max_seq, head_dim]
    v_cache: Vec<Vec<Vec<Vec<u16>>>>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
}

/// Q8_0 KV Cache
pub struct KVCacheQ8_0 {
    k_cache: Vec<Vec<Vec<Vec<BlockQ8_0>>>>, // [n_layers, n_kv_heads, max_seq, n_blocks]
    v_cache: Vec<Vec<Vec<Vec<BlockQ8_0>>>>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    n_blocks_per_pos: usize, // head_dim / 32
}

/// Q4_0 KV Cache
pub struct KVCacheQ4_0 {
    k_cache: Vec<Vec<Vec<Vec<BlockQ4_0>>>>,
    v_cache: Vec<Vec<Vec<Vec<BlockQ4_0>>>>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    n_blocks_per_pos: usize,
}

/// Q4_1 KV Cache
pub struct KVCacheQ4_1 {
    k_cache: Vec<Vec<Vec<Vec<BlockQ4_1>>>>,
    v_cache: Vec<Vec<Vec<Vec<BlockQ4_1>>>>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    n_blocks_per_pos: usize,
}

/// Q5_0 KV Cache
pub struct KVCacheQ5_0 {
    k_cache: Vec<Vec<Vec<Vec<BlockQ5_0>>>>,
    v_cache: Vec<Vec<Vec<Vec<BlockQ5_0>>>>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    n_blocks_per_pos: usize,
}

/// Q5_1 KV Cache
pub struct KVCacheQ5_1 {
    k_cache: Vec<Vec<Vec<Vec<BlockQ5_1>>>>,
    v_cache: Vec<Vec<Vec<Vec<BlockQ5_1>>>>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    n_blocks_per_pos: usize,
}

impl KVCacheF32 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        // For MLA (Multi-head Latent Attention), K and V have different dimensions
        // K uses key_length, V uses value_length
        // For standard attention, both use head_dim
        let k_head_dim = if config.key_length > 0 {
            config.key_length // MLA uses key_length
        } else {
            config.head_dim
        };
        let v_head_dim = if config.value_length > 0 {
            config.value_length // MLA uses value_length
        } else {
            config.head_dim
        };

        KVCacheF32 {
            k_cache: vec![
                vec![
                    vec![vec![0.0; k_head_dim]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![0.0; v_head_dim]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            k_head_dim: k_head_dim,
            v_head_dim: v_head_dim,
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
        // For MLA, head_dim = key_length
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            // Resize cache entry if needed
            if self.k_cache[layer][head][self.seq_len].len() != head_dim {
                self.k_cache[layer][head][self.seq_len] = vec![0.0; head_dim];
            }
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
        // For MLA, head_dim = value_length
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            // Resize cache entry if needed
            if self.v_cache[layer][head][self.seq_len].len() != head_dim {
                self.v_cache[layer][head][self.seq_len] = vec![0.0; head_dim];
            }
            self.v_cache[layer][head][self.seq_len]
                .copy_from_slice(&v[src_start..src_end]);
        }
    }
}

// ============================================================================
// F16 KV Cache
// ============================================================================

impl KVCacheF16 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let head_dim = config.head_dim;

        KVCacheF16 {
            k_cache: vec![
                vec![
                    vec![vec![0u16; head_dim]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![0u16; head_dim]; max_seq];
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

impl KVCacheBase for KVCacheF16 {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                for pos in 0..self.max_seq {
                    self.k_cache[layer][head][pos].fill(0);
                    self.v_cache[layer][head][pos].fill(0);
                }
            }
        }
    }

    fn format(&self) -> KVCacheFormat {
        KVCacheFormat::F16
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_f16(&self.k_cache[layer][head][pos], &mut out);
        out
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_f16(&self.v_cache[layer][head][pos], &mut out);
        out
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_f16(
                &k[src_start..src_end],
                &mut self.k_cache[layer][head][self.seq_len],
            );
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_f16(
                &v[src_start..src_end],
                &mut self.v_cache[layer][head][self.seq_len],
            );
        }
    }
}

// ============================================================================
// Q8_0 KV Cache
// ============================================================================

impl KVCacheQ8_0 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let head_dim = config.head_dim;
        let n_blocks_per_pos = head_dim.div_ceil(QK);

        // Initialize with zero blocks
        let zero_block = BlockQ8_0 { d: 0, qs: [0; QK] };

        KVCacheQ8_0 {
            k_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            max_seq,
            n_blocks_per_pos,
        }
    }
}

impl KVCacheBase for KVCacheQ8_0 {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        let zero_block = BlockQ8_0 { d: 0, qs: [0; QK] };
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                for pos in 0..self.max_seq {
                    for b in 0..self.n_blocks_per_pos {
                        self.k_cache[layer][head][pos][b] = zero_block;
                        self.v_cache[layer][head][pos][b] = zero_block;
                    }
                }
            }
        }
    }

    fn format(&self) -> KVCacheFormat {
        KVCacheFormat::Q8_0
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        let blocks = &self.k_cache[layer][head][pos];
        dequantize_row_q8_0(blocks, &mut out);
        out
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        let blocks = &self.v_cache[layer][head][pos];
        dequantize_row_q8_0(blocks, &mut out);
        out
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q8_0(
                &k[src_start..src_end],
                &mut self.k_cache[layer][head][self.seq_len],
            );
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q8_0(
                &v[src_start..src_end],
                &mut self.v_cache[layer][head][self.seq_len],
            );
        }
    }
}

// ============================================================================
// Q4_0 KV Cache
// ============================================================================

impl KVCacheQ4_0 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let head_dim = config.head_dim;
        let n_blocks_per_pos = head_dim.div_ceil(QK);
        let zero_block = BlockQ4_0 { d: 0, qs: [0; QK / 2] };

        KVCacheQ4_0 {
            k_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            max_seq,
            n_blocks_per_pos,
        }
    }
}

impl KVCacheBase for KVCacheQ4_0 {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        let zero_block = BlockQ4_0 { d: 0, qs: [0; QK / 2] };
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                for pos in 0..self.max_seq {
                    for b in 0..self.n_blocks_per_pos {
                        self.k_cache[layer][head][pos][b] = zero_block;
                        self.v_cache[layer][head][pos][b] = zero_block;
                    }
                }
            }
        }
    }

    fn format(&self) -> KVCacheFormat {
        KVCacheFormat::Q4_0
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q4_0(&self.k_cache[layer][head][pos], &mut out);
        out
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q4_0(&self.v_cache[layer][head][pos], &mut out);
        out
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q4_0(
                &k[src_start..src_end],
                &mut self.k_cache[layer][head][self.seq_len],
            );
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q4_0(
                &v[src_start..src_end],
                &mut self.v_cache[layer][head][self.seq_len],
            );
        }
    }
}

// ============================================================================
// Q4_1 KV Cache
// ============================================================================

impl KVCacheQ4_1 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let head_dim = config.head_dim;
        let n_blocks_per_pos = head_dim.div_ceil(QK);
        let zero_block = BlockQ4_1 { d: 0, m: 0, qs: [0; QK / 2] };

        KVCacheQ4_1 {
            k_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            max_seq,
            n_blocks_per_pos,
        }
    }
}

impl KVCacheBase for KVCacheQ4_1 {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        let zero_block = BlockQ4_1 { d: 0, m: 0, qs: [0; QK / 2] };
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                for pos in 0..self.max_seq {
                    for b in 0..self.n_blocks_per_pos {
                        self.k_cache[layer][head][pos][b] = zero_block;
                        self.v_cache[layer][head][pos][b] = zero_block;
                    }
                }
            }
        }
    }

    fn format(&self) -> KVCacheFormat {
        KVCacheFormat::Q4_1
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q4_1(&self.k_cache[layer][head][pos], &mut out);
        out
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q4_1(&self.v_cache[layer][head][pos], &mut out);
        out
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q4_1(
                &k[src_start..src_end],
                &mut self.k_cache[layer][head][self.seq_len],
            );
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q4_1(
                &v[src_start..src_end],
                &mut self.v_cache[layer][head][self.seq_len],
            );
        }
    }
}

// ============================================================================
// Q5_0 KV Cache
// ============================================================================

impl KVCacheQ5_0 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let head_dim = config.head_dim;
        let n_blocks_per_pos = head_dim.div_ceil(QK);
        let zero_block = BlockQ5_0 { d: 0, qh: [0; 4], qs: [0; QK / 2] };

        KVCacheQ5_0 {
            k_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            max_seq,
            n_blocks_per_pos,
        }
    }
}

impl KVCacheBase for KVCacheQ5_0 {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        let zero_block = BlockQ5_0 { d: 0, qh: [0; 4], qs: [0; QK / 2] };
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                for pos in 0..self.max_seq {
                    for b in 0..self.n_blocks_per_pos {
                        self.k_cache[layer][head][pos][b] = zero_block;
                        self.v_cache[layer][head][pos][b] = zero_block;
                    }
                }
            }
        }
    }

    fn format(&self) -> KVCacheFormat {
        KVCacheFormat::Q5_0
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q5_0(&self.k_cache[layer][head][pos], &mut out);
        out
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q5_0(&self.v_cache[layer][head][pos], &mut out);
        out
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q5_0(
                &k[src_start..src_end],
                &mut self.k_cache[layer][head][self.seq_len],
            );
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q5_0(
                &v[src_start..src_end],
                &mut self.v_cache[layer][head][self.seq_len],
            );
        }
    }
}

// ============================================================================
// Q5_1 KV Cache
// ============================================================================

impl KVCacheQ5_1 {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let head_dim = config.head_dim;
        let n_blocks_per_pos = head_dim.div_ceil(QK);
        let zero_block = BlockQ5_1 { d: 0, m: 0, qh: [0; 4], qs: [0; QK / 2] };

        KVCacheQ5_1 {
            k_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            v_cache: vec![
                vec![
                    vec![vec![zero_block; n_blocks_per_pos]; max_seq];
                    config.n_kv_heads
                ];
                config.n_layers
            ],
            seq_len: 0,
            n_layers: config.n_layers,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            max_seq,
            n_blocks_per_pos,
        }
    }
}

impl KVCacheBase for KVCacheQ5_1 {
    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        let zero_block = BlockQ5_1 { d: 0, m: 0, qh: [0; 4], qs: [0; QK / 2] };
        for layer in 0..self.n_layers {
            for head in 0..self.n_kv_heads {
                for pos in 0..self.max_seq {
                    for b in 0..self.n_blocks_per_pos {
                        self.k_cache[layer][head][pos][b] = zero_block;
                        self.v_cache[layer][head][pos][b] = zero_block;
                    }
                }
            }
        }
    }

    fn format(&self) -> KVCacheFormat {
        KVCacheFormat::Q5_1
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q5_1(&self.k_cache[layer][head][pos], &mut out);
        out
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.head_dim];
        dequantize_row_q5_1(&self.v_cache[layer][head][pos], &mut out);
        out
    }

    fn set_k(
        &mut self,
        layer: usize,
        k: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q5_1(
                &k[src_start..src_end],
                &mut self.k_cache[layer][head][self.seq_len],
            );
        }
    }

    fn set_v(
        &mut self,
        layer: usize,
        v: &[f32],
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        for head in 0..n_kv_heads {
            let src_start = head * head_dim;
            let src_end = src_start + head_dim;
            quantize_row_q5_1(
                &v[src_start..src_end],
                &mut self.v_cache[layer][head][self.seq_len],
            );
        }
    }
}

/// Generic KV Cache (enum over different formats)
pub enum KVCache {
    F32(KVCacheF32),
    F16(KVCacheF16),
    Q8_0(KVCacheQ8_0),
    Q4_0(KVCacheQ4_0),
    Q4_1(KVCacheQ4_1),
    Q5_0(KVCacheQ5_0),
    Q5_1(KVCacheQ5_1),
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
            KVCacheFormat::F16 => {
                KVCache::F16(KVCacheF16::new(config, max_seq))
            }
            KVCacheFormat::Q8_0 => {
                KVCache::Q8_0(KVCacheQ8_0::new(config, max_seq))
            }
            KVCacheFormat::Q4_0 => {
                KVCache::Q4_0(KVCacheQ4_0::new(config, max_seq))
            }
            KVCacheFormat::Q4_1 => {
                KVCache::Q4_1(KVCacheQ4_1::new(config, max_seq))
            }
            KVCacheFormat::Q5_0 => {
                KVCache::Q5_0(KVCacheQ5_0::new(config, max_seq))
            }
            KVCacheFormat::Q5_1 => {
                KVCache::Q5_1(KVCacheQ5_1::new(config, max_seq))
            }
            KVCacheFormat::BF16 => {
                // BF16 not yet implemented, fall back to F16
                eprintln!(
                    "Warning: KV cache format BF16 not implemented, using F16"
                );
                KVCache::F16(KVCacheF16::new(config, max_seq))
            }
        }
    }

    pub fn advance(&mut self) {
        match self {
            KVCache::F32(cache) => cache.seq_len += 1,
            KVCache::F16(cache) => cache.seq_len += 1,
            KVCache::Q8_0(cache) => cache.seq_len += 1,
            KVCache::Q4_0(cache) => cache.seq_len += 1,
            KVCache::Q4_1(cache) => cache.seq_len += 1,
            KVCache::Q5_0(cache) => cache.seq_len += 1,
            KVCache::Q5_1(cache) => cache.seq_len += 1,
        }
    }
}

// Implement KVCacheBase for KVCache enum
impl KVCacheBase for KVCache {
    fn seq_len(&self) -> usize {
        match self {
            KVCache::F32(cache) => cache.seq_len(),
            KVCache::F16(cache) => cache.seq_len(),
            KVCache::Q8_0(cache) => cache.seq_len(),
            KVCache::Q4_0(cache) => cache.seq_len(),
            KVCache::Q4_1(cache) => cache.seq_len(),
            KVCache::Q5_0(cache) => cache.seq_len(),
            KVCache::Q5_1(cache) => cache.seq_len(),
        }
    }

    fn clear(&mut self) {
        match self {
            KVCache::F32(cache) => cache.clear(),
            KVCache::F16(cache) => cache.clear(),
            KVCache::Q8_0(cache) => cache.clear(),
            KVCache::Q4_0(cache) => cache.clear(),
            KVCache::Q4_1(cache) => cache.clear(),
            KVCache::Q5_0(cache) => cache.clear(),
            KVCache::Q5_1(cache) => cache.clear(),
        }
    }

    fn format(&self) -> KVCacheFormat {
        match self {
            KVCache::F32(cache) => cache.format(),
            KVCache::F16(cache) => cache.format(),
            KVCache::Q8_0(cache) => cache.format(),
            KVCache::Q4_0(cache) => cache.format(),
            KVCache::Q4_1(cache) => cache.format(),
            KVCache::Q5_0(cache) => cache.format(),
            KVCache::Q5_1(cache) => cache.format(),
        }
    }

    fn get_k(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        match self {
            KVCache::F32(cache) => cache.get_k(layer, head, pos),
            KVCache::F16(cache) => cache.get_k(layer, head, pos),
            KVCache::Q8_0(cache) => cache.get_k(layer, head, pos),
            KVCache::Q4_0(cache) => cache.get_k(layer, head, pos),
            KVCache::Q4_1(cache) => cache.get_k(layer, head, pos),
            KVCache::Q5_0(cache) => cache.get_k(layer, head, pos),
            KVCache::Q5_1(cache) => cache.get_k(layer, head, pos),
        }
    }

    fn get_v(&self, layer: usize, head: usize, pos: usize) -> Vec<f32> {
        match self {
            KVCache::F32(cache) => cache.get_v(layer, head, pos),
            KVCache::F16(cache) => cache.get_v(layer, head, pos),
            KVCache::Q8_0(cache) => cache.get_v(layer, head, pos),
            KVCache::Q4_0(cache) => cache.get_v(layer, head, pos),
            KVCache::Q4_1(cache) => cache.get_v(layer, head, pos),
            KVCache::Q5_0(cache) => cache.get_v(layer, head, pos),
            KVCache::Q5_1(cache) => cache.get_v(layer, head, pos),
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
            KVCache::F16(cache) => cache.set_k(layer, k, n_kv_heads, head_dim),
            KVCache::Q8_0(cache) => cache.set_k(layer, k, n_kv_heads, head_dim),
            KVCache::Q4_0(cache) => cache.set_k(layer, k, n_kv_heads, head_dim),
            KVCache::Q4_1(cache) => cache.set_k(layer, k, n_kv_heads, head_dim),
            KVCache::Q5_0(cache) => cache.set_k(layer, k, n_kv_heads, head_dim),
            KVCache::Q5_1(cache) => cache.set_k(layer, k, n_kv_heads, head_dim),
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
            KVCache::F16(cache) => cache.set_v(layer, v, n_kv_heads, head_dim),
            KVCache::Q8_0(cache) => cache.set_v(layer, v, n_kv_heads, head_dim),
            KVCache::Q4_0(cache) => cache.set_v(layer, v, n_kv_heads, head_dim),
            KVCache::Q4_1(cache) => cache.set_v(layer, v, n_kv_heads, head_dim),
            KVCache::Q5_0(cache) => cache.set_v(layer, v, n_kv_heads, head_dim),
            KVCache::Q5_1(cache) => cache.set_v(layer, v, n_kv_heads, head_dim),
        }
    }
}
