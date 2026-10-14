// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GPU KV Cache: Metal buffers for zero-copy flash attention

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{Buffer, MTLResourceOptions};
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::model::config::ModelConfig;

/// GPU KV Cache for F16 format
/// Layout: [n_kv_heads, max_seq, head_dim] in F16
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct KVCacheGPU {
    /// K cache buffers per layer: [n_layers][n_kv_heads * max_seq * head_dim * 2 bytes]
    k_buffers: Vec<Rc<Buffer>>,
    /// V cache buffers per layer
    v_buffers: Vec<Rc<Buffer>>,
    /// Current sequence length
    seq_len: usize,
    /// Number of layers
    n_layers: usize,
    /// Number of KV heads
    n_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Maximum sequence length
    max_seq: usize,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl KVCacheGPU {
    /// Create a new GPU KV cache
    pub fn new(
        device: &metal::Device,
        config: &ModelConfig,
        max_seq: usize,
    ) -> Self {
        let n_layers = config.n_layers;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;

        // Each layer needs: n_kv_heads * max_seq * head_dim * 2 bytes (F16)
        let bytes_per_layer = n_kv_heads * max_seq * head_dim * 2;

        let k_buffers: Vec<Rc<Buffer>> = (0..n_layers)
            .map(|_| {
                Rc::new(device.new_buffer(
                    bytes_per_layer as u64,
                    MTLResourceOptions::StorageModeShared,
                ))
            })
            .collect();

        let v_buffers: Vec<Rc<Buffer>> = (0..n_layers)
            .map(|_| {
                Rc::new(device.new_buffer(
                    bytes_per_layer as u64,
                    MTLResourceOptions::StorageModeShared,
                ))
            })
            .collect();

        Self {
            k_buffers,
            v_buffers,
            seq_len: 0,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq,
        }
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Advance sequence length by 1
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.seq_len = 0;
        // Zero out all buffers
        for buffer in &self.k_buffers {
            unsafe {
                std::ptr::write_bytes(
                    buffer.contents() as *mut u8,
                    0,
                    buffer.length() as usize,
                );
            }
        }
        for buffer in &self.v_buffers {
            unsafe {
                std::ptr::write_bytes(
                    buffer.contents() as *mut u8,
                    0,
                    buffer.length() as usize,
                );
            }
        }
    }

    /// Get K buffer for a layer
    pub fn k_buffer(&self, layer: usize) -> &Rc<Buffer> {
        &self.k_buffers[layer]
    }

    /// Get V buffer for a layer
    pub fn v_buffer(&self, layer: usize) -> &Rc<Buffer> {
        &self.v_buffers[layer]
    }

    /// Set K values at current position for a layer
    /// k: [n_kv_heads * head_dim] in F32
    /// This converts to F16 and stores in GPU buffer
    pub fn set_k(&mut self, layer: usize, k: &[f32]) {
        let pos = self.seq_len;
        if pos >= self.max_seq {
            return;
        }

        let buffer = &self.k_buffers[layer];
        let contents = buffer.contents() as *mut u8;

        // Layout: [n_kv_heads, max_seq, head_dim]
        // Position for head h at position p: h * max_seq * head_dim + p * head_dim
        for h in 0..self.n_kv_heads {
            let src_offset = h * self.head_dim;
            let dst_offset =
                h * self.max_seq * self.head_dim * 2 + pos * self.head_dim * 2;

            // Convert F32 to F16
            let src = &k[src_offset..src_offset + self.head_dim];
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    contents.add(dst_offset) as *mut u16,
                    self.head_dim,
                )
            };

            // F32 to F16 conversion (simple, no special values)
            for (i, &v) in src.iter().enumerate() {
                dst[i] = half::f16::from_f32(v).to_bits();
            }
        }
    }

    /// Set V values at current position for a layer
    pub fn set_v(&mut self, layer: usize, v: &[f32]) {
        let pos = self.seq_len;
        if pos >= self.max_seq {
            return;
        }

        let buffer = &self.v_buffers[layer];
        let contents = buffer.contents() as *mut u8;

        for h in 0..self.n_kv_heads {
            let src_offset = h * self.head_dim;
            let dst_offset =
                h * self.max_seq * self.head_dim * 2 + pos * self.head_dim * 2;

            let src = &v[src_offset..src_offset + self.head_dim];
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    contents.add(dst_offset) as *mut u16,
                    self.head_dim,
                )
            };

            for (i, &v) in src.iter().enumerate() {
                dst[i] = half::f16::from_f32(v).to_bits();
            }
        }
    }

    /// Get K values from a layer (for debugging)
    pub fn get_k(&self, layer: usize, pos: usize) -> Vec<f32> {
        let buffer = &self.k_buffers[layer];
        let contents = buffer.contents() as *const u8;
        let mut result = vec![0.0f32; self.n_kv_heads * self.head_dim];

        for h in 0..self.n_kv_heads {
            let offset =
                h * self.max_seq * self.head_dim * 2 + pos * self.head_dim * 2;
            let src = unsafe {
                std::slice::from_raw_parts(
                    contents.add(offset) as *const u16,
                    self.head_dim,
                )
            };

            for (i, &v) in src.iter().enumerate() {
                result[h * self.head_dim + i] =
                    half::f16::from_bits(v).to_f32();
            }
        }

        result
    }

    /// Get V values from a layer (for debugging)
    pub fn get_v(&self, layer: usize, pos: usize) -> Vec<f32> {
        let buffer = &self.v_buffers[layer];
        let contents = buffer.contents() as *const u8;
        let mut result = vec![0.0f32; self.n_kv_heads * self.head_dim];

        for h in 0..self.n_kv_heads {
            let offset =
                h * self.max_seq * self.head_dim * 2 + pos * self.head_dim * 2;
            let src = unsafe {
                std::slice::from_raw_parts(
                    contents.add(offset) as *const u16,
                    self.head_dim,
                )
            };

            for (i, &v) in src.iter().enumerate() {
                result[h * self.head_dim + i] =
                    half::f16::from_bits(v).to_f32();
            }
        }

        result
    }

    /// Get n_kv_heads
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Get head_dim
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get max_seq
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
}
