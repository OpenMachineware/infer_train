// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GPU KV Cache for Flash Attention

use metal::{Buffer, Device};
use std::cell::RefCell;

use crate::model::config::ModelConfig;
use crate::quant::kv_cache::{
    quantize_row_f16, quantize_row_q4_0, quantize_row_q4_1, quantize_row_q5_0,
    quantize_row_q5_1, quantize_row_q8_0, BlockQ4_0, BlockQ4_1, BlockQ5_0,
    BlockQ5_1, BlockQ8_0, QK,
};

/// GPU KV Cache format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKVFormat {
    F16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
}

impl GpuKVFormat {
    pub fn bytes_per_elem(&self) -> usize {
        match self {
            GpuKVFormat::F16 => 2,
            GpuKVFormat::Q8_0 => 34,
            GpuKVFormat::Q4_0 => 18,
            GpuKVFormat::Q4_1 => 20,
            GpuKVFormat::Q5_0 => 22,
            GpuKVFormat::Q5_1 => 24,
        }
    }

    pub fn blocks_per_head(&self, head_dim: usize) -> usize {
        match self {
            GpuKVFormat::F16 => 0, // Not block-based
            _ => head_dim.div_ceil(QK),
        }
    }
}

/// GPU KV Cache for a single layer
pub struct GpuKVCacheLayer {
    k_buffer: Option<Buffer>,
    v_buffer: Option<Buffer>,
    capacity: usize, // max seq_len
    n_kv_heads: usize,
    head_dim: usize,
    format: GpuKVFormat,
    bytes_per_row: usize, // bytes per (seq_len, n_kv_heads) row
}

impl GpuKVCacheLayer {
    pub fn new(
        device: &Device,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        format: GpuKVFormat,
    ) -> Self {
        let bytes_per_row =
            Self::calc_bytes_per_row(n_kv_heads, head_dim, format);

        // Pre-allocate buffer
        let total_bytes = bytes_per_row * max_seq;
        let k_buffer = if total_bytes > 0 {
            Some(device.new_buffer(
                total_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ))
        } else {
            None
        };
        let v_buffer = if total_bytes > 0 {
            Some(device.new_buffer(
                total_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ))
        } else {
            None
        };

        GpuKVCacheLayer {
            k_buffer,
            v_buffer,
            capacity: max_seq,
            n_kv_heads,
            head_dim,
            format,
            bytes_per_row,
        }
    }

    fn calc_bytes_per_row(
        n_kv_heads: usize,
        head_dim: usize,
        format: GpuKVFormat,
    ) -> usize {
        match format {
            GpuKVFormat::F16 => n_kv_heads * head_dim * 2,
            GpuKVFormat::Q8_0 => {
                let blocks_per_head = head_dim.div_ceil(QK);
                n_kv_heads * blocks_per_head * std::mem::size_of::<BlockQ8_0>()
            }
            GpuKVFormat::Q4_0 => {
                let blocks_per_head = head_dim.div_ceil(QK);
                n_kv_heads * blocks_per_head * std::mem::size_of::<BlockQ4_0>()
            }
            GpuKVFormat::Q4_1 => {
                let blocks_per_head = head_dim.div_ceil(QK);
                n_kv_heads * blocks_per_head * std::mem::size_of::<BlockQ4_1>()
            }
            GpuKVFormat::Q5_0 => {
                let blocks_per_head = head_dim.div_ceil(QK);
                n_kv_heads * blocks_per_head * std::mem::size_of::<BlockQ5_0>()
            }
            GpuKVFormat::Q5_1 => {
                let blocks_per_head = head_dim.div_ceil(QK);
                n_kv_heads * blocks_per_head * std::mem::size_of::<BlockQ5_1>()
            }
        }
    }

    /// Store KV at a specific position
    /// k: [n_kv_heads * head_dim] FP32 values
    /// v: [n_kv_heads * head_dim] FP32 values
    pub fn set(&self, pos: usize, k: &[f32], v: &[f32]) {
        if pos >= self.capacity {
            return;
        }

        let offset = pos * self.bytes_per_row;

        // Quantize and store K
        if let Some(ref buffer) = self.k_buffer {
            unsafe {
                let ptr = (buffer.contents() as *mut u8).add(offset);
                self.quantize_to_buffer(k, ptr);
            }
        }

        // Quantize and store V
        if let Some(ref buffer) = self.v_buffer {
            unsafe {
                let ptr = (buffer.contents() as *mut u8).add(offset);
                self.quantize_to_buffer(v, ptr);
            }
        }
    }

    fn quantize_to_buffer(&self, src: &[f32], dst: *mut u8) {
        let head_dim = self.head_dim;
        let n_heads = self.n_kv_heads;

        unsafe {
            match self.format {
                GpuKVFormat::F16 => {
                    let ptr = dst as *mut u16;
                    for i in 0..src.len() {
                        *ptr.add(i) = half::f16::from_f32(src[i]).to_bits();
                    }
                }
                GpuKVFormat::Q8_0 => {
                    let blocks_per_head = head_dim.div_ceil(QK);
                    for head in 0..n_heads {
                        let src_start = head * head_dim;
                        let src_end = src_start + head_dim;
                        let head_data = &src[src_start..src_end];

                        let dst_offset = head
                            * blocks_per_head
                            * std::mem::size_of::<BlockQ8_0>();
                        let dst_ptr = dst.add(dst_offset) as *mut BlockQ8_0;

                        quantize_row_q8_0(
                            head_data,
                            std::slice::from_raw_parts_mut(
                                dst_ptr,
                                blocks_per_head,
                            ),
                        );
                    }
                }
                GpuKVFormat::Q4_0 => {
                    let blocks_per_head = head_dim.div_ceil(QK);
                    for head in 0..n_heads {
                        let src_start = head * head_dim;
                        let src_end = src_start + head_dim;
                        let head_data = &src[src_start..src_end];

                        let dst_offset = head
                            * blocks_per_head
                            * std::mem::size_of::<BlockQ4_0>();
                        let dst_ptr = dst.add(dst_offset) as *mut BlockQ4_0;

                        quantize_row_q4_0(
                            head_data,
                            std::slice::from_raw_parts_mut(
                                dst_ptr,
                                blocks_per_head,
                            ),
                        );
                    }
                }
                GpuKVFormat::Q4_1 => {
                    let blocks_per_head = head_dim.div_ceil(QK);
                    for head in 0..n_heads {
                        let src_start = head * head_dim;
                        let src_end = src_start + head_dim;
                        let head_data = &src[src_start..src_end];

                        let dst_offset = head
                            * blocks_per_head
                            * std::mem::size_of::<BlockQ4_1>();
                        let dst_ptr = dst.add(dst_offset) as *mut BlockQ4_1;

                        quantize_row_q4_1(
                            head_data,
                            std::slice::from_raw_parts_mut(
                                dst_ptr,
                                blocks_per_head,
                            ),
                        );
                    }
                }
                GpuKVFormat::Q5_0 => {
                    let blocks_per_head = head_dim.div_ceil(QK);
                    for head in 0..n_heads {
                        let src_start = head * head_dim;
                        let src_end = src_start + head_dim;
                        let head_data = &src[src_start..src_end];

                        let dst_offset = head
                            * blocks_per_head
                            * std::mem::size_of::<BlockQ5_0>();
                        let dst_ptr = dst.add(dst_offset) as *mut BlockQ5_0;

                        quantize_row_q5_0(
                            head_data,
                            std::slice::from_raw_parts_mut(
                                dst_ptr,
                                blocks_per_head,
                            ),
                        );
                    }
                }
                GpuKVFormat::Q5_1 => {
                    let blocks_per_head = head_dim.div_ceil(QK);
                    for head in 0..n_heads {
                        let src_start = head * head_dim;
                        let src_end = src_start + head_dim;
                        let head_data = &src[src_start..src_end];

                        let dst_offset = head
                            * blocks_per_head
                            * std::mem::size_of::<BlockQ5_1>();
                        let dst_ptr = dst.add(dst_offset) as *mut BlockQ5_1;

                        quantize_row_q5_1(
                            head_data,
                            std::slice::from_raw_parts_mut(
                                dst_ptr,
                                blocks_per_head,
                            ),
                        );
                    }
                }
            }
        }
    }

    pub fn k_buffer(&self) -> Option<&Buffer> {
        self.k_buffer.as_ref()
    }

    pub fn v_buffer(&self) -> Option<&Buffer> {
        self.v_buffer.as_ref()
    }

    pub fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }
}

/// Full GPU KV Cache for all layers
pub struct GpuKVCache {
    layers: Vec<GpuKVCacheLayer>,
    seq_len: usize,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    format: GpuKVFormat,
    device: Device,
}

impl GpuKVCache {
    pub fn new(
        device: &Device,
        config: &ModelConfig,
        max_seq: usize,
        format: GpuKVFormat,
    ) -> Self {
        let head_dim = config.hidden_size / config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let n_layers = config.n_layers;

        let layers: Vec<GpuKVCacheLayer> = (0..n_layers)
            .map(|_| {
                GpuKVCacheLayer::new(
                    device, n_kv_heads, head_dim, max_seq, format,
                )
            })
            .collect();

        GpuKVCache {
            layers,
            seq_len: 0,
            n_layers,
            n_kv_heads,
            head_dim,
            format,
            device: device.clone(),
        }
    }

    /// Store K and V for a layer
    pub fn set(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if layer < self.layers.len() {
            self.layers[layer].set(self.seq_len, k, v);
        }
    }

    /// Advance sequence length
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get K buffer for a layer
    pub fn k_buffer(&self, layer: usize) -> Option<&Buffer> {
        self.layers.get(layer).and_then(|l| l.k_buffer())
    }

    /// Get V buffer for a layer
    pub fn v_buffer(&self, layer: usize) -> Option<&Buffer> {
        self.layers.get(layer).and_then(|l| l.v_buffer())
    }

    /// Get bytes per row (for stride calculations in kernel)
    pub fn bytes_per_row(&self) -> usize {
        if let Some(layer) = self.layers.first() {
            layer.bytes_per_row()
        } else {
            0
        }
    }

    /// Get format
    pub fn format(&self) -> GpuKVFormat {
        self.format
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Get total memory usage
    pub fn total_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.k_buffer.as_ref().map(|b| b.length() as usize).unwrap_or(0)
                    * 2
            })
            .sum()
    }
}
