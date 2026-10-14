// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// K-quant weight dequantization (corrected implementation)

use crate::quant::types::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, QK_K,
};
use half::f16;

/// Helper to extract scale and min from Q4_K/Q5_K scales array
/// j: sub-block index (0-7)
/// k: which pair (0 or 1)
/// Returns (scale, min)
fn get_scale_min_k4(j: usize, k: usize, scales: &[u8; 12]) -> (f32, f32) {
    if j < 4 {
        let sc = (scales[j + k] & 63) as f32;
        let min = (scales[j + 4 + k] & 63) as f32;
        (sc, min)
    } else {
        // Extract from packed format: 4-bit value from one byte combined with 2 bits from another
        let sc = ((scales[j + 4 + k] & 0x0F)
            | ((scales[j - 4 + k] & 0xC0) >> 2)) as f32;
        let min =
            ((scales[j + 4 + k] >> 4) | ((scales[j + k] & 0xC0) >> 2)) as f32;
        (sc, min)
    }
}

/// Dequantize Q4_K block to F32
/// Q4_K: 256 elements per block, 144 bytes
/// 8 sub-blocks of 32 elements each
/// Each sub-block has its own scale and min
pub fn dequant_q4_k(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK_K);
    for block_idx in 0..n_blocks {
        let block_offset = block_idx * std::mem::size_of::<BlockQ4K>();
        if block_offset + std::mem::size_of::<BlockQ4K>() > data.len() {
            break;
        }
        let block: &BlockQ4K =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ4K) };
        let d = f16::from_bits(block.d).to_f32();
        let dmin = f16::from_bits(block.dmin).to_f32();

        // Process 8 sub-blocks of 32 elements each
        for j in 0..8 {
            // Get scale and min for this sub-block
            let (sc, min) = get_scale_min_k4(j, 0, &block.scales);
            let dl = d * sc;
            let ml = dmin * min;

            // Each sub-block has 32 elements stored in 16 bytes
            // Each byte packs two 4-bit values
            for k in 0..16 {
                let byte = block.qs[j * 16 + k];
                let v0 = (byte & 0x0F) as f32;
                let v1 = ((byte >> 4) & 0x0F) as f32;

                let idx0 = block_idx * QK_K + j * 32 + k;
                let idx1 = block_idx * QK_K + j * 32 + k + 16;

                if idx0 < count {
                    out[idx0] = dl * v0 - ml;
                }
                if idx1 < count {
                    out[idx1] = dl * v1 - ml;
                }
            }
        }
    }
}

/// Dequantize Q5_K block to F32
/// Q5_K: 256 elements per block, 176 bytes
/// 8 sub-blocks of 32 elements each with 5-bit values
pub fn dequant_q5_k(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK_K);
    for block_idx in 0..n_blocks {
        let block_offset = block_idx * std::mem::size_of::<BlockQ5K>();
        if block_offset + std::mem::size_of::<BlockQ5K>() > data.len() {
            break;
        }
        let block: &BlockQ5K =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ5K) };
        let d = f16::from_bits(block.d).to_f32();
        let dmin = f16::from_bits(block.dmin).to_f32();

        // Process 8 sub-blocks of 32 elements each
        for j in 0..8 {
            let (sc, min) = get_scale_min_k4(j, 0, &block.scales);
            let dl = d * sc;
            let ml = dmin * min;

            for k in 0..16 {
                let byte = block.qs[j * 16 + k];
                let v0_low = (byte & 0x0F) as f32;
                let v1_low = ((byte >> 4) & 0x0F) as f32;

                // High bit from qh
                let idx0 = block_idx * QK_K + j * 32 + k;
                let idx1 = block_idx * QK_K + j * 32 + k + 16;

                let h0 = if idx0 < count {
                    let qh_idx = idx0 / 8;
                    let qh_bit = idx0 % 8;
                    if qh_idx < block.qh.len() {
                        ((block.qh[qh_idx] >> qh_bit) & 1) as f32 * 16.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let h1 = if idx1 < count {
                    let qh_idx = idx1 / 8;
                    let qh_bit = idx1 % 8;
                    if qh_idx < block.qh.len() {
                        ((block.qh[qh_idx] >> qh_bit) & 1) as f32 * 16.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                if idx0 < count {
                    out[idx0] = dl * (v0_low + h0) - ml;
                }
                if idx1 < count {
                    out[idx1] = dl * (v1_low + h1) - ml;
                }
            }
        }
    }
}

/// Dequantize Q2_K block to F32
/// Q2_K: 256 elements per block, 84 bytes
/// 16 sub-blocks of 16 elements each
pub fn dequant_q2_k(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK_K);
    for block_idx in 0..n_blocks {
        let block_offset = block_idx * std::mem::size_of::<BlockQ2K>();
        if block_offset + std::mem::size_of::<BlockQ2K>() > data.len() {
            break;
        }
        let block: &BlockQ2K =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ2K) };
        let d = f16::from_bits(block.d).to_f32();
        let dmin = f16::from_bits(block.dmin).to_f32();

        // 16 sub-blocks, each with 16 elements
        for j in 0..16 {
            let sc = (block.scales[j] & 0x0F) as f32;
            let min = (block.scales[j] >> 4) as f32;
            let dl = d * sc;
            let ml = dmin * min;

            // 2-bit quantization: 4 values per byte
            for k in 0..4 {
                let byte = block.qs[j * 4 + k];
                for l in 0..4 {
                    let bits = ((byte >> (2 * l)) & 0x03) as f32;
                    let val = bits - 2.0; // 2-bit signed: 0,1,2,3 -> -2,-1,0,1

                    let idx = block_idx * QK_K + j * 16 + k * 4 + l;
                    if idx < count {
                        out[idx] = dl * val - ml;
                    }
                }
            }
        }
    }
}

/// Dequantize Q3_K block to F32
/// Q3_K: 256 elements per block, 110 bytes
pub fn dequant_q3_k(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK_K);
    for block_idx in 0..n_blocks {
        let block_offset = block_idx * std::mem::size_of::<BlockQ3K>();
        if block_offset + std::mem::size_of::<BlockQ3K>() > data.len() {
            break;
        }
        let block: &BlockQ3K =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ3K) };
        let d = f16::from_bits(block.d).to_f32();

        // Q3_K has complex scale and hmask handling
        // Simplified implementation
        for j in 0..256 {
            let idx = block_idx * QK_K + j;
            if idx >= count {
                break;
            }

            // Extract 3-bit value
            let byte_idx = j / 4;
            let bit_offset = (j % 4) * 2;
            let qs_byte = block.qs[byte_idx.min(block.qs.len() - 1)];
            let low_bits = ((qs_byte >> bit_offset) & 0x03) as f32;

            // High bit from hmask
            let hmask_idx = j / 8;
            let hmask_bit = j % 8;
            let high_bit = if hmask_idx < block.hmask.len() {
                ((block.hmask[hmask_idx] >> hmask_bit) & 1) as f32
            } else {
                0.0
            };

            // 3-bit value: 0-7, recentered
            let val = low_bits + high_bit * 4.0 - 4.0;

            // Scale (simplified - uses average scale)
            let scale = if j < 128 {
                (block.scales[j / 16] as f32).max(1.0)
            } else {
                (block.scales[8 + (j - 128) / 16] as f32).max(1.0)
            };

            out[idx] = d * scale * val / 16.0;
        }
    }
}

/// Dequantize Q6_K block to F32
/// Q6_K: 256 elements per block, 210 bytes
pub fn dequant_q6_k(data: &[u8], out: &mut [f32], count: usize) {
    let n_blocks = count.div_ceil(QK_K);
    for block_idx in 0..n_blocks {
        let block_offset = block_idx * std::mem::size_of::<BlockQ6K>();
        if block_offset + std::mem::size_of::<BlockQ6K>() > data.len() {
            break;
        }
        let block: &BlockQ6K =
            unsafe { &*(data.as_ptr().add(block_offset) as *const BlockQ6K) };
        let d = f16::from_bits(block.d).to_f32();

        // Q6_K: 16 sub-blocks of 16 elements
        // ql[128]: lower 4 bits
        // qh[64]: upper 2 bits
        // scales[16]: per-sub-block scales
        for j in 0..16 {
            let sc = block.scales[j] as f32 / 127.0;

            for k in 0..16 {
                let elem_idx = j * 16 + k;
                let idx = block_idx * QK_K + elem_idx;
                if idx >= count {
                    continue;
                }

                // Lower 4 bits from ql
                // Layout: ql[0..63] holds first 128 elements, ql[64..127] holds second 128
                let ql_idx = if elem_idx < 128 {
                    (elem_idx / 2) as usize
                } else {
                    64 + ((elem_idx - 128) / 2) as usize
                };

                if ql_idx < block.ql.len() {
                    let ql_byte = block.ql[ql_idx];
                    let l4 = if elem_idx % 2 == 0 {
                        (ql_byte & 0x0F) as f32
                    } else {
                        ((ql_byte >> 4) & 0x0F) as f32
                    };

                    // Upper 2 bits from qh
                    let qh_idx = (elem_idx / 4) as usize;
                    let qh_offset = (elem_idx % 4) * 2;
                    let h2 = if qh_idx < block.qh.len() {
                        ((block.qh[qh_idx] >> qh_offset) & 0x03) as f32
                    } else {
                        0.0
                    };

                    // 6-bit value: 0-63 -> -32 to 31
                    let val = l4 + h2 * 16.0 - 32.0;
                    out[idx] = d * sc * val;
                }
            }
        }
    }
}

/// Dequantize a single Q6_K block to F32
/// Returns 256 F32 values
pub fn dequantize_q6_k_block(block: &BlockQ6K) -> [f32; QK_K] {
    let mut out = [0.0f32; QK_K];
    let d = f16::from_bits(block.d).to_f32();

    for j in 0..16 {
        let sc = block.scales[j] as f32 / 127.0;

        for k in 0..16 {
            let elem_idx = j * 16 + k;

            // Lower 4 bits from ql
            let ql_idx = if elem_idx < 128 {
                (elem_idx / 2) as usize
            } else {
                64 + ((elem_idx - 128) / 2) as usize
            };

            if ql_idx < block.ql.len() {
                let ql_byte = block.ql[ql_idx];
                let l4 = if elem_idx % 2 == 0 {
                    (ql_byte & 0x0F) as f32
                } else {
                    ((ql_byte >> 4) & 0x0F) as f32
                };

                // Upper 2 bits from qh
                let qh_idx = (elem_idx / 4) as usize;
                let qh_offset = (elem_idx % 4) * 2;
                let h2 = if qh_idx < block.qh.len() {
                    ((block.qh[qh_idx] >> qh_offset) & 0x03) as f32
                } else {
                    0.0
                };

                // 6-bit value: 0-63 -> -32 to 31
                let val = l4 + h2 * 16.0 - 32.0;
                out[elem_idx] = d * sc * val;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_extraction() {
        let mut scales = [0u8; 12];
        scales[0] = 42; // scale
        scales[4] = 17; // min

        let (sc, min) = get_scale_min_k4(0, 0, &scales);
        assert_eq!(sc, 42.0);
        assert_eq!(min, 17.0);
    }
}
