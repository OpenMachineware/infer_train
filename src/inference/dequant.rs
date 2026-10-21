// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// K-quant weight dequantization (corrected implementation)

use crate::quant::types::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, QK_K,
};
use half::f16;

/// Helper to extract scale and min from Q4_K/Q5_K scales array
/// j: scale index (0-7)
/// Returns (scale, min)
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (f32, f32) {
    if j < 4 {
        let sc = (scales[j] & 63) as f32;
        let min = (scales[j + 4] & 63) as f32;
        (sc, min)
    } else {
        // Extract from packed format (matches llama.cpp exactly):
        // scale: lower 4 bits of scales[j+4] | (upper 2 bits of scales[j-4]) << 4
        // min: upper 4 bits of scales[j+4] | (upper 2 bits of scales[j]) << 4
        let sc = ((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)) as f32;
        let min = ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)) as f32;
        (sc, min)
    }
}

/// Dequantize Q4_K block to F32
/// Q4_K: 256 elements per block, 144 bytes
/// 4 groups of 64 elements, each group uses 2 scales
/// Layout matches llama.cpp's dequantize_row_q4_K exactly
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

        // Process 4 groups of 64 elements, matching llama.cpp layout
        // Each group uses 2 scales: scale[is] for lower nibbles, scale[is+1] for upper nibbles
        // q pointer advances by 32 bytes per group
        let mut is: usize = 0;
        let mut q_offset: usize = 0;

        for j in (0..QK_K).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, &block.scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, &block.scales);
            let d1 = d * sc0;
            let m1_val = dmin * m0;
            let d2 = d * sc1;
            let m2_val = dmin * m1;

            // Lower nibbles of q[q_offset..q_offset+31] -> elements j..j+31
            for l in 0..32 {
                let byte = block.qs[q_offset + l];
                let v = (byte & 0x0F) as f32;
                let idx = block_idx * QK_K + j + l;
                if idx < count {
                    out[idx] = d1 * v - m1_val;
                }
            }

            // Upper nibbles of q[q_offset..q_offset+31] -> elements j+32..j+63
            for l in 0..32 {
                let byte = block.qs[q_offset + l];
                let v = (byte >> 4) as f32;
                let idx = block_idx * QK_K + j + 32 + l;
                if idx < count {
                    out[idx] = d2 * v - m2_val;
                }
            }

            q_offset += 32;
            is += 2;
        }
    }
}

/// Dequantize Q5_K block to F32
/// Q5_K: 256 elements per block, 176 bytes
/// 8 sub-blocks of 32 elements each with 5-bit values
/// Layout matches NEON implementation: qh bits are reused across groups
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

        fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (f32, f32) {
            let (d, m) = if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
                let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
                (d, m)
            };
            (d as f32, m as f32)
        }

        // u1 and u2 shift left by 2 each iteration to select different qh bits
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut ql_offset: usize = 0;
        let mut is: usize = 0;

        for _j in 0..4 {
            let (scale0, min0) = get_scale_min_k4(is, &block.scales);
            let (scale1, min1) = get_scale_min_k4(is + 1, &block.scales);

            let d1 = d * scale0;
            let m1 = dmin * min0;
            let d2 = d * scale1;
            let m2 = dmin * min1;

            // First 32 elements: qs low nibbles, qh & u1
            for l in 0..32 {
                let idx = block_idx * QK_K + _j * 64 + l;
                if idx < count {
                    let q_low = (block.qs[ql_offset + l] & 0x0F) as f32;
                    let h =
                        if block.qh[l] & u1 != 0 { 16.0f32 } else { 0.0f32 };
                    out[idx] = d1 * (q_low + h) - m1;
                }
            }

            // Second 32 elements: qs high nibbles, qh & u2
            for l in 0..32 {
                let idx = block_idx * QK_K + _j * 64 + 32 + l;
                if idx < count {
                    let q_high = ((block.qs[ql_offset + l] >> 4) & 0x0F) as f32;
                    let h =
                        if block.qh[l] & u2 != 0 { 16.0f32 } else { 0.0f32 };
                    out[idx] = d2 * (q_high + h) - m2;
                }
            }

            ql_offset += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
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
/// 16 sub-blocks of 16 elements each, 3-bit quantization
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

        // Q3_K: 16 sub-blocks of 16 elements each
        // scales[12] contains 16 scales (6 bits each), packed as 4 scales per 3 bytes
        for is in 0..16 {
            // Extract 6-bit scale from scales[12]
            let byte_idx = (is / 4) * 3;
            let bit_offset = (is % 4) * 6;
            let byte_off = bit_offset / 8;
            let bit_off = bit_offset % 8;

            let scale = if bit_off <= 2 {
                ((block.scales[byte_idx + byte_off] as u32 >> bit_off) & 0x3F)
                    as u8
            } else {
                // Spans two bytes
                (((block.scales[byte_idx + byte_off] as u32 >> bit_off)
                    | ((block.scales[byte_idx + byte_off + 1] as u32)
                        << (8 - bit_off)))
                    & 0x3F) as u8
            };

            // Scale is 6-bit unsigned, converted to signed by subtracting 32
            let sc = (scale as f32) - 32.0;
            let dl = d * sc;
            let ml = 4.0 * dl; // Offset for negative quant values

            // Process 16 elements of this sub-block
            for k in 0..16 {
                let elem_idx = is * 16 + k;
                let idx = block_idx * QK_K + elem_idx;
                if idx >= count {
                    break;
                }

                // 2 bits from qs
                let qs_idx = is * 4 + k / 4; // 4 elements per qs byte
                let qs_bit = (k % 4) * 2;
                let q = ((block.qs[qs_idx] >> qs_bit) & 0x03) as f32;

                // 1 bit from hmask
                let hm_idx = is * 2 + k / 8; // 8 elements per hmask byte
                let hm_bit = k % 8;
                let hbit = ((block.hmask[hm_idx] >> hm_bit) & 1) as i32;

                // If hbit is set, v = dl * q
                // If hbit is not set, v = dl * q - ml = dl * (q - 4)
                let v = dl * q - if hbit == 1 { 0.0 } else { ml };

                out[idx] = v;
            }
        }
    }
}

/// Dequantize Q6_K block to F32
/// Q6_K: 256 elements per block, 210 bytes
/// 16 sub-blocks of 16 elements each, 6-bit quantization
/// Uses interleaved ql/qh layout
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

        // Q6_K layout from llama.cpp:
        // 2 groups of 128 elements each
        // Each group processes 32 values at a time, producing 4 outputs per iteration
        let ql = &block.ql;
        let qh = &block.qh;
        let sc = &block.scales;

        for n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;

                // Construct 6-bit values from 4-bit ql + 2-bit qh
                let q1 = ((ql[(n / 128) * 64 + l] & 0x0F) as i32
                    | (((qh[(n / 128) * 32 + l] >> 0) & 3) as i32) << 4)
                    - 32;
                let q2 = ((ql[(n / 128) * 64 + l + 32] & 0x0F) as i32
                    | (((qh[(n / 128) * 32 + l] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = ((ql[(n / 128) * 64 + l] >> 4) as i32
                    | (((qh[(n / 128) * 32 + l] >> 4) & 3) as i32) << 4)
                    - 32;
                let q4 = ((ql[(n / 128) * 64 + l + 32] >> 4) as i32
                    | (((qh[(n / 128) * 32 + l] >> 6) & 3) as i32) << 4)
                    - 32;

                let idx0 = block_idx * QK_K + n + l;
                let idx1 = block_idx * QK_K + n + l + 32;
                let idx2 = block_idx * QK_K + n + l + 64;
                let idx3 = block_idx * QK_K + n + l + 96;

                if idx0 < count {
                    out[idx0] =
                        d * sc[(n / 128) * 8 + is + 0] as f32 * q1 as f32;
                }
                if idx1 < count {
                    out[idx1] =
                        d * sc[(n / 128) * 8 + is + 2] as f32 * q2 as f32;
                }
                if idx2 < count {
                    out[idx2] =
                        d * sc[(n / 128) * 8 + is + 4] as f32 * q3 as f32;
                }
                if idx3 < count {
                    out[idx3] =
                        d * sc[(n / 128) * 8 + is + 6] as f32 * q4 as f32;
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

        let (sc, min) = get_scale_min_k4(0, &scales);
        assert_eq!(sc, 42.0);
        assert_eq!(min, 17.0);
    }
}
