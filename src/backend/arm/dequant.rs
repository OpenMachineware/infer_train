// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Dequantization: Quantized formats -> F32

use crate::types::{BlockQ4_K, BlockQ5_K, BlockQ6_K, QK_K};
use half::f16;

/// Get scale and min values for Q4_K from scales array
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q4_K block to F32
pub fn dequant_q4_k(data: &[u8], out: &mut [f32], count: usize) {
    let nb = count / QK_K;
    let block_size = std::mem::size_of::<BlockQ4_K>();

    for i in 0..nb {
        let block: &BlockQ4_K = unsafe {
            &*(data.as_ptr().add(i * block_size) as *const BlockQ4_K)
        };

        let d = f16::from_bits(block.d).to_f32();
        let min = f16::from_bits(block.dmin).to_f32();

        let q = &block.qs;
        let mut is = 0;
        let mut out_idx = 0;
        let mut q_offset = 0;

        for _j in 0..QK_K / 64 {
            let (sc0, m0) = get_scale_min_k4(is, &block.scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, &block.scales);

            let d1 = d * (sc0 as f32);
            let m1_v = min * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_v = min * (m1 as f32);

            // Lower 4 bits
            for l in 0..32 {
                out[i * QK_K + out_idx + l] =
                    d1 * ((q[q_offset + l] & 0x0F) as f32) - m1_v;
            }
            out_idx += 32;

            // Upper 4 bits
            for l in 0..32 {
                out[i * QK_K + out_idx + l] =
                    d2 * ((q[q_offset + l] >> 4) as f32) - m2_v;
            }
            out_idx += 32;

            q_offset += 32;
            is += 2;
        }
    }
}

/// Dequantize a single Q4_K block to F32
/// Returns array of 256 F32 values
pub fn dequantize_q4_k_block(block: &BlockQ4_K) -> [f32; QK_K] {
    let mut result = [0.0f32; QK_K];

    let d = f16::from_bits(block.d).to_f32();
    let min = f16::from_bits(block.dmin).to_f32();

    let q = &block.qs;
    let mut is = 0;
    let mut out_idx = 0;
    let mut q_offset = 0; // Track q offset

    for _j in 0..QK_K / 64 {
        let (sc0, m0) = get_scale_min_k4(is, &block.scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, &block.scales);

        let d1 = d * (sc0 as f32);
        let m1_v = min * (m0 as f32);
        let d2 = d * (sc1 as f32);
        let m2_v = min * (m1 as f32);

        // Lower 4 bits
        for l in 0..32 {
            result[out_idx + l] = d1 * ((q[q_offset + l] & 0x0F) as f32) - m1_v;
        }
        out_idx += 32;

        // Upper 4 bits
        for l in 0..32 {
            result[out_idx + l] = d2 * ((q[q_offset + l] >> 4) as f32) - m2_v;
        }
        out_idx += 32;

        q_offset += 32; // Move q offset forward
        is += 2;
    }

    result
}

/// Dequantize Q5_K block to F32
/// Q5_K uses 5 bits per value: 4 bits from qs + 1 bit from qh
/// Matches llama.cpp's dequantize_row_q5_K exactly
pub fn dequant_q5_k(data: &[u8], out: &mut [f32], count: usize) {
    let nb = count / QK_K;
    let block_size = std::mem::size_of::<BlockQ5_K>();

    for i in 0..nb {
        let block: &BlockQ5_K = unsafe {
            &*(data.as_ptr().add(i * block_size) as *const BlockQ5_K)
        };

        let d = f16::from_bits(block.d).to_f32();
        let min = f16::from_bits(block.dmin).to_f32();

        let qs = &block.qs;
        let qh = &block.qh;

        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        // Output position within this block
        let mut y_idx = 0;

        for j in 0..QK_K / 64 {
            let (sc0, m0) = get_scale_min_k4(is, &block.scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, &block.scales);

            let d1 = d * (sc0 as f32);
            let m1_v = min * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_v = min * (m1 as f32);

            // qs advances by 32 each iteration (not by 64!)
            // Each iteration reads 32 bytes of qs and produces 64 values
            let qs_offset = j * 32;

            // Lower 4 bits + high bit from qh
            // Note: qh is NOT advanced, always accessed with l in 0..32
            for l in 0..32 {
                let q5_val = (qs[qs_offset + l] & 0x0F) as u8
                    + if qh[l] & u1 != 0 { 16 } else { 0 };
                out[i * QK_K + y_idx + l] = d1 * (q5_val as f32) - m1_v;
            }
            y_idx += 32;

            // Upper 4 bits + high bit from qh
            for l in 0..32 {
                let q5_val = (qs[qs_offset + l] >> 4)
                    + if qh[l] & u2 != 0 { 16 } else { 0 };
                out[i * QK_K + y_idx + l] = d2 * (q5_val as f32) - m2_v;
            }
            y_idx += 32;

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Dequantize a single Q5_K block to F32
/// Returns array of 256 F32 values
/// Matches llama.cpp's dequantize_row_q5_K exactly
pub fn dequant_q5_k_block(block: &BlockQ5_K) -> [f32; QK_K] {
    let mut result = [0.0f32; QK_K];

    let d = f16::from_bits(block.d).to_f32();
    let min = f16::from_bits(block.dmin).to_f32();

    let qs = &block.qs;
    let qh = &block.qh;

    let mut is = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    let mut y_idx = 0;

    for j in 0..QK_K / 64 {
        let (sc0, m0) = get_scale_min_k4(is, &block.scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, &block.scales);

        let d1 = d * (sc0 as f32);
        let m1_v = min * (m0 as f32);
        let d2 = d * (sc1 as f32);
        let m2_v = min * (m1 as f32);

        let qs_offset = j * 32;

        for l in 0..32 {
            let q5_val = (qs[qs_offset + l] & 0x0F) as u8
                + if qh[l] & u1 != 0 { 16 } else { 0 };
            result[y_idx + l] = d1 * (q5_val as f32) - m1_v;
        }
        y_idx += 32;

        for l in 0..32 {
            let q5_val =
                (qs[qs_offset + l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
            result[y_idx + l] = d2 * (q5_val as f32) - m2_v;
        }
        y_idx += 32;

        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }

    result
}

/// Dequantize Q6_K block to F32
/// Used for embedding lookup (Q6_K embedding -> F32 hidden)
/// Matches llama.cpp's dequantize_row_q6_K exactly
pub fn dequant_q6_k(data: &[u8], out: &mut [f32], count: usize) {
    let nb = count / QK_K;
    let block_size = std::mem::size_of::<BlockQ6_K>();

    for i in 0..nb {
        let block: &BlockQ6_K = unsafe {
            &*(data.as_ptr().add(i * block_size) as *const BlockQ6_K)
        };

        let d = f16::from_bits(block.d).to_f32();

        // Process 256 values in two 128-value chunks
        // Matches llama.cpp's loop structure exactly
        for n in 0..2 {
            let ql_offset = n * 64; // ql pointer advances 64 per 128 values
            let qh_offset = n * 32; // qh pointer advances 32 per 128 values
            let sc_offset = n * 8; // scales pointer advances 8 per 128 values
            let y_offset = n * 128; // output pointer advances 128 per 128 values

            for l in 0..32 {
                let is = l / 16; // 0 for l<16, 1 for l>=16

                // Extract 4 values from ql and qh
                // ql[l] lower 4 bits + qh[l] bits 0-1 -> q1
                // ql[l+32] lower 4 bits + qh[l] bits 2-3 -> q2
                // ql[l] upper 4 bits + qh[l] bits 4-5 -> q3
                // ql[l+32] upper 4 bits + qh[l] bits 6-7 -> q4
                let ql_l = block.ql[ql_offset + l] as i32;
                let ql_l32 = block.ql[ql_offset + 32 + l] as i32;
                let qh_l = block.qh[qh_offset + l] as i32;

                let q1 = ((ql_l & 0xF) | ((qh_l & 3) << 4)) - 32;
                let q2 = (((ql_l32) & 0xF) | (((qh_l >> 2) & 3) << 4)) - 32;
                let q3 = ((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
                let q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

                // Get scales: 4 scales per 32 values
                let sc = &block.scales[sc_offset..];
                let s1 = sc[is] as f32;
                let s2 = sc[is + 2] as f32;
                let s3 = sc[is + 4] as f32;
                let s4 = sc[is + 6] as f32;

                // Write output
                let base = i * QK_K + y_offset;
                out[base + l] = d * s1 * (q1 as f32);
                out[base + 32 + l] = d * s2 * (q2 as f32);
                out[base + 64 + l] = d * s3 * (q3 as f32);
                out[base + 96 + l] = d * s4 * (q4 as f32);
            }
        }
    }
}

/// Dequantize a single Q6_K block to F32
/// Returns array of 256 F32 values
/// Matches llama.cpp's dequantize_row_q6_K exactly
pub fn dequantize_q6_k_block(block: &BlockQ6_K) -> [f32; QK_K] {
    let mut result = [0.0f32; QK_K];

    let d = f16::from_bits(block.d).to_f32();

    // Process 256 values in two 128-value chunks
    for n in 0..2 {
        let ql_offset = n * 64;
        let qh_offset = n * 32;
        let sc_offset = n * 8;
        let y_offset = n * 128;

        for l in 0..32 {
            let is = l / 16;

            let ql_l = block.ql[ql_offset + l] as i32;
            let ql_l32 = block.ql[ql_offset + 32 + l] as i32;
            let qh_l = block.qh[qh_offset + l] as i32;

            let q1 = ((ql_l & 0xF) | ((qh_l & 3) << 4)) - 32;
            let q2 = (((ql_l32) & 0xF) | (((qh_l >> 2) & 3) << 4)) - 32;
            let q3 = ((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
            let q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

            let sc = &block.scales[sc_offset..];
            let s1 = sc[is] as f32;
            let s2 = sc[is + 2] as f32;
            let s3 = sc[is + 4] as f32;
            let s4 = sc[is + 6] as f32;

            let base = y_offset;
            result[base + l] = d * s1 * (q1 as f32);
            result[base + 32 + l] = d * s2 * (q2 as f32);
            result[base + 64 + l] = d * s3 * (q3 as f32);
            result[base + 96 + l] = d * s4 * (q4 as f32);
        }
    }

    result
}

/// Dequantize a single row of Q6_K embedding to F32
/// This is used for embedding lookup
pub fn dequantize_row_q6_k(
    data: &[u8],
    row_idx: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let nb = hidden_size / QK_K;
    let block_size = std::mem::size_of::<BlockQ6_K>();

    let mut result = vec![0.0f32; hidden_size];

    for i in 0..nb {
        let block: &BlockQ6_K = unsafe {
            &*(data.as_ptr().add((row_idx * nb + i) * block_size)
                as *const BlockQ6_K)
        };

        let block_vals = dequantize_q6_k_block(block);
        result[i * QK_K..(i + 1) * QK_K].copy_from_slice(&block_vals);
    }

    result
}
