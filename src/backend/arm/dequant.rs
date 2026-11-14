// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Dequantization: Quantized formats -> F32

use crate::types::{BlockQ6_K, QK_K};
use half::f16;

/// Dequantize Q6_K block to F32
/// Used for embedding lookup (Q6_K embedding -> F32 hidden)
pub fn dequant_q6_k(data: &[u8], out: &mut [f32], count: usize) {
    let nb = count / QK_K;
    let block_size = std::mem::size_of::<BlockQ6_K>();

    for i in 0..nb {
        let block: &BlockQ6_K = unsafe {
            &*(data.as_ptr().add(i * block_size) as *const BlockQ6_K)
        };

        let d = f16::from_bits(block.d).to_f32();

        // Dequantize 256 values from this block
        for j in 0..16 {
            let scale = block.scales[j] as f32;

            for k in 0..16 {
                let idx = j * 16 + k;

                // Combine lower 4 bits and upper 2 bits
                let lower4 = block.ql[j * 8 + (k % 8)] as i32;
                let upper2 = (block.qh[j * 4 + (k / 8)] >> (4 * ((k / 2) % 2)))
                    as i32
                    & 0x03;
                let v = (lower4 | (upper2 << 4)) as i32 - 32;

                out[i * QK_K + idx] = d * scale * v as f32;
            }
        }
    }
}

/// Dequantize a single Q6_K block to F32
/// Returns array of 256 F32 values
pub fn dequantize_q6_k_block(block: &BlockQ6_K) -> [f32; QK_K] {
    let mut result = [0.0f32; QK_K];

    let d = f16::from_bits(block.d).to_f32();

    for j in 0..16 {
        let scale = block.scales[j] as f32;

        for k in 0..16 {
            let idx = j * 16 + k;

            // Combine lower 4 bits and upper 2 bits
            let lower4 = block.ql[j * 8 + (k % 8)] as i32;
            let upper2 = (block.qh[j * 4 + (k / 8)] >> (4 * ((k / 2) % 2)))
                as i32
                & 0x03;
            let v = (lower4 | (upper2 << 4)) as i32 - 32;

            result[idx] = d * scale * v as f32;
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
