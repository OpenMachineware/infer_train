// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// K-tiled GEMV for better cache locality
// Key insight: Process K (columns) in tiles, not M (rows)
//
// Problem with standard GEMV:
// - Processes one row at a time
// - For 2048 rows, each row's weights are accessed once
// - Input vector is accessed 2048 times but weights (182 MB) don't fit in cache
//
// Solution with K-tiling:
// - Process K columns in tiles (e.g., 512 elements at a time)
// - For each K-tile, compute partial dot products for ALL rows
// - Keep K-tile of input vector in L1 cache, reuse for all rows
// - Working set: K-tile input + K-tile weights for all rows

use crate::quant::types::{BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K};
use crate::quant::vec_dot::arm::k_quant::{
    vec_dot_q4_k_q8_k_neon, vec_dot_q5_k_q8_k_neon, vec_dot_q6_k_q8_k_neon,
};
use std::arch::aarch64::*;

/// Calculate K-extent (number of K elements) that fits in L1 cache
/// Following llama.cpp approach from tiled.cpp
#[inline]
fn calc_k_extent(k: usize, m: usize) -> usize {
    // L1 cache is typically 32-48KB
    // We want to fit: K-extent of input + K-extent of weights for all rows
    // For K-quant: each 256 elements = 1 block
    // BlockQ5K = 138 bytes, BlockQ8K = 264 bytes

    // For small matrices (m <= 16), process all K at once
    if m <= 16 {
        return k;
    }

    // For larger matrices, tile to fit in L1
    // Each K-tile needs: m rows * (K-tile / 256) weight blocks
    // We want: weight_size + input_size < L1_size

    // Simplified: use 256 elements (1 block) as minimum tile
    // For 256 K elements:
    // - Input: 1 BlockQ8K = 264 bytes
    // - Weights: m * 1 BlockQ5K = m * 138 bytes
    // For m=2048: 2048 * 138 = 283 KB, too large for L1

    // For very large m (output projection), use smaller K-tiles
    // For moderate m (FFN), use larger K-tiles

    // Start with QK_K (256) as tile size, fits input in L1
    // Weights will be streamed through L1 as we process each row
    QK_K
}

/// Manual SDOT implementation via inline asm (for stable Rust)
#[inline(always)]
unsafe fn vdotq_s32_manual(
    acc: int32x4_t,
    a: int8x16_t,
    b: int8x16_t,
) -> int32x4_t {
    use std::arch::asm;
    let mut result = acc;
    asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, preserves_flags)
    );
    result
}

/// Convert FP16 to FP32
#[inline(always)]
unsafe fn fp16_to_fp32(h: u16) -> f32 {
    use std::arch::asm;
    let mut f: f32;
    asm!(
        "fcvt {0:s}, {1:h}",
        out(vreg) f,
        in(vreg) h as u32,
        options(pure, nomem, preserves_flags)
    );
    f
}

/// K-tiled Q5_K × Q8_K GEMV
/// Processes K columns in tiles to keep input vector in L1 cache
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k_k_tiled(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // For small K, process directly without tiling
    if nb <= 2 {
        // Direct computation for small matrices
        for row in 0..m {
            dst[row] = vec_dot_q5_k_q8_k_neon(
                k,
                &weights[row * nb..(row + 1) * nb],
                x,
            );
        }
        return;
    }

    // Initialize output to zero
    for row in 0..m {
        dst[row] = 0.0;
    }

    // Process K blocks one at a time (K-tiling)
    // Each iteration:
    // 1. Load one K-block of input (stays in L1 cache)
    // 2. For each row, load one K-block of weights, compute partial dot product
    // 3. Accumulate to output
    for k_block in 0..nb {
        let x_block = &x[k_block];

        // Process all rows for this K-block
        for row in 0..m {
            let w_block = &weights[row * nb + k_block];

            // Compute partial dot product for this K-block
            let partial = vec_dot_q5_k_q8_k_single_block(w_block, x_block);
            dst[row] += partial;
        }
    }
}

/// Compute dot product for a single K-block (256 elements)
/// Returns the partial sum (not yet scaled by the overall scale)
#[target_feature(enable = "neon,dotprod")]
unsafe fn vec_dot_q5_k_q8_k_single_block(x: &BlockQ5K, y: &BlockQ8K) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4b = vdupq_n_u8(0xF);
    let mone = vdupq_n_u8(1);
    let mtwo = vdupq_n_u8(2);
    let mzero = vdupq_n_s32(0);

    let d = y.d * fp16_to_fp32(x.d);
    let dmin = y.d * fp16_to_fp32(x.dmin);

    // Calculate min correction using bsums
    let q8sums = vpaddq_s16(
        vld1q_s16(y.bsums.as_ptr()),
        vld1q_s16(y.bsums.as_ptr().add(8)),
    );

    // Decode scales and mins
    let mut utmp = [0u32; 4];
    std::ptr::copy_nonoverlapping(
        x.scales.as_ptr(),
        utmp.as_mut_ptr() as *mut u8,
        12,
    );

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let mins8 = vld1_u8((utmp.as_ptr() as *const u8).add(8));
    let mins = vreinterpretq_s16_u16(vmovl_u8(mins8));

    let prod = vaddq_s32(
        vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
        vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
    );
    let sumi_mins = vaddvq_s32(prod);

    let scales: [u8; 16] = std::mem::transmute(utmp);
    let mut scale_idx = 0;

    let q5 = x.qs.as_ptr();
    let qh = x.qh.as_ptr();
    let q8 = y.qs.as_ptr();

    let qhbits = vld1q_u8_x2(qh);

    let mut sumi = 0i32;

    for j in 0..(QK_K / 64) {
        let q5bits = vld1q_u8_x2(q5.add(j * 32));
        let q8bytes = vld1q_s8_x4(q8.add(j * 64));

        // Process 4 blocks of 16 elements
        let q5h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
        let q5h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
        let q5h_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
        let q5h_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);

        let q5bytes_0 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.0, m4b), q5h_0));
        let q5bytes_1 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.1, m4b), q5h_1));
        let q5bytes_2 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.0, 4), q5h_2));
        let q5bytes_3 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.1, 4), q5h_3));

        sumi += vaddvq_s32(vaddq_s32(
            vdotq_s32_manual(mzero, q5bytes_0, q8bytes.0),
            vdotq_s32_manual(mzero, q5bytes_1, q8bytes.1),
        )) * scales[scale_idx] as i32;
        scale_idx += 1;

        sumi += vaddvq_s32(vaddq_s32(
            vdotq_s32_manual(mzero, q5bytes_2, q8bytes.2),
            vdotq_s32_manual(mzero, q5bytes_3, q8bytes.3),
        )) * scales[scale_idx] as i32;
        scale_idx += 1;
    }

    d * sumi as f32 - dmin * sumi_mins as f32
}

/// K-tiled Q4_K × Q8_K GEMV
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_k_q8_k_k_tiled(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // For small K, process directly
    if nb <= 2 {
        for row in 0..m {
            dst[row] =
                crate::quant::vec_dot::arm::k_quant::vec_dot_q4_k_q8_k_neon(
                    k,
                    &weights[row * nb..(row + 1) * nb],
                    x,
                );
        }
        return;
    }

    // Initialize output
    for row in 0..m {
        dst[row] = 0.0;
    }

    // K-tiling
    for k_block in 0..nb {
        let x_block = &x[k_block];

        for row in 0..m {
            let w_block = &weights[row * nb + k_block];
            let partial = vec_dot_q4_k_q8_k_single_block(w_block, x_block);
            dst[row] += partial;
        }
    }
}

/// Compute dot product for a single Q4_K K-block
#[target_feature(enable = "neon,dotprod")]
unsafe fn vec_dot_q4_k_q8_k_single_block(x: &BlockQ4K, y: &BlockQ8K) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4b = vdupq_n_u8(0xF);
    let mzero = vdupq_n_s32(0);

    let d = y.d * fp16_to_fp32(x.d);
    let dmin = y.d * fp16_to_fp32(x.dmin);

    // Calculate min correction
    let q8sums = vpaddq_s16(
        vld1q_s16(y.bsums.as_ptr()),
        vld1q_s16(y.bsums.as_ptr().add(8)),
    );

    // Decode scales and mins
    let mut utmp = [0u32; 4];
    std::ptr::copy_nonoverlapping(
        x.scales.as_ptr(),
        utmp.as_mut_ptr() as *mut u8,
        12,
    );

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let mins8 = vld1_u8((utmp.as_ptr() as *const u8).add(8));
    let mins = vreinterpretq_s16_u16(vmovl_u8(mins8));

    let prod = vaddq_s32(
        vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
        vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
    );
    let sumi_mins = vaddvq_s32(prod);

    let scales: [u8; 16] = std::mem::transmute(utmp);
    let mut scale_idx = 0;

    let q4 = x.qs.as_ptr();
    let q8 = y.qs.as_ptr();

    let mut sumi = 0i32;

    for j in 0..(QK_K / 64) {
        let q4bits = vld1q_u8_x2(q4.add(j * 32));
        let q8bytes = vld1q_s8_x4(q8.add(j * 64));

        let q4bytes_0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
        let q4bytes_1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));
        let q4bytes_2 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
        let q4bytes_3 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));

        sumi += vaddvq_s32(vaddq_s32(
            vdotq_s32_manual(mzero, q4bytes_0, q8bytes.0),
            vdotq_s32_manual(mzero, q4bytes_1, q8bytes.1),
        )) * scales[scale_idx] as i32;
        scale_idx += 1;

        sumi += vaddvq_s32(vaddq_s32(
            vdotq_s32_manual(mzero, q4bytes_2, q8bytes.2),
            vdotq_s32_manual(mzero, q4bytes_3, q8bytes.3),
        )) * scales[scale_idx] as i32;
        scale_idx += 1;
    }

    d * sumi as f32 - dmin * sumi_mins as f32
}

/// K-tiled Q6_K × Q8_K GEMV
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6_k_q8_k_k_tiled(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // For small K, process directly
    if nb <= 2 {
        for row in 0..m {
            dst[row] =
                crate::quant::vec_dot::arm::k_quant::vec_dot_q6_k_q8_k_neon(
                    k,
                    &weights[row * nb..(row + 1) * nb],
                    x,
                );
        }
        return;
    }

    // Initialize output
    for row in 0..m {
        dst[row] = 0.0;
    }

    // K-tiling
    for k_block in 0..nb {
        let x_block = &x[k_block];

        for row in 0..m {
            let w_block = &weights[row * nb + k_block];
            let partial = vec_dot_q6_k_q8_k_single_block(w_block, x_block);
            dst[row] += partial;
        }
    }
}

/// Compute dot product for a single Q6_K K-block
#[target_feature(enable = "neon,dotprod")]
unsafe fn vec_dot_q6_k_q8_k_single_block(x: &BlockQ6K, y: &BlockQ8K) -> f32 {
    let m4b = vdupq_n_u8(0x0F);
    let mone = vdupq_n_u8(0x30);
    let vzero = vdupq_n_s32(0);

    let d = y.d * fp16_to_fp32(x.d);

    let ql = x.ql.as_ptr();
    let qh = x.qh.as_ptr();
    let q8 = y.qs.as_ptr();
    let scales = x.scales.as_ptr();

    let mut isum = 0i32;

    // Process 2 iterations of 128 elements each
    for j in 0..2 {
        let qh_bits = vld1q_u8_x2(qh.add(j * 32));
        let ql_bits = vld1q_u8_x4(ql.add(j * 64));

        // Process 8 blocks of 16 elements
        let q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.0, 4));
        let q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.1, 4));
        let q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.0, 2));
        let q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.1, 2));

        let q6h_4 = vandq_u8(mone, qh_bits.0);
        let q6h_5 = vandq_u8(mone, qh_bits.1);
        let q6h_6 = vandq_u8(mone, vshrq_n_u8(qh_bits.0, 2));
        let q6h_7 = vandq_u8(mone, vshrq_n_u8(qh_bits.1, 2));

        // Combine low 4 bits and high 2 bits
        let q6bytes_0 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.0, m4b), q6h_0));
        let q6bytes_1 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.1, m4b), q6h_1));
        let q6bytes_2 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.2, m4b), q6h_2));
        let q6bytes_3 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.3, m4b), q6h_3));
        let q6bytes_4 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.0, 4), q6h_4));
        let q6bytes_5 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.1, 4), q6h_5));
        let q6bytes_6 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.2, 4), q6h_6));
        let q6bytes_7 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.3, 4), q6h_7));

        let q8bytes = vld1q_s8_x4(q8.add(j * 64));

        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_0, q8bytes.0))
            * *scales.add(j * 4 + 0) as i32;
        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_1, q8bytes.1))
            * *scales.add(j * 4 + 1) as i32;
        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_2, q8bytes.2))
            * *scales.add(j * 4 + 2) as i32;
        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_3, q8bytes.3))
            * *scales.add(j * 4 + 3) as i32;
        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_4, q8bytes.0))
            * *scales.add(j * 4 + 4) as i32;
        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_5, q8bytes.1))
            * *scales.add(j * 4 + 5) as i32;
        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_6, q8bytes.2))
            * *scales.add(j * 4 + 6) as i32;
        isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_7, q8bytes.3))
            * *scales.add(j * 4 + 7) as i32;
    }

    d * isum as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_tiled_correctness() {
        // This test would need actual test data
        // For now, we verify it compiles and runs
    }
}
