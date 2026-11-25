// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// K-extent GEMV implementation for decode (single token) optimization
// Based on llama.cpp's narrow path in tiled.cpp

use crate::types::fp16_to_fp32;
use crate::types::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K, QK_K};
use std::arch::aarch64::*;

/// Tile dimensions matching llama.cpp
const TILE_K: usize = 256; // one QK_K block
const MICRO: usize = 16; // microtile edge
const NB_QK: usize = 8; // subblocks per 256-K for Q4_K/Q5_K

/// Calculate optimal k_extent for narrow path (n_rows <= 16)
/// From llama.cpp: fit weights in 31KB to leave room for scales
#[inline]
fn narrow_k_extent(n_rows: usize, k: usize) -> usize {
    let l1 = 31 * 1024; // 31KB for weights, leaving room for scales
    let rows = if n_rows < MICRO { MICRO } else { n_rows };
    let mut ke = l1 / (MICRO + rows);
    ke &= !(TILE_K - 1); // floor to multiple of 256

    // Find a k_extent that divides k evenly
    while ke >= TILE_K && k % ke != 0 {
        ke -= TILE_K;
    }

    if ke >= TILE_K { ke } else { TILE_K }
}

/// K-extent GEMV for Q5_K x Q8_K (decode path)
/// Optimized for single token (n=1) with K-tiling for L1 cache reuse
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q5k_q8k_k_extent(
    weight_data: &[u8],
    weight_stride: usize, // bytes per row
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
) {
    let nb = input.len(); // number of K blocks
    let k = nb * QK_K;

    // Initialize output to zero (we'll accumulate)
    for i in 0..out_features {
        output[i] = 0.0;
    }

    // Calculate k_extent for decode (n=1)
    let k_extent = narrow_k_extent(1, k);
    let num_k = k_extent / TILE_K;

    // Process K dimension in k_extent chunks
    // Key: load k_extent weights once, compute all rows
    for k0 in (0..nb).step_by(num_k) {
        let k_end = (k0 + num_k).min(nb);
        let actual_num_k = k_end - k0;

        // Process each output row
        for i in 0..out_features {
            let mut sum = 0.0f32;

            // Dot product for this K chunk
            for slab in 0..actual_num_k {
                let row_offset = i * weight_stride;
                let blk_offset = (k0 + slab) * std::mem::size_of::<BlockQ5_K>();
                let total_offset = row_offset + blk_offset;

                if total_offset + std::mem::size_of::<BlockQ5_K>()
                    <= weight_data.len()
                {
                    let w_blk = &*(weight_data.as_ptr().add(total_offset)
                        as *const BlockQ5_K);
                    let in_blk = &input[k0 + slab];

                    // Manual dot product for single block
                    sum += dot_q5k_q8k_single_block(w_blk, in_blk);
                }
            }

            // Accumulate to output
            output[i] += sum;
        }
    }
}

/// Dot product for single Q5_K x Q8_K block
#[target_feature(enable = "neon,dotprod")]
unsafe fn dot_q5k_q8k_single_block(x: &BlockQ5_K, y: &BlockQ8_K) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

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

    // Calculate dot product
    let scales = utmp.as_ptr() as *const u8;
    let m4b = vdupq_n_u8(0xF);
    let mone = vdupq_n_u8(1);
    let mtwo = vdupq_n_u8(2);
    let mzero = vdupq_n_s32(0);

    let q5 = x.qs.as_ptr();
    let qh = x.qh.as_ptr();
    let q8 = y.qs.as_ptr();

    let mut qhbits = vld1q_u8_x2(qh);
    let mut sumi = 0i32;

    for j in 0..(QK_K / 64) {
        let q5bits = vld1q_u8_x2(q5.add(j * 32));
        let q8bytes = vld1q_s8_x4(q8.add(j * 64));

        // Extract high bits
        let q5h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
        let q5h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
        let q5h_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
        let q5h_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);

        // Shift qhbits for next iteration
        qhbits.0 = vshrq_n_u8(qhbits.0, 2);
        qhbits.1 = vshrq_n_u8(qhbits.1, 2);

        // Combine low nibbles with high bits
        let q5l_0 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.0, m4b), q5h_0));
        let q5l_1 =
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.1, m4b), q5h_1));
        let q5l_2 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.0, 4), q5h_2));
        let q5l_3 =
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.1, 4), q5h_3));

        // Manual SDOT using inline assembly
        let p0 = vdotq_s32_manual(mzero, q5l_0, q8bytes.0);
        let p1 = vdotq_s32_manual(p0, q5l_1, q8bytes.1);
        let p2 = vdotq_s32_manual(p1, q5l_2, q8bytes.2);
        let p3 = vdotq_s32_manual(p2, q5l_3, q8bytes.3);

        sumi += vaddvq_s32(p3) * *scales.add(j) as i32;
    }

    d * sumi as f32 - dmin * sumi_mins as f32
}

/// Manual vdotq_s32 implementation using SDOT instruction
#[allow(asm_sub_register)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn vdotq_s32_manual(
    acc: int32x4_t,
    a: int8x16_t,
    b: int8x16_t,
) -> int32x4_t {
    let mut result = acc;
    std::arch::asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, preserves_flags)
    );
    result
}

/// Simple GEMV fallback
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemv_q5k_q8k_simple(
    weights: &[BlockQ5_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
) {
    let nb = input.len();
    for i in 0..output.len() {
        let row_blocks: &[BlockQ5_K] =
            std::slice::from_raw_parts(weights.as_ptr().add(i * nb), nb);
        output[i] = vec_dot_q5k_q8k_partial(row_blocks, input);
    }
}

/// Partial vec_dot for K-chunk (single block for now)
#[target_feature(enable = "neon,dotprod")]
unsafe fn vec_dot_q5k_q8k_partial(x: &[BlockQ5_K], y: &[BlockQ8_K]) -> f32 {
    // Use existing vec_dot for simplicity
    // In a full implementation, this would handle partial K
    super::vec_dot::vec_dot_q5_k_q8_k(x, y)
}
