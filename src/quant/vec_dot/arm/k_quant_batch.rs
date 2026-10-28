// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Interleaved weight layout for CPU inference - matches llama.cpp exactly
//!
//! Key insight: llama.cpp processes 8 rows at once by interleaving weights
//! so that loading one cache line (64 bytes) gives data for all 8 rows.
//!
//! Structure: block_q4_Kx8 from llama.cpp
//! - d[8]: super-block scales for 8 rows
//! - dmin[8]: super-block mins for 8 rows
//! - scales[96]: 6-bit packed scales/mins
//! - qs[1024]: interleaved 4-bit quants
//!
//! Total: 1152 bytes = 8 × 144 bytes (standard Q4_K block size)

use super::fp::fp16_to_fp32;
use crate::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use std::arch::aarch64::*;

/// Q4_K interleaved block: 8 rows stored together
/// Layout matches llama.cpp's block_q4_Kx8 exactly
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4Kx8 {
    pub d: [u16; 8],      // Super-block scales for 8 rows (16 bytes)
    pub dmin: [u16; 8],   // Super-block mins for 8 rows (16 bytes)
    pub scales: [u8; 96], // Scales and mins, 6-bit packed (96 bytes)
    pub qs: [u8; 1024],   // 4-bit quants, interleaved (1024 bytes)
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4Kx8>() == 1152,
    "BlockQ4Kx8 size mismatch"
);

/// Repack 8 Q4_K blocks into interleaved format
/// Direct port of llama.cpp's make_block_q4_Kx8 from repack.cpp
pub fn repack_q4_k_to_interleaved(input: &[BlockQ4K; 8]) -> BlockQ4Kx8 {
    let mut out: BlockQ4Kx8 = unsafe { std::mem::zeroed() };

    // Copy d and dmin values
    for i in 0..8 {
        out.d[i] = input[i].d;
        out.dmin[i] = input[i].dmin;
    }

    // Interleave qs bytes - take 8 bytes from each row cyclically
    // This ensures loading 64 bytes gives data for all 8 rows
    const BLCK_SIZE_INTERLEAVE: usize = 8;
    const END: usize = QK_K * 4 / BLCK_SIZE_INTERLEAVE; // 128 iterations

    for i in 0..END {
        let src_id = i % 8;
        let src_offset = (i / 8) * BLCK_SIZE_INTERLEAVE;
        let dst_offset = i * BLCK_SIZE_INTERLEAVE;

        // Copy 8 bytes from input[src_id].qs[src_offset..src_offset+8]
        out.qs[dst_offset..dst_offset + BLCK_SIZE_INTERLEAVE].copy_from_slice(
            &input[src_id].qs[src_offset..src_offset + BLCK_SIZE_INTERLEAVE],
        );
    }

    // Repack scales and mins - 6-bit packed values
    // First 48 bytes: scales/mins for subblocks 0-3
    for i in 0..4 {
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];

        for j in 0..8 {
            s[j] = input[j].scales[i] & 63;
            m[j] = input[j].scales[i + 4] & 63;
        }

        out.scales[i * 12] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    // Next 48 bytes: scales/mins for subblocks 4-7
    for i in 0..4 {
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];

        for j in 0..8 {
            s[j] = ((input[j].scales[i] & 192) >> 2)
                | (input[j].scales[i + 8] & 15);
            m[j] = ((input[j].scales[i + 4] & 192) >> 2)
                | ((input[j].scales[i + 8] & 240) >> 4);
        }

        let base = i * 12 + 48;
        out.scales[base] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[base + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[base + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[base + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[base + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[base + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[base + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[base + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[base + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[base + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[base + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[base + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    out
}

/// Manual vdotq_s32 implementation using native SDOT instruction
#[allow(asm_sub_register)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn vdotq_s32_manual(
    a: int32x4_t,
    b: int8x16_t,
    c: int8x16_t,
) -> int32x4_t {
    let mut result = a;
    std::arch::asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) b,
        in(vreg) c,
        options(pure, nomem, preserves_flags)
    );
    result
}

/// Interleaved GEMV for Q4_K: process 8 rows at once
/// Direct port of llama.cpp's ggml_gemv_q4_K_8x8_q8_K from arch/arm/repack.cpp
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q4_k_interleaved(
    weights: &[BlockQ4Kx8],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    num_rows: usize,
) {
    let nb = k / QK_K;
    const NCOLS_INTERLEAVED: usize = 8;
    const COL_PAIRS: usize = NCOLS_INTERLEAVED / 2;

    let m4b = vdupq_n_u8(0x0f);

    // Process 8 rows at a time
    for row_group_start in (0..num_rows).step_by(NCOLS_INTERLEAVED) {
        let x_group = row_group_start / NCOLS_INTERLEAVED;

        // Accumulators for 8 rows (2 groups of 4)
        let mut acc_f32 = [vdupq_n_f32(0.0f32); 2];

        // Process each block of K elements
        for b in 0..nb {
            let w_block = &weights[x_group * nb + b];
            let x_block = &x[b];

            // Load super-block scales (convert u16 f16 bits to f32)
            // Use scalar conversion and then build vectors
            let d_f32: [f32; 8] =
                std::array::from_fn(|i| fp16_to_fp32(w_block.d[i]));
            let dmin_f32: [f32; 8] =
                std::array::from_fn(|i| fp16_to_fp32(w_block.dmin[i]));

            let q8_d = x_block.d;

            // Build scale vectors for rows 0-3 and 4-7
            let sb_scale_0 = vmulq_n_f32(vld1q_f32(d_f32.as_ptr()), q8_d);
            let sb_scale_1 =
                vmulq_n_f32(vld1q_f32(d_f32.as_ptr().add(4)), q8_d);

            let sb_min_0 = vmulq_n_f32(vld1q_f32(dmin_f32.as_ptr()), q8_d);
            let sb_min_1 =
                vmulq_n_f32(vld1q_f32(dmin_f32.as_ptr().add(4)), q8_d);

            // Bias accumulators
            let mut bias_acc = [vdupq_n_s32(0); 2];

            // Compute bsums
            let bsums = vpaddq_s16(
                vld1q_s16(x_block.bsums.as_ptr()),
                vld1q_s16(x_block.bsums.as_ptr().add(8)),
            );
            let bsums_arr: [i16; 8] = std::mem::transmute(bsums);

            // Process 4 subblocks (each 64 elements)
            for sb in 0..4 {
                let mut acc_lo = [vdupq_n_s32(0); COL_PAIRS];
                let mut acc_hi = [vdupq_n_s32(0); COL_PAIRS];

                // Decode scales and mins for this subblock
                // The scales are 6-bit packed in a complex format
                let offset = sb * 24;
                let mut aux_scales = [0u8; 16];
                let mut aux_mins = [0u8; 16];

                // Decode scales for the 8 columns
                // First 12 bytes: scales/mins for first 4 columns
                // Next 12 bytes: scales/mins for next 4 columns
                for col_set in 0..2 {
                    let scale_ptr =
                        w_block.scales.as_ptr().add(offset + col_set * 12);

                    // Read 12 bytes as 3 u32
                    let mut aux = [0u32; 3]; // Only 3 u32 needed for 12 bytes
                    std::ptr::copy_nonoverlapping(
                        scale_ptr,
                        aux.as_mut_ptr() as *mut u8,
                        12,
                    );

                    // Decode using llama.cpp's formula
                    const KMASK1: u32 = 0x3f3f3f3f;
                    const KMASK2: u32 = 0x0f0f0f0f;
                    const KMASK3: u32 = 0x03030303;

                    let scales_0_3 = aux[0] & KMASK1;
                    let scales_4_7 =
                        (aux[2] & KMASK2) | (((aux[0] >> 6) & KMASK3) << 4);

                    // Store scales as u8 (they're 6-bit unsigned, stored in low 6 bits)
                    std::ptr::copy_nonoverlapping(
                        &scales_0_3 as *const u32 as *const u8,
                        aux_scales.as_mut_ptr().add(col_set * 8),
                        4,
                    );
                    std::ptr::copy_nonoverlapping(
                        &scales_4_7 as *const u32 as *const u8,
                        aux_scales.as_mut_ptr().add(col_set * 8 + 4),
                        4,
                    );

                    // Decode mins (from llama.cpp: mins_0_3 = sm[1] & kmask1, mins_4_7 uses sm[2])
                    let mins_0_3 = aux[1] & KMASK1;
                    let mins_4_7 = ((aux[2] >> 4) & KMASK2)
                        | (((aux[1] >> 6) & KMASK3) << 4);

                    std::ptr::copy_nonoverlapping(
                        &mins_0_3 as *const u32 as *const u8,
                        aux_mins.as_mut_ptr().add(col_set * 8),
                        4,
                    );
                    std::ptr::copy_nonoverlapping(
                        &mins_4_7 as *const u32 as *const u8,
                        aux_mins.as_mut_ptr().add(col_set * 8 + 4),
                        4,
                    );
                }

                // Widen scales to int16x8
                let q4sb_scales_0 =
                    vmovl_s8(vld1_s8(aux_scales.as_ptr() as *const i8));
                let q4sb_scales_1 =
                    vmovl_s8(vld1_s8(aux_scales.as_ptr().add(8) as *const i8));

                // Widen mins to int16x8
                let q4sb_mins_0 =
                    vreinterpretq_s16_u16(vmovl_u8(vld1_u8(aux_mins.as_ptr())));
                let q4sb_mins_1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(
                    aux_mins.as_ptr().add(8),
                )));

                // Load Q4 quants base
                let q4_base = w_block.qs.as_ptr().add(sb * QK_K);
                let q8_base = x_block.qs.as_ptr().add(sb * 64);

                // Load Q8 values (duplicated for interleaved columns)
                let mut q8_qs: [int8x16_t; 8] = [vdupq_n_s8(0); 8];
                for i in 0..8 {
                    // Load 8 int8 values and duplicate across 16 lanes
                    let val = std::ptr::read(q8_base.add(i * 8) as *const u64);
                    q8_qs[i] = vreinterpretq_s8_u8(vreinterpretq_u8_u64(
                        vdupq_n_u64(val),
                    ));
                }

                // Process column pairs (0-1, 2-3, 4-5, 6-7)
                for cp in 0..COL_PAIRS {
                    let q4_qs_0 = vld1q_u8(q4_base.add(16 * cp));
                    let q4_qs_1 = vld1q_u8(q4_base.add(16 * cp + 64));
                    let q4_qs_2 = vld1q_u8(q4_base.add(16 * cp + 128));
                    let q4_qs_3 = vld1q_u8(q4_base.add(16 * cp + 192));

                    // Low nibbles
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_qs_0, m4b)),
                        q8_qs[0],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_qs_1, m4b)),
                        q8_qs[1],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_qs_2, m4b)),
                        q8_qs[2],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_qs_3, m4b)),
                        q8_qs[3],
                    );

                    // High nibbles
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_0, 4)),
                        q8_qs[4],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_1, 4)),
                        q8_qs[5],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_2, 4)),
                        q8_qs[6],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_3, 4)),
                        q8_qs[7],
                    );
                }

                // Apply scales and accumulate
                // p = 0 -> rows 0123, p = 1 -> rows 4567
                for p in 0..2 {
                    let group_scales_lo = if p == 0 {
                        vget_low_s16(q4sb_scales_0)
                    } else {
                        vget_high_s16(q4sb_scales_0)
                    };
                    let group_scales_hi = if p == 0 {
                        vget_low_s16(q4sb_scales_1)
                    } else {
                        vget_high_s16(q4sb_scales_1)
                    };
                    let sb_scale = if p == 0 { sb_scale_0 } else { sb_scale_1 };

                    let sumf_0 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(group_scales_lo),
                        vpaddq_s32(acc_lo[p * 2], acc_lo[p * 2 + 1]),
                    ));
                    acc_f32[p] = vfmaq_f32(acc_f32[p], sb_scale, sumf_0);

                    let sumf_1 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(group_scales_hi),
                        vpaddq_s32(acc_hi[p * 2], acc_hi[p * 2 + 1]),
                    ));
                    acc_f32[p] = vfmaq_f32(acc_f32[p], sb_scale, sumf_1);
                }

                // Bias correction
                let bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb]);
                let bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);

                bias_acc[0] = vmlal_s16(
                    bias_acc[0],
                    bsums_vec_lo,
                    vget_low_s16(q4sb_mins_0),
                );
                bias_acc[0] = vmlal_s16(
                    bias_acc[0],
                    bsums_vec_hi,
                    vget_low_s16(q4sb_mins_1),
                );
                bias_acc[1] = vmlal_s16(
                    bias_acc[1],
                    bsums_vec_lo,
                    vget_high_s16(q4sb_mins_0),
                );
                bias_acc[1] = vmlal_s16(
                    bias_acc[1],
                    bsums_vec_hi,
                    vget_high_s16(q4sb_mins_1),
                );
            }

            // Apply bias correction
            acc_f32[0] =
                vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0);
            acc_f32[1] =
                vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_1);
        }

        // Store results
        vst1q_f32(dst.as_mut_ptr().add(row_group_start), acc_f32[0]);
        vst1q_f32(dst.as_mut_ptr().add(row_group_start + 4), acc_f32[1]);
    }
}

// Type alias for backward compatibility
pub type BlockQ4KInterleaved = BlockQ4Kx8;

/// Repack Q4_K weights to interleaved format (batch version)
pub fn repack_q4_k_batch(
    src: &[BlockQ4K],
    n_rows: usize,
    n_blocks_per_row: usize,
) -> Vec<BlockQ4Kx8> {
    assert!(n_rows % 8 == 0, "Rows must be multiple of 8 for interleaving");

    let n_interleaved = (n_rows / 8) * n_blocks_per_row;
    let mut dst = Vec::with_capacity(n_interleaved);

    for row_group in (0..n_rows).step_by(8) {
        for block_idx in 0..n_blocks_per_row {
            // Gather 8 blocks from 8 rows
            let mut blocks: [BlockQ4K; 8] = unsafe { std::mem::zeroed() };
            for row_in_group in 0..8 {
                let src_idx =
                    (row_group + row_in_group) * n_blocks_per_row + block_idx;
                blocks[row_in_group] = src[src_idx];
            }
            dst.push(repack_q4_k_to_interleaved(&blocks));
        }
    }

    dst
}

/// Interleaved GEMV using the new format
pub unsafe fn vec_dot_q4_k_q8_k_interleaved(
    weights_interleaved: &[BlockQ4Kx8],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    num_rows: usize,
) {
    gemv_q4_k_interleaved(weights_interleaved, x, dst, k, num_rows);
}
