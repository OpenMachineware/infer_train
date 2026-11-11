// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Interleaved weight layout for CPU inference - matches llama.cpp exactly
//!
//! Key insight: llama.cpp processes 8 rows at once by interleaving weights
//! so that loading one cache line (64 bytes) gives data for all 8 rows.
//!
//! # Hardware Support
//!
//! | CPU | Feature | Kernel Style | Notes |
//! |-----|---------|--------------|-------|
//! | Apple M1 | dotprod | 8x8 (vld1q_dup_s64 + vdotq_s32) | Works on M1 |
//! | Apple M2/M3/M4 | i8mm | 8x8 (same as M1) | i8mm not needed for K-quant |
//! | x86 AVX2 | AVX2 | Not implemented yet | TODO: AVX2 intrinsics |
//!
//! Note: For K-quant formats, the "8x8" kernel style uses `vld1q_dup_s64` to load
//! 8 bytes and duplicate to 16 bytes, then `vdotq_s32` for dot product. This works
//! on both M1 (dotprod) and M2+ (i8mm). The i8mm SMMLA instruction is only beneficial
//! for Q8_0 format, not K-quant.
//!
//! Structure: block_q4_Kx8 from llama.cpp
//! - d[8]: super-block scales for 8 rows
//! - dmin[8]: super-block mins for 8 rows
//! - scales[96]: 6-bit packed scales/mins
//! - qs[1024]: interleaved 4-bit quants
//!
//! Total: 1152 bytes = 8 × 144 bytes (standard Q4_K block size)

use super::fp::fp16_to_fp32;
use crate::quant::types::{
    BlockQ4_0, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K,
};
use std::arch::aarch64::*;

/// Q4_K interleaved block: 8 rows stored together
/// Layout matches llama.cpp's block_q4_Kx8 exactly
#[repr(C, align(64))]
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

/// Load 8 bytes and duplicate to fill 16-byte vector
/// Equivalent to vld1q_dup_s64 in C
/// Used for 8x8 kernel style - loads 8 Q8 values and duplicates
/// to process all 8 interleaved Q4 columns with the same Q8 data.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vld1q_dup_s64(ptr: *const i8) -> int8x16_t {
    let val = vld1_s64(ptr as *const i64);
    vreinterpretq_s8_s64(vcombine_s64(val, val))
}

/// Prefetch data to L1 cache (keep in cache)
/// Uses ARM64 PRFM PLDL1KEEP instruction
#[target_feature(enable = "neon")]
#[inline]
unsafe fn prefetch_l1_keep(ptr: *const u8) {
    std::arch::asm!(
        "prfm pldl1keep, [{0}]",
        in(reg) ptr,
        options(nomem, preserves_flags, nostack)
    );
}

/// Prefetch data to L2 cache (streaming access)
/// Uses ARM64 PRFM PLDL2STRM instruction
#[target_feature(enable = "neon")]
#[inline]
unsafe fn prefetch_l2_stream(ptr: *const u8) {
    std::arch::asm!(
        "prfm pldl2strm, [{0}]",
        in(reg) ptr,
        options(nomem, preserves_flags, nostack)
    );
}

/// Interleaved GEMV for Q4_K: process 8 rows at once
/// Direct port of llama.cpp's ggml_gemv_q4_K_8x8_q8_K from arch/arm/repack.cpp
/// Uses the 8x8 kernel style with duplicated Q8 values (compatible with M1 and M2+)
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

            // Prefetch next block's weight and activation data
            // Prefetch distance of 1 block (256 elements = 1 KB of weights)
            if b + 1 < nb {
                let w_next = &weights[x_group * nb + b + 1];
                let x_next = &x[b + 1];
                // Prefetch weight qs data (main data, large)
                prefetch_l2_stream(w_next.qs.as_ptr());
                // Prefetch Q8 activation data
                prefetch_l1_keep(x_next.qs.as_ptr() as *const u8);
            }

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

                // Load Q8 values - DUPLICATE 8 bytes into 16-byte vector (8x8 style)
                // Each q8_qs[i] holds 8 Q8 values duplicated to 16 bytes
                // This allows using vdotq_s32 instead of vdotq_laneq_s32
                let mut q8_qs: [int8x16_t; 8] = [vdupq_n_s8(0); 8];
                for i in 0..8 {
                    q8_qs[i] = vld1q_dup_s64(q8_base.add(i * 8));
                }

                // Process all 4 column pairs
                // cp=0: columns 0-1, cp=1: columns 2-3, cp=2: columns 4-5, cp=3: columns 6-7
                for cp in 0..COL_PAIRS {
                    // Determine qs offsets based on column pair
                    let qs_offset = 16 * cp;

                    // Load qs from 4 offsets (0, 64, 128, 192) + qs_offset
                    // Each covers 16 bytes = 32 nibbles = 32 elements for low/high
                    let q4_0 = vld1q_u8(q4_base.add(qs_offset));
                    let q4_1 = vld1q_u8(q4_base.add(qs_offset + 64));
                    let q4_2 = vld1q_u8(q4_base.add(qs_offset + 128));
                    let q4_3 = vld1q_u8(q4_base.add(qs_offset + 192));

                    // Low nibbles: (qs & 0x0f)
                    // Process with q8_qs[0..3] for elements 0-31 of subblock
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_0, m4b)),
                        q8_qs[0],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_1, m4b)),
                        q8_qs[1],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_2, m4b)),
                        q8_qs[2],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vandq_u8(q4_3, m4b)),
                        q8_qs[3],
                    );

                    // High nibbles: (qs >> 4)
                    // Process with q8_qs[4..7] for elements 32-63 of subblock
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_0, 4)),
                        q8_qs[4],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_1, 4)),
                        q8_qs[5],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_2, 4)),
                        q8_qs[6],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_3, 4)),
                        q8_qs[7],
                    );
                }

                // Apply scales and accumulate
                // Combine results from pairs of column pairs (matching llama.cpp's vpaddq_s32 pattern)
                // i=0: combine cp=0 and cp=1 (columns 0-3)
                // i=1: combine cp=2 and cp=3 (columns 4-7)
                for i in 0..2 {
                    let p = i * 2; // p=0 or p=2
                    let group_scales_lo = if i == 0 {
                        vget_low_s16(q4sb_scales_0)
                    } else {
                        vget_high_s16(q4sb_scales_0)
                    };
                    let group_scales_hi = if i == 0 {
                        vget_low_s16(q4sb_scales_1)
                    } else {
                        vget_high_s16(q4sb_scales_1)
                    };
                    let sb_scale = if i == 0 { sb_scale_0 } else { sb_scale_1 };

                    // Pair add: combine results from two column pairs
                    let sum_lo = vpaddq_s32(acc_lo[p], acc_lo[p + 1]);
                    let sum_hi = vpaddq_s32(acc_hi[p], acc_hi[p + 1]);

                    let sumf_0 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(group_scales_lo),
                        sum_lo,
                    ));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale, sumf_0);

                    let sumf_1 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(group_scales_hi),
                        sum_hi,
                    ));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale, sumf_1);
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

/// Q5_K interleaved block: 8 rows stored together
/// Layout matches llama.cpp's block_q5_Kx8
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ5Kx8 {
    pub d: [u16; 8],      // Super-block scales for 8 rows (16 bytes)
    pub dmin: [u16; 8],   // Super-block mins for 8 rows (16 bytes)
    pub scales: [u8; 96], // Scales and mins, 6-bit packed (96 bytes)
    pub qh: [u8; 256],    // High bits, interleaved (256 bytes)
    pub qs: [u8; 1024],   // Low 4-bit quants, interleaved (1024 bytes)
}

const _: () = assert!(
    std::mem::size_of::<BlockQ5Kx8>() == 1408,
    "BlockQ5Kx8 size mismatch"
);

/// Repack 8 Q5_K blocks into interleaved format
/// Direct port of llama.cpp's make_block_q5_Kx8 from repack.cpp
pub fn repack_q5_k_to_interleaved(input: &[BlockQ5K; 8]) -> BlockQ5Kx8 {
    let mut out: BlockQ5Kx8 = unsafe { std::mem::zeroed() };

    // Copy d and dmin values
    for i in 0..8 {
        out.d[i] = input[i].d;
        out.dmin[i] = input[i].dmin;
    }

    // Interleave qs bytes - same as Q4_K
    const BLCK_SIZE_INTERLEAVE: usize = 8;
    const END: usize = QK_K * 4 / BLCK_SIZE_INTERLEAVE; // 128 iterations

    for i in 0..END {
        let src_id = i % 8;
        let src_offset = (i / 8) * BLCK_SIZE_INTERLEAVE;
        let dst_offset = i * BLCK_SIZE_INTERLEAVE;

        out.qs[dst_offset..dst_offset + BLCK_SIZE_INTERLEAVE].copy_from_slice(
            &input[src_id].qs[src_offset..src_offset + BLCK_SIZE_INTERLEAVE],
        );
    }

    // Interleave qh bytes - same pattern as qs, but with fewer iterations
    // qh is 32 bytes per row (1/4 of qs), so we need end/4 iterations
    const QH_END: usize = END / 4; // 32 iterations for qh

    for i in 0..QH_END {
        let src_id = i % 8;
        let src_offset = (i / 8) * BLCK_SIZE_INTERLEAVE;
        let dst_offset = i * BLCK_SIZE_INTERLEAVE;

        out.qh[dst_offset..dst_offset + BLCK_SIZE_INTERLEAVE].copy_from_slice(
            &input[src_id].qh[src_offset..src_offset + BLCK_SIZE_INTERLEAVE],
        );
    }

    // Repack scales and mins - same 6-bit packed format as Q4_K
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

/// Interleaved GEMV for Q5_K: process 8 rows at once
/// Direct port of llama.cpp's ggml_gemv_q5_K_8x8_q8_K from arch/arm/repack.cpp
/// Uses the 8x8 kernel style with duplicated Q8 values
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q5_k_interleaved(
    weights: &[BlockQ5Kx8],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    num_rows: usize,
) {
    let nb = k / QK_K;
    const NCOLS_INTERLEAVED: usize = 8;
    const COL_PAIRS: usize = 4; // 4 column pairs, each processes 2 rows

    let m4b = vdupq_n_u8(0x0f);
    let mone = vdupq_n_u8(1);
    let mtwo = vdupq_n_u8(2);

    // Process 8 rows at a time
    for row_group_start in (0..num_rows).step_by(NCOLS_INTERLEAVED) {
        let x_group = row_group_start / NCOLS_INTERLEAVED;

        // Accumulators for 2 groups of 4 rows each
        let mut acc_f32 = [vdupq_n_f32(0.0f32); 2];

        // Process each block of K elements
        for b in 0..nb {
            let w_block = &weights[x_group * nb + b];
            let x_block = &x[b];

            // Prefetch next block's weight and activation data
            if b + 1 < nb {
                let w_next = &weights[x_group * nb + b + 1];
                let x_next = &x[b + 1];
                prefetch_l2_stream(w_next.qs.as_ptr());
                prefetch_l1_keep(x_next.qs.as_ptr() as *const u8);
            }

            // Load super-block scales (convert u16 f16 bits to f32)
            // fp16_to_fp32 uses NEON fcvt internally
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

            // Compute bsums
            let bsums = vpaddq_s16(
                vld1q_s16(x_block.bsums.as_ptr()),
                vld1q_s16(x_block.bsums.as_ptr().add(8)),
            );
            let bsums_arr: [i16; 8] = std::mem::transmute(bsums);

            // Pre-load qh for all 4 column pairs
            let mut qh: [[uint8x16_t; 4]; COL_PAIRS] =
                [[vdupq_n_u8(0); 4]; COL_PAIRS];
            for cp in 0..COL_PAIRS {
                for i in 0..4 {
                    qh[cp][i] =
                        vld1q_u8(w_block.qh.as_ptr().add(i * 32 + 16 * cp));
                }
            }

            // Process 4 subblocks (each 64 elements)
            for sb in 0..4 {
                let mut acc_lo = [vdupq_n_s32(0); COL_PAIRS];
                let mut acc_hi = [vdupq_n_s32(0); COL_PAIRS];

                // Decode scales and mins for this subblock
                let offset = sb * 24;
                let mut aux_scales = [0u8; 16];
                let mut aux_mins = [0u8; 16];

                for col_set in 0..2 {
                    let scale_ptr =
                        w_block.scales.as_ptr().add(offset + col_set * 12);

                    let mut aux = [0u32; 3];
                    std::ptr::copy_nonoverlapping(
                        scale_ptr,
                        aux.as_mut_ptr() as *mut u8,
                        12,
                    );

                    const KMASK1: u32 = 0x3f3f3f3f;
                    const KMASK2: u32 = 0x0f0f0f0f;
                    const KMASK3: u32 = 0x03030303;

                    let scales_0_3 = aux[0] & KMASK1;
                    let scales_4_7 =
                        (aux[2] & KMASK2) | (((aux[0] >> 6) & KMASK3) << 4);

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

                let q5sb_scales_0 =
                    vmovl_s8(vld1_s8(aux_scales.as_ptr() as *const i8));
                let q5sb_scales_1 =
                    vmovl_s8(vld1_s8(aux_scales.as_ptr().add(8) as *const i8));
                let q5sb_mins_0 =
                    vreinterpretq_s16_u16(vmovl_u8(vld1_u8(aux_mins.as_ptr())));
                let q5sb_mins_1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(
                    aux_mins.as_ptr().add(8),
                )));

                // Load Q5 quants base for this subblock
                let qs_base = w_block.qs.as_ptr().add(sb * QK_K);

                // Load Q8 values - DUPLICATE 8 bytes into 16-byte vector
                let q8_base = x_block.qs.as_ptr().add(sb * 64);
                let mut q8_qs: [int8x16_t; 8] = [vdupq_n_s8(0); 8];
                for i in 0..8 {
                    q8_qs[i] = vld1q_dup_s64(q8_base.add(i * 8));
                }

                // Process all 4 column pairs
                for cp in 0..COL_PAIRS {
                    // Determine qs offsets based on column pair
                    let qs_offset = 16 * cp;

                    // Load qs from 4 offsets (0, 64, 128, 192) + qs_offset
                    let qs_0 = vld1q_u8(qs_base.add(qs_offset));
                    let qs_1 = vld1q_u8(qs_base.add(qs_offset + 64));
                    let qs_2 = vld1q_u8(qs_base.add(qs_offset + 128));
                    let qs_3 = vld1q_u8(qs_base.add(qs_offset + 192));

                    // Extract high bits from qh[cp]
                    let hbit_lo_0 = vandq_u8(qh[cp][0], mone);
                    let hbit_lo_1 = vandq_u8(qh[cp][1], mone);
                    let hbit_lo_2 = vandq_u8(qh[cp][2], mone);
                    let hbit_lo_3 = vandq_u8(qh[cp][3], mone);
                    let hbit_hi_0 = vshlq_n_u8(vandq_u8(qh[cp][0], mtwo), 3);
                    let hbit_hi_1 = vshlq_n_u8(vandq_u8(qh[cp][1], mtwo), 3);
                    let hbit_hi_2 = vshlq_n_u8(vandq_u8(qh[cp][2], mtwo), 3);
                    let hbit_hi_3 = vshlq_n_u8(vandq_u8(qh[cp][3], mtwo), 3);

                    // Shift qh for next subblock
                    qh[cp][0] = vshrq_n_u8(qh[cp][0], 2);
                    qh[cp][1] = vshrq_n_u8(qh[cp][1], 2);
                    qh[cp][2] = vshrq_n_u8(qh[cp][2], 2);
                    qh[cp][3] = vshrq_n_u8(qh[cp][3], 2);

                    // Low nibbles: (qs & 0x0f) | (hbit_lo << 4)
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(qs_0, m4b),
                            hbit_lo_0,
                            4,
                        )),
                        q8_qs[0],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(qs_1, m4b),
                            hbit_lo_1,
                            4,
                        )),
                        q8_qs[1],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(qs_2, m4b),
                            hbit_lo_2,
                            4,
                        )),
                        q8_qs[2],
                    );
                    acc_lo[cp] = vdotq_s32_manual(
                        acc_lo[cp],
                        vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(qs_3, m4b),
                            hbit_lo_3,
                            4,
                        )),
                        q8_qs[3],
                    );

                    // High nibbles: (qs >> 4) | hbit_hi
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8(qs_0, 4),
                            hbit_hi_0,
                        )),
                        q8_qs[4],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8(qs_1, 4),
                            hbit_hi_1,
                        )),
                        q8_qs[5],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8(qs_2, 4),
                            hbit_hi_2,
                        )),
                        q8_qs[6],
                    );
                    acc_hi[cp] = vdotq_s32_manual(
                        acc_hi[cp],
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8(qs_3, 4),
                            hbit_hi_3,
                        )),
                        q8_qs[7],
                    );
                }

                // Apply scales using pairwise add
                // p=0 -> rows 0123, p=2 -> rows 4567
                let bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb]);
                let bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);

                for (i, p) in [0, 2].iter().enumerate() {
                    let p = *p;
                    let group_scales_lo = if p == 0 {
                        vget_low_s16(q5sb_scales_0)
                    } else {
                        vget_high_s16(q5sb_scales_0)
                    };
                    let group_scales_hi = if p == 0 {
                        vget_low_s16(q5sb_scales_1)
                    } else {
                        vget_high_s16(q5sb_scales_1)
                    };
                    let group_mins_lo = if p == 0 {
                        vget_low_s16(q5sb_mins_0)
                    } else {
                        vget_high_s16(q5sb_mins_0)
                    };
                    let group_mins_hi = if p == 0 {
                        vget_low_s16(q5sb_mins_1)
                    } else {
                        vget_high_s16(q5sb_mins_1)
                    };
                    let sb_scale = if p == 0 { sb_scale_0 } else { sb_scale_1 };
                    let sb_min = if p == 0 { sb_min_0 } else { sb_min_1 };

                    // Pairwise add column pairs
                    let sumf_0 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(group_scales_lo),
                        vpaddq_s32(acc_lo[p], acc_lo[p + 1]),
                    ));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale, sumf_0);

                    let sumf_1 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(group_scales_hi),
                        vpaddq_s32(acc_hi[p], acc_hi[p + 1]),
                    ));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale, sumf_1);

                    // Fused bias
                    let bias = vmull_s16(bsums_vec_lo, group_mins_lo);
                    let bias = vmlal_s16(bias, bsums_vec_hi, group_mins_hi);
                    acc_f32[i] =
                        vmlsq_f32(acc_f32[i], sb_min, vcvtq_f32_s32(bias));
                }
            }
        }

        // Store results
        vst1q_f32(dst.as_mut_ptr().add(row_group_start), acc_f32[0]);
        vst1q_f32(dst.as_mut_ptr().add(row_group_start + 4), acc_f32[1]);
    }
}

/// Repack Q5_K weights to interleaved format (batch version)
pub fn repack_q5_k_batch(
    src: &[BlockQ5K],
    n_rows: usize,
    n_blocks_per_row: usize,
) -> Vec<BlockQ5Kx8> {
    assert!(n_rows % 8 == 0, "Rows must be multiple of 8 for interleaving");

    let n_interleaved = (n_rows / 8) * n_blocks_per_row;
    let mut dst = Vec::with_capacity(n_interleaved);

    for row_group in (0..n_rows).step_by(8) {
        for block_idx in 0..n_blocks_per_row {
            // Gather 8 blocks from 8 rows
            let mut blocks: [BlockQ5K; 8] = unsafe { std::mem::zeroed() };
            for row_in_group in 0..8 {
                let src_idx =
                    (row_group + row_in_group) * n_blocks_per_row + block_idx;
                blocks[row_in_group] = src[src_idx];
            }
            dst.push(repack_q5_k_to_interleaved(&blocks));
        }
    }

    dst
}

/// Interleaved GEMV using the new format
pub unsafe fn vec_dot_q5_k_q8_k_interleaved(
    weights_interleaved: &[BlockQ5Kx8],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    num_rows: usize,
) {
    gemv_q5_k_interleaved(weights_interleaved, x, dst, k, num_rows);
}

/// Q6_K interleaved block: 8 rows stored together
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ6Kx8 {
    pub d: [u16; 8],       // Super-block scales for 8 rows (16 bytes)
    pub scales: [i8; 128], // Scales: QK_K / 16 * 8 = 128 bytes
    pub ql: [u8; 1024],    // Lower 4 bits: QK_K / 2 * 8 = 1024 bytes
    pub qh: [u8; 512],     // Upper 2 bits: QK_K / 4 * 8 = 512 bytes
}

const _: () = assert!(
    std::mem::size_of::<BlockQ6Kx8>() == 1728,
    "BlockQ6Kx8 size mismatch"
);

/// Repack 8 Q6_K blocks into interleaved format
pub fn repack_q6_k_to_interleaved(input: &[BlockQ6K; 8]) -> BlockQ6Kx8 {
    let mut out: BlockQ6Kx8 = unsafe { std::mem::zeroed() };

    // Copy d values (scales)
    for i in 0..8 {
        out.d[i] = input[i].d;
    }

    // Copy scales - 16 bytes per row, 8 rows = 128 bytes
    for i in 0..8 {
        out.scales[i * 16..(i + 1) * 16].copy_from_slice(&input[i].scales);
    }

    // Interleave ql bytes - 128 bytes per row, 8 rows
    // Same pattern as other interleaved formats: 8 bytes from each row cyclically
    const QL_BLCK_SIZE: usize = 8;
    const QL_END: usize = 128 / QL_BLCK_SIZE * 8; // 16 iterations × 8 rows
    for i in 0..QL_END {
        let src_id = i % 8;
        let src_offset = (i / 8) * QL_BLCK_SIZE;
        let dst_offset = i * QL_BLCK_SIZE;
        out.ql[dst_offset..dst_offset + QL_BLCK_SIZE].copy_from_slice(
            &input[src_id].ql[src_offset..src_offset + QL_BLCK_SIZE],
        );
    }

    // Interleave qh bytes - 64 bytes per row, 8 rows
    const QH_BLCK_SIZE: usize = 8;
    const QH_END: usize = 64 / QH_BLCK_SIZE * 8; // 8 iterations × 8 rows
    for i in 0..QH_END {
        let src_id = i % 8;
        let src_offset = (i / 8) * QH_BLCK_SIZE;
        let dst_offset = i * QH_BLCK_SIZE;
        out.qh[dst_offset..dst_offset + QH_BLCK_SIZE].copy_from_slice(
            &input[src_id].qh[src_offset..src_offset + QH_BLCK_SIZE],
        );
    }

    out
}

// ===== IQ4_XS Interleaved Format =====

use crate::quant::types::BlockIQ4XS;

/// IQ4_XS interleaved block: 8 rows stored together
/// Each IQ4_XS block is 136 bytes:
/// - d: u16 (2 bytes)
/// - scales_h: u16 (2 bytes)
/// - scales_l: [u8; 4] (4 bytes)
/// - qs: [u8; 128] (128 bytes)
///
/// Interleaved layout (1088 bytes = 8 × 136):
/// - d: [u16; 8] (16 bytes)
/// - scales_h: [u16; 8] (16 bytes)
/// - scales_l: [u8; 32] (32 bytes)
/// - qs: [u8; 1024] (1024 bytes) - interleaved for cache locality
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct BlockIQ4XSx8 {
    pub d: [u16; 8], // Super-block scales for 8 rows (16 bytes)
    pub scales_h: [u16; 8], // High bits of scales for 8 rows (16 bytes)
    pub scales_l: [u8; 32], // Low bits of scales, 4 per row × 8 rows (32 bytes)
    pub qs: [u8; 1024], // 4-bit quants, interleaved (1024 bytes)
}

const _: () = assert!(
    std::mem::size_of::<BlockIQ4XSx8>() == 1088,
    "BlockIQ4XSx8 size mismatch"
);

/// Repack 8 IQ4_XS blocks into interleaved format
pub fn repack_iq4_xs_to_interleaved(input: &[BlockIQ4XS; 8]) -> BlockIQ4XSx8 {
    let mut out: BlockIQ4XSx8 = unsafe { std::mem::zeroed() };

    // Copy d values (scales)
    for i in 0..8 {
        out.d[i] = input[i].d;
    }

    // Copy scales_h values
    for i in 0..8 {
        out.scales_h[i] = input[i].scales_h;
    }

    // Copy scales_l values - 4 bytes per block, 8 blocks = 32 bytes
    for i in 0..8 {
        out.scales_l[i * 4..(i + 1) * 4].copy_from_slice(&input[i].scales_l);
    }

    // Interleave qs bytes - same pattern as Q4_K
    // Take 8 bytes from each row cyclically
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

    out
}

/// Interleaved GEMV for IQ4_XS × Q8_K
/// Processes 8 rows at a time for better cache locality
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_iq4_xs_interleaved(
    weights: &[BlockIQ4XSx8],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    num_rows: usize,
) {
    let nb = k / QK_K;
    const NCOLS_INTERLEAVED: usize = 8;

    // Load kvalues lookup table
    let values = vld1q_s8(crate::quant::kv_cache::KVALUES_IQ4NL_I8.as_ptr());

    // Process 8 rows at a time
    for row_group_start in (0..num_rows).step_by(NCOLS_INTERLEAVED) {
        let x_group = row_group_start / NCOLS_INTERLEAVED;

        // Accumulators for 8 rows (as i64 to avoid overflow)
        let mut sumi = [0i64; 8];

        // Process each block of K elements
        for b in 0..nb {
            let w_block = &weights[x_group * nb + b];
            let x_block = &x[b];

            // Prefetch next block's weight and activation data
            if b + 1 < nb {
                let w_next = &weights[x_group * nb + b + 1];
                let x_next = &x[b + 1];
                prefetch_l2_stream(w_next.qs.as_ptr());
                prefetch_l1_keep(x_next.qs.as_ptr() as *const u8);
            }

            // Load super-block scales
            let d_f32: [f32; 8] =
                std::array::from_fn(|i| fp16_to_fp32(w_block.d[i]));
            let x_d = x_block.d;

            // Extract scales for 8 rows for all 4 subblocks
            // Each subblock has 2 scales (ls1, ls2), each covering 32 elements
            let mut row_scales = [[0i32; 8]; 8]; // [scale_idx][row]

            for r in 0..8 {
                let scales_l = &w_block.scales_l[r * 4..r * 4 + 4];
                let mut h = w_block.scales_h[r];

                for sb in 0..4 {
                    let ls1 = ((scales_l[sb] & 0x0F) as i32
                        | (((h as i32) << 4) & 0x30))
                        - 32;
                    let ls2 = (((scales_l[sb] >> 4) as i32)
                        | (((h as i32) << 2) & 0x30))
                        - 32;
                    h >>= 4;

                    row_scales[sb * 2][r] = ls1;
                    row_scales[sb * 2 + 1][r] = ls2;
                }
            }

            // Process 4 subblocks (each 64 elements = 32 packed bytes)
            for sb in 0..4 {
                // Load Q8 values for this subblock (64 elements)
                let q8_base = x_block.qs.as_ptr().add(sb * 64);
                let q8_0 = vld1q_s8(q8_base);
                let q8_1 = vld1q_s8(q8_base.add(16));
                let q8_2 = vld1q_s8(q8_base.add(32));
                let q8_3 = vld1q_s8(q8_base.add(48));

                // Interleaved layout: qs is organized as:
                // For iteration i (0..127):
                //   dst[i*8..i*8+8] = input[src_id][src_offset..src_offset+8]
                // where src_id = i % 8, src_offset = (i / 8) * 8
                //
                // For subblock sb, the qs bytes for row r are at:
                // - In original row: offset sb*32 + j*8 for j in 0..3 (32 bytes total)
                // - In interleaved: offset where i = sb*32 + j with i % 8 == r
                //   So j = (r - sb*32 % 8 + k*8) % 32 for k in 0..3
                //   dst_offset = i*8 = (sb*32 + j)*8

                // For each row, process 2 chunks of 16 bytes (32 elements) each
                // This matches the reference: 32 bytes = 64 elements per subblock
                // Each chunk: 16 bytes = 32 elements (16 from low + 16 from high nibbles)
                // Chunk 0: elements 0-31 (ls1), Chunk 1: elements 32-63 (ls2)
                for chunk in 0..2 {
                    // For iteration i: src_row = i % 8, src_offset = (i / 8) * 8
                    // To load 16 bytes for row r at offset o:
                    // We need two 8-byte loads: at offsets o and o+8
                    // i = (o/8)*8 + r and i = ((o+8)/8)*8 + r = (o/8+1)*8 + r

                    // For subblock sb, offset sb*32 + chunk*16:
                    // First 8 bytes: i = (sb*4 + chunk*2)*8 + r, dst = (sb*4 + chunk*2)*64 + r*8
                    // Next 8 bytes: i = (sb*4 + chunk*2 + 1)*8 + r, dst = (sb*4 + chunk*2 + 1)*64 + r*8

                    for r in 0..8 {
                        // Load 16 bytes of Q4 (2 × 8 bytes)
                        let q4_ptr_0 = w_block
                            .qs
                            .as_ptr()
                            .add((sb * 4 + chunk * 2) * 64 + r * 8);
                        let q4_ptr_1 = w_block
                            .qs
                            .as_ptr()
                            .add((sb * 4 + chunk * 2 + 1) * 64 + r * 8);

                        let q4_8bytes_0 = vld1_u8(q4_ptr_0);
                        let q4_8bytes_1 = vld1_u8(q4_ptr_1);

                        // Combine into 16-byte vector
                        let q4bits = vcombine_u8(q4_8bytes_0, q4_8bytes_1);

                        // Dequantize: lookup low and high nibbles
                        // Low nibbles: 16 elements, High nibbles: 16 elements
                        let q4b_0 = vqtbl1q_s8(
                            values,
                            vandq_u8(q4bits, vdupq_n_u8(0x0F)),
                        );
                        let q4b_1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits, 4));

                        // Load Q8 values
                        // chunk 0: need q8_0 and q8_1 (elements 0-31)
                        // chunk 1: need q8_2 and q8_3 (elements 32-63)
                        let (q8_a, q8_b) = match chunk {
                            0 => (q8_0, q8_1),
                            _ => (q8_2, q8_3),
                        };

                        // Dot products
                        let dot_0 = vaddvq_s32(vdotq_s32_manual(
                            vdupq_n_s32(0),
                            q4b_0,
                            q8_a,
                        ));
                        let dot_1 = vaddvq_s32(vdotq_s32_manual(
                            vdupq_n_s32(0),
                            q4b_1,
                            q8_b,
                        ));

                        // Scale: chunk 0 uses ls1, chunk 1 uses ls2
                        let scale_idx = sb * 2 + chunk;
                        sumi[r] += (dot_0 + dot_1) as i64
                            * row_scales[scale_idx][r] as i64;
                    }
                }
            }

            // Apply scales and store results
            for r in 0..8 {
                let row_idx = row_group_start + r;
                if row_idx < num_rows {
                    let scale = d_f32[r] * x_d;
                    dst[row_idx] += scale * sumi[r] as f32;
                }
            }
        }
    }
}

/// Interleaved GEMV for Q6_K × Q8_K
/// Processes 8 rows at a time for better cache locality
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q6_k_interleaved(
    weights: &[BlockQ6Kx8],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    num_rows: usize,
) {
    let nb = k / QK_K;
    const NCOLS_INTERLEAVED: usize = 8;

    let m4b = vdupq_n_u8(0x0F);
    let mone = vdupq_n_u8(0x30);

    // Process 8 rows at a time
    for row_group_start in (0..num_rows).step_by(NCOLS_INTERLEAVED) {
        let x_group = row_group_start / NCOLS_INTERLEAVED;

        // Accumulators for 8 rows
        let mut sumi = [0i64; 8];

        // Process each block of K elements
        for b in 0..nb {
            let w_block = &weights[x_group * nb + b];
            let x_block = &x[b];

            // Prefetch next block's weight and activation data
            if b + 1 < nb {
                let w_next = &weights[x_group * nb + b + 1];
                let x_next = &x[b + 1];
                prefetch_l2_stream(w_next.ql.as_ptr());
                prefetch_l1_keep(x_next.qs.as_ptr() as *const u8);
            }

            // Load super-block scales
            let d_f32: [f32; 8] =
                std::array::from_fn(|i| fp16_to_fp32(w_block.d[i]));
            let x_d = x_block.d;

            // Process 2 iterations of 128 elements each (matching reference)
            for j in 0..2 {
                // Q8 values for this iteration (128 elements)
                // Reference uses: q8[j*64..j*64+63] and q8[j*64+32..j*64+95]
                // We need to use the SAME Q8 data for all rows (shared x_block)
                let q8_base = x_block.qs.as_ptr().add(j * 64);

                // For each row, process all 8 scales for this iteration
                for r in 0..8 {
                    // Interleaved layout:
                    // ql[i*8..i*8+8] = row i%8's ql[(i/8)*8..(i/8)*8+8]
                    // For iteration j: need rows' ql[j*64..j*64+64]
                    // Chunk c: row r's ql[c*8..c*8+8] is at offset (j*8 + c)*64 + r*8
                    // Simplified: offset = j*512 + c*64 + r*8

                    let mut ql_bytes = [0u8; 64];
                    for c in 0..8 {
                        let offset = j * 512 + c * 64 + r * 8;
                        std::ptr::copy_nonoverlapping(
                            w_block.ql.as_ptr().add(offset),
                            ql_bytes.as_mut_ptr().add(c * 8),
                            8,
                        );
                    }

                    // qh layout:
                    // qh[i*8..i*8+8] = row i%8's qh[(i/8)*8..(i/8)*8+8]
                    // For iteration j: need rows' qh[j*32..j*32+32]
                    // Chunk c: row r's qh[c*8..c*8+8] is at offset (j*4 + c)*64 + r*8
                    // Simplified: offset = j*256 + c*64 + r*8

                    let mut qh_bytes = [0u8; 32];
                    for c in 0..4 {
                        let offset = j * 256 + c * 64 + r * 8;
                        std::ptr::copy_nonoverlapping(
                            w_block.qh.as_ptr().add(offset),
                            qh_bytes.as_mut_ptr().add(c * 8),
                            8,
                        );
                    }

                    // Load ql and qh into NEON vectors
                    let ql_bits = [
                        vld1q_u8(ql_bytes.as_ptr()),
                        vld1q_u8(ql_bytes.as_ptr().add(16)),
                        vld1q_u8(ql_bytes.as_ptr().add(32)),
                        vld1q_u8(ql_bytes.as_ptr().add(48)),
                    ];

                    let qh_bits = [
                        vld1q_u8(qh_bytes.as_ptr()),
                        vld1q_u8(qh_bytes.as_ptr().add(16)),
                    ];

                    // Extract high 2 bits for each of 8 blocks (matching reference)
                    let q6h_0 = vshlq_n_u8(vandq_u8(mone, qh_bits[0]), 4);
                    let q6h_1 = vshlq_n_u8(vandq_u8(mone, qh_bits[1]), 4);
                    let q6h_2 = vshlq_n_u8(
                        vandq_u8(mone, vshrq_n_u8(qh_bits[0], 2)),
                        4,
                    );
                    let q6h_3 = vshlq_n_u8(
                        vandq_u8(mone, vshrq_n_u8(qh_bits[1], 2)),
                        4,
                    );
                    let q6h_4 = vshlq_n_u8(
                        vandq_u8(mone, vshrq_n_u8(qh_bits[0], 4)),
                        4,
                    );
                    let q6h_5 = vshlq_n_u8(
                        vandq_u8(mone, vshrq_n_u8(qh_bits[1], 4)),
                        4,
                    );
                    let q6h_6 = vshlq_n_u8(
                        vandq_u8(mone, vshrq_n_u8(qh_bits[0], 6)),
                        4,
                    );
                    let q6h_7 = vshlq_n_u8(
                        vandq_u8(mone, vshrq_n_u8(qh_bits[1], 6)),
                        4,
                    );

                    // Combine low 4 bits and high 2 bits (matching reference)
                    let q6b_0 = vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(ql_bits[0], m4b),
                        q6h_0,
                    ));
                    let q6b_1 = vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(ql_bits[1], m4b),
                        q6h_1,
                    ));
                    let q6b_2 = vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(ql_bits[2], m4b),
                        q6h_2,
                    ));
                    let q6b_3 = vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(ql_bits[3], m4b),
                        q6h_3,
                    ));
                    let q6b_4 = vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(ql_bits[0], 4),
                        q6h_4,
                    ));
                    let q6b_5 = vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(ql_bits[1], 4),
                        q6h_5,
                    ));
                    let q6b_6 = vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(ql_bits[2], 4),
                        q6h_6,
                    ));
                    let q6b_7 = vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8(ql_bits[3], 4),
                        q6h_7,
                    ));

                    // Load Q8 for this iteration
                    let q8bytes = vld1q_s8_x4(q8_base);
                    let q8bytes2 = vld1q_s8_x4(q8_base.add(32));

                    // Dot products with scales
                    let scale_base = j * 8;
                    let scales_r = &w_block.scales[r * 16..(r + 1) * 16];

                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_0,
                        q8bytes.0,
                    )) as i64
                        * scales_r[scale_base] as i64;
                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_1,
                        q8bytes.1,
                    )) as i64
                        * scales_r[scale_base + 1] as i64;
                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_2,
                        q8bytes.2,
                    )) as i64
                        * scales_r[scale_base + 2] as i64;
                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_3,
                        q8bytes.3,
                    )) as i64
                        * scales_r[scale_base + 3] as i64;
                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_4,
                        q8bytes2.0,
                    )) as i64
                        * scales_r[scale_base + 4] as i64;
                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_5,
                        q8bytes2.1,
                    )) as i64
                        * scales_r[scale_base + 5] as i64;
                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_6,
                        q8bytes2.2,
                    )) as i64
                        * scales_r[scale_base + 6] as i64;
                    sumi[r] += vaddvq_s32(vdotq_s32_manual(
                        vdupq_n_s32(0),
                        q6b_7,
                        q8bytes2.3,
                    )) as i64
                        * scales_r[scale_base + 7] as i64;
                }
            }

            // Apply scales and accumulate results
            for r in 0..8 {
                let row_idx = row_group_start + r;
                if row_idx < num_rows {
                    let scale = d_f32[r] * x_d;
                    dst[row_idx] += scale * sumi[r] as f32;
                }
            }
        }
    }
}

// ===== Q4_0 Interleaved Format =====

/// Q4_0 interleaved block: 8 rows stored together
/// Each Q4_0 block is 18 bytes:
/// - d: u16 (2 bytes)
/// - qs: [u8; 16] (16 bytes, 32 elements packed)
///
/// Interleaved layout (144 bytes = 8 × 18):
/// - d: [u16; 8] (16 bytes)
/// - qs: [u8; 128] (128 bytes) - interleaved for cache locality
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0x8 {
    pub d: [u16; 8],   // Scales for 8 rows (16 bytes)
    pub qs: [u8; 128], // Interleaved 4-bit quants (128 bytes)
}

const _: () = assert!(
    std::mem::size_of::<BlockQ4_0x8>() == 192,
    "BlockQ4_0x8 size mismatch"
);

/// Repack 8 Q4_0 blocks into interleaved format
pub fn repack_q4_0_to_interleaved(input: &[BlockQ4_0; 8]) -> BlockQ4_0x8 {
    let mut out: BlockQ4_0x8 = unsafe { std::mem::zeroed() };

    // Copy d values (scales)
    for i in 0..8 {
        out.d[i] = input[i].d;
    }

    // Interleave qs bytes - take 8 bytes from each row cyclically
    // This ensures loading 64 bytes gives data for all 8 rows
    // Each row has 16 bytes of qs, so 2 iterations per row
    const BLCK_SIZE_INTERLEAVE: usize = 8;

    for i in 0..16 {
        let src_id = i % 8;
        let src_offset = (i / 8) * BLCK_SIZE_INTERLEAVE;
        let dst_offset = i * BLCK_SIZE_INTERLEAVE;

        out.qs[dst_offset..dst_offset + BLCK_SIZE_INTERLEAVE].copy_from_slice(
            &input[src_id].qs[src_offset..src_offset + BLCK_SIZE_INTERLEAVE],
        );
    }

    out
}

/// Interleaved GEMV for Q4_0 × Q8_K
/// Processes 8 rows at a time for better cache locality
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q4_0_interleaved(
    weights: &[BlockQ4_0x8],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    num_rows: usize,
) {
    let nb = k / QK_K;
    const NCOLS_INTERLEAVED: usize = 8;

    let m4b = vdupq_n_u8(0x0F);
    let s8b = vdupq_n_s8(0x8);

    // Process 8 rows at a time
    for row_group_start in (0..num_rows).step_by(NCOLS_INTERLEAVED) {
        let x_group = row_group_start / NCOLS_INTERLEAVED;

        // Accumulators for 8 rows
        let mut sumi = [0i64; 8];

        // Process each block of K elements
        for b in 0..nb {
            let w_block = &weights[x_group * nb + b];
            let x_block = &x[b];

            // Prefetch next block's weight and activation data
            if b + 1 < nb {
                let w_next = &weights[x_group * nb + b + 1];
                let x_next = &x[b + 1];
                prefetch_l2_stream(w_next.qs.as_ptr());
                prefetch_l1_keep(x_next.qs.as_ptr() as *const u8);
            }

            // Load scales
            let d_f32: [f32; 8] =
                std::array::from_fn(|i| fp16_to_fp32(w_block.d[i]));
            let x_d = x_block.d;

            // Process 8 subblocks (32 elements each, 16 bytes of Q4)
            // Q4_0 has QK4_0 = 32 elements per block, QK_K = 256 elements
            // So QK_K / QK4_0 = 8 blocks per QK_K block
            for sb in 0..8 {
                // Q8 values for this subblock (32 elements)
                let q8_base = x_block.qs.as_ptr().add(sb * 32);
                let q8_0 = vld1q_s8(q8_base);
                let q8_1 = vld1q_s8(q8_base.add(16));

                // For each row, load interleaved Q4 data
                for r in 0..8 {
                    // Interleaved layout:
                    // qs[i*8..i*8+8] = row i%8's qs[(i/8)*8..(i/8)*8+8]
                    // For subblock sb: need row r's qs[sb*16..sb*16+16]
                    // Each subblock has 16 bytes = 2 × 8-byte chunks
                    // First chunk: i = sb*2, src_row = i%8, offset = (i/8)*8
                    // Second chunk: i = sb*2+1

                    let q4_ptr_0 =
                        w_block.qs.as_ptr().add((sb * 2) * 64 + r * 8);
                    let q4_ptr_1 =
                        w_block.qs.as_ptr().add((sb * 2 + 1) * 64 + r * 8);

                    let q4_0 = vld1_u8(q4_ptr_0);
                    let q4_1 = vld1_u8(q4_ptr_1);
                    let q4bits = vcombine_u8(q4_0, q4_1);

                    // Unpack 4-bit to 8-bit and subtract bias
                    let q4l = vreinterpretq_s8_u8(vandq_u8(q4bits, m4b));
                    let q4h = vreinterpretq_s8_u8(vshrq_n_u8(q4bits, 4));
                    let q4ls = vsubq_s8(q4l, s8b);
                    let q4hs = vsubq_s8(q4h, s8b);

                    // Dot product
                    let dot = vaddvq_s32(vdotq_s32_manual(
                        vdotq_s32_manual(vdupq_n_s32(0), q4ls, q8_0),
                        q4hs,
                        q8_1,
                    ));
                    sumi[r] += dot as i64;
                }
            }

            // Apply scales
            for r in 0..8 {
                let row_idx = row_group_start + r;
                if row_idx < num_rows {
                    let scale = d_f32[r] * x_d;
                    dst[row_idx] += scale * sumi[r] as f32;
                }
            }
        }
    }
}

/// Repack Q4_0 weights to interleaved format (batch version)
pub fn repack_q4_0_batch(
    src: &[BlockQ4_0],
    n_rows: usize,
    n_blocks_per_row: usize,
) -> Vec<BlockQ4_0x8> {
    assert!(n_rows % 8 == 0, "Rows must be multiple of 8 for interleaving");

    let n_interleaved = (n_rows / 8) * n_blocks_per_row;
    let mut dst = Vec::with_capacity(n_interleaved);

    for row_group in (0..n_rows).step_by(8) {
        for block_idx in 0..n_blocks_per_row {
            // Gather 8 blocks from 8 rows
            let mut blocks: [BlockQ4_0; 8] = unsafe { std::mem::zeroed() };
            for row_in_group in 0..8 {
                let src_idx =
                    (row_group + row_in_group) * n_blocks_per_row + block_idx;
                blocks[row_in_group] = src[src_idx];
            }
            dst.push(repack_q4_0_to_interleaved(&blocks));
        }
    }

    dst
}
