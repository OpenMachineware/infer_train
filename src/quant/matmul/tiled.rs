// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// Tiled GEMV/GEMM implementation matching llama.cpp architecture
//
// Key design from llama.cpp:
// 1. 256x256 macrotile, 16x16 microtile
// 2. Per-thread workspace (512KB) for tile buffers
// 3. Narrow path for decode (batch <= 16) with long contiguous reads
// 4. Accumulator buffer to avoid direct dst writes

use crate::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use std::arch::aarch64::*;
use std::arch::asm;

/// Manual vdotq_s32 implementation using native SDOT instruction
/// vdotq_s32 is unstable in Rust, so we use inline assembly
#[allow(asm_sub_register)]
#[target_feature(enable = "neon,dotprod")]
unsafe fn vdotq_s32_manual(
    acc: int32x4_t,
    a: int8x16_t,
    b: int8x16_t,
) -> int32x4_t {
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

/// Tile dimensions matching llama.cpp
pub const TILE_K: usize = 256; // one QK_K block
pub const TILE_ROWS: usize = 256; // max window rows
pub const MICRO: usize = 16; // microtile edge
pub const NB_Q5K: usize = 8; // subblocks per 256-K for Q5_K (32-wide subblocks)

/// Per-thread workspace for tiled matmul
/// Size: ~512KB per thread
#[repr(C, align(64))]
pub struct TiledWorkspace {
    /// Weight tile (src0): quantized weights unpacked to uint8
    pub src0_q: [u8; TILE_ROWS * TILE_K],
    /// Weight scales (per-subblock, 8 per 256-K for Q5_K)
    pub src0_scales: [i32; TILE_ROWS * NB_Q5K],
    /// Weight mins (per-subblock, for Q5_K)
    pub src0_mins: [i32; TILE_ROWS * NB_Q5K],
    /// Weight super-block scales (per 256-K)
    pub src0_d: [f32; TILE_ROWS],
    /// Weight super-block mins (per 256-K)
    pub src0_dmin: [f32; TILE_ROWS],
    /// Input tile (src1): Q8_K data unpacked
    pub src1_q: [i8; TILE_ROWS * TILE_K],
    /// Input scales (per 256-K)
    pub src1_d: [f32; TILE_ROWS],
    /// Input sums (per-16-block)
    pub src1_bsums: [i32; (TILE_K / 16) * TILE_ROWS],
    /// Accumulator buffer
    pub acc: [f32; TILE_ROWS * TILE_ROWS],
}

impl TiledWorkspace {
    pub fn new() -> Self {
        Self {
            src0_q: [0u8; TILE_ROWS * TILE_K],
            src0_scales: [0i32; TILE_ROWS * NB_Q5K],
            src0_mins: [0i32; TILE_ROWS * NB_Q5K],
            src0_d: [0.0f32; TILE_ROWS],
            src0_dmin: [0.0f32; TILE_ROWS],
            src1_q: [0i8; TILE_ROWS * TILE_K],
            src1_d: [0.0f32; TILE_ROWS],
            src1_bsums: [0i32; (TILE_K / 16) * TILE_ROWS],
            acc: [0.0f32; TILE_ROWS * TILE_ROWS],
        }
    }

    /// Clear accumulator
    #[inline]
    pub fn clear_acc(&mut self, n_rows: usize, n_cols: usize) {
        for i in 0..n_rows * n_cols {
            self.acc[i] = 0.0;
        }
    }
}

impl Default for TiledWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Global thread-local workspace pool
thread_local! {
    static WORKSPACE: std::cell::RefCell<TiledWorkspace> =
        std::cell::RefCell::new(TiledWorkspace::new());
}

/// Unpack Q5_K weights to tile format
/// Matches llama.cpp tiled_unpack_src0(block_q5_K)
#[target_feature(enable = "neon")]
pub unsafe fn unpack_src0_q5k(
    weights: &[BlockQ5K],
    row_start: usize,
    n_rows: usize,
    kblk_start: usize,
    num_k: usize,
    ws: &mut TiledWorkspace,
) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let nb = weights.len() / n_rows.max(1);

    for slab in 0..num_k {
        for r in 0..n_rows {
            let row_idx = row_start + r;
            let block_idx = row_idx * nb + kblk_start + slab;

            if block_idx >= weights.len() {
                continue;
            }

            let blk = &weights[block_idx];

            let d_off = slab * MICRO + r;
            let q_off = r * num_k * TILE_K + slab * TILE_K;
            let s_off = r * num_k * NB_Q5K + slab * NB_Q5K;

            // Unpack scales
            ws.src0_d[d_off] = blk.d as f32;
            ws.src0_dmin[d_off] = blk.dmin as f32;

            // Decode scales and mins from packed 12-byte format
            let mut utmp = [0u32; 4];
            std::ptr::copy_nonoverlapping(
                blk.scales.as_ptr(),
                utmp.as_mut_ptr() as *mut u8,
                12,
            );

            utmp[3] =
                ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let scales = &utmp[0] as *const u32 as *const u8;
            let mins = &utmp[2] as *const u32 as *const u8;

            for s in 0..NB_Q5K {
                ws.src0_scales[s_off + s] = *scales.add(s) as i32;
                ws.src0_mins[s_off + s] = *mins.add(s) as i32;
            }

            // Unpack 5-bit quantized values to 8-bit
            // Q5_K uses: qs (low 4 bits), qh (high bit)
            // Use NEON for vectorized unpack

            // Unpack 4-bit nibbles from qs
            for i in 0..4 {
                let src_offset = i * 32;
                let dst_lo_offset = i * 64;
                let dst_hi_offset = i * 64 + 32;

                // Load 32 bytes from qs
                let v = vld1q_u8(blk.qs.as_ptr().add(src_offset));

                // Extract low and high nibbles
                let lo = vandq_u8(v, vdupq_n_u8(0x0F));
                let hi = vshrq_n_u8(vandq_u8(v, vdupq_n_u8(0xF0)), 4);

                vst1q_u8(ws.src0_q.as_mut_ptr().add(q_off + dst_lo_offset), lo);
                vst1q_u8(ws.src0_q.as_mut_ptr().add(q_off + dst_hi_offset), hi);
            }

            // Add 5th bit from qh
            // qh[i] contains bit i for elements 2*i and 2*i+1
            for i in 0..4 {
                let qh_val = blk.qh[i];
                let base_offset = q_off + i * 64;

                for j in 0..32 {
                    let high_bit =
                        if (qh_val >> j) & 1 != 0 { 16u8 } else { 0u8 };

                    ws.src0_q[base_offset + j] |= high_bit;
                    ws.src0_q[base_offset + 32 + j] |= high_bit;
                }
            }
        }
    }
}

/// Unpack Q8_K input to tile format
#[target_feature(enable = "neon")]
pub unsafe fn unpack_src1_q8k(
    x: &[BlockQ8K],
    row_start: usize,
    n_rows: usize,
    kblk_start: usize,
    num_k: usize,
    ws: &mut TiledWorkspace,
) {
    let nb = if n_rows > 0 { x.len() / n_rows } else { x.len() };

    for slab in 0..num_k {
        for r in 0..n_rows.max(1) {
            let row_idx = row_start + r;
            let block_idx = if n_rows > 0 {
                row_idx * nb + kblk_start + slab
            } else {
                kblk_start + slab
            };

            if block_idx >= x.len() {
                continue;
            }

            let blk = &x[block_idx];

            let d_off = slab * MICRO + r;
            let q_off = r * num_k * TILE_K + slab * TILE_K;
            let bs_off = slab * MICRO + r * num_k * MICRO;

            ws.src1_d[d_off] = blk.d as f32;

            // Copy quantized values (already int8)
            std::ptr::copy_nonoverlapping(
                blk.qs.as_ptr() as *const i8,
                ws.src1_q.as_mut_ptr().add(q_off),
                QK_K,
            );

            // Copy block sums
            for i in 0..16 {
                ws.src1_bsums[bs_off + i] = blk.bsums[i] as i32;
            }
        }
    }
}

/// Run 16x16 microkernel for Q5_K x Q8_K
/// Computes: acc[i,j] += sum over subblocks of (scale_i * dot(q0_i, q1_j) - min_i * bsum_j) * d0_i * d1_j
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn run_microtile_q5k_q8k(
    ws: &TiledWorkspace,
    tile_i0: usize, // offset in src0 rows
    tile_j0: usize, // offset in src1 cols
    num_k: usize,
    slab: usize,
    n_rows: usize,
    n_cols: usize,
) -> [[f32; 16]; 16] {
    let mut acc = [[0.0f32; 16]; 16];

    // Calculate strides
    let qk_stride = num_k * TILE_K;
    let qk_off = slab * TILE_K;
    let nb_stride = num_k * NB_Q5K;
    let nb_off = slab * NB_Q5K;
    let bs_stride = if num_k == 1 { TILE_ROWS } else { MICRO };
    let bs_off = slab * MICRO;

    // Process each subblock (32-wide for Q5_K)
    for s in 0..NB_Q5K {
        for j in 0..n_cols {
            let br = tile_j0 + j;

            // Get bsum for this subblock
            let mut bsum = 0i32;
            for u in 0..2 {
                bsum += ws.src1_bsums[(bs_off + s * 2 + u) * bs_stride + br];
            }

            for i in 0..n_rows {
                let ar = tile_i0 + i;
                let d_off = slab * MICRO + ar;

                // Get pointers to quantized data
                let q0 =
                    ws.src0_q.as_ptr().add(ar * qk_stride + qk_off + s * 32);
                let q1 =
                    ws.src1_q.as_ptr().add(br * qk_stride + qk_off + s * 32);

                // Compute dot product using SDOT
                let mut dot_acc = vdupq_n_s32(0);

                // Process 32 elements as 2x16
                for chunk in 0..2 {
                    let offset = chunk * 16;

                    // Load uint8 weights and int8 inputs
                    let w = vld1q_u8(q0.add(offset));
                    let x = vld1q_s8(q1.add(offset));

                    // Convert uint8 to int8 for SDOT
                    // Note: weights are 0-31 range, we need signed
                    // Use vreinterpret to treat as signed
                    let w_signed = vreinterpretq_s8_u8(w);

                    // SDOT: accumulate int8 x int8
                    dot_acc = vdotq_s32_manual(dot_acc, w_signed, x);
                }

                // Extract sum from vector
                let raw = vaddvq_s32(dot_acc);

                // Apply bias (weights are stored as value + 16, so subtract 16*bsum)
                let corr = raw - 16 * bsum;

                // Apply scale
                let scale = ws.src0_scales[ar * nb_stride + nb_off + s];
                let scaled = corr * scale;

                // Apply super-block scale and min
                let d = ws.src0_d[d_off];
                let dmin = ws.src0_dmin[d_off];
                let min = ws.src0_mins[ar * nb_stride + nb_off + s];
                let min_correction = min * bsum;

                acc[i][j] +=
                    d * (scaled as f32) - dmin * (min_correction as f32);
            }
        }
    }

    // Apply input scale
    for j in 0..n_cols {
        let d1 = ws.src1_d[slab * MICRO + tile_j0 + j];
        for i in 0..n_rows {
            acc[i][j] *= d1;
        }
    }

    acc
}

/// Tiled Q5_K x Q8_K GEMM
/// For GEMV, set n = 1 (single input vector)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5k_q8k_tiled(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
) {
    let nb = k / QK_K;

    WORKSPACE.with(|ws_cell| {
        let mut ws = ws_cell.borrow_mut();

        // Determine path: narrow (N <= 16) or standard
        if n <= MICRO {
            // Narrow path: optimize for decode
            matmul_q5k_q8k_narrow(&mut ws, weights, x, dst, m, n, nb);
        } else {
            // Standard path: tiled 256x256
            matmul_q5k_q8k_standard(&mut ws, weights, x, dst, m, n, nb);
        }
    });
}

/// Narrow path: N <= 16 (small batch), optimize for decode
#[target_feature(enable = "neon,dotprod")]
unsafe fn matmul_q5k_q8k_narrow(
    ws: &mut TiledWorkspace,
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    m: usize,
    n: usize,
    nb: usize,
) {
    // For narrow path, use longer contiguous K reads
    let num_k = if n <= 4 { 4.min(nb) } else { 2.min(nb) };
    if num_k == 0 {
        return;
    }

    // Clear accumulator once
    ws.clear_acc(m, n);

    // Process K dimension in num_k chunks
    for k0 in (0..nb).step_by(num_k) {
        let k_end = (k0 + num_k).min(nb);
        let actual_num_k = k_end - k0;

        // Unpack input vectors (only once per K chunk)
        unpack_src1_q8k(x, 0, n, k0, actual_num_k, ws);

        // Process weight rows in 16-row chunks
        for iir0 in (0..m).step_by(MICRO) {
            let ir0_end = (iir0 + MICRO).min(m);
            let n_src0 = ir0_end - iir0;

            // Unpack weights for this chunk
            unpack_src0_q5k(weights, iir0, n_src0, k0, actual_num_k, ws);

            // Run microkernel for each slab
            for slab in 0..actual_num_k {
                let result = run_microtile_q5k_q8k(
                    ws,
                    0, // relative to unpacked chunk
                    0,
                    actual_num_k,
                    slab,
                    n_src0,
                    n,
                );

                // Accumulate to workspace
                for i in 0..n_src0 {
                    for j in 0..n {
                        ws.acc[(iir0 + i) * n + j] += result[i][j];
                    }
                }
            }
        }
    }

    // Store to dst
    for i in 0..m {
        for j in 0..n {
            dst[i * n + j] = ws.acc[i * n + j];
        }
    }
}

/// Standard tiled path: 256x256 tiles
#[target_feature(enable = "neon,dotprod")]
unsafe fn matmul_q5k_q8k_standard(
    ws: &mut TiledWorkspace,
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    m: usize,
    n: usize,
    nb: usize,
) {
    // Process 256x256 macrotiles
    for iir1 in (0..n).step_by(TILE_ROWS) {
        let ir1_end = (iir1 + TILE_ROWS).min(n);
        let n_src1 = ir1_end - iir1;

        for iir0 in (0..m).step_by(TILE_ROWS) {
            let ir0_end = (iir0 + TILE_ROWS).min(m);
            let n_src0 = ir0_end - iir0;

            // Clear accumulator
            ws.clear_acc(n_src0, n_src1);

            // Process K dimension one block at a time
            for ib in 0..nb {
                // Unpack src0 and src1 for this K block
                unpack_src0_q5k(weights, iir0, n_src0, ib, 1, ws);
                unpack_src1_q8k(x, iir1, n_src1, ib, 1, ws);

                // Run 16x16 microtiles
                for ir1 in (0..n_src1).step_by(MICRO) {
                    for ir0 in (0..n_src0).step_by(MICRO) {
                        let r1_end = (ir1 + MICRO).min(n_src1);
                        let r0_end = (ir0 + MICRO).min(n_src0);

                        let result = run_microtile_q5k_q8k(
                            ws,
                            ir0,
                            ir1,
                            1,
                            0,
                            r0_end - ir0,
                            r1_end - ir1,
                        );

                        // Accumulate
                        for i in 0..(r0_end - ir0) {
                            for j in 0..(r1_end - ir1) {
                                ws.acc[(ir0 + i) * TILE_ROWS + ir1 + j] +=
                                    result[i][j];
                            }
                        }
                    }
                }
            }

            // Store accumulator to dst
            for ir1 in 0..n_src1 {
                for ir0 in 0..n_src0 {
                    dst[(iir0 + ir0) * n + (iir1 + ir1)] =
                        ws.acc[ir0 * TILE_ROWS + ir1];
                }
            }
        }
    }
}

/// GEMV wrapper (single vector)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q5k_q8k_tiled(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    matmul_q5k_q8k_tiled(weights, x, dst, k, m, 1);
}
