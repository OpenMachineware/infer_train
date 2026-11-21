// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Tiled matrix multiplication structures and unpack/repack routines
// Based on llama.cpp's tiled implementation
//
// Key concepts:
// - src0 (weights): [n_rows][k_blocks], each block has 256 elements
// - src1 (activations): [n_tokens][k_blocks], each block has 256 elements
// - For narrow path (decode), n_tokens is small (1), n_rows is large
// - We process K in chunks to improve cache reuse
// - Unpack quantized data to tile structures for SIMD-friendly access

use crate::types::*;
use half::f16;

// ============================================================================
// Constants (matching llama.cpp)
// ============================================================================

/// Each block covers 256 elements (QK_K)
pub const TILED_TILE_K: usize = 256;

/// Maximum tile rows
pub const TILED_TILE_ROWS: usize = 256;

/// Microtile edge (16x16)
pub const TILED_MICRO: usize = 16;

/// Per-thread workspace slot (512KB)
pub const TILED_WS_SLOT: usize = 512 * 1024;

/// Maximum subblocks per 256-element block
pub const NB_MAX: usize = TILED_TILE_K / 16;

// ============================================================================
// Tile structures
// ============================================================================

/// Source 0 tile: weight side.
/// For Q5_K: NB=8 (32-element subblocks)
/// For Q6_K: NB=16 (16-element subblocks)
#[derive(Debug)]
pub struct TiledTileSrc0 {
    /// Unsigned quant codes, widened to uint8
    /// Layout: [row][k_offset] where k_offset = slab * TILED_TILE_K + element
    /// Size: TILED_TILE_ROWS * TILED_TILE_K = 256 * 256 = 64KB
    pub q: Vec<u8>,

    /// One d from each input block, widened to f32
    /// Layout: [slab * TILED_MICRO + row] for multi-k, or [row] for single k
    pub d: Vec<f32>,

    /// dmin from each input block (if applicable), widened to f32
    pub dmin: Vec<f32>,

    /// Per-subblock scale, stored as i32
    /// Layout: [row * nb_stride + slab * NB + s]
    pub scales: Vec<i32>,

    /// Per-subblock min, used when HAS_MIN
    pub mins: Vec<i32>,
}

impl TiledTileSrc0 {
    /// Create a new tile for given dimensions
    pub fn new(num_k: usize, n_rows: usize, nb_per_block: usize) -> Self {
        Self {
            q: vec![0u8; n_rows * num_k * TILED_TILE_K],
            d: vec![0.0f32; num_k * TILED_MICRO.max(n_rows)],
            dmin: vec![0.0f32; num_k * TILED_MICRO.max(n_rows)],
            scales: vec![0i32; n_rows * nb_per_block * num_k],
            mins: vec![0i32; n_rows * nb_per_block * num_k],
        }
    }

    /// Get q index for (row, slab, element)
    #[inline]
    fn q_idx(&self, row: usize, slab: usize, elem: usize) -> usize {
        row * (self.q.len() / TILED_TILE_ROWS) + slab * TILED_TILE_K + elem
    }
}

/// Source 1 tile: built from q8_K (activations)
#[derive(Debug)]
pub struct TiledTileSrc1 {
    /// q8 codes, one byte per element
    /// Layout: [token_row][slab * TILED_TILE_K + element]
    pub q: Vec<i8>,

    /// Per-16 code sums from q8_k, widened to i32
    /// Layout: [(slab * 16 + s) * bs_stride + token_row]
    /// where s is subblock index within the K-block
    pub bsums: Vec<i32>,

    /// f32 scale (converted from fp16 once)
    pub d: Vec<f32>,
}

impl TiledTileSrc1 {
    /// Create a new tile for given dimensions
    pub fn new(num_k: usize, n_rows: usize) -> Self {
        let bs_stride = n_rows.max(TILED_MICRO);
        Self {
            q: vec![0i8; n_rows * num_k * TILED_TILE_K],
            bsums: vec![0i32; num_k * (TILED_TILE_K / TILED_MICRO) * bs_stride],
            d: vec![0.0f32; num_k * TILED_MICRO.max(n_rows)],
        }
    }
}

// ============================================================================
// Unpack primitives (scalar fallback, NEON versions would use intrinsics)
// ============================================================================

/// Unpack packed 4-bit codes -> low nibbles (lo) + high nibbles (hi)
/// Each 32-byte input produces 32 low nibbles and 32 high nibbles
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tiled_unpk_nib4_neon(src: *const u8, lo: *mut u8, hi: *mut u8) {
    let m4b = vdupq_n_u8(0x0F);

    // Process 32 bytes in two 16-byte chunks
    let (v0, v1) = unsafe { (vld1q_u8(src), vld1q_u8(src.add(16))) };

    // Extract low and high nibbles
    let lo0 = vandq_u8(v0, m4b);
    let lo1 = vandq_u8(v1, m4b);
    let hi0 = vshrq_n_u8(v0, 4);
    let hi1 = vshrq_n_u8(v1, 4);

    // Store results
    unsafe {
        vst1q_u8(lo, lo0);
        vst1q_u8(lo.add(16), lo1);
        vst1q_u8(hi, hi0);
        vst1q_u8(hi.add(16), hi1);
    }
}

/// Scalar fallback for tiled_unpk_nib4
#[inline]
pub fn tiled_unpk_nib4(src: &[u8], lo: &mut [u8], hi: &mut [u8]) {
    for i in 0..32 {
        lo[i] = src[i] & 0x0F;
        hi[i] = src[i] >> 4;
    }
}

/// Unpack 2-bit values at bit offset S
#[inline]
pub fn tiled_unpk_2bit<const S: usize>(src: &[u8], dst: &mut [u8]) {
    for i in 0..32 {
        dst[i] = (src[i] >> S) & 0x03;
    }
}

/// OR the 1-bit value at bit offset S of src into bit offset D of dst (NEON)
/// Using macro to generate specialized versions for each S value
#[cfg(target_arch = "aarch64")]
macro_rules! impl_tiled_unpk_or_neon {
    ($name:ident, $S:expr) => {
        #[target_feature(enable = "neon")]
        #[inline]
        pub unsafe fn $name<const D: i32>(dst: *mut u8, src: *const u8) {
            let shift_d = vdupq_n_s8(D as i8);
            let mone = vdupq_n_u8(1);

            let (d0, d1, s0, s1) = unsafe {
                (
                    vld1q_u8(dst),
                    vld1q_u8(dst.add(16)),
                    vld1q_u8(src),
                    vld1q_u8(src.add(16)),
                )
            };

            let or0 = vreinterpretq_u8_s8(vshlq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<$S>(s0), mone)),
                shift_d,
            ));
            let or1 = vreinterpretq_u8_s8(vshlq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<$S>(s1), mone)),
                shift_d,
            ));

            unsafe {
                vst1q_u8(dst, vorrq_u8(d0, or0));
                vst1q_u8(dst.add(16), vorrq_u8(d1, or1));
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_tiled_unpk_or_neon!(tiled_unpk_or_s1_neon, 1);
#[cfg(target_arch = "aarch64")]
impl_tiled_unpk_or_neon!(tiled_unpk_or_s2_neon, 2);
#[cfg(target_arch = "aarch64")]
impl_tiled_unpk_or_neon!(tiled_unpk_or_s3_neon, 3);
#[cfg(target_arch = "aarch64")]
impl_tiled_unpk_or_neon!(tiled_unpk_or_s4_neon, 4);
#[cfg(target_arch = "aarch64")]
impl_tiled_unpk_or_neon!(tiled_unpk_or_s5_neon, 5);
#[cfg(target_arch = "aarch64")]
impl_tiled_unpk_or_neon!(tiled_unpk_or_s6_neon, 6);
#[cfg(target_arch = "aarch64")]
impl_tiled_unpk_or_neon!(tiled_unpk_or_s7_neon, 7);

/// OR the bit 0 of src into bit D of dst (no shift needed)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tiled_unpk_or_s0_neon<const D: i32>(
    dst: *mut u8,
    src: *const u8,
) {
    let shift_d = vdupq_n_s8(D as i8);
    let mone = vdupq_n_u8(1);

    let (d0, d1, s0, s1) = unsafe {
        (
            vld1q_u8(dst),
            vld1q_u8(dst.add(16)),
            vld1q_u8(src),
            vld1q_u8(src.add(16)),
        )
    };

    // No shift needed for S=0, just mask and shift left
    let or0 = vreinterpretq_u8_s8(vshlq_s8(
        vreinterpretq_s8_u8(vandq_u8(s0, mone)),
        shift_d,
    ));
    let or1 = vreinterpretq_u8_s8(vshlq_s8(
        vreinterpretq_s8_u8(vandq_u8(s1, mone)),
        shift_d,
    ));

    unsafe {
        vst1q_u8(dst, vorrq_u8(d0, or0));
        vst1q_u8(dst.add(16), vorrq_u8(d1, or1));
    }
}

/// OR the M-bit value at bit offset S of src into bit offset D of dst
#[inline]
pub fn tiled_unpk_or<const S: usize, const D: usize, const M: usize>(
    dst: &mut [u8],
    src: &[u8],
) {
    for i in 0..32 {
        dst[i] |= ((src[i] >> S) & (M as u8)) << D;
    }
}

// ============================================================================
// Scale decoding for K-quant formats
// ============================================================================

/// Decode Q5_K scales from 12-byte packed format
/// Returns (scales, mins) arrays, each of 8 values
pub fn decode_q5k_scales(packed: &[u8; 12]) -> ([i32; 8], [i32; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp: [u32; 4] = [0; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            packed.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let scales_bytes: [u8; 8] =
        unsafe { std::ptr::read(utmp.as_ptr() as *const [u8; 8]) };
    let mins_bytes: [u8; 8] =
        unsafe { std::ptr::read(utmp[2..].as_ptr() as *const [u8; 8]) };

    let mut scales = [0i32; 8];
    let mut mins = [0i32; 8];
    for i in 0..8 {
        scales[i] = scales_bytes[i] as i32;
        mins[i] = mins_bytes[i] as i32;
    }
    (scales, mins)
}

// ============================================================================
// Main unpack functions
// ============================================================================

/// Unpack Q5_K weight blocks into TiledTileSrc0
///
/// # Arguments
/// * `weights` - Raw weight bytes
/// * `col_stride` - Byte stride between rows
/// * `n_rows` - Number of rows to unpack (window: iir0..ir0_end)
/// * `tile` - Destination tile
/// * `num_k` - Number of K-blocks to unpack per row (k_extent_blocks)
/// * `kblk_start` - Starting K-block index in the source
/// * `blocks_per_row` - Number of blocks per weight row
pub fn tiled_unpack_src0_q5_k(
    weights: &[u8],
    col_stride: usize,
    n_rows: usize,
    tile: &mut TiledTileSrc0,
    num_k: usize,
    kblk_start: usize,
    blocks_per_row: usize,
) {
    const NB: usize = 8; // Q5_K has 32-element subblocks, 8 per 256
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_K>();

    for slab in 0..num_k {
        for r in 0..n_rows {
            // Calculate source offset
            let src_row_offset = r * col_stride;
            let src_block_offset = (kblk_start + slab) * BLOCK_SIZE;
            let src_ptr = unsafe {
                weights.as_ptr().add(src_row_offset + src_block_offset)
            };
            let x: &BlockQ5_K = unsafe { &*(src_ptr as *const BlockQ5_K) };

            // Tile offsets
            let d_off = slab * TILED_MICRO + r;
            let q_off = r * num_k * TILED_TILE_K + slab * TILED_TILE_K;
            let s_off = r * NB * num_k + slab * NB;

            // Decode d and dmin
            tile.d[d_off] = f16::to_f32(f16::from_bits(x.d));
            tile.dmin[d_off] = f16::to_f32(f16::from_bits(x.dmin));

            // Decode scales and mins
            let (scales, mins) = decode_q5k_scales(&x.scales);
            for s in 0..NB {
                tile.scales[s_off + s] = scales[s];
                tile.mins[s_off + s] = mins[s];
            }

            // Extract 5-bit codes (4 low bits + 1 high bit)
            // Same as llama.cpp:
            // - q[0..32] = qs[0..32] low nibble, qh bit 0 OR'd in
            // - q[32..64] = qs[0..32] high nibble, qh bit 1 OR'd in
            // - etc.
            let q = &mut tile.q[q_off..q_off + TILED_TILE_K];

            // Unpack 4-bit nibbles from qs (NEON optimized)
            {
                let q_ptr = q.as_mut_ptr();
                let qs_ptr = x.qs.as_ptr();
                let qh_ptr = x.qh.as_ptr();

                // Unpack all 128 bytes of qs
                // qs[0..32] -> q[0..32] (lo), q[32..64] (hi)
                // qs[32..64] -> q[64..96] (lo), q[96..128] (hi)
                // etc.
                for chunk in 0..4 {
                    let src_off = chunk * 32;
                    let lo_off = chunk * 64;
                    let hi_off = lo_off + 32;
                    unsafe {
                        tiled_unpk_nib4_neon(
                            qs_ptr.add(src_off),
                            q_ptr.add(lo_off),
                            q_ptr.add(hi_off),
                        );
                    }
                }

                // Add 5th bit from qh
                unsafe {
                    tiled_unpk_or_s0_neon::<4>(q_ptr.add(0), qh_ptr);
                    tiled_unpk_or_s1_neon::<4>(q_ptr.add(32), qh_ptr);
                    tiled_unpk_or_s2_neon::<4>(q_ptr.add(64), qh_ptr);
                    tiled_unpk_or_s3_neon::<4>(q_ptr.add(96), qh_ptr);
                    tiled_unpk_or_s4_neon::<4>(q_ptr.add(128), qh_ptr);
                    tiled_unpk_or_s5_neon::<4>(q_ptr.add(160), qh_ptr);
                    tiled_unpk_or_s6_neon::<4>(q_ptr.add(192), qh_ptr);
                    tiled_unpk_or_s7_neon::<4>(q_ptr.add(224), qh_ptr);
                }
            }
        }
    }
}

/// Unpack Q6_K weight blocks into TiledTileSrc0
pub fn tiled_unpack_src0_q6_k(
    weights: &[u8],
    col_stride: usize,
    n_rows: usize,
    tile: &mut TiledTileSrc0,
    num_k: usize,
    kblk_start: usize,
    blocks_per_row: usize,
) {
    const NB: usize = 16; // Q6_K has 16-element subblocks, 16 per 256
    const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6_K>();

    for slab in 0..num_k {
        for r in 0..n_rows {
            let src_row_offset = r * col_stride;
            let src_block_offset = (kblk_start + slab) * BLOCK_SIZE;
            let src_ptr = unsafe {
                weights.as_ptr().add(src_row_offset + src_block_offset)
            };
            let x: &BlockQ6_K = unsafe { &*(src_ptr as *const BlockQ6_K) };

            let d_off = slab * TILED_MICRO + r;
            let q_off = r * num_k * TILED_TILE_K + slab * TILED_TILE_K;
            let s_off = r * NB * num_k + slab * NB;

            tile.d[d_off] = f16::to_f32(f16::from_bits(x.d));
            tile.dmin[d_off] = 0.0;

            // 6-bit code = 4 low bits (ql) | 2 high bits (qh)
            // Layout from llama.cpp:
            // For each half: [ql_lo(0:32)] [ql_lo(32:64)] [ql_hi(0:32)] [ql_hi(32:64)]
            let q = &mut tile.q[q_off..q_off + TILED_TILE_K];

            for half in 0..2 {
                let out = &mut q[128 * half..128 * half + 128];

                let ql_off = 64 * half;
                let qh_half = &x.qh[32 * half..32 * half + 32];

                // ql low nibbles at out[0..32] and out[32..64]
                for i in 0..32 {
                    out[i] = x.ql[ql_off + i] & 0x0F;
                    out[32 + i] = x.ql[ql_off + 32 + i] & 0x0F;
                }

                // ql high nibbles at out[64..96] and out[96..128]
                for i in 0..32 {
                    out[64 + i] = x.ql[ql_off + i] >> 4;
                    out[96 + i] = x.ql[ql_off + 32 + i] >> 4;
                }

                // Add 2 high bits from qh
                tiled_unpk_or::<0, 4, 3>(&mut out[0..32], qh_half);
                tiled_unpk_or::<2, 4, 3>(&mut out[32..64], qh_half);
                tiled_unpk_or::<4, 4, 3>(&mut out[64..96], qh_half);
                tiled_unpk_or::<6, 4, 3>(&mut out[96..128], qh_half);
            }

            for s in 0..NB {
                tile.scales[s_off + s] = x.scales[s] as i8 as i32;
                tile.mins[s_off + s] = 0;
            }
        }
    }
}

/// Unpack Q8_K activation blocks into TiledTileSrc1
///
/// # Arguments
/// * `input` - Q8_K blocks for all tokens
/// * `blocks_per_token` - Number of K-blocks per token (n_blocks)
/// * `n_tokens` - Number of tokens
/// * `tile` - Destination tile
/// * `kblk_start` - Starting K-block index
/// * `num_k` - Number of K-blocks to unpack
pub fn tiled_unpack_src1_q8_k(
    input: &[BlockQ8_K],
    blocks_per_token: usize,
    n_tokens: usize,
    tile: &mut TiledTileSrc1,
    kblk_start: usize,
    num_k: usize,
) {
    let bs_stride = n_tokens.max(TILED_MICRO);
    let n16 = TILED_TILE_K / TILED_MICRO;

    for slab in 0..num_k {
        for t in 0..n_tokens {
            // Source: token t, block (kblk_start + slab)
            let src_block_idx = t * blocks_per_token + kblk_start + slab;
            let x = &input[src_block_idx];

            // Tile layout: [token][slab][element]
            let q_dst_off = t * num_k * TILED_TILE_K + slab * TILED_TILE_K;

            // Copy quant codes
            for (i, &v) in x.qs.iter().enumerate() {
                tile.q[q_dst_off + i] = v as i8;
            }

            // bsums: [(slab * n16 + s) * bs_stride + t]
            for s in 0..n16 {
                tile.bsums[(slab * n16 + s) * bs_stride + t] =
                    x.bsums[s] as i32;
            }

            // d (already f32 in BlockQ8_K)
            tile.d[slab * TILED_MICRO + t] = x.d;
        }
    }
}

// ============================================================================
// NEON optimized compute kernels
// ============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON SDOT helper for computing dot product of two 16-element arrays
/// Returns sum of (a[i] * b[i]) for i in 0..16
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn dot_i8x16_neon(a: *const u8, b: *const i8) -> i32 {
    let (va, vb, vzero) =
        unsafe { (vld1q_u8(a), vld1q_s8(b as *const i8), vdupq_n_s32(0)) };
    // Convert u8 to s8 for SDOT (reinterpret, no conversion cost)
    let va_s8 = vreinterpretq_s8_u8(va);
    let vresult = unsafe { vdotq_s32_manual(vzero, va_s8, vb) };
    vaddvq_s32(vresult)
}

/// NEON SDOT helper for 32-element dot product (used by Q5_K)
/// Returns sum of (a[i] * b[i]) for i in 0..32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn dot_i8x32_neon(a: *const u8, b: *const i8) -> i32 {
    // Load two 16-element vectors
    let (va0, va1, vb0, vb1, vzero) = unsafe {
        (
            vld1q_u8(a),
            vld1q_u8(a.add(16)),
            vld1q_s8(b as *const i8),
            vld1q_s8(b.add(16) as *const i8),
            vdupq_n_s32(0),
        )
    };

    // Convert u8 to s8
    let va0_s8 = vreinterpretq_s8_u8(va0);
    let va1_s8 = vreinterpretq_s8_u8(va1);

    // SDOT for each 16-element chunk
    let (vresult0, vresult1) = unsafe {
        (
            vdotq_s32_manual(vzero, va0_s8, vb0),
            vdotq_s32_manual(vzero, va1_s8, vb1),
        )
    };

    // Sum results
    vaddvq_s32(vaddq_s32(vresult0, vresult1))
}

// Import the manual SDOT implementation from vec_dot module
#[cfg(target_arch = "aarch64")]
use super::vec_dot::vdotq_s32_manual;

// ============================================================================
// Microkernel implementations
// ============================================================================

/// Compute dot products for Q5_K using unpacked tiles (NEON optimized)
///
/// Q5_K format:
/// - 8 subblocks of 32 elements each
/// - Each code is 5 bits: value in [0, 31], NO BIAS
/// - Has scale and min per subblock
///
/// Following llama.cpp's approach (BIAS = 0 for Q5_K):
/// - raw = sum(q5 * q8) - no bias subtraction
/// - scales_raw = scale * raw
/// - mins_bsum = min * bsum
/// - acc += weight.d * scales_raw - weight.dmin * mins_bsum
/// - Final result: acc * activation.d
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn tiled_compute_q5_k_neon(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    const NB: usize = 8;
    const SUBBLK: usize = 32;
    const NS: usize = SUBBLK / 16; // bsums per subblock = 2

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];
            let weight_dmin = src0.dmin[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;
                    let min = src0.mins[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    // NEON SDOT for 32-element dot product
                    let raw = unsafe {
                        dot_i8x32_neon(
                            src0.q.as_ptr().add(w_off),
                            src1.q.as_ptr().add(a_off),
                        )
                    };

                    // Apply scale
                    let scales_raw = scale * raw;

                    // Get bsum for min correction
                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let mins_bsum = min * bsum;

                    // Accumulate with weight's d and dmin
                    acc_val += weight_d * (scales_raw as f32)
                        - weight_dmin * (mins_bsum as f32);
                }

                // Apply activation's d at the end
                let act_d = src1.d[slab * TILED_MICRO + t];
                acc[r * n_tokens + t] += act_d * acc_val;
            }
        }
    }
}

/// Scalar fallback for Q5_K compute (for non-NEON platforms)
#[cfg(not(target_arch = "aarch64"))]
pub fn tiled_compute_q5_k(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    const NB: usize = 8;
    const SUBBLK: usize = 32;
    const NS: usize = SUBBLK / 16;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];
            let weight_dmin = src0.dmin[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;
                    let min = src0.mins[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let mut raw = 0i32;
                    for k in 0..SUBBLK {
                        let w = src0.q[w_off + k];
                        let a = src1.q[a_off + k];
                        raw += (w as i32) * (a as i32);
                    }

                    let scales_raw = scale * raw;
                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let mins_bsum = min * bsum;

                    acc_val += weight_d * (scales_raw as f32)
                        - weight_dmin * (mins_bsum as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                acc[r * n_tokens + t] += act_d * acc_val;
            }
        }
    }
}

/// Dispatch to NEON or scalar version based on CPU features
#[cfg(target_arch = "aarch64")]
pub fn tiled_compute_q5_k(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    // Apple M series and most ARMv8.2+ CPUs support dotprod
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        unsafe {
            tiled_compute_q5_k_neon(src0, src1, n_rows, n_tokens, num_k, acc);
        }
    } else {
        // Fallback to scalar version
        tiled_compute_q5_k_scalar(src0, src1, n_rows, n_tokens, num_k, acc);
    }
}

/// Direct version that writes to output in [token][row] order
/// Avoids creating temporary partial array
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn tiled_compute_q5_k_direct_neon(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    output: &mut [f32],
    out_features: usize,
    row_offset: usize,
) {
    const NB: usize = 8;
    const SUBBLK: usize = 32;
    const NS: usize = SUBBLK / 16;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];
            let weight_dmin = src0.dmin[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;
                    let min = src0.mins[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let raw = dot_i8x32_neon(
                        src0.q.as_ptr().add(w_off),
                        src1.q.as_ptr().add(a_off),
                    );

                    let scales_raw = scale * raw;
                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let mins_bsum = min * bsum;

                    acc_val += weight_d * (scales_raw as f32)
                        - weight_dmin * (mins_bsum as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                // Write directly to output in [token][row] order
                let out_idx = t * out_features + row_offset + r;
                output[out_idx] += act_d * acc_val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn tiled_compute_q5_k_direct(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    output: &mut [f32],
    out_features: usize,
    row_offset: usize,
) {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        unsafe {
            tiled_compute_q5_k_direct_neon(
                src0,
                src1,
                n_rows,
                n_tokens,
                num_k,
                output,
                out_features,
                row_offset,
            );
        }
    } else {
        // Scalar fallback
        tiled_compute_q5_k_direct_scalar(
            src0,
            src1,
            n_rows,
            n_tokens,
            num_k,
            output,
            out_features,
            row_offset,
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn tiled_compute_q5_k_direct_scalar(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    output: &mut [f32],
    out_features: usize,
    row_offset: usize,
) {
    const NB: usize = 8;
    const SUBBLK: usize = 32;
    const NS: usize = SUBBLK / 16;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];
            let weight_dmin = src0.dmin[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;
                    let min = src0.mins[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let mut raw = 0i32;
                    for k in 0..SUBBLK {
                        raw += (src0.q[w_off + k] as i32)
                            * (src1.q[a_off + k] as i32);
                    }

                    let scales_raw = scale * raw;
                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let mins_bsum = min * bsum;

                    acc_val += weight_d * (scales_raw as f32)
                        - weight_dmin * (mins_bsum as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                let out_idx = t * out_features + row_offset + r;
                output[out_idx] += act_d * acc_val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn tiled_compute_q5_k_scalar(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    const NB: usize = 8;
    const SUBBLK: usize = 32;
    const NS: usize = SUBBLK / 16;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];
            let weight_dmin = src0.dmin[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;
                    let min = src0.mins[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let mut raw = 0i32;
                    for k in 0..SUBBLK {
                        let w = src0.q[w_off + k];
                        let a = src1.q[a_off + k];
                        raw += (w as i32) * (a as i32);
                    }

                    let scales_raw = scale * raw;
                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let mins_bsum = min * bsum;

                    acc_val += weight_d * (scales_raw as f32)
                        - weight_dmin * (mins_bsum as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                acc[r * n_tokens + t] += act_d * acc_val;
            }
        }
    }
}

/// Helper to get bsum for a subblock
#[inline]
fn get_bsum(
    src1: &TiledTileSrc1,
    s: usize,
    ns: usize,
    bs_stride: usize,
    bs_off: usize,
    token: usize,
) -> i32 {
    let mut bsum = 0i32;
    for u in 0..ns {
        let bsum_idx = (bs_off + s * ns + u) * bs_stride + token;
        bsum += src1.bsums.get(bsum_idx).copied().unwrap_or(0);
    }
    bsum
}

/// Compute dot products for Q6_K using unpacked tiles (NEON optimized)
///
/// Q6_K format:
/// - 16 subblocks of 16 elements each
/// - Each code is 6 bits: value in [0, 63], bias by subtracting 32
/// - Has scale per subblock, no min
///
/// Following llama.cpp's approach:
/// - raw = sum(q6 * q8) for each subblock
/// - corr = raw - BIAS * bsum
/// - scales_raw = scale * corr
/// - acc += weight.d * scales_raw
/// - Final result: acc * activation.d
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn tiled_compute_q6_k_neon(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    const NB: usize = 16;
    const SUBBLK: usize = 16;
    const NS: usize = SUBBLK / 16; // bsums per subblock = 1
    const BIAS: i32 = 32;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    // NEON SDOT for 16-element dot product
                    let raw = unsafe {
                        dot_i8x16_neon(
                            src0.q.as_ptr().add(w_off),
                            src1.q.as_ptr().add(a_off),
                        )
                    };

                    // Get bsum and apply bias correction
                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let corr = raw - BIAS * bsum;

                    // Apply scale
                    let scales_raw = scale * corr;

                    acc_val += weight_d * (scales_raw as f32);
                }

                // Apply activation's d at the end
                let act_d = src1.d[slab * TILED_MICRO + t];
                acc[r * n_tokens + t] += act_d * acc_val;
            }
        }
    }
}

/// Scalar fallback for Q6_K compute
#[cfg(target_arch = "aarch64")]
fn tiled_compute_q6_k_scalar(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    const NB: usize = 16;
    const SUBBLK: usize = 16;
    const NS: usize = SUBBLK / 16;
    const BIAS: i32 = 32;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let mut raw = 0i32;
                    for k in 0..SUBBLK {
                        let w = src0.q[w_off + k];
                        let a = src1.q[a_off + k];
                        raw += (w as i32) * (a as i32);
                    }

                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let corr = raw - BIAS * bsum;
                    let scales_raw = scale * corr;

                    acc_val += weight_d * (scales_raw as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                acc[r * n_tokens + t] += act_d * acc_val;
            }
        }
    }
}

/// Dispatch to NEON or scalar version based on CPU features
#[cfg(target_arch = "aarch64")]
pub fn tiled_compute_q6_k(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        unsafe {
            tiled_compute_q6_k_neon(src0, src1, n_rows, n_tokens, num_k, acc);
        }
    } else {
        tiled_compute_q6_k_scalar(src0, src1, n_rows, n_tokens, num_k, acc);
    }
}

/// Direct version for Q6_K that writes to output in [token][row] order
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn tiled_compute_q6_k_direct_neon(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    output: &mut [f32],
    out_features: usize,
    row_offset: usize,
) {
    const NB: usize = 16;
    const SUBBLK: usize = 16;
    const NS: usize = SUBBLK / 16;
    const BIAS: i32 = 32;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let raw = dot_i8x16_neon(
                        src0.q.as_ptr().add(w_off),
                        src1.q.as_ptr().add(a_off),
                    );

                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let corr = raw - BIAS * bsum;
                    let scales_raw = scale * corr;

                    acc_val += weight_d * (scales_raw as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                let out_idx = t * out_features + row_offset + r;
                output[out_idx] += act_d * acc_val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn tiled_compute_q6_k_direct(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    output: &mut [f32],
    out_features: usize,
    row_offset: usize,
) {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        unsafe {
            tiled_compute_q6_k_direct_neon(
                src0,
                src1,
                n_rows,
                n_tokens,
                num_k,
                output,
                out_features,
                row_offset,
            );
        }
    } else {
        tiled_compute_q6_k_direct_scalar(
            src0,
            src1,
            n_rows,
            n_tokens,
            num_k,
            output,
            out_features,
            row_offset,
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn tiled_compute_q6_k_direct_scalar(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    output: &mut [f32],
    out_features: usize,
    row_offset: usize,
) {
    const NB: usize = 16;
    const SUBBLK: usize = 16;
    const NS: usize = SUBBLK / 16;
    const BIAS: i32 = 32;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let mut raw = 0i32;
                    for k in 0..SUBBLK {
                        raw += (src0.q[w_off + k] as i32)
                            * (src1.q[a_off + k] as i32);
                    }

                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let corr = raw - BIAS * bsum;
                    let scales_raw = scale * corr;

                    acc_val += weight_d * (scales_raw as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                let out_idx = t * out_features + row_offset + r;
                output[out_idx] += act_d * acc_val;
            }
        }
    }
}

/// Scalar version for non-NEON platforms
#[cfg(not(target_arch = "aarch64"))]
pub fn tiled_compute_q6_k(
    src0: &TiledTileSrc0,
    src1: &TiledTileSrc1,
    n_rows: usize,
    n_tokens: usize,
    num_k: usize,
    acc: &mut [f32],
) {
    const NB: usize = 16;
    const SUBBLK: usize = 16;
    const NS: usize = SUBBLK / 16;
    const BIAS: i32 = 32;

    let bs_stride = if num_k == 1 { TILED_TILE_ROWS } else { TILED_MICRO };

    for slab in 0..num_k {
        let qk_off = slab * TILED_TILE_K;
        let nb_off = slab * NB;
        let bs_off = slab * TILED_MICRO;

        for r in 0..n_rows {
            let d_off = slab * TILED_MICRO + r;
            let weight_d = src0.d[d_off];

            for t in 0..n_tokens {
                let mut acc_val = 0.0f32;

                for s in 0..NB {
                    let scale = src0.scales[r * NB * num_k + nb_off + s] as i32;

                    let w_off = r * num_k * TILED_TILE_K + qk_off + s * SUBBLK;
                    let a_off = t * num_k * TILED_TILE_K + qk_off + s * SUBBLK;

                    let mut raw = 0i32;
                    for k in 0..SUBBLK {
                        let w = src0.q[w_off + k];
                        let a = src1.q[a_off + k];
                        raw += (w as i32) * (a as i32);
                    }

                    let bsum = get_bsum(src1, s, NS, bs_stride, bs_off, t);
                    let corr = raw - BIAS * bsum;
                    let scales_raw = scale * corr;

                    acc_val += weight_d * (scales_raw as f32);
                }

                let act_d = src1.d[slab * TILED_MICRO + t];
                acc[r * n_tokens + t] += act_d * acc_val;
            }
        }
    }
}

// ============================================================================
// High-level tiled GEMV functions
// ============================================================================

/// Tiled matrix multiplication for Q5_K
///
/// This is the entry point that integrates unpack/repack into K-extent processing.
pub fn mul_mat_q5_k_tiled(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
) {
    let n_blocks = in_features / QK_K;
    let block_size = std::mem::size_of::<BlockQ5_K>();

    // K-extent: how many K-blocks to process at once
    let k_extent_blocks = compute_k_extent_blocks(n_tokens, block_size);

    // Initialize output
    output.fill(0.0);

    // Process output rows in windows of TILE_ROWS
    const TILE_ROWS: usize = 256;

    for iir0 in (0..out_features).step_by(TILE_ROWS) {
        let ir0_end = (iir0 + TILE_ROWS).min(out_features);
        let window_rows = ir0_end - iir0;

        // Process K dimension in chunks
        for k0 in (0..n_blocks).step_by(k_extent_blocks) {
            let k_end = (k0 + k_extent_blocks).min(n_blocks);
            let num_k = k_end - k0;

            // Create tile for this K-chunk
            let mut src0_tile = TiledTileSrc0::new(num_k, window_rows, 8);

            // Unpack weights for this K-chunk
            tiled_unpack_src0_q5_k(
                weights,
                col_stride,
                window_rows,
                &mut src0_tile,
                num_k,
                k0,
                n_blocks,
            );

            // Unpack activations for this K-chunk
            let mut src1_tile = TiledTileSrc1::new(num_k, n_tokens);
            tiled_unpack_src1_q8_k(
                input,
                n_blocks,
                n_tokens,
                &mut src1_tile,
                k0,
                num_k,
            );

            // Compute partial products directly into output
            // We pass the output slice starting at the correct offset
            // The compute function writes in [token][row] order matching output
            tiled_compute_q5_k_direct(
                &src0_tile,
                &src1_tile,
                window_rows,
                n_tokens,
                num_k,
                output,
                out_features,
                iir0,
            );
        }
    }
}

/// Tiled matrix multiplication for Q6_K
pub fn mul_mat_q6_k_tiled(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
) {
    let n_blocks = in_features / QK_K;
    let block_size = std::mem::size_of::<BlockQ6_K>();

    let k_extent_blocks = compute_k_extent_blocks(n_tokens, block_size);

    output.fill(0.0);

    const TILE_ROWS: usize = 256;

    for iir0 in (0..out_features).step_by(TILE_ROWS) {
        let ir0_end = (iir0 + TILE_ROWS).min(out_features);
        let window_rows = ir0_end - iir0;

        for k0 in (0..n_blocks).step_by(k_extent_blocks) {
            let k_end = (k0 + k_extent_blocks).min(n_blocks);
            let num_k = k_end - k0;

            let mut src0_tile = TiledTileSrc0::new(num_k, window_rows, 16);

            tiled_unpack_src0_q6_k(
                weights,
                col_stride,
                window_rows,
                &mut src0_tile,
                num_k,
                k0,
                n_blocks,
            );

            let mut src1_tile = TiledTileSrc1::new(num_k, n_tokens);
            tiled_unpack_src1_q8_k(
                input,
                n_blocks,
                n_tokens,
                &mut src1_tile,
                k0,
                num_k,
            );

            let mut partial = vec![0.0f32; window_rows * n_tokens];
            tiled_compute_q6_k_direct(
                &src0_tile,
                &src1_tile,
                window_rows,
                n_tokens,
                num_k,
                output,
                out_features,
                iir0,
            );
        }
    }
}

/// Compute K-extent in blocks for narrow path
fn compute_k_extent_blocks(n_tokens: usize, block_size: usize) -> usize {
    const L1_BUDGET: usize = 31 * 1024;
    const TILE_K: usize = 256;

    let n_rows = n_tokens.max(16);
    let k_extent = L1_BUDGET / (16 * block_size / 256 + n_rows);
    let k_extent = (k_extent / TILE_K) * TILE_K;
    let k_extent_blocks = k_extent / TILE_K;

    k_extent_blocks.max(1)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_creation() {
        let tile = TiledTileSrc0::new(1, 256, 8);
        assert_eq!(tile.q.len(), 256 * 256);
        assert_eq!(tile.scales.len(), 256 * 8);
    }

    #[test]
    fn test_unpk_nib4() {
        let src: [u8; 32] = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33,
            0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
            0xFF, 0x00, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        ];

        let mut lo = [0u8; 32];
        let mut hi = [0u8; 32];

        tiled_unpk_nib4(&src, &mut lo, &mut hi);

        assert_eq!(lo[0], 0x02);
        assert_eq!(hi[0], 0x01);
        assert_eq!(lo[1], 0x04);
        assert_eq!(hi[1], 0x03);
    }

    #[test]
    fn test_unpk_or() {
        let mut dst = [0u8; 32];
        let src = [0xFFu8; 32];

        tiled_unpk_or::<0, 4, 1>(&mut dst, &src);

        for i in 0..32 {
            assert_eq!(dst[i], 0x10);
        }
    }

    #[test]
    fn test_decode_q5k_scales() {
        // Test with known values
        let packed = [
            0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33,
            0x44,
        ];
        let (scales, mins) = decode_q5k_scales(&packed);

        // Just verify it runs without panicking
        println!("scales: {:?}", scales);
        println!("mins: {:?}", mins);
    }
}
