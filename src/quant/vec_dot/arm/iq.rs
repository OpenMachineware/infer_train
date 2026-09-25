use std::arch::aarch64::*;

use crate::quant::types::{BlockIQ1M, BlockIQ1S, BlockIQ2S, BlockIQ2XS, BlockIQ2XXS, BlockIQ3S, BlockIQ3XXS, BlockIQ4NL, BlockIQ4XS, BlockQ8K, QK_K};
use crate::quant::vec_dot::tables::{IQ1S_DELTA, IQ1S_GRID, IQ1M_DELTA, IQ2_S_GRID, IQ2_XS_GRID, IQ2_XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};

use super::fp::fp16_to_fp32;

/// Lookup table for IQ4_NL: maps 4-bit values to actual quantized values
static KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const QK4_NL: usize = 32;

/// Manual vdotq_s32 implementation using native SDOT instruction
#[target_feature(enable = "neon,dotprod")]
unsafe fn vdotq_s32_manual(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
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

/// IQ4_NL × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq4_nl_q8_0_neon(n: usize, x: &[BlockIQ4NL], y: &[crate::quant::types::BlockQ8_0]) -> f32 {
    let nb = n / QK4_NL;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    // Load the lookup table into a NEON register
    let values = vld1q_s8(KVALUES_IQ4NL.as_ptr());

    let mut sum = 0.0f32;
    let mut ib = 0;

    // Process 2 blocks at a time
    while ib + 1 < nb {
        let q4bits_0 = vld1q_u8(x[ib].qs.as_ptr());
        let q4bits_1 = vld1q_u8(x[ib + 1].qs.as_ptr());

        let q8b_0 = vld1q_s8_x2(y[ib].qs.as_ptr());
        let q8b_1 = vld1q_s8_x2(y[ib + 1].qs.as_ptr());

        // Lookup 4-bit values using TBL instruction
        let q4b_0 = vqtbl1q_s8(values, vandq_u8(q4bits_0, m4b));
        let q4b_1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits_0, 4));
        let q4b_2 = vqtbl1q_s8(values, vandq_u8(q4bits_1, m4b));
        let q4b_3 = vqtbl1q_s8(values, vshrq_n_u8(q4bits_1, 4));

        let prod_0 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_0, q8b_0.0), q4b_1, q8b_0.1);
        let prod_1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_2, q8b_1.0), q4b_3, q8b_1.1);

        sum += fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d) * vaddvq_s32(prod_0) as f32;
        sum += fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d) * vaddvq_s32(prod_1) as f32;

        ib += 2;
    }

    // Handle remainder
    for i in ib..nb {
        let d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += y[i].qs[j] as i32 * KVALUES_IQ4NL[(x[i].qs[j] & 0x0F) as usize] as i32;
            sumi += y[i].qs[j + 16] as i32 * KVALUES_IQ4NL[(x[i].qs[j] >> 4) as usize] as i32;
        }
        sum += d * sumi as f32;
    }

    sum
}

/// IQ3_XXS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon(n: usize, x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let q3 = x[i].qs.as_ptr();
        let gas = q3.add(QK_K / 4); // Signs and scales start after grid indices
        let q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        // Process 2 blocks of 32 elements at a time (64 elements per iteration)
        for ib32 in (0..QK_K / 32).step_by(2) {
            // Load Q8 values
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));
            let q8b2 = vld1q_s8_x4(q8.add((ib32 + 1) * 32));

            // Load signs/scales (2 uint32_t per 32-element block)
            let aux32_0 = std::ptr::read_unaligned(gas.add(ib32 * 4) as *const u32);
            let aux32_1 = std::ptr::read_unaligned(gas.add((ib32 + 1) * 4) as *const u32);

            // Load grid indices (16 indices per 32-element block)
            let q3_ptr = q3.add(ib32 * 16);
            let grid_vals_0: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(0) as usize],
                IQ3XXS_GRID[*q3_ptr.add(1) as usize],
                IQ3XXS_GRID[*q3_ptr.add(2) as usize],
                IQ3XXS_GRID[*q3_ptr.add(3) as usize],
            ];
            let grid_vals_1: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(4) as usize],
                IQ3XXS_GRID[*q3_ptr.add(5) as usize],
                IQ3XXS_GRID[*q3_ptr.add(6) as usize],
                IQ3XXS_GRID[*q3_ptr.add(7) as usize],
            ];
            let grid_vals_2: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(8) as usize],
                IQ3XXS_GRID[*q3_ptr.add(9) as usize],
                IQ3XXS_GRID[*q3_ptr.add(10) as usize],
                IQ3XXS_GRID[*q3_ptr.add(11) as usize],
            ];
            let grid_vals_3: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(12) as usize],
                IQ3XXS_GRID[*q3_ptr.add(13) as usize],
                IQ3XXS_GRID[*q3_ptr.add(14) as usize],
                IQ3XXS_GRID[*q3_ptr.add(15) as usize],
            ];
            let aux32x4_0 = vld1q_u32(grid_vals_0.as_ptr());
            let aux32x4_1 = vld1q_u32(grid_vals_1.as_ptr());
            let aux32x4_2 = vld1q_u32(grid_vals_2.as_ptr());
            let aux32x4_3 = vld1q_u32(grid_vals_3.as_ptr());

            // Load signs from keven_signs_q2xs (8 bytes per sign index)
            let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;
            let q3s_0 = vcombine_s8(
                vld1_s8(signs64.add(((aux32_0 >> 0) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32_0 >> 7) & 127) as usize * 8)),
            );
            let q3s_1 = vcombine_s8(
                vld1_s8(signs64.add(((aux32_0 >> 14) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32_0 >> 21) & 127) as usize * 8)),
            );
            let q3s_2 = vcombine_s8(
                vld1_s8(signs64.add(((aux32_1 >> 0) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32_1 >> 7) & 127) as usize * 8)),
            );
            let q3s_3 = vcombine_s8(
                vld1_s8(signs64.add(((aux32_1 >> 14) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32_1 >> 21) & 127) as usize * 8)),
            );

            // Apply signs to grid values
            let q3s_0 = vmulq_s8(q3s_0, vreinterpretq_s8_u32(aux32x4_0));
            let q3s_1 = vmulq_s8(q3s_1, vreinterpretq_s8_u32(aux32x4_1));
            let q3s_2 = vmulq_s8(q3s_2, vreinterpretq_s8_u32(aux32x4_2));
            let q3s_3 = vmulq_s8(q3s_3, vreinterpretq_s8_u32(aux32x4_3));

            // Dot product with Q8 values
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_0, q8b.0), q3s_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_2, q8b2.0), q3s_3, q8b2.1);

            // Apply scale: 0.5 + (aux32 >> 28)
            let scale1 = 0.5f32 + ((aux32_0 >> 28) as f32);
            let scale2 = 0.5f32 + ((aux32_1 >> 28) as f32);

            sumf1 += vaddvq_s32(p1) as f32 * scale1;
            sumf2 += vaddvq_s32(p2) as f32 * scale2;
        }

        sum += d * (sumf1 + sumf2) * 0.5;
    }

    sum
}

/// IQ3_S × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_s_q8_k_neon(n: usize, x: &[BlockIQ3S], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    // Masks for sign processing (matching llama.cpp k_mask1, k_mask2)
    static K_MASK1_0: [u8; 16] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f];
    static K_MASK1_1: [u8; 16] = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f];
    static K_MASK2: [u8; 16] = [0x01u8; 16];

    let mask1 = uint8x16x2_t(
        vld1q_u8(K_MASK1_0.as_ptr()),
        vld1q_u8(K_MASK1_1.as_ptr()),
    );
    let mask2 = vld1q_u8(K_MASK2.as_ptr());
    let m1 = vdupq_n_u8(1);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut qs = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let mut signs = x[i].signs.as_ptr() as *const u16;
        let q8 = y[i].qs.as_ptr();

        // Decode scales
        let mut scales32 = [0u32; 2];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), scales32.as_mut_ptr() as *mut u8, 4);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;
        let scales8 = scales32.as_ptr() as *const u8;

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        // Process 2 blocks of 32 elements at a time
        for ib32 in (0..QK_K / 32).step_by(2) {
            // Load Q8 values (64 bytes for 2 blocks of 32)
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));

            // Load grid indices (16 bytes)
            let idx_l = vld1q_u8(qs);
            qs = qs.add(16);

            // Shift amounts: [8, 7, 6, 5, 4, 3, 2, 1]
            let hshift = vld1q_s16([8i16, 7, 6, 5, 4, 3, 2, 1].as_ptr());
            let m256 = vdupq_n_u16(256);

            // Combine low bits from qs with high bits from qh
            let qh_bits_0 = vdupq_n_u16(*qh.add(ib32) as u16);
            let qh_bits_1 = vdupq_n_u16(*qh.add(ib32 + 1) as u16);

            // Index for low 8 bytes: idx + ((qh >> shift) & 1) << 8
            let idx_lo = vmovl_u8(vget_low_u8(idx_l));
            let idx_hi = vmovl_u8(vget_high_u8(idx_l));

            // Shift qh by different amounts for each element, mask with 256, and OR into index
            let idx_final_0 = vorrq_u16(idx_lo, vandq_u16(vshlq_u16(qh_bits_0, hshift), m256));
            let idx_final_1 = vorrq_u16(idx_hi, vandq_u16(vshlq_u16(qh_bits_1, hshift), m256));

            // Extract indices for table lookup
            let indices_0: [u16; 8] = std::mem::transmute(idx_final_0);
            let indices_1: [u16; 8] = std::mem::transmute(idx_final_1);

            // Load grid values using indices (this could be optimized with gather, but we use scalar for now)
            let grid_vals_0: [u32; 4] = [
                IQ3S_GRID[indices_0[0] as usize],
                IQ3S_GRID[indices_0[1] as usize],
                IQ3S_GRID[indices_0[2] as usize],
                IQ3S_GRID[indices_0[3] as usize],
            ];
            let grid_vals_1: [u32; 4] = [
                IQ3S_GRID[indices_0[4] as usize],
                IQ3S_GRID[indices_0[5] as usize],
                IQ3S_GRID[indices_0[6] as usize],
                IQ3S_GRID[indices_0[7] as usize],
            ];
            let grid_vals_2: [u32; 4] = [
                IQ3S_GRID[indices_1[0] as usize],
                IQ3S_GRID[indices_1[1] as usize],
                IQ3S_GRID[indices_1[2] as usize],
                IQ3S_GRID[indices_1[3] as usize],
            ];
            let grid_vals_3: [u32; 4] = [
                IQ3S_GRID[indices_1[4] as usize],
                IQ3S_GRID[indices_1[5] as usize],
                IQ3S_GRID[indices_1[6] as usize],
                IQ3S_GRID[indices_1[7] as usize],
            ];

            let aux32x4_0 = vld1q_u32(grid_vals_0.as_ptr());
            let aux32x4_1 = vld1q_u32(grid_vals_1.as_ptr());
            let aux32x4_2 = vld1q_u32(grid_vals_2.as_ptr());
            let aux32x4_3 = vld1q_u32(grid_vals_3.as_ptr());

            // Load signs (2 u16 per block, 4 total)
            let s0 = *signs.add(0);
            let s1 = *signs.add(1);
            let s2 = *signs.add(2);
            let s3 = *signs.add(3);
            signs = signs.add(4);

            // Process signs using TBL
            let vs0 = vreinterpretq_u8_u32(vdupq_n_u32(s0 as u32 | ((s1 as u32) << 16)));
            let vs1 = vandq_u8(vqtbl1q_u8(vs0, mask1.1), mask2);
            let vs0 = vandq_u8(vqtbl1q_u8(vs0, mask1.0), mask2);
            let vs0 = vorrq_u8(vceqq_u8(vs0, mask2), m1);
            let vs1 = vorrq_u8(vceqq_u8(vs1, mask2), m1);

            // Apply signs to grid values
            let q3s_0 = vmulq_s8(vreinterpretq_s8_u8(vs0), vreinterpretq_s8_u32(aux32x4_0));
            let q3s_1 = vmulq_s8(vreinterpretq_s8_u8(vs1), vreinterpretq_s8_u32(aux32x4_1));

            // Process second block of signs
            let vs2 = vreinterpretq_u8_u32(vdupq_n_u32(s2 as u32 | ((s3 as u32) << 16)));
            let vs3 = vandq_u8(vqtbl1q_u8(vs2, mask1.1), mask2);
            let vs2 = vandq_u8(vqtbl1q_u8(vs2, mask1.0), mask2);
            let vs2 = vorrq_u8(vceqq_u8(vs2, mask2), m1);
            let vs3 = vorrq_u8(vceqq_u8(vs3, mask2), m1);

            let q3s_2 = vmulq_s8(vreinterpretq_s8_u8(vs2), vreinterpretq_s8_u32(aux32x4_2));
            let q3s_3 = vmulq_s8(vreinterpretq_s8_u8(vs3), vreinterpretq_s8_u32(aux32x4_3));

            // Dot product
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_0, q8b.0), q3s_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_2, q8b.2), q3s_3, q8b.3);

            sumi1 += vaddvq_s32(p1) * *scales8.add(ib32 / 2) as i32;
            sumi2 += vaddvq_s32(p2) * *scales8.add(ib32 / 2 + 4) as i32;
        }

        sumf += d * (sumi1 + sumi2) as f32;
    }

    sumf
}

/// IQ1_S × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq1_s_q8_k_neon(n: usize, x: &[BlockIQ1S], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut qs = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();
        let bsums = y[i].bsums.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        let mut sumi3 = 0i32;

        // Process 8 blocks of 32 elements, 2 at a time
        for ib in (0..QK_K / 32).step_by(2) {
            // Load Q8 values (64 bytes for 2 blocks)
            let q8b = vld1q_s8_x4(q8.add(ib * 32));

            // Load grid indices (8 bytes per 2 blocks)
            let qs_vals: [u8; 8] = [
                *qs.add(0), *qs.add(1), *qs.add(2), *qs.add(3),
                *qs.add(4), *qs.add(5), *qs.add(6), *qs.add(7),
            ];
            qs = qs.add(8);

            // Load high bits from qh
            let qh0 = *qh.add(ib) as u32;
            let qh1 = *qh.add(ib + 1) as u32;

            // Compute grid indices and load values
            // Index formula: qs[j] | ((qh << shift) & 0x700)
            let idx_0: [usize; 8] = [
                (qs_vals[0] as usize) | (((qh0 << 8) & 0x700) as usize),
                (qs_vals[1] as usize) | (((qh0 << 5) & 0x700) as usize),
                (qs_vals[2] as usize) | (((qh0 << 2) & 0x700) as usize),
                (qs_vals[3] as usize) | (((qh0 >> 1) & 0x700) as usize),
                (qs_vals[4] as usize) | (((qh1 << 8) & 0x700) as usize),
                (qs_vals[5] as usize) | (((qh1 << 5) & 0x700) as usize),
                (qs_vals[6] as usize) | (((qh1 << 2) & 0x700) as usize),
                (qs_vals[7] as usize) | (((qh1 >> 1) & 0x700) as usize),
            ];

            // Load grid values (each entry is 8 bytes = 8 int8 values)
            // Combine two consecutive entries into int8x16_t
            let q1b_0 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[0]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[1]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_1 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[2]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[3]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_2 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[4]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[5]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_3 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[6]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[7]].to_le_bytes().as_ptr() as *const i8),
            );

            // Dot product
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q1b_0, q8b.0), q1b_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q1b_2, q8b.2), q1b_3, q8b.3);

            // Compute scale: 2*((qh >> 12) & 7) + 1
            let ls1 = 2 * ((qh0 >> 12) & 7) as i32 + 1;
            let ls2 = 2 * ((qh1 >> 12) & 7) as i32 + 1;

            sumi1 += vaddvq_s32(p1) * ls1;
            sumi2 += vaddvq_s32(p2) * ls2;

            // Delta term: sum of Q8 values * scale * sign
            let sign1 = if (qh0 & 0x8000) != 0 { -1i32 } else { 1i32 };
            let sign2 = if (qh1 & 0x8000) != 0 { -1i32 } else { 1i32 };

            sumi3 += (*bsums.add(2 * ib) as i32 + *bsums.add(2 * ib + 1) as i32) * ls1 * sign1;
            sumi3 += (*bsums.add(2 * ib + 2) as i32 + *bsums.add(2 * ib + 3) as i32) * ls2 * sign2;
        }

        sumf += d * (sumi1 + sumi2 + (IQ1S_DELTA * sumi3 as f32) as i32) as f32;
    }

    sumf
}

/// Helper to get delta vector by index
#[target_feature(enable = "neon,dotprod")]
unsafe fn get_delta(deltas: &int8x16x4_t, idx: u8) -> int8x16_t {
    match idx {
        0 => deltas.0,
        1 => deltas.1,
        2 => deltas.2,
        3 => deltas.3,
        _ => deltas.0, // Should never happen
    }
}

/// IQ1_M × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq1_m_q8_k_neon(n: usize, x: &[BlockIQ1M], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);
    let mask = vdupq_n_s32(0x7);
    let mone = vdupq_n_s32(1);

    // Delta vectors for handling sign combinations
    let deltas: int8x16x4_t = int8x16x4_t(
        vcombine_s8(vdup_n_s8(1), vdup_n_s8(1)),
        vcombine_s8(vdup_n_s8(-1), vdup_n_s8(1)),
        vcombine_s8(vdup_n_s8(1), vdup_n_s8(-1)),
        vcombine_s8(vdup_n_s8(-1), vdup_n_s8(-1)),
    );

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let mut qs = x[i].qs.as_ptr();
        let mut qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();
        let sc = x[i].scales.as_ptr() as *const u16;

        // Extract merged scale from 4 u16 values
        let scale_u16 = (*sc.add(0) >> 12) | ((*sc.add(1) >> 8) & 0x00f0) | ((*sc.add(2) >> 4) & 0x0f00) | (*sc.add(3) & 0xf000);

        let mut sumi1 = vdupq_n_s32(0);
        let mut sumi2 = vdupq_n_s32(0);

        // Process 8 blocks of 32 elements, 2 at a time
        for ib in (0..QK_K / 32).step_by(2) {
            // Load grid indices (8 bytes per 2 blocks)
            let qs_vals: [u8; 8] = [
                *qs.add(0), *qs.add(1), *qs.add(2), *qs.add(3),
                *qs.add(4), *qs.add(5), *qs.add(6), *qs.add(7),
            ];
            qs = qs.add(8);

            // Load high bits from qh (4 bytes per 2 blocks)
            let qh_vals: [u8; 4] = [*qh.add(0), *qh.add(1), *qh.add(2), *qh.add(3)];
            qh = qh.add(4);

            // Compute grid indices
            let idx: [usize; 8] = [
                (qs_vals[0] as usize) | (((qh_vals[0] as u32) << 8) & 0x700) as usize,
                (qs_vals[1] as usize) | (((qh_vals[0] as u32) << 4) & 0x700) as usize,
                (qs_vals[2] as usize) | (((qh_vals[1] as u32) << 8) & 0x700) as usize,
                (qs_vals[3] as usize) | (((qh_vals[1] as u32) << 4) & 0x700) as usize,
                (qs_vals[4] as usize) | (((qh_vals[2] as u32) << 8) & 0x700) as usize,
                (qs_vals[5] as usize) | (((qh_vals[2] as u32) << 4) & 0x700) as usize,
                (qs_vals[6] as usize) | (((qh_vals[3] as u32) << 8) & 0x700) as usize,
                (qs_vals[7] as usize) | (((qh_vals[3] as u32) << 4) & 0x700) as usize,
            ];

            // Load grid values
            let q1b_0 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[0]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[1]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_1 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[2]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[3]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_2 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[4]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[5]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_3 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[6]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[7]].to_le_bytes().as_ptr() as *const i8),
            );

            // Load Q8 values
            let q8b = vld1q_s8_x4(q8.add(ib * 32));

            // Dot product
            let p1 = vpaddq_s32(vdotq_s32_manual(vzero, q1b_0, q8b.0), vdotq_s32_manual(vzero, q1b_1, q8b.1));
            let p2 = vpaddq_s32(vdotq_s32_manual(vzero, q1b_2, q8b.2), vdotq_s32_manual(vzero, q1b_3, q8b.3));
            let p12 = vpaddq_s32(p1, p2);

            // Extract delta bits
            let qh32 = (qh_vals[0] as u32) | ((qh_vals[1] as u32) << 8) | ((qh_vals[2] as u32) << 16) | ((qh_vals[3] as u32) << 24);
            let aux32 = ((qh32 >> 3) & 0x01010101) | ((qh32 >> 6) & 0x02020202);
            let aux8 = aux32.to_le_bytes();

            // Delta dot products using aux8 indices
            let p3 = vpaddq_s32(
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[0]), q8b.0),
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[1]), q8b.1),
            );
            let p4 = vpaddq_s32(
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[2]), q8b.2),
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[3]), q8b.3),
            );
            let p34 = vpaddq_s32(p3, p4);

            // Load scales (4 scales per 2 blocks, packed in one u16)
            let sc_val = *sc.add(ib / 2);
            let scales_arr: [i32; 4] = [
                (sc_val >> 0) as i32,
                (sc_val >> 3) as i32,
                (sc_val >> 6) as i32,
                (sc_val >> 9) as i32,
            ];
            let scales_4 = vld1q_s32(scales_arr.as_ptr());
            let scales_4 = vaddq_s32(vshlq_n_s32(vandq_s32(scales_4, mask), 1), mone);

            sumi1 = vmlaq_s32(sumi1, scales_4, p12);
            sumi2 = vmlaq_s32(sumi2, scales_4, p34);
        }

        sumf += y[i].d * fp16_to_fp32(scale_u16) * (vaddvq_s32(sumi1) as f32 + IQ1M_DELTA * vaddvq_s32(sumi2) as f32);
    }

    sumf
}

/// IQ2_XXS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq2_xxs_q8_k_neon(n: usize, x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut q2 = x[i].qs.as_ptr() as *const u32;
        let q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        // Process 8 blocks of 32 elements, 2 at a time
        for ib32 in (0..QK_K / 32).step_by(2) {
            // Load Q8 values (64 bytes for 2 blocks)
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));

            // Load q2 values (4 u32 for 2 blocks)
            let aux32: [u32; 4] = [
                *q2.add(0), *q2.add(1), *q2.add(2), *q2.add(3),
            ];
            q2 = q2.add(4);

            let aux8: [u8; 16] = std::mem::transmute(aux32);

            // Load grid values using indices from aux8
            let q2u_0 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[0] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[1] as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_1 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[2] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[3] as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_2 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[8] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[9] as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_3 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[10] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[11] as usize].to_le_bytes().as_ptr() as *const i8),
            );

            // Load signs from KEVEN_SIGNS_Q2XS
            let q2s_0 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[1] >> 0) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[1] >> 7) & 127) as usize * 8)),
            );
            let q2s_1 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[1] >> 14) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[1] >> 21) & 127) as usize * 8)),
            );
            let q2s_2 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[3] >> 0) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[3] >> 7) & 127) as usize * 8)),
            );
            let q2s_3 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[3] >> 14) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[3] >> 21) & 127) as usize * 8)),
            );

            // Apply signs
            let q2u_0 = vmulq_s8(q2u_0, q2s_0);
            let q2u_1 = vmulq_s8(q2u_1, q2s_1);
            let q2u_2 = vmulq_s8(q2u_2, q2s_2);
            let q2u_3 = vmulq_s8(q2u_3, q2s_3);

            // Dot product
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q2u_0, q8b.0), q2u_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q2u_2, q8b.2), q2u_3, q8b.3);

            // Apply scale
            let scale1 = 0.5f32 + ((aux32[1] >> 28) as f32);
            let scale2 = 0.5f32 + ((aux32[3] >> 28) as f32);

            sumf1 += vaddvq_s32(p1) as f32 * scale1;
            sumf2 += vaddvq_s32(p2) as f32 * scale2;
        }

        sumf += d * (sumf1 + sumf2) * 0.25;
    }

    sumf
}

/// IQ2_XS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq2_xs_q8_k_neon(n: usize, x: &[BlockIQ2XS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let q2 = x[i].qs.as_ptr();
        let q8 = y[i].qs.as_ptr();

        // Process scales
        let scales8 = vld1_u8(x[i].scales.as_ptr());
        let scales_l = vand_u8(scales8, vdup_n_u8(0xf));
        let scales_h = vshr_n_u8(scales8, 4);
        let scales = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));
        let scales = vaddq_u8(vshlq_n_u8(scales, 1), vdupq_n_u8(1));
        let scales1 = vmovl_u8(vget_low_u8(scales));
        let scales2 = vmovl_u8(vget_high_u8(scales));
        let scales32_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales1)));
        let scales32_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales1)));
        let scales32_2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales2)));
        let scales32_3 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales2)));

        let mut sumi = vdupq_n_s32(0);

        // Process 4 blocks of 64 elements
        for ib64 in 0..(QK_K / 64) {
            let q8b = vld1q_s8_x4(q8.add(ib64 * 64));

            // Load q2 values (8 u16 for 64 elements)
            let q2_vals: [u16; 8] = [
                *q2.add(ib64 * 8 + 0),
                *q2.add(ib64 * 8 + 1),
                *q2.add(ib64 * 8 + 2),
                *q2.add(ib64 * 8 + 3),
                *q2.add(ib64 * 8 + 4),
                *q2.add(ib64 * 8 + 5),
                *q2.add(ib64 * 8 + 6),
                *q2.add(ib64 * 8 + 7),
            ];

            // Load grid values using (q2 & 511) as index
            let q2u_0 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[0] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[1] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_1 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[2] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[3] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_2 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[4] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[5] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_3 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[6] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[7] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );

            // Load signs using (q2 >> 9) as index
            let q2s_0 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[0] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[1] >> 9) as usize * 8)),
            );
            let q2s_1 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[2] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[3] >> 9) as usize * 8)),
            );
            let q2s_2 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[4] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[5] >> 9) as usize * 8)),
            );
            let q2s_3 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[6] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[7] >> 9) as usize * 8)),
            );

            // Apply signs
            let q2u_0 = vmulq_s8(q2u_0, q2s_0);
            let q2u_1 = vmulq_s8(q2u_1, q2s_1);
            let q2u_2 = vmulq_s8(q2u_2, q2s_2);
            let q2u_3 = vmulq_s8(q2u_3, q2s_3);

            // Dot product
            let p1 = vdotq_s32_manual(vzero, q2u_0, q8b.0);
            let p2 = vdotq_s32_manual(vzero, q2u_1, q8b.1);
            let p3 = vdotq_s32_manual(vzero, q2u_2, q8b.2);
            let p4 = vdotq_s32_manual(vzero, q2u_3, q8b.3);
            let p = vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4));

            // Select appropriate scale vector based on block index
            let scale_vec = match ib64 {
                0 => scales32_0,
                1 => scales32_1,
                2 => scales32_2,
                3 => scales32_3,
                _ => scales32_0,
            };

            sumi = vmlaq_s32(sumi, p, scale_vec);
        }

        sumf += d * vaddvq_s32(sumi) as f32 * 0.125;
    }

    sumf
}

/// IQ2_S × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq2_s_q8_k_neon(n: usize, x: &[BlockIQ2S], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    // Masks for sign processing
    static K_MASK1: [u8; 32] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    ];
    static K_MASK2: [u8; 16] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    ];

    let mask1 = uint8x16x2_t(
        vld1q_u8(K_MASK1.as_ptr()),
        vld1q_u8(K_MASK1.as_ptr().add(16)),
    );
    let mask2 = vld1q_u8(K_MASK2.as_ptr());
    let m1 = vdupq_n_u8(1);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;

        let mut qs = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let mut signs = (x[i].qs.as_ptr().add(QK_K / 8)) as *const u16;
        let q8 = y[i].qs.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        // Process 8 blocks of 32 elements, 2 at a time
        for ib32 in (0..QK_K / 32).step_by(2) {
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));

            // Load grid indices (8 bytes)
            let qs_vals: [u8; 8] = [
                *qs.add(0), *qs.add(1), *qs.add(2), *qs.add(3),
                *qs.add(4), *qs.add(5), *qs.add(6), *qs.add(7),
            ];
            qs = qs.add(8);

            // Load high bits
            let qh0 = *qh.add(ib32) as u32;
            let qh1 = *qh.add(ib32 + 1) as u32;

            // Compute grid indices and load values
            let q2s_0 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[0] as usize | ((qh0 << 8) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[1] as usize | ((qh0 << 6) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2s_1 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[2] as usize | ((qh0 << 4) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[3] as usize | ((qh0 << 2) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2s_2 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[4] as usize | ((qh1 << 8) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[5] as usize | ((qh1 << 6) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2s_3 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[6] as usize | ((qh1 << 4) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[7] as usize | ((qh1 << 2) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );

            // Load signs (4 u16 values)
            let s0 = *signs.add(0);
            let s1 = *signs.add(1);
            let s2 = *signs.add(2);
            let s3 = *signs.add(3);
            signs = signs.add(4);

            // Process signs using TBL
            let vs_0 = vreinterpretq_u8_u32(vdupq_n_u32(s0 as u32 | ((s1 as u32) << 16)));
            let vs_1 = vandq_u8(vqtbl1q_u8(vs_0, mask1.1), mask2);
            let vs_0 = vandq_u8(vqtbl1q_u8(vs_0, mask1.0), mask2);
            let vs_0 = vceqq_u8(vs_0, mask2);
            let vs_1 = vceqq_u8(vs_1, mask2);

            // Apply signs
            let q2s_0 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_0, m1)), q2s_0);
            let q2s_1 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_1, m1)), q2s_1);

            // Process second block signs
            let vs_2 = vreinterpretq_u8_u32(vdupq_n_u32(s2 as u32 | ((s3 as u32) << 16)));
            let vs_3 = vandq_u8(vqtbl1q_u8(vs_2, mask1.1), mask2);
            let vs_2 = vandq_u8(vqtbl1q_u8(vs_2, mask1.0), mask2);
            let vs_2 = vceqq_u8(vs_2, mask2);
            let vs_3 = vceqq_u8(vs_3, mask2);

            let q2s_2 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_2, m1)), q2s_2);
            let q2s_3 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_3, m1)), q2s_3);

            // Dot product
            let p1 = vdotq_s32_manual(vzero, q2s_0, q8b.0);
            let p2 = vdotq_s32_manual(vzero, q2s_1, q8b.1);
            let p3 = vdotq_s32_manual(vzero, q2s_2, q8b.2);
            let p4 = vdotq_s32_manual(vzero, q2s_3, q8b.3);

            // Apply scales
            sumi1 += vaddvq_s32(p1) * (1 + 2 * (x[i].scales[ib32] as i32 & 0xf));
            sumi2 += vaddvq_s32(p2) * (1 + 2 * (x[i].scales[ib32] as i32 >> 4));
            sumi1 += vaddvq_s32(p3) * (1 + 2 * (x[i].scales[ib32 + 1] as i32 & 0xf));
            sumi2 += vaddvq_s32(p4) * (1 + 2 * (x[i].scales[ib32 + 1] as i32 >> 4));
        }

        sumf += d * (sumi1 + sumi2) as f32 * 0.125;
    }

    sumf
}

/// IQ4_XS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq4_xs_q8_k_neon(n: usize, x: &[BlockIQ4XS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);
    let values = vld1q_s8(KVALUES_IQ4NL.as_ptr());

    let mut sum = 0.0f32;

    for ibl in 0..nb {
        let q4 = x[ibl].qs.as_ptr();
        let q8 = y[ibl].qs.as_ptr();
        let mut h = x[ibl].scales_h;

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        for ib in 0..(QK_K / 64) {
            let q4bits = vld1q_u8_x2(q4.add(ib * 32));
            let q8bytes = vld1q_s8_x4(q8.add(ib * 64));

            let q4b_0 = vqtbl1q_s8(values, vandq_u8(q4bits.0, m4b));
            let q4b_1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.0, 4));
            let q4b_2 = vqtbl1q_s8(values, vandq_u8(q4bits.1, m4b));
            let q4b_3 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.1, 4));

            let prod_1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_0, q8bytes.0), q4b_1, q8bytes.1);
            let prod_2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_2, q8bytes.2), q4b_3, q8bytes.3);

            let ls1 = ((x[ibl].scales_l[ib] & 0x0F) as i32 | ((h as i32 & 0x0F) << 4)) - 32;
            let ls2 = ((x[ibl].scales_l[ib] >> 4) as i32 | ((h as i32 >> 4) << 4)) - 32;
            h >>= 8;

            sumi1 += vaddvq_s32(prod_1) * ls1;
            sumi2 += vaddvq_s32(prod_2) * ls2;
        }

        sum += fp16_to_fp32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2) as f32;
    }

    sum
}
