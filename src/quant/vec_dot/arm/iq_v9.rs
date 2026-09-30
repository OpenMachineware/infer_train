// IQ3_XXS v9 - Exact llama.cpp NEON intrinsics copy
// No compiler optimization differences - direct intrinsics

use super::fp::fp16_to_fp32;
use super::iq::{vdotq_s32_manual, vld1q_u32_from_scalars};
use crate::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use crate::quant::vec_dot::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};
use std::arch::aarch64::*;

/// IQ3_XXS × Q8_K (exact llama.cpp NEON copy)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v9(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;

    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;
    let grid = IQ3XXS_GRID.as_ptr();

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut q3 = x[i].qs.as_ptr();
        let mut gas = q3.add(QK_K / 4);
        let mut q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        // Process 2 blocks at a time (matches llama.cpp loop)
        for _ in 0..2 {
            // Q8 load - 4 vectors of 16 int8
            let q8b_0 = vld1q_s8(q8);
            let q8b_1 = vld1q_s8(q8.add(16));
            let q8b_2 = vld1q_s8(q8.add(32));
            let q8b_3 = vld1q_s8(q8.add(48));
            q8 = q8.add(64);

            // aux32 load
            let aux32_0 = std::ptr::read_unaligned(gas as *const u32);
            let aux32_1 = std::ptr::read_unaligned(gas.add(4) as *const u32);
            gas = gas.add(8);

            // Grid lookup - exactly like llama.cpp
            let aux32x4_0 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(0) as usize),
                *grid.add(*q3.add(1) as usize),
                *grid.add(*q3.add(2) as usize),
                *grid.add(*q3.add(3) as usize),
            );
            let aux32x4_1 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(4) as usize),
                *grid.add(*q3.add(5) as usize),
                *grid.add(*q3.add(6) as usize),
                *grid.add(*q3.add(7) as usize),
            );
            let aux32x4_2 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(8) as usize),
                *grid.add(*q3.add(9) as usize),
                *grid.add(*q3.add(10) as usize),
                *grid.add(*q3.add(11) as usize),
            );
            let aux32x4_3 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(12) as usize),
                *grid.add(*q3.add(13) as usize),
                *grid.add(*q3.add(14) as usize),
                *grid.add(*q3.add(15) as usize),
            );
            q3 = q3.add(16);

            // Sign loading - exactly like llama.cpp
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

            // Dot products
            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(vdupq_n_s32(0), q3s_0, q8b_0),
                q3s_1,
                q8b_1,
            );
            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(vdupq_n_s32(0), q3s_2, q8b_2),
                q3s_3,
                q8b_3,
            );

            // Accumulate with scales
            sumf1 += vaddvq_s32(p1) as f32 * (0.5 + (aux32_0 >> 28) as f32);
            sumf2 += vaddvq_s32(p2) as f32 * (0.5 + (aux32_1 >> 28) as f32);
        }

        sumf += d * (sumf1 + sumf2);
    }

    0.5 * sumf
}
