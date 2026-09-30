// IQ3_XXS v6 - Minimal assembly, let compiler optimize the rest
// Focus on eliminating the 2 mov instructions

use super::super::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};
use super::fp::fp16_to_fp32;
use super::iq::vld1q_u32_from_scalars;
use crate::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use std::arch::aarch64::*;
use std::arch::asm;

/// Manual vdotq_s32 using SDOT instruction
#[target_feature(enable = "neon,dotprod")]
#[inline]
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

/// IQ3_XXS × Q8_K (v6 - eliminate mov instructions)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v6(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;

    let mut sum = 0.0f32;
    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;
    let grid = IQ3XXS_GRID.as_ptr();
    let half = 0.5f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut q3 = x[i].qs.as_ptr();
        let mut gas = q3.add(QK_K / 4);
        let mut q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        for _ in 0..4 {
            // Q8 load
            let q8b = vld1q_s8_x4(q8);
            q8 = q8.add(64);

            // Inline assembly block for aux32 loading and sign extraction
            // This keeps aux32 values in registers without mov
            let mut aux32_0: u32;
            let mut aux32_1: u32;
            let (s0, s1, s2, s3, s4, s5, s6, s7): (
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
            );

            asm!(
                // Load aux32 with post-increment
                "ldp w2, w0, [{gas}], #8",
                // Extract signs directly from w2/w0 (no intermediate mov)
                "and {s0}, x2, #0x7f",
                "ubfx {s1}, x2, #7, #7",
                "ubfx {s2}, x2, #14, #7",
                "ubfx {s3}, x2, #21, #7",
                "and {s4}, x0, #0x7f",
                "ubfx {s5}, x0, #7, #7",
                "ubfx {s6}, x0, #14, #7",
                "ubfx {s7}, x0, #21, #7",
                // Extract scales
                "lsr {aux0}, w2, #28",
                "lsr {aux1}, w0, #28",
                gas = inout(reg) gas,
                s0 = out(reg) s0,
                s1 = out(reg) s1,
                s2 = out(reg) s2,
                s3 = out(reg) s3,
                s4 = out(reg) s4,
                s5 = out(reg) s5,
                s6 = out(reg) s6,
                s7 = out(reg) s7,
                aux0 = out(reg) aux32_0,
                aux1 = out(reg) aux32_1,
                out("w2") _,
                out("w0") _,
                options(pure, nomem, preserves_flags)
            );

            // Grid lookup
            let v0 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(0) as usize),
                *grid.add(*q3.add(1) as usize),
                *grid.add(*q3.add(2) as usize),
                *grid.add(*q3.add(3) as usize),
            );
            let v1 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(4) as usize),
                *grid.add(*q3.add(5) as usize),
                *grid.add(*q3.add(6) as usize),
                *grid.add(*q3.add(7) as usize),
            );
            let v2 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(8) as usize),
                *grid.add(*q3.add(9) as usize),
                *grid.add(*q3.add(10) as usize),
                *grid.add(*q3.add(11) as usize),
            );
            let v3 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(12) as usize),
                *grid.add(*q3.add(13) as usize),
                *grid.add(*q3.add(14) as usize),
                *grid.add(*q3.add(15) as usize),
            );
            q3 = q3.add(16);

            // Sign application
            let q3s_0 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(s0 as usize * 8)),
                    vld1_s8(signs64.add(s1 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v0),
            );
            let q3s_1 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(s2 as usize * 8)),
                    vld1_s8(signs64.add(s3 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v1),
            );
            let q3s_2 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(s4 as usize * 8)),
                    vld1_s8(signs64.add(s5 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v2),
            );
            let q3s_3 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(s6 as usize * 8)),
                    vld1_s8(signs64.add(s7 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v3),
            );

            // Dot products
            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(vdupq_n_s32(0), q3s_0, q8b.0),
                q3s_1,
                q8b.1,
            );
            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(vdupq_n_s32(0), q3s_2, q8b.2),
                q3s_3,
                q8b.3,
            );

            // Accumulate with extracted scales
            sumf1 += vaddvq_s32(p1) as f32 * (half + aux32_0 as f32);
            sumf2 += vaddvq_s32(p2) as f32 * (half + aux32_1 as f32);
        }

        sum += d * (sumf1 + sumf2);
    }

    0.5 * sum
}
