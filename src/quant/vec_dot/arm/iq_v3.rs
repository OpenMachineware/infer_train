// IQ3_XXS v3 - Compact assembly matching llama.cpp's inner loop
// Focus on minimizing instruction count by using the same scheduling

use super::super::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};
use super::fp::fp16_to_fp32;
use super::iq::vld1q_u32_from_scalars;
use crate::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use std::arch::aarch64::*;
use std::arch::asm;

/// Manual SDOT instruction
#[target_feature(enable = "neon,dotprod")]
#[inline]
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

/// IQ3_XXS × Q8_K - v3 with compact assembly for sign extraction
/// Key: Use compact ubfx sequence + ldr.d + mov.d
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v3(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let grid = IQ3XXS_GRID.as_ptr();
    let signs = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;

    let mut sumf = 0.0f32;
    let half = 0.5f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let qs = x[i].qs.as_ptr();
        let gas = qs.add(QK_K / 4);
        let q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        // Process 4 iterations (ib32 = 0, 2, 4, 6) -> 4 x 64 bytes Q8
        for ib in 0..4 {
            // Load Q8: 64 bytes
            let q8b = vld1q_s8_x4(q8.add(ib * 64));

            // Load aux32: 2 x uint32_t (compact method)
            let aux32_0 =
                std::ptr::read_unaligned(gas.add(ib * 8) as *const u32);
            let aux32_1 =
                std::ptr::read_unaligned(gas.add(ib * 8 + 4) as *const u32);

            // Grid lookup: load 4 vectors using ld1.s (like llama.cpp)
            let qs_ptr = qs.add(ib * 16);

            // Use compact assembly for sign extraction and application
            let s0: int8x16_t;
            let s1: int8x16_t;
            let s2: int8x16_t;
            let s3: int8x16_t;

            // Grid lookup using intrinsics (simpler)
            let v0 = vld1q_u32_from_scalars(
                *grid.add(*qs_ptr.add(0) as usize),
                *grid.add(*qs_ptr.add(1) as usize),
                *grid.add(*qs_ptr.add(2) as usize),
                *grid.add(*qs_ptr.add(3) as usize),
            );
            let v1 = vld1q_u32_from_scalars(
                *grid.add(*qs_ptr.add(4) as usize),
                *grid.add(*qs_ptr.add(5) as usize),
                *grid.add(*qs_ptr.add(6) as usize),
                *grid.add(*qs_ptr.add(7) as usize),
            );
            let v2 = vld1q_u32_from_scalars(
                *grid.add(*qs_ptr.add(8) as usize),
                *grid.add(*qs_ptr.add(9) as usize),
                *grid.add(*qs_ptr.add(10) as usize),
                *grid.add(*qs_ptr.add(11) as usize),
            );
            let v3 = vld1q_u32_from_scalars(
                *grid.add(*qs_ptr.add(12) as usize),
                *grid.add(*qs_ptr.add(13) as usize),
                *grid.add(*qs_ptr.add(14) as usize),
                *grid.add(*qs_ptr.add(15) as usize),
            );

            // Sign extraction using compact ubfx + ldr.d + mov.d
            let s0: int8x16_t;
            let s1: int8x16_t;
            let s2: int8x16_t;
            let s3: int8x16_t;

            asm!(
                // Extract sign indices
                "ubfx x4, {aux0}, #0, #7",
                "ubfx x5, {aux0}, #7, #7",
                "ubfx x6, {aux0}, #14, #7",
                "ubfx x7, {aux0}, #21, #7",
                "ubfx x8, {aux1}, #0, #7",
                "ubfx x9, {aux1}, #7, #7",
                "ubfx x10, {aux1}, #14, #7",
                "ubfx x11, {aux1}, #21, #7",

                // Load sign vectors (8 bytes each)
                "ldr d16, [{signs}, x4, lsl #3]",
                "ldr d17, [{signs}, x5, lsl #3]",
                "ldr d18, [{signs}, x6, lsl #3]",
                "ldr d19, [{signs}, x7, lsl #3]",
                "ldr d20, [{signs}, x8, lsl #3]",
                "ldr d21, [{signs}, x9, lsl #3]",
                "ldr d22, [{signs}, x10, lsl #3]",
                "ldr d23, [{signs}, x11, lsl #3]",

                // Combine: mov.d v16[1], v17[0]
                "mov.d v16[1], v17[0]",
                "mov.d v18[1], v19[0]",
                "mov.d v20[1], v21[0]",
                "mov.d v22[1], v23[0]",

                // Apply signs to grid values
                "mul.16b v0, v16, v0",
                "mul.16b v1, v18, v1",
                "mul.16b v2, v20, v2",
                "mul.16b v3, v22, v3",

                aux0 = in(reg) aux32_0 as u64,
                aux1 = in(reg) aux32_1 as u64,
                signs = in(reg) signs,
                out("v0") s0,
                out("v1") s1,
                out("v2") s2,
                out("v3") s3,
                out("x4") _,
                out("x5") _,
                out("x6") _,
                out("x7") _,
                out("x8") _,
                out("x9") _,
                out("x10") _,
                out("x11") _,
                out("d16") _,
                out("d17") _,
                out("d18") _,
                out("d19") _,
                out("d20") _,
                out("d21") _,
                out("d22") _,
                out("d23") _,
                options(preserves_flags, nomem)
            );

            // Dot products
            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(vdupq_n_s32(0), s0, q8b.0),
                s1,
                q8b.1,
            );
            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(vdupq_n_s32(0), s2, q8b.2),
                s3,
                q8b.3,
            );

            // Accumulate
            sumf1 += vaddvq_s32(p1) as f32 * (half + (aux32_0 >> 28) as f32);
            sumf2 += vaddvq_s32(p2) as f32 * (half + (aux32_1 >> 28) as f32);
        }

        sumf += d * (sumf1 + sumf2);
    }

    0.5 * sumf
}
