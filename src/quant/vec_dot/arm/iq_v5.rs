// IQ3_XXS v5 - Match llama.cpp's exact Q8 loading pattern
// Use ld1.16b with post-increment for efficient Q8 loading

use super::super::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};
use super::fp::fp16_to_fp32;
use super::iq::vld1q_u32_from_scalars;
use crate::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use std::arch::aarch64::*;
use std::arch::asm;

/// Manual SDOT implementation (avoids unstable feature)
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

/// Load 4 x int8x16_t vectors using ld1.16b with post-increment (matches llama.cpp)
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vld1q_s8_x4_ld1(
    ptr: *const i8,
) -> (int8x16_t, int8x16_t, int8x16_t, int8x16_t) {
    let v0: int8x16_t;
    let v1: int8x16_t;
    let v2: int8x16_t;
    let v3: int8x16_t;
    let mut ptr_mut = ptr as *mut i8;

    asm!(
        "ld1.16b {{v0, v1, v2, v3}}, [{ptr}], #64",
        ptr = inout(reg) ptr_mut,
        out("v0") v0,
        out("v1") v1,
        out("v2") v2,
        out("v3") v3,
        options(pure, nomem, preserves_flags)
    );

    (v0, v1, v2, v3)
}

/// Load aux32 pair with post-increment
#[target_feature(enable = "neon")]
#[inline]
unsafe fn ldp_u32_with_increment(ptr: *const u8) -> (u32, u32, *const u8) {
    let mut ptr_mut = ptr as *mut u8;
    let aux0: u32;
    let aux1: u32;

    asm!(
        "ldp {aux0:w}, {aux1:w}, [{ptr}], #8",
        ptr = inout(reg) ptr_mut,
        aux0 = out(reg) aux0,
        aux1 = out(reg) aux1,
        options(pure, nomem, preserves_flags)
    );

    (aux0, aux1, ptr_mut)
}

/// Load aux32, extract signs and scales in ONE assembly block
/// Uses w2/w0 directly without saving to Rust variables (KEY OPTIMIZATION)
#[target_feature(enable = "neon")]
#[inline]
unsafe fn ldp_extract_signs_scales(
    ptr: *const u8,
) -> ((u64, u64, u64, u64), (u64, u64, u64, u64), (u32, u32), *const u8) {
    let mut ptr_mut = ptr as *mut u8;
    let s0: u64;
    let s1: u64;
    let s2: u64;
    let s3: u64;
    let s4: u64;
    let s5: u64;
    let s6: u64;
    let s7: u64;
    let scale0: u32;
    let scale1: u32;

    asm!(
        "ldp w2, w0, [{ptr}], #8",
        "and {s0}, x2, #0x7f",
        "ubfx {s1}, x2, #7, #7",
        "ubfx {s2}, x2, #14, #7",
        "ubfx {s3}, x2, #21, #7",
        "lsr {scale0}, x2, #28",
        "and {s4}, x0, #0x7f",
        "ubfx {s5}, x0, #7, #7",
        "ubfx {s6}, x0, #14, #7",
        "ubfx {s7}, x0, #21, #7",
        "lsr {scale1}, x0, #28",
        ptr = inout(reg) ptr_mut,
        s0 = out(reg) s0,
        s1 = out(reg) s1,
        s2 = out(reg) s2,
        s3 = out(reg) s3,
        s4 = out(reg) s4,
        s5 = out(reg) s5,
        s6 = out(reg) s6,
        s7 = out(reg) s7,
        scale0 = out(reg) scale0,
        scale1 = out(reg) scale1,
        out("w2") _,
        out("w0") _,
        options(pure, nomem, preserves_flags)
    );

    ((s0, s1, s2, s3), (s4, s5, s6, s7), (scale0, scale1), ptr_mut)
}

/// IQ3_XXS × Q8_K (v5 - optimized Q8 loading)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v5(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;

    let mut sum = 0.0f32;
    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;
    let grid = IQ3XXS_GRID.as_ptr();
    let half = 0.5f32;
    let zero = vdupq_n_s32(0);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut q3 = x[i].qs.as_ptr();
        let mut gas = q3.add(QK_K / 4);
        let mut q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        for _ in 0..4 {
            // Q8 load using ld1.16b with post-increment (KEY OPTIMIZATION)
            let (q8b_0, q8b_1, q8b_2, q8b_3) = vld1q_s8_x4_ld1(q8);
            q8 = q8.add(64);

            // aux32 load + sign + scale extraction in ONE assembly block (KEY OPTIMIZATION)
            // w2/w0 are never saved to Rust variables - no mov instructions!
            let ((s0, s1, s2, s3), (s4, s5, s6, s7), (scale0, scale1), new_gas) =
                ldp_extract_signs_scales(gas);
            gas = new_gas;

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

            // Dot products using SDOT
            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(zero, q3s_0, q8b_0),
                q3s_1,
                q8b_1,
            );
            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(zero, q3s_2, q8b_2),
                q3s_3,
                q8b_3,
            );

            // Accumulate - use pre-extracted scales (no lsr in Rust code)
            sumf1 += vaddvq_s32(p1) as f32 * (half + scale0 as f32);
            sumf2 += vaddvq_s32(p2) as f32 * (half + scale1 as f32);
        }

        sum += d * (sumf1 + sumf2) * half;
    }

    sum
}
