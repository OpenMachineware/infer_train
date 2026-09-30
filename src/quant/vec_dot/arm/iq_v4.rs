// IQ3_XXS v4 - Match llama.cpp's exact grid lookup pattern
// Use ld1.s instead of vcreate_u64 for grid loading

use super::super::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};
use super::fp::fp16_to_fp32;
use crate::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use std::arch::aarch64::*;
use std::arch::asm;

/// Manual SDOT
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

/// Load 4 grid values using ld1.s (matches llama.cpp's ggml_vld1q_u32)
/// This is much more efficient than vcreate_u64 + vcombine_u32
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_grid_4values(
    grid: *const u32,
    idx0: usize,
    idx1: usize,
    idx2: usize,
    idx3: usize,
) -> uint32x4_t {
    let mut v: uint32x4_t;

    // Load using ldr + ld1.s (like llama.cpp)
    asm!(
        "ldr s0, [{grid}]",
        "ldr s1, [{grid1}]",
        "ld1.s {{v0}}[1], [{grid2}]",
        "ld1.s {{v0}}[2], [{grid3}]",
        grid = in(reg) grid.add(idx0),
        grid1 = in(reg) grid.add(idx1),
        grid2 = in(reg) grid.add(idx2),
        grid3 = in(reg) grid.add(idx3),
        out("v0") v,
        out("s1") _,
        options(pure, nomem, preserves_flags)
    );

    v
}

/// IQ3_XXS × Q8_K (v4 - optimized grid lookup)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v4(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let grid = IQ3XXS_GRID.as_ptr();
    let signs = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;

    let mut sumf = 0.0f32;
    let half = 0.5f32;
    let zero = vdupq_n_s32(0);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let qs = x[i].qs.as_ptr();
        let gas = qs.add(64);
        let q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        for ib in 0..4 {
            // Load Q8
            let q8b = vld1q_s8_x4(q8.add(ib * 64));

            // Load aux32
            let aux32_0 =
                std::ptr::read_unaligned(gas.add(ib * 8) as *const u32);
            let aux32_1 =
                std::ptr::read_unaligned(gas.add(ib * 8 + 4) as *const u32);

            // Grid lookup - use efficient ld1.s method
            let qs_off = qs.add(ib * 16);
            let v0 = load_grid_4values(
                grid,
                *qs_off.add(0) as usize,
                *qs_off.add(1) as usize,
                *qs_off.add(2) as usize,
                *qs_off.add(3) as usize,
            );
            let v1 = load_grid_4values(
                grid,
                *qs_off.add(4) as usize,
                *qs_off.add(5) as usize,
                *qs_off.add(6) as usize,
                *qs_off.add(7) as usize,
            );
            let v2 = load_grid_4values(
                grid,
                *qs_off.add(8) as usize,
                *qs_off.add(9) as usize,
                *qs_off.add(10) as usize,
                *qs_off.add(11) as usize,
            );
            let v3 = load_grid_4values(
                grid,
                *qs_off.add(12) as usize,
                *qs_off.add(13) as usize,
                *qs_off.add(14) as usize,
                *qs_off.add(15) as usize,
            );

            // Sign extraction using ubfx
            let (s0, s1, s2, s3): (u64, u64, u64, u64);
            let (s4, s5, s6, s7): (u64, u64, u64, u64);
            asm!(
                "ubfx {s0}, {aux0}, #0, #7",
                "ubfx {s1}, {aux0}, #7, #7",
                "ubfx {s2}, {aux0}, #14, #7",
                "ubfx {s3}, {aux0}, #21, #7",
                "ubfx {s4}, {aux1}, #0, #7",
                "ubfx {s5}, {aux1}, #7, #7",
                "ubfx {s6}, {aux1}, #14, #7",
                "ubfx {s7}, {aux1}, #21, #7",
                aux0 = in(reg) aux32_0 as u64,
                aux1 = in(reg) aux32_1 as u64,
                s0 = out(reg) s0,
                s1 = out(reg) s1,
                s2 = out(reg) s2,
                s3 = out(reg) s3,
                s4 = out(reg) s4,
                s5 = out(reg) s5,
                s6 = out(reg) s6,
                s7 = out(reg) s7,
                options(pure, nomem, preserves_flags)
            );

            // Sign application
            let q3s_0 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs.add(s0 as usize * 8)),
                    vld1_s8(signs.add(s1 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v0),
            );
            let q3s_1 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs.add(s2 as usize * 8)),
                    vld1_s8(signs.add(s3 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v1),
            );
            let q3s_2 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs.add(s4 as usize * 8)),
                    vld1_s8(signs.add(s5 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v2),
            );
            let q3s_3 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs.add(s6 as usize * 8)),
                    vld1_s8(signs.add(s7 as usize * 8)),
                ),
                vreinterpretq_s8_u32(v3),
            );

            // Dot products
            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(zero, q3s_0, q8b.0),
                q3s_1,
                q8b.1,
            );
            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(zero, q3s_2, q8b.2),
                q3s_3,
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
