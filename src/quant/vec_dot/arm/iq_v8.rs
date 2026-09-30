// IQ3_XXS v8 - Efficient grid loading like IQ2_XXS

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

/// Load grid values: construct vector from 4 grid entries
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_grid_4entries(
    grid: *const u32,
    idx0: u8,
    idx1: u8,
    idx2: u8,
    idx3: u8,
) -> int8x16_t {
    let bytes0 = (*grid.add(idx0 as usize)).to_le_bytes();
    let bytes1 = (*grid.add(idx1 as usize)).to_le_bytes();
    let bytes2 = (*grid.add(idx2 as usize)).to_le_bytes();
    let bytes3 = (*grid.add(idx3 as usize)).to_le_bytes();

    // Construct vector directly from bytes
    vcombine_s8(
        vcreate_s8(u64::from_le_bytes([
            bytes0[0], bytes0[1], bytes0[2], bytes0[3], bytes1[0], bytes1[1],
            bytes1[2], bytes1[3],
        ])),
        vcreate_s8(u64::from_le_bytes([
            bytes2[0], bytes2[1], bytes2[2], bytes2[3], bytes3[0], bytes3[1],
            bytes3[2], bytes3[3],
        ])),
    )
}

/// IQ3_XXS × Q8_K (v8 - efficient grid loading)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v8(
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
            // Q8 load
            let q8b = vld1q_s8_x4(q8);
            q8 = q8.add(64);

            // aux32 load
            let aux32_0 = *(gas as *const u32);
            let aux32_1 = *(gas.add(4) as *const u32);
            gas = gas.add(8);

            // Grid lookup - load directly like IQ2_XXS
            let v0 = vreinterpretq_u32_s8(load_grid_4entries(
                grid,
                *q3.add(0),
                *q3.add(1),
                *q3.add(2),
                *q3.add(3),
            ));
            let v1 = vreinterpretq_u32_s8(load_grid_4entries(
                grid,
                *q3.add(4),
                *q3.add(5),
                *q3.add(6),
                *q3.add(7),
            ));
            let v2 = vreinterpretq_u32_s8(load_grid_4entries(
                grid,
                *q3.add(8),
                *q3.add(9),
                *q3.add(10),
                *q3.add(11),
            ));
            let v3 = vreinterpretq_u32_s8(load_grid_4entries(
                grid,
                *q3.add(12),
                *q3.add(13),
                *q3.add(14),
                *q3.add(15),
            ));
            q3 = q3.add(16);

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

        sum += d * (sumf1 + sumf2) * half;
    }

    sum
}
