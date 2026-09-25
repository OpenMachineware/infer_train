use std::arch::aarch64::*;

use crate::quant::types::{BlockQ8K, BlockTQ1_0, BlockTQ2_0, QK_K};

use super::fp::fp16_to_fp32;

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

/// TQ2_0 × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_tq2_0_q8_k_neon(n: usize, x: &[BlockTQ2_0], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let m3 = vdupq_n_u8(3);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let q2 = x[i].qs.as_ptr();
        let q8 = y[i].qs.as_ptr();

        let mut sumi0 = vdupq_n_s32(0);
        let mut sumi1 = vdupq_n_s32(0);

        // Process 64 bytes (256 elements, 2 bits each, 4 elements per byte)
        for j in (0..64).step_by(32) {
            let qx0 = vld1q_u8(q2.add(j));
            let qx1 = vld1q_u8(q2.add(j + 16));

            // Extract 2-bit values by shifting and masking
            let sqx0 = vreinterpretq_s8_u8(vandq_u8(qx0, m3));
            let sqx1 = vreinterpretq_s8_u8(vandq_u8(qx1, m3));
            let sqx2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx0, 2), m3));
            let sqx3 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx1, 2), m3));
            let sqx4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx0, 4), m3));
            let sqx5 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx1, 4), m3));
            let sqx6 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx0, 6), m3));
            let sqx7 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx1, 6), m3));

            // Load Q8 values
            let qy0 = vld1q_s8(q8.add(j * 4));
            let qy1 = vld1q_s8(q8.add(j * 4 + 16));
            let qy2 = vld1q_s8(q8.add(j * 4 + 32));
            let qy3 = vld1q_s8(q8.add(j * 4 + 48));
            let qy4 = vld1q_s8(q8.add(j * 4 + 64));
            let qy5 = vld1q_s8(q8.add(j * 4 + 80));
            let qy6 = vld1q_s8(q8.add(j * 4 + 96));
            let qy7 = vld1q_s8(q8.add(j * 4 + 112));

            // Dot products - accumulate in two registers for better ILP
            sumi0 = vdotq_s32_manual(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32_manual(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32_manual(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32_manual(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32_manual(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32_manual(sumi1, sqx5, qy5);
            sumi0 = vdotq_s32_manual(sumi0, sqx6, qy6);
            sumi1 = vdotq_s32_manual(sumi1, sqx7, qy7);
        }

        // Subtract bsums (for trinary representation bias)
        let ysum0 = vld1q_s16(y[i].bsums.as_ptr());
        let ysum1 = vld1q_s16(y[i].bsums.as_ptr().add(8));

        sumi0 = vaddq_s32(sumi0, sumi1);
        sumi0 = vsubq_s32(sumi0, vpaddlq_s16(vaddq_s16(ysum0, ysum1)));

        let d = fp16_to_fp32(x[i].d) * y[i].d;
        sumf += d * vaddvq_s32(sumi0) as f32;
    }

    sumf
}

/// TQ1_0 × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_tq1_0_q8_k_neon(n: usize, x: &[BlockTQ1_0], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;

    // Shift table for the last 16 elements
    static K_SHIFT: [u8; 16] = [1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27];
    let shift = vld1q_u8(K_SHIFT.as_ptr());

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let q1 = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();

        // Use dual accumulators for better ILP
        let mut sumi0 = vdupq_n_s32(0);
        let mut sumi1 = vdupq_n_s32(0);

        // First 32 bytes encode 160 elements (5 elements per byte)
        {
            let qx0 = vld1q_u8(q1.add(0));
            let qx1 = vld1q_u8(q1.add(16));

            // Multiply by powers of 3 to extract individual trits
            let qx2 = vmulq_u8(qx0, vdupq_n_u8(3));
            let qx3 = vmulq_u8(qx1, vdupq_n_u8(3));
            let qx4 = vmulq_u8(qx0, vdupq_n_u8(9));
            let qx5 = vmulq_u8(qx1, vdupq_n_u8(9));
            let qx6 = vmulq_u8(qx0, vdupq_n_u8(27));
            let qx7 = vmulq_u8(qx1, vdupq_n_u8(27));
            let qx8 = vmulq_u8(qx0, vdupq_n_u8(81));
            let qx9 = vmulq_u8(qx1, vdupq_n_u8(81));

            // Divide by 64 (shift right 6) to get the top 2 bits
            // This is equivalent to: (qx * 3 / 2 + qx / 2) >> 6 = round(qx * 1.5) >> 6
            let sqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx0, vshrq_n_u8(qx0, 1)), 6));
            let sqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx1, vshrq_n_u8(qx1, 1)), 6));
            let sqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx2, vshrq_n_u8(qx2, 1)), 6));
            let sqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx3, vshrq_n_u8(qx3, 1)), 6));
            let sqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx4, vshrq_n_u8(qx4, 1)), 6));
            let sqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx5, vshrq_n_u8(qx5, 1)), 6));
            let sqx6 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx6, vshrq_n_u8(qx6, 1)), 6));
            let sqx7 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx7, vshrq_n_u8(qx7, 1)), 6));
            let sqx8 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx8, vshrq_n_u8(qx8, 1)), 6));
            let sqx9 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx9, vshrq_n_u8(qx9, 1)), 6));

            // Load Q8 values
            let qy0 = vld1q_s8(q8.add(0));
            let qy1 = vld1q_s8(q8.add(16));
            let qy2 = vld1q_s8(q8.add(32));
            let qy3 = vld1q_s8(q8.add(48));
            let qy4 = vld1q_s8(q8.add(64));
            let qy5 = vld1q_s8(q8.add(80));
            let qy6 = vld1q_s8(q8.add(96));
            let qy7 = vld1q_s8(q8.add(112));
            let qy8 = vld1q_s8(q8.add(128));
            let qy9 = vld1q_s8(q8.add(144));

            // Dot products with dual accumulators
            sumi0 = vdotq_s32_manual(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32_manual(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32_manual(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32_manual(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32_manual(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32_manual(sumi1, sqx5, qy5);
            sumi0 = vdotq_s32_manual(sumi0, sqx6, qy6);
            sumi1 = vdotq_s32_manual(sumi1, sqx7, qy7);
            sumi0 = vdotq_s32_manual(sumi0, sqx8, qy8);
            sumi1 = vdotq_s32_manual(sumi1, sqx9, qy9);
        }

        // Last 16 bytes encode 80 elements
        {
            let qx0 = vld1q_u8(q1.add(32));

            // Extract trits for different positions
            let qx1 = vmulq_u8(qx0, vdupq_n_u8(3));
            let qx2 = vmulq_u8(qx0, vdupq_n_u8(9));
            let qx3 = vmulq_u8(qx0, vdupq_n_u8(27));
            let qx4 = vmulq_u8(qx0, vdupq_n_u8(81));

            // Load high bits
            let qh_val = std::ptr::read_unaligned(qh as *const u32);
            let qx5 = vmulq_u8(vreinterpretq_u8_u32(vdupq_n_u32(qh_val)), shift);

            // Extract trits
            let sqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx0, vshrq_n_u8(qx0, 1)), 6));
            let sqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx1, vshrq_n_u8(qx1, 1)), 6));
            let sqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx2, vshrq_n_u8(qx2, 1)), 6));
            let sqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx3, vshrq_n_u8(qx3, 1)), 6));
            let sqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx4, vshrq_n_u8(qx4, 1)), 6));
            let sqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx5, vshrq_n_u8(qx5, 1)), 6));

            // Load Q8 values
            let qy0 = vld1q_s8(q8.add(160));
            let qy1 = vld1q_s8(q8.add(176));
            let qy2 = vld1q_s8(q8.add(192));
            let qy3 = vld1q_s8(q8.add(208));
            let qy4 = vld1q_s8(q8.add(224));
            let qy5 = vld1q_s8(q8.add(240));

            // Dot products with dual accumulators
            sumi0 = vdotq_s32_manual(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32_manual(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32_manual(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32_manual(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32_manual(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32_manual(sumi1, sqx5, qy5);
        }

        // Combine accumulators and subtract bsums (for trinary -1, 0, 1 representation)
        let ysum0 = vld1q_s16(y[i].bsums.as_ptr());
        let ysum1 = vld1q_s16(y[i].bsums.as_ptr().add(8));

        sumi0 = vaddq_s32(sumi0, sumi1);
        sumi0 = vsubq_s32(sumi0, vpaddlq_s16(vaddq_s16(ysum0, ysum1)));

        let d = fp16_to_fp32(x[i].d) * y[i].d;
        sumf += d * vaddvq_s32(sumi0) as f32;
    }

    sumf
}
