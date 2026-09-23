use crate::quant::types::*;
use crate::quant::vec_dot::arm::*;
use std::arch::aarch64::*;
use std::arch::asm;

/// Manual SDOT implementation via inline asm
#[inline(always)]
unsafe fn vdotq_s32_manual(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
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

/// Optimized Q4_K × Q8_K matrix-vector multiplication
/// Uses block-tiling and processes multiple rows at once
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_k_q8_k(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // Process in chunks of 16 rows to improve cache utilization
    const BLCK_1: usize = 16;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // Process rows in blocks
    for ib1 in (0..m).step_by(BLCK_1) {
        let ib1_end = (ib1 + BLCK_1).min(m);

        // Initialize sums for each row in this block - use stack array
        let mut row_sums = [0.0f32; BLCK_1];

        // Process all blocks of K dimension
        for ib in 0..nb {
            let x_block = &x[ib];
            let q8 = x_block.qs.as_ptr();
            let d_x = x_block.d;

            // Precompute q8-related values once for all rows
            let q8sums = vpaddq_s16(
                vld1q_s16(x_block.bsums.as_ptr()),
                vld1q_s16(x_block.bsums.as_ptr().add(8))
            );

            // Process each row in the block
            for ir in ib1..ib1_end {
                let w_block = &weights[ir * nb + ib];

                let d = d_x * half::f16::from_bits(w_block.d).to_f32();
                let dmin = d_x * half::f16::from_bits(w_block.dmin).to_f32();

                // Decode scales and mins
                let mut utmp = [0u32; 4];
                std::ptr::copy_nonoverlapping(w_block.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);

                let mut mins8 = vdup_n_u32(0);
                mins8 = vset_lane_u32(utmp[1] & KMASK1, mins8, 0);
                mins8 = vset_lane_u32(((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4), mins8, 1);

                utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
                utmp[0] &= KMASK1;

                let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
                let prod = vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                    vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins))
                );
                let min_sum = vaddvq_s32(prod);

                let scales = utmp.as_ptr() as *const u8;
                let q4 = w_block.qs.as_ptr();

                let mut sumi1 = 0i32;
                let mut sumi2 = 0i32;

                for j in 0..(QK_K / 64) {
                    let q4bits = vld1q_u8_x2(q4.add(j * 32));
                    let q8bytes_0 = vld1q_s8_x2(q8.add(j * 64));

                    let q4l_0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
                    let q4l_1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));

                    let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4l_0, q8bytes_0.0), q4l_1, q8bytes_0.1);
                    sumi1 += vaddvq_s32(p1) * *scales.add(j * 2) as i32;

                    let q8bytes_1 = vld1q_s8_x2(q8.add(j * 64 + 32));
                    let q4h_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
                    let q4h_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));

                    let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4h_0, q8bytes_1.0), q4h_1, q8bytes_1.1);
                    sumi2 += vaddvq_s32(p2) * *scales.add(j * 2 + 1) as i32;
                }

                row_sums[ir - ib1] += d * (sumi1 + sumi2) as f32 - dmin * min_sum as f32;
            }
        }

        // Store results
        for ir in ib1..ib1_end {
            dst[ir] = row_sums[ir - ib1];
        }
    }
}

/// Q6_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q6_k_q8_k(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_q6_k_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q5_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q5_k_q8_k(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_q5_k_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q3_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q3_k_q8_k(
    weights: &[BlockQ3K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_q3_k_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q2_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q2_k_q8_k(
    weights: &[BlockQ2K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_q2_k_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q4_0 × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_q4_0_q8_0(
    weights: &[BlockQ4_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    for i in 0..m {
        let row_sum = vec_dot_q4_0_q8_0_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q5_0 × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_q5_0_q8_0(
    weights: &[BlockQ5_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK8_0;

    for i in 0..m {
        let row_sum = vec_dot_q5_0_q8_0_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q4_1 × Q8_1 matrix-vector multiplication
pub unsafe fn matmul_q4_1_q8_1(
    weights: &[BlockQ4_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    for i in 0..m {
        let row_sum = vec_dot_q4_1_q8_1_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q5_1 × Q8_1 matrix-vector multiplication
pub unsafe fn matmul_q5_1_q8_1(
    weights: &[BlockQ5_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK8_0;

    for i in 0..m {
        let row_sum = vec_dot_q5_1_q8_1_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

// ===== IQ series matmul =====

/// IQ4_XS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq4_xs_q8_k(
    weights: &[BlockIQ4XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq4_xs_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ4_NL × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_iq4_nl_q8_0(
    weights: &[BlockIQ4NL],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    for i in 0..m {
        let row_sum = vec_dot_iq4_nl_q8_0_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ3_XXS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq3_xxs_q8_k(
    weights: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq3_xxs_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ3_S × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq3_s_q8_k(
    weights: &[BlockIQ3S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq3_s_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ2_XXS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq2_xxs_q8_k(
    weights: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq2_xxs_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ2_XS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq2_xs_q8_k(
    weights: &[BlockIQ2XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq2_xs_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ2_S × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq2_s_q8_k(
    weights: &[BlockIQ2S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq2_s_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ1_S × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq1_s_q8_k(
    weights: &[BlockIQ1S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq1_s_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// IQ1_M × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq1_m_q8_k(
    weights: &[BlockIQ1M],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_iq1_m_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

// ===== TQ series matmul =====

/// TQ2_0 × Q8_K matrix-vector multiplication
pub unsafe fn matmul_tq2_0_q8_k(
    weights: &[BlockTQ2_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_tq2_0_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// TQ1_0 × Q8_K matrix-vector multiplication
pub unsafe fn matmul_tq1_0_q8_k(
    weights: &[BlockTQ1_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for i in 0..m {
        let row_sum = vec_dot_tq1_0_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}
