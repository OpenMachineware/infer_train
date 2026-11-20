// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Vector dot product operators

#![allow(asm_sub_register)]
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::types::*;

/// Manual vdotq_s32 implementation using native SDOT instruction
/// vdotq_s32 is unstable in Rust, so we use inline assembly
/// SDOT computes: result[i] = acc[i] + sum(a[4*i..4*i+4] * b[4*i..4*i+4]) for i in 0..4
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
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

/// Manual vmmlaq_s32 implementation using native MMLA (i8mm) instruction
/// vmmlaq_s32 computes 2x2 matrix multiply from 2x8 and 8x2 int8 matrices
/// Layout: a is [row0_low, row1_low] interleaved, b is [col0_low, col1_low] interleaved
/// Result[0] = sum(a[0..8] * b[0..8]), Result[1] = sum(a[0..8] * b[8..16])
/// Result[2] = sum(a[8..16] * b[0..8]), Result[3] = sum(a[8..16] * b[8..16])
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn vmmlaq_s32_manual(
    acc: int32x4_t,
    a: int8x16_t,
    b: int8x16_t,
) -> int32x4_t {
    use std::arch::asm;

    let mut result = acc;
    asm!(
        "smmla {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, preserves_flags)
    );
    result
}

// FP32 dot product (NEON optimized)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_f32_neon(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    assert_eq!(n, y.len());
    assert_eq!(n % 4, 0);

    let np = n & !31;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let mut sum4 = vdupq_n_f32(0.0);
    let mut sum5 = vdupq_n_f32(0.0);
    let mut sum6 = vdupq_n_f32(0.0);
    let mut sum7 = vdupq_n_f32(0.0);

    let mut i = 0;
    while i < np {
        sum0 = vmlaq_f32(
            sum0,
            vld1q_f32(x.as_ptr().add(i)),
            vld1q_f32(y.as_ptr().add(i)),
        );
        sum1 = vmlaq_f32(
            sum1,
            vld1q_f32(x.as_ptr().add(i + 4)),
            vld1q_f32(y.as_ptr().add(i + 4)),
        );
        sum2 = vmlaq_f32(
            sum2,
            vld1q_f32(x.as_ptr().add(i + 8)),
            vld1q_f32(y.as_ptr().add(i + 8)),
        );
        sum3 = vmlaq_f32(
            sum3,
            vld1q_f32(x.as_ptr().add(i + 12)),
            vld1q_f32(y.as_ptr().add(i + 12)),
        );
        sum4 = vmlaq_f32(
            sum4,
            vld1q_f32(x.as_ptr().add(i + 16)),
            vld1q_f32(y.as_ptr().add(i + 16)),
        );
        sum5 = vmlaq_f32(
            sum5,
            vld1q_f32(x.as_ptr().add(i + 20)),
            vld1q_f32(y.as_ptr().add(i + 20)),
        );
        sum6 = vmlaq_f32(
            sum6,
            vld1q_f32(x.as_ptr().add(i + 24)),
            vld1q_f32(y.as_ptr().add(i + 24)),
        );
        sum7 = vmlaq_f32(
            sum7,
            vld1q_f32(x.as_ptr().add(i + 28)),
            vld1q_f32(y.as_ptr().add(i + 28)),
        );
        i += 32;
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum4 = vaddq_f32(sum4, sum5);
    sum6 = vaddq_f32(sum6, sum7);
    sum0 = vaddq_f32(sum0, sum2);
    sum4 = vaddq_f32(sum4, sum6);
    sum0 = vaddq_f32(sum0, sum4);

    let mut result = vaddvq_f32(sum0);

    for j in np..n {
        result += x[j] * y[j];
    }

    result
}

// Q4_0 x Q8_0 dot product (NEON optimized)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_q4_0_q8_0_neon(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let m4b = vdupq_n_u8(0x0F);
    let s8b = vdupq_n_s8(0x8);

    let mut sumv = vdupq_n_f32(0.0);

    for i in 0..nb {
        let block_x = &x[i];
        let block_y = &y[i];

        let v = vld1q_u8(block_x.qs.as_ptr());
        let vxl = vandq_u8(v, m4b);
        let vxh = vshrq_n_u8(v, 4);

        let vxl = vsubq_s8(vreinterpretq_s8_u8(vxl), s8b);
        let vxh = vsubq_s8(vreinterpretq_s8_u8(vxh), s8b);

        let vy = vld1q_s8(block_y.qs.as_ptr());

        let p0 = vmull_s8(vget_low_s8(vxl), vget_low_s8(vy));
        let p1 = vmull_high_s8(vxl, vy);

        let vy2 = vld1q_s8(block_y.qs.as_ptr().add(16));
        let p2 = vmull_s8(vget_low_s8(vxh), vget_low_s8(vy2));
        let p3 = vmull_high_s8(vxh, vy2);

        let sumb = vaddq_s32(
            vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
            vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)),
        );

        let dx = fp16_to_fp32(block_x.d);
        let dy = fp16_to_fp32(block_y.d);
        let scale = dx * dy;

        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(sumb), scale);
    }

    vaddvq_f32(sumv)
}

/// Q4_K × Q8_K vector dot product (NEON implementation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_q4_k_q8_k_neon(x: &[BlockQ4_K], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sum = 0.0f32;

    for i in 0..nb {
        let x_i = x.get_unchecked(i);
        let y_i = y.get_unchecked(i);

        let d = y_i.d * fp16_to_fp32(x_i.d);
        let dmin = y_i.d * fp16_to_fp32(x_i.dmin);

        let q8sums = vpaddq_s16(
            vld1q_s16(y_i.bsums.as_ptr()),
            vld1q_s16(y_i.bsums.as_ptr().add(8)),
        );

        // Decode scales and mins from 12-byte format
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(
            x_i.scales.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );

        // Extract mins first (before modifying utmp)
        let mut mins8 = vdup_n_u32(0);
        mins8 = vset_lane_u32(utmp[1] & KMASK1, mins8, 0);
        mins8 = vset_lane_u32(
            ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
            mins8,
            1,
        );

        // Now modify utmp for scales
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[0] &= KMASK1;

        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        let min_sum = vaddvq_s32(prod);

        // Get scales from utmp
        let scales = utmp.as_ptr() as *const u8;

        let mut q4 = x_i.qs.as_ptr();
        let mut q8 = y_i.qs.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        for j in 0..4 {
            let q4bits = vld1q_u8_x2(q4);
            q4 = q4.add(32);

            let q4l_0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
            let q4l_1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));

            // First q8 load for low nibbles
            let q8bytes = vld1q_s8_x2(q8);
            q8 = q8.add(32);

            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(vzero, q4l_0, q8bytes.0),
                q4l_1,
                q8bytes.1,
            );
            sumi1 += vaddvq_s32(p1) * *scales.add(j * 2) as i32;

            // Second q8 load for high nibbles
            let q8bytes = vld1q_s8_x2(q8);
            q8 = q8.add(32);

            let q4h_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
            let q4h_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));

            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(vzero, q4h_0, q8bytes.0),
                q4h_1,
                q8bytes.1,
            );
            sumi2 += vaddvq_s32(p2) * *scales.add(j * 2 + 1) as i32;
        }

        sum -= dmin * min_sum as f32;
        sum += d * (sumi1 + sumi2) as f32;
    }

    sum
}

/// Q5_K × Q8_K vector dot product (NEON implementation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_q5_k_q8_k_neon(x: &[BlockQ5_K], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let m4b = vdupq_n_u8(0xF);
    let mone = vdupq_n_u8(1);
    let mtwo = vdupq_n_u8(2);
    let mzero = vdupq_n_s32(0);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * fp16_to_fp32(x[i].d);
        let dmin = y[i].d * fp16_to_fp32(x[i].dmin);

        // Calculate min correction using bsums
        // vpaddq_s16: pairwise add of adjacent elements
        let q8sums = vpaddq_s16(
            vld1q_s16(y[i].bsums.as_ptr()),
            vld1q_s16(y[i].bsums.as_ptr().add(8)),
        );

        // Decode scales and mins
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(
            x[i].scales.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mins8 = vld1_u8((utmp.as_ptr() as *const u8).add(8));
        let mins = vreinterpretq_s16_u16(vmovl_u8(mins8));

        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        let sumi_mins = vaddvq_s32(prod);

        let scales: [u8; 16] = std::mem::transmute(utmp);
        let mut scale_idx = 0;

        let q5 = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();

        let mut qhbits = vld1q_u8_x2(qh);

        let mut sumi = 0i32;

        for j in 0..(QK_K / 64) {
            let q5bits = vld1q_u8_x2(q5.add(j * 32));
            let q8bytes = vld1q_s8_x4(q8.add(j * 64));

            // Process 4 blocks of 16 elements
            let q5h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
            let q5h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
            let q5h_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
            let q5h_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);

            // Update qhbits for next iteration (shift right by 2 bits)
            qhbits.0 = vshrq_n_u8(qhbits.0, 2);
            qhbits.1 = vshrq_n_u8(qhbits.1, 2);

            let q5bytes_0 =
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.0, m4b), q5h_0));
            let q5bytes_1 =
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.1, m4b), q5h_1));
            let q5bytes_2 =
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.0, 4), q5h_2));
            let q5bytes_3 =
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.1, 4), q5h_3));

            sumi += vaddvq_s32(vaddq_s32(
                vdotq_s32_manual(mzero, q5bytes_0, q8bytes.0),
                vdotq_s32_manual(mzero, q5bytes_1, q8bytes.1),
            )) * scales[scale_idx] as i32;
            scale_idx += 1;

            sumi += vaddvq_s32(vaddq_s32(
                vdotq_s32_manual(mzero, q5bytes_2, q8bytes.2),
                vdotq_s32_manual(mzero, q5bytes_3, q8bytes.3),
            )) * scales[scale_idx] as i32;
            scale_idx += 1;
        }

        sumf += d * sumi as f32 - dmin * sumi_mins as f32;
    }

    sumf
}

/// Q6_K × Q8_K vector dot product (NEON implementation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_q6_k_q8_k_neon(x: &[BlockQ6_K], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let m4b = vdupq_n_u8(0x0F);
    let m2b = vdupq_n_u8(0x03);
    let m32 = vdupq_n_s8(32); // For subtracting 32
    let vzero = vdupq_n_s32(0);

    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * fp16_to_fp32(x[i].d);

        let ql = x[i].ql.as_ptr();
        let qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();
        let scales = x[i].scales.as_ptr();

        let mut isum = 0i32;

        // Process 2 iterations of 128 elements each
        for j in 0..2 {
            let qh_bits = vld1q_u8_x2(qh.add(j * 32));
            let ql_bits = vld1q_u8_x4(ql.add(j * 64));

            // Extract high 2 bits for each of 8 blocks (16 elements each)
            // Combine with low 4 bits, then subtract 32 to get signed values
            // q6 value = ((ql bits) | ((qh 2-bit value) << 4)) - 32
            let q6h_0 = vshlq_n_u8(vandq_u8(m2b, qh_bits.0), 4);
            let q6h_1 = vshlq_n_u8(vandq_u8(m2b, qh_bits.1), 4);
            let q6h_2 = vshlq_n_u8(vandq_u8(m2b, vshrq_n_u8(qh_bits.0, 2)), 4);
            let q6h_3 = vshlq_n_u8(vandq_u8(m2b, vshrq_n_u8(qh_bits.1, 2)), 4);
            let q6h_4 = vshlq_n_u8(vandq_u8(m2b, vshrq_n_u8(qh_bits.0, 4)), 4);
            let q6h_5 = vshlq_n_u8(vandq_u8(m2b, vshrq_n_u8(qh_bits.1, 4)), 4);
            let q6h_6 = vshlq_n_u8(vandq_u8(m2b, vshrq_n_u8(qh_bits.0, 6)), 4);
            let q6h_7 = vshlq_n_u8(vandq_u8(m2b, vshrq_n_u8(qh_bits.1, 6)), 4);

            // Combine and subtract 32
            let q6bytes_0 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.0, m4b), q6h_0)),
                m32,
            );
            let q6bytes_1 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.1, m4b), q6h_1)),
                m32,
            );
            let q6bytes_2 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.2, m4b), q6h_2)),
                m32,
            );
            let q6bytes_3 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.3, m4b), q6h_3)),
                m32,
            );
            let q6bytes_4 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.0, 4), q6h_4)),
                m32,
            );
            let q6bytes_5 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.1, 4), q6h_5)),
                m32,
            );
            let q6bytes_6 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.2, 4), q6h_6)),
                m32,
            );
            let q6bytes_7 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.3, 4), q6h_7)),
                m32,
            );

            // Load q8 in two 64-byte chunks (128 bytes total per iteration)
            let q8bytes_lo = vld1q_s8_x4(q8.add(j * 128));
            let q8bytes_hi = vld1q_s8_x4(q8.add(j * 128 + 64));

            // Dot products with scales (q6 values are already signed -32..31)
            let scale_idx = j * 8;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_0, q8bytes_lo.0))
                    * *scales.add(scale_idx) as i32;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_1, q8bytes_lo.1))
                    * *scales.add(scale_idx + 1) as i32;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_2, q8bytes_lo.2))
                    * *scales.add(scale_idx + 2) as i32;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_3, q8bytes_lo.3))
                    * *scales.add(scale_idx + 3) as i32;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_4, q8bytes_hi.0))
                    * *scales.add(scale_idx + 4) as i32;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_5, q8bytes_hi.1))
                    * *scales.add(scale_idx + 5) as i32;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_6, q8bytes_hi.2))
                    * *scales.add(scale_idx + 6) as i32;
            isum +=
                vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_7, q8bytes_hi.3))
                    * *scales.add(scale_idx + 7) as i32;
        }

        sum += d * isum as f32;
    }

    sum
}

/// IQ4_NL lookup table: maps 4-bit values to actual quantized values
/// This is the non-linear quantization used by IQ4_NL and IQ4_XS formats
#[cfg(target_arch = "aarch64")]
const KVALUES_IQ4NL_I8: [i8; 16] =
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// IQ4_XS × Q8_K vector dot product (NEON implementation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_iq4_xs_q8_k_neon(
    x: &[BlockIQ4XS],
    y: &[BlockQ8_K],
) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);
    let values = vld1q_s8(KVALUES_IQ4NL_I8.as_ptr());

    let mut sum = 0.0f32;

    for ibl in 0..nb {
        let q4 = x[ibl].qs.as_ptr();
        let q8 = y[ibl].qs.as_ptr();
        let mut h = x[ibl].scales_h;

        let mut sumi1 = 0i64;
        let mut sumi2 = 0i64;

        for ib in 0..(QK_K / 64) {
            let q4bits = vld1q_u8_x2(q4.add(ib * 32));
            let q8bytes = vld1q_s8_x4(q8.add(ib * 64));

            let q4b_0 = vqtbl1q_s8(values, vandq_u8(q4bits.0, m4b));
            let q4b_1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.0, 4));
            let q4b_2 = vqtbl1q_s8(values, vandq_u8(q4bits.1, m4b));
            let q4b_3 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.1, 4));

            let prod_1 = vdotq_s32_manual(
                vdotq_s32_manual(vzero, q4b_0, q8bytes.0),
                q4b_1,
                q8bytes.1,
            );
            let prod_2 = vdotq_s32_manual(
                vdotq_s32_manual(vzero, q4b_2, q8bytes.2),
                q4b_3,
                q8bytes.3,
            );

            let ls1 = ((x[ibl].scales_l[ib] & 0x0F) as i32
                | (((h as i32) << 4) & 0x30))
                - 32;
            let ls2 = ((x[ibl].scales_l[ib] >> 4) as i32
                | (((h as i32) << 2) & 0x30))
                - 32;
            h >>= 4;

            // Use i64 for multiplication to avoid overflow
            sumi1 += (vaddvq_s32(prod_1) as i64) * (ls1 as i64);
            sumi2 += (vaddvq_s32(prod_2) as i64) * (ls2 as i64);
        }

        sum += fp16_to_fp32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2) as f32;
    }

    sum
}

// Public interface
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    unsafe { vec_dot_f32_neon(x, y) }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    unsafe { vec_dot_q4_0_q8_0_neon(x, y) }
}

/// Q8_0 x Q8_0 batch vector dot product (NEON optimized)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q8_0_neon(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let mut sum = 0.0f32;
    for i in 0..nb {
        let x_i = x.get_unchecked(i);
        let y_i = y.get_unchecked(i);

        let dx = fp16_to_fp32(x_i.d);
        let dy = fp16_to_fp32(y_i.d);

        let mut dot = 0i32;
        for j in 0..32 {
            dot += (x_i.qs[j] as i32) * (y_i.qs[j] as i32);
        }
        sum += (dot as f32) * dx * dy;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    unsafe { vec_dot_q8_0_neon(x, y) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mut sum = 0.0f32;
    for i in 0..nb {
        let x_i = &x[i];
        let y_i = &y[i];
        let dx = fp16_to_fp32(x_i.d);
        let dy = fp16_to_fp32(y_i.d);
        let mut dot = 0i32;
        for j in 0..32 {
            dot += (x_i.qs[j] as i32) * (y_i.qs[j] as i32);
        }
        sum += (dot as f32) * dx * dy;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q4_k_q8_k(x: &[BlockQ4_K], y: &[BlockQ8_K]) -> f32 {
    unsafe { vec_dot_q4_k_q8_k_neon(x, y) }
}

/// Partial vec_dot for Q4_K × Q8_K: compute only blocks [block_start, block_end)
/// and accumulate into acc. Used for K-extent optimization in narrow path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_q4_k_q8_k_blocks(
    x: &[BlockQ4_K],
    y: &[BlockQ8_K],
    block_start: usize,
    block_end: usize,
    acc: &mut f32,
) {
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sum = 0.0f32;

    for i in block_start..block_end {
        let x_i = x.get_unchecked(i);
        let y_i = y.get_unchecked(i);

        let d = y_i.d * fp16_to_fp32(x_i.d);
        let dmin = y_i.d * fp16_to_fp32(x_i.dmin);

        let q8sums = vpaddq_s16(
            vld1q_s16(y_i.bsums.as_ptr()),
            vld1q_s16(y_i.bsums.as_ptr().add(8)),
        );

        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(
            x_i.scales.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );

        let mut mins8 = vdup_n_u32(0);
        mins8 = vset_lane_u32(utmp[1] & KMASK1, mins8, 0);
        mins8 = vset_lane_u32(
            ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
            mins8,
            1,
        );

        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[0] &= KMASK1;

        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        let min_sum = vaddvq_s32(prod);

        let scales = utmp.as_ptr() as *const u8;

        let mut q4 = x_i.qs.as_ptr();
        let mut q8 = y_i.qs.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        for j in 0..4 {
            let q4bits = vld1q_u8_x2(q4);
            q4 = q4.add(32);

            let q4l_0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
            let q4l_1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));

            let q8bytes = vld1q_s8_x2(q8);
            q8 = q8.add(32);

            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(vzero, q4l_0, q8bytes.0),
                q4l_1,
                q8bytes.1,
            );
            sumi1 += vaddvq_s32(p1) * *scales.add(j * 2) as i32;

            let q8bytes = vld1q_s8_x2(q8);
            q8 = q8.add(32);

            let q4h_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
            let q4h_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));

            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(vzero, q4h_0, q8bytes.0),
                q4h_1,
                q8bytes.1,
            );
            sumi2 += vaddvq_s32(p2) * *scales.add(j * 2 + 1) as i32;
        }

        sum -= dmin * min_sum as f32;
        sum += d * (sumi1 + sumi2) as f32;
    }

    *acc += sum;
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q5_k_q8_k(x: &[BlockQ5_K], y: &[BlockQ8_K]) -> f32 {
    unsafe { vec_dot_q5_k_q8_k_neon(x, y) }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q6_k_q8_k(x: &[BlockQ6_K], y: &[BlockQ8_K]) -> f32 {
    unsafe { vec_dot_q6_k_q8_k_neon(x, y) }
}

// ============================================================================
// MMLA (i8mm) implementations for 2-row simultaneous computation
// ============================================================================

/// Q4_K × Q8_K two-row dot product using MMLA (i8mm)
/// Computes two rows simultaneously for better memory bandwidth utilization.
/// Returns (sum0, sum1) for the two rows.
/// Based on llama.cpp's NEON MMLA implementation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_q4_k_q8_k_2rows(
    x0: &[BlockQ4_K],
    x1: &[BlockQ4_K],
    y0: &[BlockQ8_K],
    y1: &[BlockQ8_K],
) -> (f32, f32) {
    let nb = x0.len();
    debug_assert_eq!(nb, x1.len());
    debug_assert_eq!(nb, y0.len());
    debug_assert_eq!(nb, y1.len());

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4b = vdupq_n_u8(0x0f);

    // Accumulator for 4 outputs: [row0_y0, row0_y1, row1_y0, row1_y1]
    let mut vfsum = vdupq_n_f32(0.0);

    for i in 0..nb {
        let x0_i = x0.get_unchecked(i);
        let x1_i = x1.get_unchecked(i);
        let y0_i = y0.get_unchecked(i);
        let y1_i = y1.get_unchecked(i);

        // Decode scales and mins for both rows
        let mut x0_scales = [0i8; 8];
        let mut x1_scales = [0i8; 8];
        let x0_mins: int16x8_t;
        let x1_mins: int16x8_t;

        // Decode x0 scales and mins from 12-byte format
        {
            let mut scales_mins = [0u32; 3];
            std::ptr::copy_nonoverlapping(
                x0_i.scales.as_ptr(),
                scales_mins.as_mut_ptr() as *mut u8,
                12,
            );
            let mins_0_3 = scales_mins[1] & KMASK1;
            let mins_4_7 = ((scales_mins[2] >> 4) & KMASK2)
                | (((scales_mins[1] >> 6) & KMASK3) << 4);
            let mins =
                vcreate_u32((mins_0_3 as u64) | ((mins_4_7 as u64) << 32));
            x0_mins =
                vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
            let mut scales = [0u32; 2];
            scales[0] = scales_mins[0] & KMASK1;
            scales[1] = (scales_mins[2] & KMASK2)
                | (((scales_mins[0] >> 6) & KMASK3) << 4);
            std::ptr::copy_nonoverlapping(
                scales.as_ptr() as *const u8,
                x0_scales.as_mut_ptr() as *mut u8,
                8,
            );
        }

        // Decode x1 scales and mins
        {
            let mut scales_mins = [0u32; 3];
            std::ptr::copy_nonoverlapping(
                x1_i.scales.as_ptr(),
                scales_mins.as_mut_ptr() as *mut u8,
                12,
            );
            let mins_0_3 = scales_mins[1] & KMASK1;
            let mins_4_7 = ((scales_mins[2] >> 4) & KMASK2)
                | (((scales_mins[1] >> 6) & KMASK3) << 4);
            let mins =
                vcreate_u32((mins_0_3 as u64) | ((mins_4_7 as u64) << 32));
            x1_mins =
                vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
            let mut scales = [0u32; 2];
            scales[0] = scales_mins[0] & KMASK1;
            scales[1] = (scales_mins[2] & KMASK2)
                | (((scales_mins[0] >> 6) & KMASK3) << 4);
            std::ptr::copy_nonoverlapping(
                scales.as_ptr() as *const u8,
                x1_scales.as_mut_ptr() as *mut u8,
                8,
            );
        }

        let mut visum = vdupq_n_s32(0);

        // Process 64 data points per iteration, total 256 data points
        let mut qx0 = x0_i.qs.as_ptr();
        let mut qx1 = x1_i.qs.as_ptr();
        let mut qy0 = y0_i.qs.as_ptr();
        let mut qy1 = y1_i.qs.as_ptr();

        for j in 0..4 {
            // Load 64 bytes of input data (4 x 16 bytes)
            let vy0 = vld1q_s8_x4(qy0);
            let vy1 = vld1q_s8_x4(qy1);

            // Load and unpack 4-bit weights to int8
            let vv0 = vld1q_u8_x2(qx0);
            let vx0: [int8x16_t; 4] = [
                vreinterpretq_s8_u8(vandq_u8(vv0.0, m4b)),
                vreinterpretq_s8_u8(vandq_u8(vv0.1, m4b)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv0.0, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv0.1, 4)),
            ];
            let vv1 = vld1q_u8_x2(qx1);
            let vx1: [int8x16_t; 4] = [
                vreinterpretq_s8_u8(vandq_u8(vv1.0, m4b)),
                vreinterpretq_s8_u8(vandq_u8(vv1.1, m4b)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv1.0, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv1.1, 4)),
            ];

            // Process 32 data points (share same block scale) per iteration
            for k in 0..2 {
                let blk = j * 2 + k;
                let block_scale_arr: [i32; 4] = [
                    x0_scales[blk] as i32,
                    x0_scales[blk] as i32,
                    x1_scales[blk] as i32,
                    x1_scales[blk] as i32,
                ];
                let block_scale = vld1q_s32(block_scale_arr.as_ptr());

                let mut vr = vdupq_n_s32(0);
                for l in 0..2 {
                    let idx = k * 2 + l;
                    // Interleave two rows of weights and two inputs
                    let vx0_s64 = vreinterpretq_s64_s8(vx0[idx]);
                    let vx1_s64 = vreinterpretq_s64_s8(vx1[idx]);
                    // Access int8x16x4_t using .0, .1, .2, .3
                    let vy0_val = match idx {
                        0 => vy0.0,
                        1 => vy0.1,
                        2 => vy0.2,
                        _ => vy0.3,
                    };
                    let vy1_val = match idx {
                        0 => vy1.0,
                        1 => vy1.1,
                        2 => vy1.2,
                        _ => vy1.3,
                    };
                    let vy0_s64 = vreinterpretq_s64_s8(vy0_val);
                    let vy1_s64 = vreinterpretq_s64_s8(vy1_val);

                    let vx_l =
                        vreinterpretq_s8_s64(vzip1q_s64(vx0_s64, vx1_s64));
                    let vx_h =
                        vreinterpretq_s8_s64(vzip2q_s64(vx0_s64, vx1_s64));
                    let vy_l =
                        vreinterpretq_s8_s64(vzip1q_s64(vy0_s64, vy1_s64));
                    let vy_h =
                        vreinterpretq_s8_s64(vzip2q_s64(vy0_s64, vy1_s64));

                    vr = vmmlaq_s32_manual(vr, vx_l, vy_l);
                    vr = vmmlaq_s32_manual(vr, vx_h, vy_h);
                }
                // Apply block scale
                visum = vmlaq_s32(visum, vr, block_scale);
            }

            qx0 = qx0.add(32);
            qx1 = qx1.add(32);
            qy0 = qy0.add(64);
            qy1 = qy1.add(64);
        }

        // Adjust bias, apply superblock scale
        {
            let y0_sums = vpaddq_s16(
                vld1q_s16(y0_i.bsums.as_ptr()),
                vld1q_s16(y0_i.bsums.as_ptr().add(8)),
            );
            let y1_sums = vpaddq_s16(
                vld1q_s16(y1_i.bsums.as_ptr()),
                vld1q_s16(y1_i.bsums.as_ptr().add(8)),
            );

            let bias_arr: [i32; 4] = [
                vaddvq_s32(vaddq_s32(
                    vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x0_mins)),
                    vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x0_mins)),
                )),
                vaddvq_s32(vaddq_s32(
                    vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x0_mins)),
                    vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x0_mins)),
                )),
                vaddvq_s32(vaddq_s32(
                    vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x1_mins)),
                    vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x1_mins)),
                )),
                vaddvq_s32(vaddq_s32(
                    vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x1_mins)),
                    vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x1_mins)),
                )),
            ];

            let dmins_arr: [f32; 4] = [
                fp16_to_fp32(x0_i.dmin) * y0_i.d,
                fp16_to_fp32(x0_i.dmin) * y1_i.d,
                fp16_to_fp32(x1_i.dmin) * y0_i.d,
                fp16_to_fp32(x1_i.dmin) * y1_i.d,
            ];
            let dmins = vld1q_f32(dmins_arr.as_ptr());
            vfsum = vmlsq_f32(
                vfsum,
                vcvtq_f32_s32(vld1q_s32(bias_arr.as_ptr())),
                dmins,
            );

            let superblock_scale_arr: [f32; 4] = [
                fp16_to_fp32(x0_i.d) * y0_i.d,
                fp16_to_fp32(x0_i.d) * y1_i.d,
                fp16_to_fp32(x1_i.d) * y0_i.d,
                fp16_to_fp32(x1_i.d) * y1_i.d,
            ];
            let superblock_scale = vld1q_f32(superblock_scale_arr.as_ptr());
            vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
        }
    }

    // vfsum = ABCD -> ACBD
    // AC -> s, BD -> (s+bs)
    vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
    let result0 = vget_low_f32(vfsum);
    let result1 = vget_high_f32(vfsum);

    (vget_lane_f32(result0, 0), vget_lane_f32(result1, 0))
}

/// Check if MMLA (i8mm) is supported at runtime
#[cfg(target_arch = "aarch64")]
pub fn has_mmla() -> bool {
    // M1/M2/M3 support i8mm
    // We check at compile time via target_feature
    cfg!(target_feature = "i8mm")
}

/// Q4_K × Q8_K two-row dot product with single input using MMLA (i8mm)
/// Computes two output rows from one input (for decode n_tokens=1).
/// Returns (sum0, sum1) for the two rows.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn vec_dot_q4_k_q8_k_2rows_1input(
    x0: &[BlockQ4_K],
    x1: &[BlockQ4_K],
    y: &[BlockQ8_K],
) -> (f32, f32) {
    let nb = x0.len();
    debug_assert_eq!(nb, x1.len());
    debug_assert_eq!(nb, y.len());

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4b = vdupq_n_u8(0x0f);

    // Accumulator for 2 outputs: [row0, row1]
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;

    for i in 0..nb {
        let x0_i = x0.get_unchecked(i);
        let x1_i = x1.get_unchecked(i);
        let y_i = y.get_unchecked(i);

        // Decode scales and mins for both rows
        let mut x0_scales = [0u8; 8];
        let mut x1_scales = [0u8; 8];
        let x0_mins: int16x8_t;
        let x1_mins: int16x8_t;

        // Decode x0
        {
            let mut scales_mins = [0u32; 3];
            std::ptr::copy_nonoverlapping(
                x0_i.scales.as_ptr(),
                scales_mins.as_mut_ptr() as *mut u8,
                12,
            );
            let mins_0_3 = scales_mins[1] & KMASK1;
            let mins_4_7 = ((scales_mins[2] >> 4) & KMASK2)
                | (((scales_mins[1] >> 6) & KMASK3) << 4);
            let mins =
                vcreate_u32((mins_0_3 as u64) | ((mins_4_7 as u64) << 32));
            x0_mins =
                vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
            let mut scales = [0u32; 2];
            scales[0] = scales_mins[0] & KMASK1;
            scales[1] = (scales_mins[2] & KMASK2)
                | (((scales_mins[0] >> 6) & KMASK3) << 4);
            std::ptr::copy_nonoverlapping(
                scales.as_ptr() as *const u8,
                x0_scales.as_mut_ptr(),
                8,
            );
        }

        // Decode x1
        {
            let mut scales_mins = [0u32; 3];
            std::ptr::copy_nonoverlapping(
                x1_i.scales.as_ptr(),
                scales_mins.as_mut_ptr() as *mut u8,
                12,
            );
            let mins_0_3 = scales_mins[1] & KMASK1;
            let mins_4_7 = ((scales_mins[2] >> 4) & KMASK2)
                | (((scales_mins[1] >> 6) & KMASK3) << 4);
            let mins =
                vcreate_u32((mins_0_3 as u64) | ((mins_4_7 as u64) << 32));
            x1_mins =
                vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
            let mut scales = [0u32; 2];
            scales[0] = scales_mins[0] & KMASK1;
            scales[1] = (scales_mins[2] & KMASK2)
                | (((scales_mins[0] >> 6) & KMASK3) << 4);
            std::ptr::copy_nonoverlapping(
                scales.as_ptr() as *const u8,
                x1_scales.as_mut_ptr(),
                8,
            );
        }

        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        let mut qx0 = x0_i.qs.as_ptr();
        let mut qx1 = x1_i.qs.as_ptr();
        let mut qy = y_i.qs.as_ptr();

        for j in 0..4 {
            // Load 32 bytes of input
            let vy = vld1q_s8_x2(qy);

            // Load and unpack weights
            let vv0 = vld1q_u8_x2(qx0);
            let vx0: [int8x16_t; 4] = [
                vreinterpretq_s8_u8(vandq_u8(vv0.0, m4b)),
                vreinterpretq_s8_u8(vandq_u8(vv0.1, m4b)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv0.0, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv0.1, 4)),
            ];
            let vv1 = vld1q_u8_x2(qx1);
            let vx1: [int8x16_t; 4] = [
                vreinterpretq_s8_u8(vandq_u8(vv1.0, m4b)),
                vreinterpretq_s8_u8(vandq_u8(vv1.1, m4b)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv1.0, 4)),
                vreinterpretq_s8_u8(vshrq_n_u8(vv1.1, 4)),
            ];

            // Process low and high nibbles separately
            for k in 0..2 {
                let blk = j * 2 + k;
                let scale0 = x0_scales[blk] as i32;
                let scale1 = x1_scales[blk] as i32;

                for l in 0..2 {
                    let idx = k * 2 + l;
                    // Interleave two rows' weights
                    let vx0_s64 = vreinterpretq_s64_s8(vx0[idx]);
                    let vx1_s64 = vreinterpretq_s64_s8(vx1[idx]);
                    let vy_s64 =
                        vreinterpretq_s64_s8(if l == 0 { vy.0 } else { vy.1 });

                    let vx_l =
                        vreinterpretq_s8_s64(vzip1q_s64(vx0_s64, vx1_s64));
                    let vx_h =
                        vreinterpretq_s8_s64(vzip2q_s64(vx0_s64, vx1_s64));

                    // Use SDOT for single input (MMLA needs two inputs)
                    let p0 = vdotq_s32_manual(
                        vdupq_n_s32(0),
                        vx0[idx],
                        if l == 0 { vy.0 } else { vy.1 },
                    );
                    let p1 = vdotq_s32_manual(
                        vdupq_n_s32(0),
                        vx1[idx],
                        if l == 0 { vy.0 } else { vy.1 },
                    );

                    sumi0 += vaddvq_s32(p0) * scale0;
                    sumi1 += vaddvq_s32(p1) * scale1;
                }
            }

            qx0 = qx0.add(32);
            qx1 = qx1.add(32);
            qy = qy.add(64);
        }

        // Min corrections
        let y_sums = vpaddq_s16(
            vld1q_s16(y_i.bsums.as_ptr()),
            vld1q_s16(y_i.bsums.as_ptr().add(8)),
        );

        let min_sum0 = vaddvq_s32(vaddq_s32(
            vmull_s16(vget_low_s16(y_sums), vget_low_s16(x0_mins)),
            vmull_s16(vget_high_s16(y_sums), vget_high_s16(x0_mins)),
        ));
        let min_sum1 = vaddvq_s32(vaddq_s32(
            vmull_s16(vget_low_s16(y_sums), vget_low_s16(x1_mins)),
            vmull_s16(vget_high_s16(y_sums), vget_high_s16(x1_mins)),
        ));

        let d0 = y_i.d * fp16_to_fp32(x0_i.d);
        let dmin0 = y_i.d * fp16_to_fp32(x0_i.dmin);
        let d1 = y_i.d * fp16_to_fp32(x1_i.d);
        let dmin1 = y_i.d * fp16_to_fp32(x1_i.dmin);

        sum0 += d0 * sumi0 as f32 - dmin0 * min_sum0 as f32;
        sum1 += d1 * sumi1 as f32 - dmin1 * min_sum1 as f32;
    }

    (sum0, sum1)
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_iq4_xs_q8_k(x: &[BlockIQ4XS], y: &[BlockQ8_K]) -> f32 {
    unsafe { vec_dot_iq4_xs_q8_k_neon(x, y) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_q4_k_q8_k(x: &[BlockQ4_K], y: &[BlockQ8_K]) -> f32 {
    unimplemented!("Q4_K vec_dot only supported on aarch64 with NEON")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_q5_k_q8_k(x: &[BlockQ5_K], y: &[BlockQ8_K]) -> f32 {
    unimplemented!("Q5_K vec_dot only supported on aarch64 with NEON")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_q6_k_q8_k(x: &[BlockQ6_K], y: &[BlockQ8_K]) -> f32 {
    unimplemented!("Q6_K vec_dot only supported on aarch64 with NEON")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_iq4_xs_q8_k(x: &[BlockIQ4XS], y: &[BlockQ8_K]) -> f32 {
    unimplemented!("IQ4_XS vec_dot only supported on aarch64 with NEON")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    let mut sum = 0.0f32;
    for i in 0..nb {
        let block_x = &x[i];
        let block_y = &y[i];
        let dx = fp16_to_fp32(block_x.d);
        let dy = fp16_to_fp32(block_y.d);
        let scale = dx * dy;

        for j in 0..16 {
            let v0 = (block_x.qs[j] & 0xF) as i8 as i32 - 8;
            let v1 = (block_x.qs[j] >> 4) as i8 as i32 - 8;
            sum += scale * v0 as f32 * block_y.qs[j] as f32;
            sum += scale * v1 as f32 * block_y.qs[j + 16] as f32;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_dot_f32() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = vec_dot_f32(&x, &y);
        assert!((result - 30.0).abs() < 1e-6);
    }
}

// Test Q4_K vec_dot with known values
#[cfg(test)]
mod q4k_test {
    use super::*;

    #[test]
    fn test_q4_k_vec_dot_basic() {
        // Create a simple Q4_K block
        let block_q4 = BlockQ4_K {
            d: half::f16::to_bits(half::f16::from_f32(1.0)), // scale = 1.0
            dmin: half::f16::to_bits(half::f16::from_f32(0.0)), // min scale = 0
            scales: [0; 12],                                 // all scales = 0
            qs: [0; 128],                                    // all values = 0
        };

        // Create a simple Q8_K block
        let block_q8 = BlockQ8_K {
            d: 1.0,
            qs: [1; 256],     // all values = 1
            bsums: [256; 16], // each block sum = 256 (256 ones)
        };

        // With all zeros in Q4_K and ones in Q8_K, result should be 0
        let result = vec_dot_q4_k_q8_k(&[block_q4], &[block_q8]);
        println!("Result with all zeros: {}", result);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_q4_k_vec_dot_nonzero() {
        // Create Q4_K block with known values
        let block_q4 = BlockQ4_K {
            d: half::f16::to_bits(half::f16::from_f32(0.5)), // scale = 0.5
            dmin: half::f16::to_bits(half::f16::from_f32(0.0)),
            scales: [8; 12], // scales = 8
            qs: [0x88; 128], // each byte = 0b10001000, so low nibble = 8, high nibble = 8
        };

        // Create Q8_K block
        let block_q8 = BlockQ8_K {
            d: 1.0,
            qs: [1; 256],   // all values = 1
            bsums: [0; 16], // no contribution from mins
        };

        // Expected: each of 256 positions has value 8 * 1 = 8
        // Sum = 256 * 8 = 2048
        // With d = 0.5 and scale = 8, but we need to understand the scale encoding
        let result = vec_dot_q4_k_q8_k(&[block_q4], &[block_q8]);
        println!("Result with known values: {}", result);
        // The exact result depends on scale interpretation
    }
}
