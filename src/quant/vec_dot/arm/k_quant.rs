use std::arch::aarch64::*;

use crate::quant::types::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K};

use super::fp::fp16_to_fp32;

/// Manual vdotq_s32 implementation using native SDOT instruction
/// vdotq_s32 is unstable in Rust, so we use inline assembly
/// SDOT computes: result[i] = acc[i] + sum(a[4*i..4*i+4] * b[4*i..4*i+4]) for i in 0..4
/// This is ~4x faster than the vmull+vpaddl sequence.
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

/// Q2_K × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q2_k_q8_k_neon(n: usize, x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let m3 = vdupq_n_u8(0x03);
    let m4 = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * fp16_to_fp32(x[i].d);
        let dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        let q2 = x[i].qs.as_ptr();
        let q8 = y[i].qs.as_ptr();
        let sc = x[i].scales.as_ptr();

        // Load scales and mins (packed in same byte: low 4bits = scale, high 4bits = min)
        let mins_and_scales = vld1q_u8(sc);
        let scales = vandq_u8(mins_and_scales, m4);
        let mins = vshrq_n_u8(mins_and_scales, 4);

        // Calculate min correction using bsums
        let q8sums = vld1q_s16(y[i].bsums.as_ptr());
        let mins16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
        let mins16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));

        let s0 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16), vget_low_s16(q8sums)),
            vmull_s16(vget_high_s16(mins16), vget_high_s16(q8sums))
        );
        let s1 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16_hi), vget_low_s16(vld1q_s16(y[i].bsums.as_ptr().add(8)))),
            vmull_s16(vget_high_s16(mins16_hi), vget_high_s16(vld1q_s16(y[i].bsums.as_ptr().add(8))))
        );
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1)) as f32;

        // Process 256 elements (128 bytes of Q2 data)
        let mut isum = 0i32;
        let mut is = 0;

        // Store scales for scalar access
        let scales_arr: [u8; 16] = std::mem::transmute(scales);

        for j in 0..(QK_K / 128) {
            let q2bits = vld1q_u8_x2(q2.add(j * 32));
            let q8bytes = vld1q_s8_x2(q8.add(j * 64));

            // Decode 2-bit values (4 per byte)
            let q2bytes_0 = vreinterpretq_s8_u8(vandq_u8(q2bits.0, m3));
            let q2bytes_1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.0, 2), m3));

            // Dot products with scales
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes_0, q8bytes.0)) * scales_arr[is] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes_1, q8bytes.1)) * scales_arr[is + 1] as i32;

            // Process second half
            let q2bytes_4 = vreinterpretq_s8_u8(vandq_u8(q2bits.1, m3));
            let q2bytes_5 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.1, 2), m3));

            let q8bytes2 = vld1q_s8_x2(q8.add(j * 64 + 32));

            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes_4, q8bytes2.0)) * scales_arr[is + 2] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes_5, q8bytes2.1)) * scales_arr[is + 3] as i32;

            is += 4;
        }

        sum += d * isum as f32;
    }

    sum
}

/// Q4_K × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_k_q8_k_neon(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
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

        // Calculate min correction first (matching llama.cpp order)
        let q8sums = vpaddq_s16(vld1q_s16(y_i.bsums.as_ptr()), vld1q_s16(y_i.bsums.as_ptr().add(8)));

        // Decode scales and mins from 12-byte format
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(x_i.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);

        // Extract mins first (before modifying utmp)
        let mut mins8 = vdup_n_u32(0);
        mins8 = vset_lane_u32(utmp[1] & KMASK1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4), mins8, 1);

        // Now modify utmp for scales
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[0] &= KMASK1;

        // Calculate min using vmull_s16 (matching llama.cpp)
        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins))
        );
        let min_sum = vaddvq_s32(prod);

        // Get scales - load all 8 scales at once to avoid pointer arithmetic in loop
        let sc = std::slice::from_raw_parts(utmp.as_ptr() as *const u8, 8);

        // Process 256 elements
        let mut q4 = x_i.qs.as_ptr();
        let mut q8 = y_i.qs.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        for j in 0..4 {
            let q4bits = vld1q_u8_x2(q4);
            let q8bytes = vld1q_s8_x2(q8);

            let q4l_0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
            let q4l_1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));

            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4l_0, q8bytes.0), q4l_1, q8bytes.1);
            sumi1 += vaddvq_s32(p1) * sc[j * 2] as i32;

            let q8bytes_1 = vld1q_s8_x2(q8.add(32));
            let q4h_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
            let q4h_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));

            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4h_0, q8bytes_1.0), q4h_1, q8bytes_1.1);
            sumi2 += vaddvq_s32(p2) * sc[j * 2 + 1] as i32;

            q4 = q4.add(32);
            q8 = q8.add(64);
        }

        sum -= dmin * min_sum as f32;
        sum += d * (sumi1 + sumi2) as f32;
    }

    sum
}

/// Q3_K × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q3_k_q8_k_neon(n: usize, x: &[BlockQ3K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let m3b = vdupq_n_u8(0x3);
    let vzero = vdupq_n_s32(0);

    let m0 = vdupq_n_u8(1);
    let m1 = vshlq_n_u8(m0, 1);
    let m2 = vshlq_n_u8(m0, 2);
    let m3 = vshlq_n_u8(m0, 3);

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    const M32: i8 = 32;

    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * fp16_to_fp32(x[i].d);

        let q3 = x[i].qs.as_ptr();
        let qh = x[i].hmask.as_ptr();
        let q8 = y[i].qs.as_ptr();

        let qhbits = vld1q_u8_x2(qh);

        let mut isum = 0i32;

        // Decode scales from 12-byte format
        let mut aux = [0u32; 3];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), aux.as_mut_ptr() as *mut u8, 12);

        let mut utmp = [0u32; 4];
        utmp[3] = ((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4);
        utmp[2] = ((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4);
        utmp[1] = (aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4);
        utmp[0] = (aux[0] & KMASK2) | ((aux[2] & KMASK1) << 4);

        let mut scales: [i8; 16] = std::mem::transmute(utmp);
        for j in 0..16 {
            scales[j] -= M32;
        }

        let mut scale_ptr = 0;

        for j in 0..(QK_K / 128) {
            let q3bits = vld1q_u8_x2(q3.add(j * 32));
            let q8bytes_1 = vld1q_s8_x4(q8.add(j * 64));
            let q8bytes_2 = vld1q_s8_x4(q8.add(j * 64 + 32));

            // Process 4 groups of 16 elements
            let q3h_0 = vshlq_n_u8(vbicq_u8(m0, qhbits.0), 2);
            let q3h_1 = vshlq_n_u8(vbicq_u8(m0, qhbits.1), 2);
            let q3h_2 = vshlq_n_u8(vbicq_u8(m1, qhbits.0), 1);
            let q3h_3 = vshlq_n_u8(vbicq_u8(m1, qhbits.1), 1);

            let q3bytes_0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.0, m3b)), vreinterpretq_s8_u8(q3h_0));
            let q3bytes_1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.1, m3b)), vreinterpretq_s8_u8(q3h_1));
            let q3bytes_2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 2), m3b)), vreinterpretq_s8_u8(q3h_2));
            let q3bytes_3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 2), m3b)), vreinterpretq_s8_u8(q3h_3));

            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_0, q8bytes_1.0)) * scales[scale_ptr] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_1, q8bytes_1.1)) * scales[scale_ptr + 1] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_2, q8bytes_1.2)) * scales[scale_ptr + 2] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_3, q8bytes_1.3)) * scales[scale_ptr + 3] as i32;

            scale_ptr += 4;

            let q3h_4 = vbicq_u8(m2, qhbits.0);
            let q3h_5 = vbicq_u8(m2, qhbits.1);
            let q3h_6 = vshrq_n_u8(vbicq_u8(m3, qhbits.0), 1);
            let q3h_7 = vshrq_n_u8(vbicq_u8(m3, qhbits.1), 1);

            let q3bytes_4 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 4), m3b)), vreinterpretq_s8_u8(q3h_4));
            let q3bytes_5 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 4), m3b)), vreinterpretq_s8_u8(q3h_5));
            let q3bytes_6 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.0, 6), m3b)), vreinterpretq_s8_u8(q3h_6));
            let q3bytes_7 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.1, 6), m3b)), vreinterpretq_s8_u8(q3h_7));

            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_4, q8bytes_2.0)) * scales[scale_ptr] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_5, q8bytes_2.1)) * scales[scale_ptr + 1] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_6, q8bytes_2.2)) * scales[scale_ptr + 2] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q3bytes_7, q8bytes_2.3)) * scales[scale_ptr + 3] as i32;

            scale_ptr += 4;
        }

        sum += d * isum as f32;
    }

    sum
}

/// Q5_K × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_k_q8_k_neon(n: usize, x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
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
        let q8sums = vpaddq_s16(vld1q_s16(y[i].bsums.as_ptr()), vld1q_s16(y[i].bsums.as_ptr().add(8)));

        // Decode scales and mins
        let mut utmp = [0u32; 4];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mins8 = vld1_u8((utmp.as_ptr() as *const u8).add(8));
        let mins = vreinterpretq_s16_u16(vmovl_u8(mins8));

        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins))
        );
        let sumi_mins = vaddvq_s32(prod);

        let scales: [u8; 16] = std::mem::transmute(utmp);
        let mut scale_idx = 0;

        let q5 = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();

        let qhbits = vld1q_u8_x2(qh);

        let mut sumi = 0i32;

        for j in 0..(QK_K / 64) {
            let q5bits = vld1q_u8_x2(q5.add(j * 32));
            let q8bytes = vld1q_s8_x4(q8.add(j * 64));

            // Process 4 blocks of 16 elements
            let q5h_0 = vshlq_n_u8(vandq_u8(mone, qhbits.0), 4);
            let q5h_1 = vshlq_n_u8(vandq_u8(mone, qhbits.1), 4);
            let q5h_2 = vshlq_n_u8(vandq_u8(mtwo, qhbits.0), 3);
            let q5h_3 = vshlq_n_u8(vandq_u8(mtwo, qhbits.1), 3);

            let q5bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.0, m4b), q5h_0));
            let q5bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.1, m4b), q5h_1));
            let q5bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.0, 4), q5h_2));
            let q5bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.1, 4), q5h_3));

            sumi += vaddvq_s32(vaddq_s32(
                vdotq_s32_manual(mzero, q5bytes_0, q8bytes.0),
                vdotq_s32_manual(mzero, q5bytes_1, q8bytes.1)
            )) * scales[scale_idx] as i32;
            scale_idx += 1;

            sumi += vaddvq_s32(vaddq_s32(
                vdotq_s32_manual(mzero, q5bytes_2, q8bytes.2),
                vdotq_s32_manual(mzero, q5bytes_3, q8bytes.3)
            )) * scales[scale_idx] as i32;
            scale_idx += 1;
        }

        sumf += d * sumi as f32 - dmin * sumi_mins as f32;
    }

    sumf
}

/// Q6_K × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q6_k_q8_k_neon(n: usize, x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let m4b = vdupq_n_u8(0x0F);
    let mone = vdupq_n_u8(0x30);
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

            // Process 8 blocks of 16 elements
            let q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.0, 4));
            let q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.1, 4));
            let q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.0, 2));
            let q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.1, 2));

            let q6h_4 = vandq_u8(mone, qh_bits.0);
            let q6h_5 = vandq_u8(mone, qh_bits.1);
            let q6h_6 = vandq_u8(mone, vshrq_n_u8(qh_bits.0, 2));
            let q6h_7 = vandq_u8(mone, vshrq_n_u8(qh_bits.1, 2));

            // Combine low 4 bits and high 2 bits
            let q6bytes_0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.0, m4b), q6h_0));
            let q6bytes_1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.1, m4b), q6h_1));
            let q6bytes_2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.2, m4b), q6h_2));
            let q6bytes_3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.3, m4b), q6h_3));

            let q6bytes_4 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.0, 4), q6h_4));
            let q6bytes_5 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.1, 4), q6h_5));
            let q6bytes_6 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.2, 4), q6h_6));
            let q6bytes_7 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.3, 4), q6h_7));

            let q8bytes = vld1q_s8_x4(q8.add(j * 64));
            let q8bytes2 = vld1q_s8_x4(q8.add(j * 64 + 32));

            // Dot products with scales
            let scale_idx = j * 8;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_0, q8bytes.0)) * *scales.add(scale_idx) as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_1, q8bytes.1)) * *scales.add(scale_idx + 1) as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_2, q8bytes.2)) * *scales.add(scale_idx + 2) as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_3, q8bytes.3)) * *scales.add(scale_idx + 3) as i32;

            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_4, q8bytes2.0)) * *scales.add(scale_idx + 4) as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_5, q8bytes2.1)) * *scales.add(scale_idx + 5) as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_6, q8bytes2.2)) * *scales.add(scale_idx + 6) as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q6bytes_7, q8bytes2.3)) * *scales.add(scale_idx + 7) as i32;
        }

        sum += d * isum as f32;
    }

    sum
}
