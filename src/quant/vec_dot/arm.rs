use std::arch::aarch64::*;

/// Convert FP16 to FP32 using NEON fcvt instruction
#[inline(always)]
pub unsafe fn fp16_to_fp32(bits: u16) -> f32 {
    // Use NEON fcvt instruction for efficient conversion
    let f16_vec = vdup_n_u16(bits);
    let f16 = vreinterpret_f16_u16(f16_vec);
    let f32_vec = vcvt_f32_f16(f16);
    vgetq_lane_f32(f32_vec, 0)
}

/// FP32 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_fp32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 floats at a time
    let chunks = n / 4;
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        sum = vmlaq_f32(sum, va, vb);
    }

    // Horizontal sum
    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 4)..n {
        remainder += a[i] * b[i];
    }

    result + remainder
}

/// FP16 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_fp16_neon(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 8 FP16 at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let va = vld1q_u16(a.as_ptr().add(i * 8));
        let vb = vld1q_u16(b.as_ptr().add(i * 8));

        // Convert FP16 to FP32 (lower and upper halves)
        let va_f32_lo = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(va)));
        let va_f32_hi = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(va)));
        let vb_f32_lo = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(vb)));
        let vb_f32_hi = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(vb)));

        // Multiply and accumulate
        sum = vmlaq_f32(sum, va_f32_lo, vb_f32_lo);
        sum = vmlaq_f32(sum, va_f32_hi, vb_f32_hi);
    }

    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 8)..n {
        let a_f32 = fp16_to_fp32(a[i]);
        let b_f32 = fp16_to_fp32(b[i]);
        remainder += a_f32 * b_f32;
    }

    result + remainder
}

/// BF16 vector dot product (NEON implementation)
/// Note: Requires ARMv8.2-A with BF16 extension
#[cfg(target_feature = "bf16")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_bf16_neon(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 8 BF16 at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let va = vld1q_u16(a.as_ptr().add(i * 8));
        let vb = vld1q_u16(b.as_ptr().add(i * 8));

        // Convert BF16 to FP32 and multiply
        let va_f32_lo = vcvt_f32_bf16(vget_low_bf16(vreinterpretq_bf16_u16(va)));
        let va_f32_hi = vcvt_f32_bf16(vget_high_bf16(vreinterpretq_bf16_u16(va)));
        let vb_f32_lo = vcvt_f32_bf16(vget_low_bf16(vreinterpretq_bf16_u16(vb)));
        let vb_f32_hi = vcvt_f32_bf16(vget_high_bf16(vreinterpretq_bf16_u16(vb)));

        sum = vmlaq_f32(sum, va_f32_lo, vb_f32_lo);
        sum = vmlaq_f32(sum, va_f32_hi, vb_f32_hi);
    }

    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 8)..n {
        let a_f32 = half::bf16::from_bits(a[i]).to_f32();
        let b_f32 = half::bf16::from_bits(b[i]).to_f32();
        remainder += a_f32 * b_f32;
    }

    result + remainder
}

/// BF16 vector dot product (NEON fallback without BF16 extension)
/// Uses software conversion from BF16 to FP32
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_bf16_neon_fallback(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 BF16 at a time using scalar conversion
    // BF16 is just the upper 16 bits of FP32, so we can use shifts
    let chunks = n / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let mut a_f32 = [0.0f32; 4];
        let mut b_f32 = [0.0f32; 4];

        for j in 0..4 {
            // BF16 to FP32: shift left 16 bits
            a_f32[j] = f32::from_bits((a[idx + j] as u32) << 16);
            b_f32[j] = f32::from_bits((b[idx + j] as u32) << 16);
        }

        let va = vld1q_f32(a_f32.as_ptr());
        let vb = vld1q_f32(b_f32.as_ptr());
        sum = vmlaq_f32(sum, va, vb);
    }

    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 4)..n {
        let a_f32 = f32::from_bits((a[i] as u32) << 16);
        let b_f32 = f32::from_bits((b[i] as u32) << 16);
        remainder += a_f32 * b_f32;
    }

    result + remainder
}

// ===== Quantized vec_dot NEON implementations =====

use crate::quant::types::{BlockQ4_0, BlockQ4_1, BlockQ8_0, BlockQ8_1, QK4_0, QK8_0};

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

/// Q4_0 × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_0_q8_0_neon(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK4_0;
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);

    let m4b = vdupq_n_u8(0x0F);
    let s8b = vdupq_n_s8(0x8);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Subtract 8 (bias)
        let v0_0ls = vsubq_s8(v0_0l, s8b);
        let v0_0hs = vsubq_s8(v0_0h, s8b);
        let v0_1ls = vsubq_s8(v0_1l, s8b);
        let v0_1hs = vsubq_s8(v0_1h, s8b);

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        let p_1 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);

        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..16 {
            let v0 = (x_b.qs[j] & 0x0F) as i32 - 8;
            let v1 = (x_b.qs[j] >> 4) as i32 - 8;

            sumi0 += v0 * y_b.qs[j] as i32;
            sumi1 += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += (sumi0 + sumi1) as f32 * d;
    }

    sumf
}

/// Q8_0 × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q8_0_q8_0_neon(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK8_0;
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Load Q8 values
        let x0_l = vld1q_s8(x0.qs.as_ptr());
        let x0_h = vld1q_s8(x0.qs.as_ptr().add(16));
        let x1_l = vld1q_s8(x1.qs.as_ptr());
        let x1_h = vld1q_s8(x1.qs.as_ptr().add(16));

        let y0_l = vld1q_s8(y0.qs.as_ptr());
        let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));
        let y1_l = vld1q_s8(y1.qs.as_ptr());
        let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), x0_l, y0_l), x0_h, y0_h);
        let p_1 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), x1_l, y1_l), x1_h, y1_h);

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);

        let isum: i32 = x_b.qs.iter().zip(y_b.qs.iter())
            .map(|(&a, &b)| a as i32 * b as i32)
            .sum();

        sumf += isum as f32 * d;
    }

    sumf
}

/// Q4_1 × Q8_1 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_1_q8_1_neon(n: usize, x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    let nb = n / QK4_0;
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let mut summs = 0.0f32;

    let m4b = vdupq_n_u8(0x0F);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Add min * sum term
        let m0 = fp16_to_fp32(x0.m) * fp16_to_fp32(y0.s);
        let m1 = fp16_to_fp32(x1.m) * fp16_to_fp32(y1.s);
        summs += m0 + m1;

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit (no bias for Q4_1)
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
        let p_1 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);
        let m = fp16_to_fp32(x_b.m) * fp16_to_fp32(y_b.s);

        let mut isum = 0i32;
        for j in 0..16 {
            let v0 = (x_b.qs[j] & 0x0F) as i32;
            let v1 = (x_b.qs[j] >> 4) as i32;

            isum += v0 * y_b.qs[j] as i32;
            isum += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += isum as f32 * d + m;
    }

    sumf
}

// ===== K-quant vec_dot NEON implementations =====

use crate::quant::types::{BlockQ2K, BlockQ4K, BlockQ8K, QK_K};

/// Load 2x uint8x16_t from memory
#[inline(always)]
unsafe fn vld1q_u8_x2(ptr: *const u8) -> uint8x16x2_t {
    uint8x16x2_t(
        vld1q_u8(ptr),
        vld1q_u8(ptr.add(16))
    )
}

/// Load 2x int8x16_t from memory
#[inline(always)]
unsafe fn vld1q_s8_x2(ptr: *const i8) -> int8x16x2_t {
    int8x16x2_t(
        vld1q_s8(ptr),
        vld1q_s8(ptr.add(16))
    )
}

/// Load 4x int8x16_t from memory
#[inline(always)]
unsafe fn vld1q_s8_x4(ptr: *const i8) -> int8x16x4_t {
    int8x16x4_t(
        vld1q_s8(ptr),
        vld1q_s8(ptr.add(16)),
        vld1q_s8(ptr.add(32)),
        vld1q_s8(ptr.add(48))
    )
}

/// Load 4x uint8x16_t from memory
#[inline(always)]
unsafe fn vld1q_u8_x4(ptr: *const u8) -> uint8x16x4_t {
    uint8x16x4_t(
        vld1q_u8(ptr),
        vld1q_u8(ptr.add(16)),
        vld1q_u8(ptr.add(32)),
        vld1q_u8(ptr.add(48))
    )
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

use crate::quant::types::{BlockQ3K, BlockQ5K, BlockQ6K, BlockQ5_0, BlockQ5_1};

/// Expand 32-bit qh to two int8x16_t vectors using TABLE_B2B_1 lookup
/// Each bit in qh becomes 0x10 (if bit=0) or 0x00 (if bit=1)
/// Returns (low_16_bytes, high_16_bytes)
#[target_feature(enable = "neon,dotprod")]
unsafe fn expand_qh_to_vectors(qh: u32) -> (int8x16_t, int8x16_t) {
    let tmp: [u64; 4] = [
        TABLE_B2B_1[(qh >> 0) as usize & 0xFF],
        TABLE_B2B_1[(qh >> 8) as usize & 0xFF],
        TABLE_B2B_1[(qh >> 16) as usize & 0xFF],
        TABLE_B2B_1[(qh >> 24) as usize],
    ];

    let qhl = vld1q_s8(tmp.as_ptr() as *const i8);
    let qhh = vld1q_s8(tmp.as_ptr().add(2) as *const i8);

    (qhl, qhh)
}

/// Table for bit expansion: expand 8 bits to 8 bytes
/// For each bit: if bit is 0, output byte is 0x10; if bit is 1, output byte is 0x00
#[inline(always)]
const fn build_table_b2b_1() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        let mut val: u64 = 0;
        let mut j = 0;
        while j < 8 {
            let bit = (i >> j) & 1;
            // (!bit) << 4: if bit=0, result=0x10; if bit=1, result=0x00
            let byte = if bit == 0 { 0x10u64 } else { 0x00u64 };
            val |= byte << (j * 8);
            j += 1;
        }
        table[i] = val;
        i += 1;
    }
    table
}

static TABLE_B2B_1: [u64; 256] = build_table_b2b_1();

/// Q5_0 × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_0_q8_0_neon(n: usize, x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK4_0;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Extract 5th bit from qh (4 bytes = 32 bits for 32 elements)
        let qh0 = u32::from_le_bytes(x0.qh);
        let qh1 = u32::from_le_bytes(x1.qh);

        // Use lookup table to expand bits (unsafe for no bounds check)
        let table_ptr = TABLE_B2B_1.as_ptr();
        let tmp0: [u64; 4] = [
            *table_ptr.add(((qh0 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh0 >> 24) as usize),
        ];
        let tmp1: [u64; 4] = [
            *table_ptr.add(((qh1 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh1 >> 24) as usize),
        ];

        // Load expanded bits as int8 vectors
        let qhl0 = vld1q_s8(tmp0.as_ptr() as *const i8);
        let qhh0 = vld1q_s8(tmp0.as_ptr().add(2) as *const i8);
        let qhl1 = vld1q_s8(tmp1.as_ptr() as *const i8);
        let qhh1 = vld1q_s8(tmp1.as_ptr().add(2) as *const i8);

        // Add high bit and sub 16: result = low4bit - high_bit_expanded
        let v0_0lf = vsubq_s8(v0_0l, qhl0);
        let v0_0hf = vsubq_s8(v0_0h, qhh0);
        let v0_1lf = vsubq_s8(v0_1l, qhl1);
        let v0_1hf = vsubq_s8(v0_1h, qhh1);

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_0lf, v1_0l),
            vdotq_s32_manual(vzero, v0_0hf, v1_0h)
        );
        let p_1 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_1lf, v1_1l),
            vdotq_s32_manual(vzero, v0_1hf, v1_1h)
        );

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);

        let qh = u32::from_le_bytes(x_b.qh);
        let mut isum = 0i32;

        for j in 0..16 {
            // Bits 0-15 for low nibbles, bits 16-31 for high nibbles
            let bit_lo = ((qh >> j) & 1) as i32;
            let bit_hi = ((qh >> (j + 16)) & 1) as i32;

            let v0 = ((x_b.qs[j] & 0x0F) as i32 | (bit_lo << 4)) - 16;
            let v1 = ((x_b.qs[j] >> 4) as i32 | (bit_hi << 4)) - 16;

            isum += v0 * y_b.qs[j] as i32;
            isum += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += isum as f32 * d;
    }

    sumf
}

/// Q5_1 × Q8_1 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_1_q8_1_neon(n: usize, x: &[BlockQ5_1], y: &[BlockQ8_1]) -> f32 {
    let nb = n / QK4_0;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let mut summs = 0.0f32;

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Add min * sum term
        let m0 = fp16_to_fp32(x0.m) * fp16_to_fp32(y0.s);
        let m1 = fp16_to_fp32(x1.m) * fp16_to_fp32(y1.s);
        summs += m0 + m1;

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit (no bias for Q5_1)
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Extract 5th bit
        let qh0 = u32::from_le_bytes(x0.qh);
        let qh1 = u32::from_le_bytes(x1.qh);

        // Use lookup table to expand bits (unsafe for no bounds check)
        let table_ptr = TABLE_B2B_1.as_ptr();
        let tmp0: [u64; 4] = [
            *table_ptr.add(((qh0 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh0 >> 24) as usize),
        ];
        let tmp1: [u64; 4] = [
            *table_ptr.add(((qh1 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh1 >> 24) as usize),
        ];

        let qhl0 = vld1q_s8(tmp0.as_ptr() as *const i8);
        let qhh0 = vld1q_s8(tmp0.as_ptr().add(2) as *const i8);
        let qhl1 = vld1q_s8(tmp1.as_ptr() as *const i8);
        let qhh1 = vld1q_s8(tmp1.as_ptr().add(2) as *const i8);

        // Add high bit (for Q5_1, we add the bit, not subtract)
        let v0_0lf = vaddq_s8(v0_0l, qhl0);
        let v0_0hf = vaddq_s8(v0_0h, qhh0);
        let v0_1lf = vaddq_s8(v0_1l, qhl1);
        let v0_1hf = vaddq_s8(v0_1h, qhh1);

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_0lf, v1_0l),
            vdotq_s32_manual(vzero, v0_0hf, v1_0h)
        );
        let p_1 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_1lf, v1_1l),
            vdotq_s32_manual(vzero, v0_1hf, v1_1h)
        );

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);
        let m = fp16_to_fp32(x_b.m) * fp16_to_fp32(y_b.s);

        let qh = u32::from_le_bytes(x_b.qh);
        let mut isum = 0i32;

        for j in 0..16 {
            // Bits 0-15 for low nibbles, bits 16-31 for high nibbles
            let bit_lo = ((qh >> j) & 1) as i32;
            let bit_hi = ((qh >> (j + 16)) & 1) as i32;

            let v0 = (x_b.qs[j] & 0x0F) as i32 + (bit_lo << 4);
            let v1 = (x_b.qs[j] >> 4) as i32 + (bit_hi << 4);

            isum += v0 * y_b.qs[j] as i32;
            isum += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += isum as f32 * d + m;
    }

    sumf
}

// ===== Q3_K, Q5_K, Q6_K NEON implementations =====

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

// ===== IQ Series =====

use crate::quant::types::BlockIQ4NL;

/// Lookup table for IQ4_NL: maps 4-bit values to actual quantized values
static KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const QK4_NL: usize = 32;

/// IQ4_NL × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq4_nl_q8_0_neon(n: usize, x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK4_NL;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    // Load the lookup table into a NEON register
    let values = vld1q_s8(KVALUES_IQ4NL.as_ptr());

    let mut sum = 0.0f32;
    let mut ib = 0;

    // Process 2 blocks at a time
    while ib + 1 < nb {
        let q4bits_0 = vld1q_u8(x[ib].qs.as_ptr());
        let q4bits_1 = vld1q_u8(x[ib + 1].qs.as_ptr());

        let q8b_0 = vld1q_s8_x2(y[ib].qs.as_ptr());
        let q8b_1 = vld1q_s8_x2(y[ib + 1].qs.as_ptr());

        // Lookup 4-bit values using TBL instruction
        let q4b_0 = vqtbl1q_s8(values, vandq_u8(q4bits_0, m4b));
        let q4b_1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits_0, 4));
        let q4b_2 = vqtbl1q_s8(values, vandq_u8(q4bits_1, m4b));
        let q4b_3 = vqtbl1q_s8(values, vshrq_n_u8(q4bits_1, 4));

        let prod_0 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_0, q8b_0.0), q4b_1, q8b_0.1);
        let prod_1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_2, q8b_1.0), q4b_3, q8b_1.1);

        sum += fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d) * vaddvq_s32(prod_0) as f32;
        sum += fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d) * vaddvq_s32(prod_1) as f32;

        ib += 2;
    }

    // Handle remainder
    for i in ib..nb {
        let d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += y[i].qs[j] as i32 * KVALUES_IQ4NL[(x[i].qs[j] & 0x0F) as usize] as i32;
            sumi += y[i].qs[j + 16] as i32 * KVALUES_IQ4NL[(x[i].qs[j] >> 4) as usize] as i32;
        }
        sum += d * sumi as f32;
    }

    sum
}

use crate::quant::types::{BlockIQ4XS, BlockIQ3XXS, BlockIQ3S, BlockIQ1S, BlockIQ1M, BlockIQ2XXS, BlockIQ2XS, BlockIQ2S, BlockTQ1_0, BlockTQ2_0};
use crate::quant::vec_dot::tables::{IQ3XXS_GRID, IQ3S_GRID, KEVEN_SIGNS_Q2XS, IQ1S_GRID, IQ1S_DELTA, IQ1M_DELTA, IQ2_XXS_GRID, IQ2_XS_GRID, IQ2_S_GRID};

/// IQ3_XXS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon(n: usize, x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let q3 = x[i].qs.as_ptr();
        let gas = q3.add(QK_K / 4); // Signs and scales start after grid indices
        let q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        // Process 2 blocks of 32 elements at a time (64 elements per iteration)
        for ib32 in (0..QK_K / 32).step_by(2) {
            // Load Q8 values
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));
            let q8b2 = vld1q_s8_x4(q8.add((ib32 + 1) * 32));

            // Load signs/scales (2 uint32_t per 32-element block)
            let aux32_0 = std::ptr::read_unaligned(gas.add(ib32 * 4) as *const u32);
            let aux32_1 = std::ptr::read_unaligned(gas.add((ib32 + 1) * 4) as *const u32);

            // Load grid indices (16 indices per 32-element block)
            let q3_ptr = q3.add(ib32 * 16);
            let grid_vals_0: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(0) as usize],
                IQ3XXS_GRID[*q3_ptr.add(1) as usize],
                IQ3XXS_GRID[*q3_ptr.add(2) as usize],
                IQ3XXS_GRID[*q3_ptr.add(3) as usize],
            ];
            let grid_vals_1: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(4) as usize],
                IQ3XXS_GRID[*q3_ptr.add(5) as usize],
                IQ3XXS_GRID[*q3_ptr.add(6) as usize],
                IQ3XXS_GRID[*q3_ptr.add(7) as usize],
            ];
            let grid_vals_2: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(8) as usize],
                IQ3XXS_GRID[*q3_ptr.add(9) as usize],
                IQ3XXS_GRID[*q3_ptr.add(10) as usize],
                IQ3XXS_GRID[*q3_ptr.add(11) as usize],
            ];
            let grid_vals_3: [u32; 4] = [
                IQ3XXS_GRID[*q3_ptr.add(12) as usize],
                IQ3XXS_GRID[*q3_ptr.add(13) as usize],
                IQ3XXS_GRID[*q3_ptr.add(14) as usize],
                IQ3XXS_GRID[*q3_ptr.add(15) as usize],
            ];
            let aux32x4_0 = vld1q_u32(grid_vals_0.as_ptr());
            let aux32x4_1 = vld1q_u32(grid_vals_1.as_ptr());
            let aux32x4_2 = vld1q_u32(grid_vals_2.as_ptr());
            let aux32x4_3 = vld1q_u32(grid_vals_3.as_ptr());

            // Load signs from keven_signs_q2xs (8 bytes per sign index)
            let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;
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

            // Dot product with Q8 values
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_0, q8b.0), q3s_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_2, q8b2.0), q3s_3, q8b2.1);

            // Apply scale: 0.5 + (aux32 >> 28)
            let scale1 = 0.5f32 + ((aux32_0 >> 28) as f32);
            let scale2 = 0.5f32 + ((aux32_1 >> 28) as f32);

            sumf1 += vaddvq_s32(p1) as f32 * scale1;
            sumf2 += vaddvq_s32(p2) as f32 * scale2;
        }

        sum += d * (sumf1 + sumf2) * 0.5;
    }

    sum
}

/// IQ3_S × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq3_s_q8_k_neon(n: usize, x: &[BlockIQ3S], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    // Masks for sign processing (matching llama.cpp k_mask1, k_mask2)
    static K_MASK1_0: [u8; 16] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f];
    static K_MASK1_1: [u8; 16] = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f];
    static K_MASK2: [u8; 16] = [0x01u8; 16];

    let mask1 = uint8x16x2_t(
        vld1q_u8(K_MASK1_0.as_ptr()),
        vld1q_u8(K_MASK1_1.as_ptr()),
    );
    let mask2 = vld1q_u8(K_MASK2.as_ptr());
    let m1 = vdupq_n_u8(1);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut qs = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let mut signs = x[i].signs.as_ptr() as *const u16;
        let q8 = y[i].qs.as_ptr();

        // Decode scales
        let mut scales32 = [0u32; 2];
        std::ptr::copy_nonoverlapping(x[i].scales.as_ptr(), scales32.as_mut_ptr() as *mut u8, 4);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;
        let scales8 = scales32.as_ptr() as *const u8;

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        // Process 2 blocks of 32 elements at a time
        for ib32 in (0..QK_K / 32).step_by(2) {
            // Load Q8 values (64 bytes for 2 blocks of 32)
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));

            // Load grid indices (16 bytes)
            let idx_l = vld1q_u8(qs);
            qs = qs.add(16);

            // Shift amounts: [8, 7, 6, 5, 4, 3, 2, 1]
            let hshift = vld1q_s16([8i16, 7, 6, 5, 4, 3, 2, 1].as_ptr());
            let m256 = vdupq_n_u16(256);

            // Combine low bits from qs with high bits from qh
            let qh_bits_0 = vdupq_n_u16(*qh.add(ib32) as u16);
            let qh_bits_1 = vdupq_n_u16(*qh.add(ib32 + 1) as u16);

            // Index for low 8 bytes: idx + ((qh >> shift) & 1) << 8
            let idx_lo = vmovl_u8(vget_low_u8(idx_l));
            let idx_hi = vmovl_u8(vget_high_u8(idx_l));

            // Shift qh by different amounts for each element, mask with 256, and OR into index
            let idx_final_0 = vorrq_u16(idx_lo, vandq_u16(vshlq_u16(qh_bits_0, hshift), m256));
            let idx_final_1 = vorrq_u16(idx_hi, vandq_u16(vshlq_u16(qh_bits_1, hshift), m256));

            // Extract indices for table lookup
            let indices_0: [u16; 8] = std::mem::transmute(idx_final_0);
            let indices_1: [u16; 8] = std::mem::transmute(idx_final_1);

            // Load grid values using indices (this could be optimized with gather, but we use scalar for now)
            let grid_vals_0: [u32; 4] = [
                IQ3S_GRID[indices_0[0] as usize],
                IQ3S_GRID[indices_0[1] as usize],
                IQ3S_GRID[indices_0[2] as usize],
                IQ3S_GRID[indices_0[3] as usize],
            ];
            let grid_vals_1: [u32; 4] = [
                IQ3S_GRID[indices_0[4] as usize],
                IQ3S_GRID[indices_0[5] as usize],
                IQ3S_GRID[indices_0[6] as usize],
                IQ3S_GRID[indices_0[7] as usize],
            ];
            let grid_vals_2: [u32; 4] = [
                IQ3S_GRID[indices_1[0] as usize],
                IQ3S_GRID[indices_1[1] as usize],
                IQ3S_GRID[indices_1[2] as usize],
                IQ3S_GRID[indices_1[3] as usize],
            ];
            let grid_vals_3: [u32; 4] = [
                IQ3S_GRID[indices_1[4] as usize],
                IQ3S_GRID[indices_1[5] as usize],
                IQ3S_GRID[indices_1[6] as usize],
                IQ3S_GRID[indices_1[7] as usize],
            ];

            let aux32x4_0 = vld1q_u32(grid_vals_0.as_ptr());
            let aux32x4_1 = vld1q_u32(grid_vals_1.as_ptr());
            let aux32x4_2 = vld1q_u32(grid_vals_2.as_ptr());
            let aux32x4_3 = vld1q_u32(grid_vals_3.as_ptr());

            // Load signs (2 u16 per block, 4 total)
            let s0 = *signs.add(0);
            let s1 = *signs.add(1);
            let s2 = *signs.add(2);
            let s3 = *signs.add(3);
            signs = signs.add(4);

            // Process signs using TBL
            let vs0 = vreinterpretq_u8_u32(vdupq_n_u32(s0 as u32 | ((s1 as u32) << 16)));
            let vs1 = vandq_u8(vqtbl1q_u8(vs0, mask1.1), mask2);
            let vs0 = vandq_u8(vqtbl1q_u8(vs0, mask1.0), mask2);
            let vs0 = vorrq_u8(vceqq_u8(vs0, mask2), m1);
            let vs1 = vorrq_u8(vceqq_u8(vs1, mask2), m1);

            // Apply signs to grid values
            let q3s_0 = vmulq_s8(vreinterpretq_s8_u8(vs0), vreinterpretq_s8_u32(aux32x4_0));
            let q3s_1 = vmulq_s8(vreinterpretq_s8_u8(vs1), vreinterpretq_s8_u32(aux32x4_1));

            // Process second block of signs
            let vs2 = vreinterpretq_u8_u32(vdupq_n_u32(s2 as u32 | ((s3 as u32) << 16)));
            let vs3 = vandq_u8(vqtbl1q_u8(vs2, mask1.1), mask2);
            let vs2 = vandq_u8(vqtbl1q_u8(vs2, mask1.0), mask2);
            let vs2 = vorrq_u8(vceqq_u8(vs2, mask2), m1);
            let vs3 = vorrq_u8(vceqq_u8(vs3, mask2), m1);

            let q3s_2 = vmulq_s8(vreinterpretq_s8_u8(vs2), vreinterpretq_s8_u32(aux32x4_2));
            let q3s_3 = vmulq_s8(vreinterpretq_s8_u8(vs3), vreinterpretq_s8_u32(aux32x4_3));

            // Dot product
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_0, q8b.0), q3s_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q3s_2, q8b.2), q3s_3, q8b.3);

            sumi1 += vaddvq_s32(p1) * *scales8.add(ib32 / 2) as i32;
            sumi2 += vaddvq_s32(p2) * *scales8.add(ib32 / 2 + 4) as i32;
        }

        sumf += d * (sumi1 + sumi2) as f32;
    }

    sumf
}

/// IQ1_S × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq1_s_q8_k_neon(n: usize, x: &[BlockIQ1S], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut qs = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();
        let bsums = y[i].bsums.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        let mut sumi3 = 0i32;

        // Process 8 blocks of 32 elements, 2 at a time
        for ib in (0..QK_K / 32).step_by(2) {
            // Load Q8 values (64 bytes for 2 blocks)
            let q8b = vld1q_s8_x4(q8.add(ib * 32));

            // Load grid indices (8 bytes per 2 blocks)
            let qs_vals: [u8; 8] = [
                *qs.add(0), *qs.add(1), *qs.add(2), *qs.add(3),
                *qs.add(4), *qs.add(5), *qs.add(6), *qs.add(7),
            ];
            qs = qs.add(8);

            // Load high bits from qh
            let qh0 = *qh.add(ib) as u32;
            let qh1 = *qh.add(ib + 1) as u32;

            // Compute grid indices and load values
            // Index formula: qs[j] | ((qh << shift) & 0x700)
            let idx_0: [usize; 8] = [
                (qs_vals[0] as usize) | (((qh0 << 8) & 0x700) as usize),
                (qs_vals[1] as usize) | (((qh0 << 5) & 0x700) as usize),
                (qs_vals[2] as usize) | (((qh0 << 2) & 0x700) as usize),
                (qs_vals[3] as usize) | (((qh0 >> 1) & 0x700) as usize),
                (qs_vals[4] as usize) | (((qh1 << 8) & 0x700) as usize),
                (qs_vals[5] as usize) | (((qh1 << 5) & 0x700) as usize),
                (qs_vals[6] as usize) | (((qh1 << 2) & 0x700) as usize),
                (qs_vals[7] as usize) | (((qh1 >> 1) & 0x700) as usize),
            ];

            // Load grid values (each entry is 8 bytes = 8 int8 values)
            // Combine two consecutive entries into int8x16_t
            let q1b_0 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[0]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[1]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_1 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[2]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[3]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_2 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[4]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[5]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_3 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx_0[6]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx_0[7]].to_le_bytes().as_ptr() as *const i8),
            );

            // Dot product
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q1b_0, q8b.0), q1b_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q1b_2, q8b.2), q1b_3, q8b.3);

            // Compute scale: 2*((qh >> 12) & 7) + 1
            let ls1 = 2 * ((qh0 >> 12) & 7) as i32 + 1;
            let ls2 = 2 * ((qh1 >> 12) & 7) as i32 + 1;

            sumi1 += vaddvq_s32(p1) * ls1;
            sumi2 += vaddvq_s32(p2) * ls2;

            // Delta term: sum of Q8 values * scale * sign
            let sign1 = if (qh0 & 0x8000) != 0 { -1i32 } else { 1i32 };
            let sign2 = if (qh1 & 0x8000) != 0 { -1i32 } else { 1i32 };

            sumi3 += (*bsums.add(2 * ib) as i32 + *bsums.add(2 * ib + 1) as i32) * ls1 * sign1;
            sumi3 += (*bsums.add(2 * ib + 2) as i32 + *bsums.add(2 * ib + 3) as i32) * ls2 * sign2;
        }

        sumf += d * (sumi1 + sumi2 + (IQ1S_DELTA * sumi3 as f32) as i32) as f32;
    }

    sumf
}

/// Helper to get delta vector by index
#[target_feature(enable = "neon,dotprod")]
unsafe fn get_delta(deltas: &int8x16x4_t, idx: u8) -> int8x16_t {
    match idx {
        0 => deltas.0,
        1 => deltas.1,
        2 => deltas.2,
        3 => deltas.3,
        _ => deltas.0, // Should never happen
    }
}

/// IQ1_M × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq1_m_q8_k_neon(n: usize, x: &[BlockIQ1M], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);
    let mask = vdupq_n_s32(0x7);
    let mone = vdupq_n_s32(1);

    // Delta vectors for handling sign combinations
    let deltas: int8x16x4_t = int8x16x4_t(
        vcombine_s8(vdup_n_s8(1), vdup_n_s8(1)),
        vcombine_s8(vdup_n_s8(-1), vdup_n_s8(1)),
        vcombine_s8(vdup_n_s8(1), vdup_n_s8(-1)),
        vcombine_s8(vdup_n_s8(-1), vdup_n_s8(-1)),
    );

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let mut qs = x[i].qs.as_ptr();
        let mut qh = x[i].qh.as_ptr();
        let q8 = y[i].qs.as_ptr();
        let sc = x[i].scales.as_ptr() as *const u16;

        // Extract merged scale from 4 u16 values
        let scale_u16 = (*sc.add(0) >> 12) | ((*sc.add(1) >> 8) & 0x00f0) | ((*sc.add(2) >> 4) & 0x0f00) | (*sc.add(3) & 0xf000);

        let mut sumi1 = vdupq_n_s32(0);
        let mut sumi2 = vdupq_n_s32(0);

        // Process 8 blocks of 32 elements, 2 at a time
        for ib in (0..QK_K / 32).step_by(2) {
            // Load grid indices (8 bytes per 2 blocks)
            let qs_vals: [u8; 8] = [
                *qs.add(0), *qs.add(1), *qs.add(2), *qs.add(3),
                *qs.add(4), *qs.add(5), *qs.add(6), *qs.add(7),
            ];
            qs = qs.add(8);

            // Load high bits from qh (4 bytes per 2 blocks)
            let qh_vals: [u8; 4] = [*qh.add(0), *qh.add(1), *qh.add(2), *qh.add(3)];
            qh = qh.add(4);

            // Compute grid indices
            let idx: [usize; 8] = [
                (qs_vals[0] as usize) | (((qh_vals[0] as u32) << 8) & 0x700) as usize,
                (qs_vals[1] as usize) | (((qh_vals[0] as u32) << 4) & 0x700) as usize,
                (qs_vals[2] as usize) | (((qh_vals[1] as u32) << 8) & 0x700) as usize,
                (qs_vals[3] as usize) | (((qh_vals[1] as u32) << 4) & 0x700) as usize,
                (qs_vals[4] as usize) | (((qh_vals[2] as u32) << 8) & 0x700) as usize,
                (qs_vals[5] as usize) | (((qh_vals[2] as u32) << 4) & 0x700) as usize,
                (qs_vals[6] as usize) | (((qh_vals[3] as u32) << 8) & 0x700) as usize,
                (qs_vals[7] as usize) | (((qh_vals[3] as u32) << 4) & 0x700) as usize,
            ];

            // Load grid values
            let q1b_0 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[0]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[1]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_1 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[2]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[3]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_2 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[4]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[5]].to_le_bytes().as_ptr() as *const i8),
            );
            let q1b_3 = vcombine_s8(
                vld1_s8(IQ1S_GRID[idx[6]].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ1S_GRID[idx[7]].to_le_bytes().as_ptr() as *const i8),
            );

            // Load Q8 values
            let q8b = vld1q_s8_x4(q8.add(ib * 32));

            // Dot product
            let p1 = vpaddq_s32(vdotq_s32_manual(vzero, q1b_0, q8b.0), vdotq_s32_manual(vzero, q1b_1, q8b.1));
            let p2 = vpaddq_s32(vdotq_s32_manual(vzero, q1b_2, q8b.2), vdotq_s32_manual(vzero, q1b_3, q8b.3));
            let p12 = vpaddq_s32(p1, p2);

            // Extract delta bits
            let qh32 = (qh_vals[0] as u32) | ((qh_vals[1] as u32) << 8) | ((qh_vals[2] as u32) << 16) | ((qh_vals[3] as u32) << 24);
            let aux32 = ((qh32 >> 3) & 0x01010101) | ((qh32 >> 6) & 0x02020202);
            let aux8 = aux32.to_le_bytes();

            // Delta dot products using aux8 indices
            let p3 = vpaddq_s32(
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[0]), q8b.0),
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[1]), q8b.1),
            );
            let p4 = vpaddq_s32(
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[2]), q8b.2),
                vdotq_s32_manual(vzero, get_delta(&deltas, aux8[3]), q8b.3),
            );
            let p34 = vpaddq_s32(p3, p4);

            // Load scales (4 scales per 2 blocks, packed in one u16)
            let sc_val = *sc.add(ib / 2);
            let scales_arr: [i32; 4] = [
                (sc_val >> 0) as i32,
                (sc_val >> 3) as i32,
                (sc_val >> 6) as i32,
                (sc_val >> 9) as i32,
            ];
            let scales_4 = vld1q_s32(scales_arr.as_ptr());
            let scales_4 = vaddq_s32(vshlq_n_s32(vandq_s32(scales_4, mask), 1), mone);

            sumi1 = vmlaq_s32(sumi1, scales_4, p12);
            sumi2 = vmlaq_s32(sumi2, scales_4, p34);
        }

        sumf += y[i].d * fp16_to_fp32(scale_u16) * (vaddvq_s32(sumi1) as f32 + IQ1M_DELTA * vaddvq_s32(sumi2) as f32);
    }

    sumf
}

/// IQ2_XXS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq2_xxs_q8_k_neon(n: usize, x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut q2 = x[i].qs.as_ptr() as *const u32;
        let q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        // Process 8 blocks of 32 elements, 2 at a time
        for ib32 in (0..QK_K / 32).step_by(2) {
            // Load Q8 values (64 bytes for 2 blocks)
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));

            // Load q2 values (4 u32 for 2 blocks)
            let aux32: [u32; 4] = [
                *q2.add(0), *q2.add(1), *q2.add(2), *q2.add(3),
            ];
            q2 = q2.add(4);

            let aux8: [u8; 16] = std::mem::transmute(aux32);

            // Load grid values using indices from aux8
            let q2u_0 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[0] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[1] as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_1 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[2] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[3] as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_2 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[8] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[9] as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_3 = vcombine_s8(
                vld1_s8(IQ2_XXS_GRID[aux8[10] as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XXS_GRID[aux8[11] as usize].to_le_bytes().as_ptr() as *const i8),
            );

            // Load signs from KEVEN_SIGNS_Q2XS
            let q2s_0 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[1] >> 0) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[1] >> 7) & 127) as usize * 8)),
            );
            let q2s_1 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[1] >> 14) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[1] >> 21) & 127) as usize * 8)),
            );
            let q2s_2 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[3] >> 0) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[3] >> 7) & 127) as usize * 8)),
            );
            let q2s_3 = vcombine_s8(
                vld1_s8(signs64.add(((aux32[3] >> 14) & 127) as usize * 8)),
                vld1_s8(signs64.add(((aux32[3] >> 21) & 127) as usize * 8)),
            );

            // Apply signs
            let q2u_0 = vmulq_s8(q2u_0, q2s_0);
            let q2u_1 = vmulq_s8(q2u_1, q2s_1);
            let q2u_2 = vmulq_s8(q2u_2, q2s_2);
            let q2u_3 = vmulq_s8(q2u_3, q2s_3);

            // Dot product
            let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q2u_0, q8b.0), q2u_1, q8b.1);
            let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q2u_2, q8b.2), q2u_3, q8b.3);

            // Apply scale
            let scale1 = 0.5f32 + ((aux32[1] >> 28) as f32);
            let scale2 = 0.5f32 + ((aux32[3] >> 28) as f32);

            sumf1 += vaddvq_s32(p1) as f32 * scale1;
            sumf2 += vaddvq_s32(p2) as f32 * scale2;
        }

        sumf += d * (sumf1 + sumf2) * 0.25;
    }

    sumf
}

/// IQ2_XS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq2_xs_q8_k_neon(n: usize, x: &[BlockIQ2XS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let q2 = x[i].qs.as_ptr();
        let q8 = y[i].qs.as_ptr();

        // Process scales
        let scales8 = vld1_u8(x[i].scales.as_ptr());
        let scales_l = vand_u8(scales8, vdup_n_u8(0xf));
        let scales_h = vshr_n_u8(scales8, 4);
        let scales = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));
        let scales = vaddq_u8(vshlq_n_u8(scales, 1), vdupq_n_u8(1));
        let scales1 = vmovl_u8(vget_low_u8(scales));
        let scales2 = vmovl_u8(vget_high_u8(scales));
        let scales32_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales1)));
        let scales32_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales1)));
        let scales32_2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales2)));
        let scales32_3 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales2)));

        let mut sumi = vdupq_n_s32(0);

        // Process 4 blocks of 64 elements
        for ib64 in 0..(QK_K / 64) {
            let q8b = vld1q_s8_x4(q8.add(ib64 * 64));

            // Load q2 values (8 u16 for 64 elements)
            let q2_vals: [u16; 8] = [
                *q2.add(ib64 * 8 + 0),
                *q2.add(ib64 * 8 + 1),
                *q2.add(ib64 * 8 + 2),
                *q2.add(ib64 * 8 + 3),
                *q2.add(ib64 * 8 + 4),
                *q2.add(ib64 * 8 + 5),
                *q2.add(ib64 * 8 + 6),
                *q2.add(ib64 * 8 + 7),
            ];

            // Load grid values using (q2 & 511) as index
            let q2u_0 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[0] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[1] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_1 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[2] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[3] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_2 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[4] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[5] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2u_3 = vcombine_s8(
                vld1_s8(IQ2_XS_GRID[(q2_vals[6] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_XS_GRID[(q2_vals[7] & 511) as usize].to_le_bytes().as_ptr() as *const i8),
            );

            // Load signs using (q2 >> 9) as index
            let q2s_0 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[0] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[1] >> 9) as usize * 8)),
            );
            let q2s_1 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[2] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[3] >> 9) as usize * 8)),
            );
            let q2s_2 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[4] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[5] >> 9) as usize * 8)),
            );
            let q2s_3 = vcombine_s8(
                vld1_s8(signs64.add((q2_vals[6] >> 9) as usize * 8)),
                vld1_s8(signs64.add((q2_vals[7] >> 9) as usize * 8)),
            );

            // Apply signs
            let q2u_0 = vmulq_s8(q2u_0, q2s_0);
            let q2u_1 = vmulq_s8(q2u_1, q2s_1);
            let q2u_2 = vmulq_s8(q2u_2, q2s_2);
            let q2u_3 = vmulq_s8(q2u_3, q2s_3);

            // Dot product
            let p1 = vdotq_s32_manual(vzero, q2u_0, q8b.0);
            let p2 = vdotq_s32_manual(vzero, q2u_1, q8b.1);
            let p3 = vdotq_s32_manual(vzero, q2u_2, q8b.2);
            let p4 = vdotq_s32_manual(vzero, q2u_3, q8b.3);
            let p = vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4));

            // Select appropriate scale vector based on block index
            let scale_vec = match ib64 {
                0 => scales32_0,
                1 => scales32_1,
                2 => scales32_2,
                3 => scales32_3,
                _ => scales32_0,
            };

            sumi = vmlaq_s32(sumi, p, scale_vec);
        }

        sumf += d * vaddvq_s32(sumi) as f32 * 0.125;
    }

    sumf
}

/// IQ2_S × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq2_s_q8_k_neon(n: usize, x: &[BlockIQ2S], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let vzero = vdupq_n_s32(0);

    // Masks for sign processing
    static K_MASK1: [u8; 32] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    ];
    static K_MASK2: [u8; 16] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    ];

    let mask1 = uint8x16x2_t(
        vld1q_u8(K_MASK1.as_ptr()),
        vld1q_u8(K_MASK1.as_ptr().add(16)),
    );
    let mask2 = vld1q_u8(K_MASK2.as_ptr());
    let m1 = vdupq_n_u8(1);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;

        let mut qs = x[i].qs.as_ptr();
        let qh = x[i].qh.as_ptr();
        let mut signs = (x[i].qs.as_ptr().add(QK_K / 8)) as *const u16;
        let q8 = y[i].qs.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        // Process 8 blocks of 32 elements, 2 at a time
        for ib32 in (0..QK_K / 32).step_by(2) {
            let q8b = vld1q_s8_x4(q8.add(ib32 * 32));

            // Load grid indices (8 bytes)
            let qs_vals: [u8; 8] = [
                *qs.add(0), *qs.add(1), *qs.add(2), *qs.add(3),
                *qs.add(4), *qs.add(5), *qs.add(6), *qs.add(7),
            ];
            qs = qs.add(8);

            // Load high bits
            let qh0 = *qh.add(ib32) as u32;
            let qh1 = *qh.add(ib32 + 1) as u32;

            // Compute grid indices and load values
            let q2s_0 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[0] as usize | ((qh0 << 8) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[1] as usize | ((qh0 << 6) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2s_1 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[2] as usize | ((qh0 << 4) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[3] as usize | ((qh0 << 2) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2s_2 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[4] as usize | ((qh1 << 8) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[5] as usize | ((qh1 << 6) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );
            let q2s_3 = vcombine_s8(
                vld1_s8(IQ2_S_GRID[qs_vals[6] as usize | ((qh1 << 4) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
                vld1_s8(IQ2_S_GRID[qs_vals[7] as usize | ((qh1 << 2) & 0x300) as usize].to_le_bytes().as_ptr() as *const i8),
            );

            // Load signs (4 u16 values)
            let s0 = *signs.add(0);
            let s1 = *signs.add(1);
            let s2 = *signs.add(2);
            let s3 = *signs.add(3);
            signs = signs.add(4);

            // Process signs using TBL
            let vs_0 = vreinterpretq_u8_u32(vdupq_n_u32(s0 as u32 | ((s1 as u32) << 16)));
            let vs_1 = vandq_u8(vqtbl1q_u8(vs_0, mask1.1), mask2);
            let vs_0 = vandq_u8(vqtbl1q_u8(vs_0, mask1.0), mask2);
            let vs_0 = vceqq_u8(vs_0, mask2);
            let vs_1 = vceqq_u8(vs_1, mask2);

            // Apply signs
            let q2s_0 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_0, m1)), q2s_0);
            let q2s_1 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_1, m1)), q2s_1);

            // Process second block signs
            let vs_2 = vreinterpretq_u8_u32(vdupq_n_u32(s2 as u32 | ((s3 as u32) << 16)));
            let vs_3 = vandq_u8(vqtbl1q_u8(vs_2, mask1.1), mask2);
            let vs_2 = vandq_u8(vqtbl1q_u8(vs_2, mask1.0), mask2);
            let vs_2 = vceqq_u8(vs_2, mask2);
            let vs_3 = vceqq_u8(vs_3, mask2);

            let q2s_2 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_2, m1)), q2s_2);
            let q2s_3 = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs_3, m1)), q2s_3);

            // Dot product
            let p1 = vdotq_s32_manual(vzero, q2s_0, q8b.0);
            let p2 = vdotq_s32_manual(vzero, q2s_1, q8b.1);
            let p3 = vdotq_s32_manual(vzero, q2s_2, q8b.2);
            let p4 = vdotq_s32_manual(vzero, q2s_3, q8b.3);

            // Apply scales
            sumi1 += vaddvq_s32(p1) * (1 + 2 * (x[i].scales[ib32] as i32 & 0xf));
            sumi2 += vaddvq_s32(p2) * (1 + 2 * (x[i].scales[ib32] as i32 >> 4));
            sumi1 += vaddvq_s32(p3) * (1 + 2 * (x[i].scales[ib32 + 1] as i32 & 0xf));
            sumi2 += vaddvq_s32(p4) * (1 + 2 * (x[i].scales[ib32 + 1] as i32 >> 4));
        }

        sumf += d * (sumi1 + sumi2) as f32 * 0.125;
    }

    sumf
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
    let vzero = vdupq_n_s32(0);

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

/// IQ4_XS × Q8_K vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq4_xs_q8_k_neon(n: usize, x: &[BlockIQ4XS], y: &[BlockQ8K]) -> f32 {
    let nb = n / QK_K;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);
    let values = vld1q_s8(KVALUES_IQ4NL.as_ptr());

    let mut sum = 0.0f32;

    for ibl in 0..nb {
        let q4 = x[ibl].qs.as_ptr();
        let q8 = y[ibl].qs.as_ptr();
        let mut h = x[ibl].scales_h;

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        for ib in 0..(QK_K / 64) {
            let q4bits = vld1q_u8_x2(q4.add(ib * 32));
            let q8bytes = vld1q_s8_x4(q8.add(ib * 64));

            let q4b_0 = vqtbl1q_s8(values, vandq_u8(q4bits.0, m4b));
            let q4b_1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.0, 4));
            let q4b_2 = vqtbl1q_s8(values, vandq_u8(q4bits.1, m4b));
            let q4b_3 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.1, 4));

            let prod_1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_0, q8bytes.0), q4b_1, q8bytes.1);
            let prod_2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4b_2, q8bytes.2), q4b_3, q8bytes.3);

            let ls1 = ((x[ibl].scales_l[ib] & 0x0F) as i32 | ((h as i32 & 0x0F) << 4)) - 32;
            let ls2 = ((x[ibl].scales_l[ib] >> 4) as i32 | ((h as i32 >> 4) << 4)) - 32;
            h >>= 8;

            sumi1 += vaddvq_s32(prod_1) * ls1;
            sumi2 += vaddvq_s32(prod_2) * ls2;
        }

        sum += fp16_to_fp32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2) as f32;
    }

    sum
}
