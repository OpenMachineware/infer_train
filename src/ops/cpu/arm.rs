#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::quant::{BlockQ2K, BlockQ4K, BlockQ8K, f16_to_f32};
use crate::QK_K;

/// Q4_K scale decoding constants
const KMASK1: u32 = 0x3f3f3f3f;
const KMASK2: u32 = 0x0f0f0f0f;
const KMASK3: u32 = 0x03030303;

/// Decode Q4_K scales from 12-byte format
/// Returns (scales[8], mins[8]) as u8 arrays
#[inline]
fn decode_q4k_scales(scale_bytes: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let utmp: [u32; 3] = [
        u32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]),
        u32::from_le_bytes([scale_bytes[4], scale_bytes[5], scale_bytes[6], scale_bytes[7]]),
        u32::from_le_bytes([scale_bytes[8], scale_bytes[9], scale_bytes[10], scale_bytes[11]]),
    ];

    // Decode mins
    let mins8_0 = utmp[1] & KMASK1;
    let mins8_1 = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);

    let mins: [u8; 8] = unsafe {
        let mut arr = [0u8; 8];
        std::ptr::copy_nonoverlapping(&mins8_0 as *const u32 as *const u8, arr.as_mut_ptr(), 4);
        std::ptr::copy_nonoverlapping(&mins8_1 as *const u32 as *const u8, arr.as_mut_ptr().add(4), 4);
        arr
    };

    // Decode scales
    let scales_0 = utmp[0] & KMASK1;
    let scales_1 = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);

    let scales: [u8; 8] = unsafe {
        let mut arr = [0u8; 8];
        std::ptr::copy_nonoverlapping(&scales_0 as *const u32 as *const u8, arr.as_mut_ptr(), 4);
        std::ptr::copy_nonoverlapping(&scales_1 as *const u32 as *const u8, arr.as_mut_ptr().add(4), 4);
        arr
    };

    (scales, mins)
}

/// Q4_K × Q8_K dot product (NEON implementation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q4k_q8k(n: usize, x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    debug_assert!(n % QK_K == 0);
    let nb = n / QK_K;

    let m4b = vdupq_n_u8(0x0F);
    let mzero = vdupq_n_s32(0);

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);
        let dmin = y[i].d * f16_to_f32(x[i].dmin);

        // Load and sum Q8 bsums
        let q8sums_lo = vld1q_s16(y[i].bsums.as_ptr());
        let q8sums_hi = vld1q_s16(y[i].bsums.as_ptr().add(8));
        let q8sums = vpaddq_s16(q8sums_lo, q8sums_hi);

        // Decode scales and mins
        let (scales, mins) = decode_q4k_scales(&x[i].scales);

        // Calculate min correction
        let mins16 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vld1_u32(&mins[0] as *const u8 as *const u32))));
        let prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins16)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins16))
        );
        sumf -= dmin * vaddvq_s32(prod) as f32;

        let q4 = x[i].qs.as_ptr();
        let q8 = y[i].qs.as_ptr();

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;

        // Process 64 elements per iteration (4 iterations total)
        for j in 0..4 {
            // Load 32 bytes of 4-bit quantized values
            let q4bits0 = vld1q_u8(q4.add(j * 32));
            let q4bits1 = vld1q_u8(q4.add(j * 32 + 16));

            // Low nibble (4-bit -> 8-bit)
            let q4l0 = vreinterpretq_s8_u8(vandq_u8(q4bits0, m4b));
            let q4l1 = vreinterpretq_s8_u8(vandq_u8(q4bits1, m4b));

            // Load Q8 values
            let q8v0 = vld1q_s8(q8.add(j * 64));
            let q8v1 = vld1q_s8(q8.add(j * 64 + 16));

            // Dot product for low nibbles
            let p1 = vdotq_s32_manual(vdotq_s32_manual(mzero, q4l0, q8v0), q4l1, q8v1);
            sumi1 += vaddvq_s32(p1) * scales[2 * j] as i32;

            // High nibble (shift right by 4)
            let q4h0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits0, 4));
            let q4h1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits1, 4));

            // Load Q8 values
            let q8v2 = vld1q_s8(q8.add(j * 64 + 32));
            let q8v3 = vld1q_s8(q8.add(j * 64 + 48));

            // Dot product for high nibbles
            let p2 = vdotq_s32_manual(vdotq_s32_manual(mzero, q4h0, q8v2), q4h1, q8v3);
            sumi2 += vaddvq_s32(p2) * scales[2 * j + 1] as i32;
        }

        sumf += d * (sumi1 + sumi2) as f32;
    }

    sumf
}

/// Q2_K × Q8_K dot product (NEON implementation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q2k_q8k(n: usize, x: &[BlockQ2K], y: &[BlockQ8K]) -> f32 {
    debug_assert!(n % QK_K == 0);
    let nb = n / QK_K;

    let m3 = vdupq_n_u8(0x03);
    let m4 = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * f16_to_f32(x[i].d);
        let dmin = -y[i].d * f16_to_f32(x[i].dmin);

        let q2 = x[i].qs.as_ptr();
        let mut q8 = y[i].qs.as_ptr();
        let sc = x[i].scales.as_ptr();

        // Load scales and mins
        let mins_and_scales = vld1q_u8(sc);
        let scales = vandq_u8(mins_and_scales, m4);
        let mins = vshrq_n_u8(mins_and_scales, 4);

        // Calculate min correction
        let q8sums = vld1q_s16(y[i].bsums.as_ptr());
        let mins16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
        let s0 = vmull_s16(vget_low_s16(mins16), vget_low_s16(q8sums));
        let s1 = vmull_s16(vget_high_s16(mins16), vget_high_s16(q8sums));
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1)) as f32;

        let mut isum = 0i32;
        let mut is = 0usize;

        // Store scales in array for scalar access
        let scale_arr: [u8; 16] = std::mem::transmute(scales);

        // Process 128 elements per iteration (2 iterations total)
        for _j in 0..2 {
            let q2bits0 = vld1q_u8(q2.add(is * 2));
            let q2bits1 = vld1q_u8(q2.add(is * 2 + 16));

            // Shift = 0
            let q2bytes0 = vreinterpretq_s8_u8(vandq_u8(q2bits0, m3));
            let q2bytes1 = vreinterpretq_s8_u8(vandq_u8(q2bits1, m3));
            let q8bytes0 = vld1q_s8(q8);
            let q8bytes1 = vld1q_s8(q8.add(16));
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes0, q8bytes0)) * scale_arr[is] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes1, q8bytes1)) * scale_arr[is + 1] as i32;

            // Shift = 2
            let q2bytes0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 2), m3));
            let q2bytes1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 2), m3));
            let q8bytes0 = vld1q_s8(q8.add(32));
            let q8bytes1 = vld1q_s8(q8.add(48));
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes0, q8bytes0)) * scale_arr[is + 2] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes1, q8bytes1)) * scale_arr[is + 3] as i32;

            // Shift = 4
            let q2bytes0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 4), m3));
            let q2bytes1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 4), m3));
            let q8bytes0 = vld1q_s8(q8.add(64));
            let q8bytes1 = vld1q_s8(q8.add(80));
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes0, q8bytes0)) * scale_arr[is + 4] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes1, q8bytes1)) * scale_arr[is + 5] as i32;

            // Shift = 6
            let q2bytes0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 6), m3));
            let q2bytes1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 6), m3));
            let q8bytes0 = vld1q_s8(q8.add(96));
            let q8bytes1 = vld1q_s8(q8.add(112));
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes0, q8bytes0)) * scale_arr[is + 6] as i32;
            isum += vaddvq_s32(vdotq_s32_manual(vzero, q2bytes1, q8bytes1)) * scale_arr[is + 7] as i32;

            q8 = q8.add(128);
            is += 8;
        }

        sum += d * isum as f32;
    }

    sum
}

/// Manual implementation of vdotq_s32 (avoiding unstable feature)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn vdotq_s32_manual(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let a_lo = vmovl_s8(vget_low_s8(a));
    let a_hi = vmovl_s8(vget_high_s8(a));
    let b_lo = vmovl_s8(vget_low_s8(b));
    let b_hi = vmovl_s8(vget_high_s8(b));

    let prod_lo = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
    let prod_hi = vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo));
    let sum_lo = vaddq_s32(prod_lo, prod_hi);

    let prod_lo2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
    let prod_hi2 = vmull_s16(vget_high_s16(a_hi), vget_high_s16(b_hi));
    let sum_hi = vaddq_s32(prod_lo2, prod_hi2);

    vaddq_s32(acc, vaddq_s32(sum_lo, sum_hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_q4k_scales() {
        // Test scale decoding
        let scale_bytes: [u8; 12] = [0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x3F, 0x0F, 0x0F, 0x0F, 0x0F];
        let (scales, mins) = decode_q4k_scales(&scale_bytes);

        // Verify scales are decoded correctly
        assert!(scales.iter().all(|&s| s == 0x3F));
        assert!(mins.iter().all(|&m| m == 0x3F));
    }
}
