// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Vector dot product operators

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::types::*;

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

// Q4_K x Q8_K dot product
pub fn vec_dot_q4_k_q8_k(x: &[BlockQ4_K], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];

    for i in 0..nb {
        let block_x = &x[i];
        let block_y = &y[i];

        let mut utmp: [u32; 4] = [0; 4];
        utmp[0..3].copy_from_slice(unsafe {
            std::slice::from_raw_parts(block_x.scales.as_ptr() as *const u32, 3)
        });

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let scales: [u8; 8] = unsafe {
            std::slice::from_raw_parts(utmp[0..2].as_ptr() as *const u8, 8)
                .try_into()
                .unwrap()
        };
        let mins: [u8; 8] = unsafe {
            std::slice::from_raw_parts(utmp[2..4].as_ptr() as *const u8, 8)
                .try_into()
                .unwrap()
        };

        let mut aux8 = [0i8; 256];
        let q4 = &block_x.qs;
        for j in 0..4 {
            for l in 0..32 {
                aux8[j * 64 + l] = (q4[j * 32 + l] & 0xF) as i8;
            }
            for l in 0..32 {
                aux8[j * 64 + 32 + l] = (q4[j * 32 + l] >> 4) as i8;
            }
        }

        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += block_y.bsums[j] as i32 * mins[j / 2] as i32;
        }

        let mut aux32 = [0i32; 8];
        let q8 = &block_y.qs;
        let mut is = 0;
        for j in 0..8 {
            let scale = scales[is] as i32;
            for l in 0..8 {
                aux32[l] +=
                    scale * q8[j * 32 + l] as i32 * aux8[j * 32 + l] as i32;
            }
            for l in 0..8 {
                aux32[l] += scale
                    * q8[j * 32 + 8 + l] as i32
                    * aux8[j * 32 + 8 + l] as i32;
            }
            for l in 0..8 {
                aux32[l] += scale
                    * q8[j * 32 + 16 + l] as i32
                    * aux8[j * 32 + 16 + l] as i32;
            }
            for l in 0..8 {
                aux32[l] += scale
                    * q8[j * 32 + 24 + l] as i32
                    * aux8[j * 32 + 24 + l] as i32;
            }
            is += 1;
        }

        let d = fp16_to_fp32(block_x.d) * block_y.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }

        let dmin = fp16_to_fp32(block_x.dmin) * block_y.d;
        sums[0] -= dmin * sumi as f32;
    }

    sums.iter().sum()
}

// Q5_K x Q8_K dot product
pub fn vec_dot_q5_k_q8_k(x: &[BlockQ5_K], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];

    for i in 0..nb {
        let block_x = &x[i];
        let block_y = &y[i];

        let mut utmp: [u32; 4] = [0; 4];
        utmp[0..3].copy_from_slice(unsafe {
            std::slice::from_raw_parts(block_x.scales.as_ptr() as *const u32, 3)
        });

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let scales: [u8; 8] = unsafe {
            std::slice::from_raw_parts(utmp[0..2].as_ptr() as *const u8, 8)
                .try_into()
                .unwrap()
        };
        let mins: [u8; 8] = unsafe {
            std::slice::from_raw_parts(utmp[2..4].as_ptr() as *const u8, 8)
                .try_into()
                .unwrap()
        };

        let mut aux8 = [0i8; 256];
        let q4 = &block_x.qs;
        let qh = &block_x.qh;
        let mut m: u8 = 1;
        for j in 0..4 {
            for l in 0..32 {
                aux8[j * 64 + l] = (q4[j * 32 + l] & 0xF) as i8
                    + if qh[l] & m != 0 { 16 } else { 0 };
            }
            m <<= 1;
            for l in 0..32 {
                aux8[j * 64 + 32 + l] = (q4[j * 32 + l] >> 4) as i8
                    + if qh[l] & m != 0 { 16 } else { 0 };
            }
            m <<= 1;
        }

        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += block_y.bsums[j] as i32 * mins[j / 2] as i32;
        }

        let mut aux32 = [0i32; 8];
        let q8 = &block_y.qs;
        let mut is = 0;
        for j in 0..8 {
            let scale = scales[is] as i32;
            for l in 0..8 {
                aux32[l] +=
                    scale * q8[j * 32 + l] as i32 * aux8[j * 32 + l] as i32;
            }
            for l in 0..8 {
                aux32[l] += scale
                    * q8[j * 32 + 8 + l] as i32
                    * aux8[j * 32 + 8 + l] as i32;
            }
            for l in 0..8 {
                aux32[l] += scale
                    * q8[j * 32 + 16 + l] as i32
                    * aux8[j * 32 + 16 + l] as i32;
            }
            for l in 0..8 {
                aux32[l] += scale
                    * q8[j * 32 + 24 + l] as i32
                    * aux8[j * 32 + 24 + l] as i32;
            }
            is += 1;
        }

        let d = fp16_to_fp32(block_x.d) * block_y.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }

        let dmin = fp16_to_fp32(block_x.dmin) * block_y.d;
        sums[0] -= dmin * sumi as f32;
    }

    sums.iter().sum()
}

// Q6_K x Q8_K dot product
pub fn vec_dot_q6_k_q8_k(x: &[BlockQ6_K], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let mut sum = 0.0f32;

    for i in 0..nb {
        let d = y[i].d * fp16_to_fp32(x[i].d);
        let mut isum = 0i32;

        // Process 16 blocks of 16 elements each
        for j in 0..16 {
            for k in 0..8 {
                // Combine lower 4 bits and upper 2 bits
                let lower4 = x[i].ql[j * 8 + k] as i32;
                let upper2 =
                    (x[i].qh[j * 4 + k / 2] >> (4 * (k % 2))) as i32 & 0x03;
                let v = lower4 | (upper2 << 4);

                isum += v * y[i].qs[j * 16 + k] as i32 * x[i].scales[j] as i32;
                isum +=
                    v * y[i].qs[j * 16 + k + 8] as i32 * x[i].scales[j] as i32;
            }
        }

        sum += d * isum as f32;
    }

    sum
}

// IQ4_XS uses non-linear 4-bit quantization with a lookup table
// The KVALUES_IQ4NL_I8 table maps 4-bit indices to signed values
const KVALUES_IQ4NL_I8: [i8; 16] =
    [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];

pub fn vec_dot_iq4_xs_q8_k(x: &[BlockIQ4XS], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let mut sum = 0.0f32;

    for ibl in 0..nb {
        let mut sumi = 0i64;

        // Process 4 sub-blocks of 64 elements each
        for ib in 0..4 {
            // Extract scales for this sub-block
            let ls = ((x[ibl].scales_l[ib] & 0x0F) as i32
                | (((x[ibl].scales_h as i32) >> (ib * 4)) & 0x30))
                - 32;

            // Process 64 elements in this sub-block
            for j in 0..32 {
                // Get 4-bit values from packed bytes
                let v0_idx = (x[ibl].qs[ib * 32 + j] & 0x0F) as usize;
                let v1_idx = (x[ibl].qs[ib * 32 + j] >> 4) as usize;

                // Look up the non-linear quantized values
                let v0 = KVALUES_IQ4NL_I8[v0_idx] as i64;
                let v1 = KVALUES_IQ4NL_I8[v1_idx] as i64;

                // Multiply with Q8 values and accumulate
                let q8_0 = y[ibl].qs[ib * 64 + j] as i64;
                let q8_1 = y[ibl].qs[ib * 64 + j + 32] as i64;

                sumi += (v0 * q8_0 + v1 * q8_1) * ls as i64;
            }
        }

        sum += fp16_to_fp32(x[ibl].d) * y[ibl].d * sumi as f32;
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
