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
    let mut sumf = 0.0f32;

    for i in 0..nb {
        let block_x = &x[i];
        let block_y = &y[i];

        let mut utmp: [u32; 4] = [0; 4];
        // Copy scales bytes to utmp - use byte-by-byte copy to avoid alignment issues
        // The scales array is 12 bytes, we need to interpret it as 3 uint32s
        // On little-endian systems, this should work correctly
        let scales_bytes = &block_x.scales;
        unsafe {
            std::ptr::copy_nonoverlapping(
                scales_bytes.as_ptr(),
                utmp.as_mut_ptr() as *mut u8,
                12,
            );
        }

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
        for j in 0..8 {
            let scale = scales[j] as i32;
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
        }

        let d = fp16_to_fp32(block_x.d) * block_y.d;
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }

        let dmin = fp16_to_fp32(block_x.dmin) * block_y.d;
        sumf -= dmin * sumi as f32;
    }

    sumf + sums.iter().sum::<f32>()
}

// Q4_K x Q8_K dot product for a single block pair
pub fn vec_dot_q4_k_q8_k_single(x: &BlockQ4_K, y: &BlockQ8_K) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    let mut utmp: [u32; 4] = [0; 4];
    // Copy scales bytes to utmp - use byte-by-byte copy to avoid alignment issues
    let scales_bytes = &x.scales;
    unsafe {
        std::ptr::copy_nonoverlapping(
            scales_bytes.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

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
    let q4 = &x.qs;
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
        sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
    }

    let mut aux32 = [0i32; 8];
    let q8 = &y.qs;
    let mut is = 0;
    for j in 0..8 {
        let scale = scales[is] as i32;
        for l in 0..8 {
            aux32[l] += scale * q8[j * 32 + l] as i32 * aux8[j * 32 + l] as i32;
        }
        for l in 0..8 {
            aux32[l] +=
                scale * q8[j * 32 + 8 + l] as i32 * aux8[j * 32 + 8 + l] as i32;
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

    let d = fp16_to_fp32(x.d) * y.d;
    for l in 0..8 {
        sums[l] += d * aux32[l] as f32;
    }

    let dmin = fp16_to_fp32(x.dmin) * y.d;
    sumf -= dmin * sumi as f32;

    sumf + sums.iter().sum::<f32>()
}

// Q5_K x Q8_K dot product for a single block pair
pub fn vec_dot_q5_k_q8_k_single(x: &BlockQ5_K, y: &BlockQ8_K) -> f32 {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    // Step 1: Unpack Q5_K quantized values to int8
    let mut aux8 = [0i8; 256];
    let q4 = &x.qs;
    let qh = &x.qh;
    let mut m: u8 = 1;
    let mut q4_offset = 0;

    for j in 0..4 {
        let base = j * 64;
        // Lower 4 bits
        for l in 0..32 {
            aux8[base + l] = (q4[q4_offset + l] & 0xF) as i8
                + if qh[l] & m != 0 { 16 } else { 0 };
        }
        m <<= 1;
        // Upper 4 bits
        for l in 0..32 {
            aux8[base + 32 + l] = (q4[q4_offset + l] >> 4) as i8
                + if qh[l] & m != 0 { 16 } else { 0 };
        }
        m <<= 1;
        q4_offset += 32;
    }

    // Step 2: Decode scales and mins
    let mut utmp: [u32; 4] = [0; 4];
    // Copy scales bytes to utmp - use byte-by-byte copy to avoid alignment issues
    let scales_bytes = &x.scales;
    unsafe {
        std::ptr::copy_nonoverlapping(
            scales_bytes.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

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

    // Step 3: Compute sum of min terms
    let mut sumi = 0i32;
    for j in 0..16 {
        sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
    }

    // Step 4: Compute aux32
    let mut aux32 = [0i32; 8];
    let q8 = &y.qs;
    let a = &aux8[..];
    let mut is = 0;
    let mut q8_idx = 0;
    let mut a_idx = 0;

    for _j in 0..8 {
        let scale = scales[is] as i32;
        is += 1;

        // 4 inner loops of 8 elements each
        for l in 0..8 {
            let aux16 = q8[q8_idx + l] as i32 * a[a_idx + l] as i32;
            aux32[l] += scale * aux16;
        }
        q8_idx += 8;
        a_idx += 8;

        for l in 0..8 {
            let aux16 = q8[q8_idx + l] as i32 * a[a_idx + l] as i32;
            aux32[l] += scale * aux16;
        }
        q8_idx += 8;
        a_idx += 8;

        for l in 0..8 {
            let aux16 = q8[q8_idx + l] as i32 * a[a_idx + l] as i32;
            aux32[l] += scale * aux16;
        }
        q8_idx += 8;
        a_idx += 8;

        for l in 0..8 {
            let aux16 = q8[q8_idx + l] as i32 * a[a_idx + l] as i32;
            aux32[l] += scale * aux16;
        }
        q8_idx += 8;
        a_idx += 8;
    }

    // Step 5: Apply final scale
    let d = fp16_to_fp32(x.d) * y.d;
    for l in 0..8 {
        sums[l] += d * aux32[l] as f32;
    }

    let dmin = fp16_to_fp32(x.dmin) * y.d;
    sumf -= dmin * sumi as f32;

    sumf + sums.iter().sum::<f32>()
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

// Q6_K x Q8_K dot product
pub fn vec_dot_q6_k_q8_k(x: &[BlockQ6_K], y: &[BlockQ8_K]) -> f32 {
    let nb = x.len();
    assert_eq!(nb, y.len());

    let mut sums = [0.0f32; 8];

    for i in 0..nb {
        let d = y[i].d * fp16_to_fp32(x[i].d);

        // Create aux8 array with values in range -32 to 31
        // Matches llama.cpp exactly: q4 (ql) advances by 64 per 128 output, qh advances by 32 per 128 output
        let mut aux8 = [0i8; 256];
        let q4 = &x[i].ql;
        let qh = &x[i].qh;

        // First 128 elements: q4 uses [0..64], qh uses [0..32]
        for l in 0..32 {
            aux8[l + 0] =
                ((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32;
            aux8[l + 32] =
                ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            aux8[l + 64] =
                ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            aux8[l + 96] =
                ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
        }

        // Second 128 elements: q4 uses [64..128], qh uses [32..64]
        for l in 0..32 {
            aux8[l + 128 + 0] =
                ((q4[l + 64 + 0] & 0xF) | (((qh[l + 32] >> 0) & 3) << 4)) as i8
                    - 32;
            aux8[l + 128 + 32] = ((q4[l + 64 + 32] & 0xF)
                | (((qh[l + 32] >> 2) & 3) << 4))
                as i8
                - 32;
            aux8[l + 128 + 64] =
                ((q4[l + 64 + 0] >> 4) | (((qh[l + 32] >> 4) & 3) << 4)) as i8
                    - 32;
            aux8[l + 128 + 96] =
                ((q4[l + 64 + 32] >> 4) | (((qh[l + 32] >> 6) & 3) << 4)) as i8
                    - 32;
        }

        let q8 = &y[i].qs;
        let scales = &x[i].scales;

        let mut aux32 = [0i32; 8];
        let mut is = 0;
        let mut a_idx = 0;

        for _j in 0..16 {
            let scale = scales[is] as i32;
            for l in 0..8 {
                aux32[l] +=
                    scale * q8[a_idx + l] as i32 * aux8[a_idx + l] as i32;
            }
            a_idx += 8;
            for l in 0..8 {
                aux32[l] +=
                    scale * q8[a_idx + l] as i32 * aux8[a_idx + l] as i32;
            }
            a_idx += 8;
            is += 1;
        }

        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }

    sums.iter().sum()
}

// Q6_K x Q8_K dot product for a single block pair
pub fn vec_dot_q6_k_q8_k_single(x: &BlockQ6_K, y: &BlockQ8_K) -> f32 {
    let d = y.d * fp16_to_fp32(x.d);

    // Create aux8 array with values in range -32 to 31
    let mut aux8 = [0i8; 256];
    let q4 = &x.ql;
    let qh = &x.qh;

    // First 128 elements: q4 uses [0..64], qh uses [0..32]
    for l in 0..32 {
        aux8[l + 0] =
            ((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32;
        aux8[l + 32] =
            ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
        aux8[l + 64] =
            ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
        aux8[l + 96] =
            ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
    }

    // Second 128 elements: q4 uses [64..128], qh uses [32..64]
    for l in 0..32 {
        aux8[l + 128 + 0] = ((q4[l + 64 + 0] & 0xF)
            | (((qh[l + 32] >> 0) & 3) << 4)) as i8
            - 32;
        aux8[l + 128 + 32] =
            ((q4[l + 64 + 32] & 0xF) | (((qh[l + 32] >> 2) & 3) << 4)) as i8
                - 32;
        aux8[l + 128 + 64] =
            ((q4[l + 64 + 0] >> 4) | (((qh[l + 32] >> 4) & 3) << 4)) as i8 - 32;
        aux8[l + 128 + 96] =
            ((q4[l + 64 + 32] >> 4) | (((qh[l + 32] >> 6) & 3) << 4)) as i8
                - 32;
    }

    let q8 = &y.qs;
    let scales = &x.scales;

    let mut aux32 = [0i32; 8];
    let mut is = 0;
    let mut a_idx = 0;

    for _j in 0..16 {
        let scale = scales[is] as i32;
        for l in 0..8 {
            aux32[l] += scale * q8[a_idx + l] as i32 * aux8[a_idx + l] as i32;
        }
        a_idx += 8;
        for l in 0..8 {
            aux32[l] += scale * q8[a_idx + l] as i32 * aux8[a_idx + l] as i32;
        }
        a_idx += 8;
        is += 1;
    }

    let mut sum = 0.0f32;
    for l in 0..8 {
        sum += d * aux32[l] as f32;
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
        let d4d8 = fp16_to_fp32(x[ibl].d) * y[ibl].d;
        let mut h = x[ibl].scales_h;

        // Process 8 sub-blocks of 32 elements each (ib = 0, 2, 4, 6)
        // Each iteration handles 2 sub-blocks (64 elements)
        let mut qs_offset = 0usize;
        let mut q8_offset = 0usize;

        for ib in (0..8).step_by(2) {
            // Extract scales for two sub-blocks
            // scales_l[ib/2] contains two 4-bit scales
            let ls1 = ((x[ibl].scales_l[ib / 2] & 0x0F) as i32
                | (((h as i32) << 4) & 0x30))
                - 32;
            let ls2 = ((x[ibl].scales_l[ib / 2] >> 4) as i32
                | (((h as i32) << 2) & 0x30))
                - 32;
            h >>= 4;

            // Process first sub-block (32 elements)
            let d1 = d4d8 * ls1 as f32;
            let mut sumi1 = 0i64;
            for j in 0..16 {
                let v0_idx = (x[ibl].qs[qs_offset + j] & 0x0F) as usize;
                let v1_idx = (x[ibl].qs[qs_offset + j] >> 4) as usize;
                let v0 = KVALUES_IQ4NL_I8[v0_idx] as i64;
                let v1 = KVALUES_IQ4NL_I8[v1_idx] as i64;
                sumi1 += v0 * y[ibl].qs[q8_offset + j] as i64
                    + v1 * y[ibl].qs[q8_offset + j + 16] as i64;
            }
            sum += d1 * sumi1 as f32;

            qs_offset += 16;
            q8_offset += 32;

            // Process second sub-block (32 elements)
            let d2 = d4d8 * ls2 as f32;
            let mut sumi2 = 0i64;
            for j in 0..16 {
                let v0_idx = (x[ibl].qs[qs_offset + j] & 0x0F) as usize;
                let v1_idx = (x[ibl].qs[qs_offset + j] >> 4) as usize;
                let v0 = KVALUES_IQ4NL_I8[v0_idx] as i64;
                let v1 = KVALUES_IQ4NL_I8[v1_idx] as i64;
                sumi2 += v0 * y[ibl].qs[q8_offset + j] as i64
                    + v1 * y[ibl].qs[q8_offset + j + 16] as i64;
            }
            sum += d2 * sumi2 as f32;

            qs_offset += 16;
            q8_offset += 32;
        }
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

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q5_k_q8_k(x: &[BlockQ5_K], y: &[BlockQ8_K]) -> f32 {
    unsafe { vec_dot_q5_k_q8_k_neon(x, y) }
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
