// Dequantization kernels: Quantized formats -> FP32

use super::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// F16 Dequantization
// ============================================================================

/// Dequantize F16 to FP32 (scalar)
#[inline(never)]
pub fn dequantize_row_f16(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    for i in 0..x.len() {
        y[i] = fp16_to_fp32(x[i]);
    }
}

/// Dequantize F16 to FP32 (NEON)
#[inline(never)]
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_f16_neon(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    let n = x.len();

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        // Process 8 elements at a time
        let n8 = n / 8;
        for i in 0..n8 {
            let offset = i * 8;
            let x_u16 = vld1q_u16(x_ptr.add(offset));

            // Split into two halves and convert each
            let x_f16_low = vreinterpret_f16_u16(vget_low_u16(x_u16));
            let x_f16_high = vreinterpret_f16_u16(vget_high_u16(x_u16));

            let y_low = vcvt_f32_f16(x_f16_low);
            let y_high = vcvt_f32_f16(x_f16_high);

            vst1q_f32(y_ptr.add(offset), y_low);
            vst1q_f32(y_ptr.add(offset + 4), y_high);
        }

        // Handle remaining elements
        for i in (n8 * 8)..n {
            y[i] = fp16_to_fp32(x[i]);
        }
    }
}

// ============================================================================
// BF16 Dequantization
// ============================================================================

/// Dequantize BF16 to FP32 (scalar)
#[inline(never)]
pub fn dequantize_row_bf16(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    for i in 0..x.len() {
        y[i] = bf16_to_fp32(x[i]);
    }
}

/// Dequantize BF16 to FP32 (NEON)
/// Use NEON for all sizes - more consistent performance
#[inline(never)]
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_bf16_neon(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    let n = x.len();

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        // Process 8 elements at a time
        let n8 = n / 8;
        for i in 0..n8 {
            let offset = i * 8;
            // Load 8 u16 values
            let x_u16 = vld1q_u16(x_ptr.add(offset));
            // Extend to 32-bit
            let x_u32 = vmovl_u16(vget_low_u16(x_u16));
            let x_u32_high = vmovl_u16(vget_high_u16(x_u16));
            // Shift left by 16 to reconstruct FP32
            let y_low = vreinterpretq_f32_u32(vshlq_n_u32(x_u32, 16));
            let y_high = vreinterpretq_f32_u32(vshlq_n_u32(x_u32_high, 16));
            // Store
            vst1q_f32(y_ptr.add(offset), y_low);
            vst1q_f32(y_ptr.add(offset + 4), y_high);
        }

        // Handle remaining elements
        for i in (n8 * 8)..n {
            y[i] = bf16_to_fp32(x[i]);
        }
    }
}

// ============================================================================
// Q8_0 Dequantization
// ============================================================================

/// Dequantize Q8_0 to FP32 (scalar)
#[inline(never)]
pub fn dequantize_row_q8_0(x: &[BlockQ8_0], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK {
            y[i * QK + j] = x[i].qs[j] as f32 * d;
        }
    }
}

/// Dequantize Q8_0 to FP32 (NEON)
/// Use scalar for small sizes, SIMD for large sizes
#[inline(never)]
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_q8_0_neon(x: &[BlockQ8_0], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    // Use scalar for small data - matches llama.cpp exactly
    if nb < 128 {
        for i in 0..nb {
            let d = fp16_to_fp32(x[i].d);
            for j in 0..QK {
                y[i * QK + j] = x[i].qs[j] as f32 * d;
            }
        }
        return;
    }

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        for i in 0..nb {
            let block = &*x_ptr.add(i);
            let d = fp16_to_fp32(block.d);
            let d_vec = vdupq_n_f32(d);

            // Process all 32 values in 4 iterations of 8
            for j in 0..4 {
                let idx = j * 8;

                // Load 8 int8 values
                let qs = vld1_s8(block.qs.as_ptr().add(idx));

                // Widen int8 -> int16 -> int32
                let qs_16 = vmovl_s8(qs); // int16x8_t

                // Process low 4 and high 4 separately
                let qs_32_low = vmovl_s16(vget_low_s16(qs_16));
                let qs_32_high = vmovl_s16(vget_high_s16(qs_16));

                // Convert to float and multiply
                let y_low = vmulq_f32(vcvtq_f32_s32(qs_32_low), d_vec);
                let y_high = vmulq_f32(vcvtq_f32_s32(qs_32_high), d_vec);

                vst1q_f32(y_ptr.add(i * QK + idx), y_low);
                vst1q_f32(y_ptr.add(i * QK + idx + 4), y_high);
            }
        }
    }
}

// ============================================================================
// Q4_0 Dequantization
// ============================================================================

/// Dequantize Q4_0 to FP32 (scalar)
#[inline(never)]
pub fn dequantize_row_q4_0(x: &[BlockQ4_0], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK / 2 {
            let q = x[i].qs[j];
            let q0 = (q & 0x0F) as i32 - 8;
            let q1 = ((q >> 4) & 0x0F) as i32 - 8;
            y[i * QK + j] = q0 as f32 * d;
            y[i * QK + j + QK / 2] = q1 as f32 * d;
        }
    }
}

/// Dequantize Q4_0 to FP32 (NEON)
/// Simple scalar version - compiler auto-vectorizes
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_q4_0_neon(x: &[BlockQ4_0], y: &mut [f32]) {
    let nb = x.len();
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..(QK / 2) {
            let q0 = (x[i].qs[j] & 0x0F) as i32 - 8;
            let q1 = ((x[i].qs[j] >> 4) & 0x0F) as i32 - 8;
            y[i * QK + j] = q0 as f32 * d;
            y[i * QK + j + QK / 2] = q1 as f32 * d;
        }
    }
}

// ============================================================================
// Q4_1 Dequantization
// ============================================================================

/// Dequantize Q4_1 to FP32 (scalar)
/// Q4_1: x = d * q + m
#[inline(never)]
pub fn dequantize_row_q4_1(x: &[BlockQ4_1], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let m = fp16_to_fp32(x[i].m);
        for j in 0..QK / 2 {
            let q0 = (x[i].qs[j] & 0x0F) as i32;
            let q1 = ((x[i].qs[j] >> 4) & 0x0F) as i32;
            y[i * QK + j] = q0 as f32 * d + m;
            y[i * QK + j + QK / 2] = q1 as f32 * d + m;
        }
    }
}

// ============================================================================
// Q5_0 Dequantization
// ============================================================================

/// Dequantize Q5_0 to FP32 (scalar)
/// Q5_0: 5-bit values with extra bits in qh array
#[inline(never)]
pub fn dequantize_row_q5_0(x: &[BlockQ5_0], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for (i, block) in x.iter().enumerate() {
        let d = fp16_to_fp32(block.d);

        // Reconstruct qh as 32-bit value
        let qh = block.qh[0] as u32
              | ((block.qh[1] as u32) << 8)
              | ((block.qh[2] as u32) << 16)
              | ((block.qh[3] as u32) << 24);

        // llama.cpp layout: low nibbles go to positions 0-15, high nibbles to 16-31
        for j in 0..QK / 2 {
            // Extract 5th bits
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;

            // Combine low 4 bits with 5th bit
            let q0 = ((block.qs[j] & 0x0F) as i32 | xh0 as i32) - 16;
            let q1 = (((block.qs[j] >> 4) & 0x0F) as i32 | xh1 as i32) - 16;

            y[i * QK + j] = q0 as f32 * d;
            y[i * QK + j + QK / 2] = q1 as f32 * d;
        }
    }
}

// ============================================================================
// Q5_1 Dequantization
// ============================================================================

/// Dequantize Q5_1 to FP32 (scalar)
/// Q5_1: x = d * q + m, with 5-bit values
#[inline(never)]
pub fn dequantize_row_q5_1(x: &[BlockQ5_1], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let m = fp16_to_fp32(x[i].m);

        let qh = x[i].qh[0] as u32
              | ((x[i].qh[1] as u32) << 8)
              | ((x[i].qh[2] as u32) << 16)
              | ((x[i].qh[3] as u32) << 24);

        for j in 0..QK / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;

            let q0 = ((x[i].qs[j] & 0x0F) as i32 | xh0 as i32);
            let q1 = (((x[i].qs[j] >> 4) & 0x0F) as i32 | xh1 as i32);

            y[i * QK + j] = q0 as f32 * d + m;
            y[i * QK + j + QK / 2] = q1 as f32 * d + m;
        }
    }
}

// ============================================================================
// IQ4_NL Dequantization
// ============================================================================

/// Dequantize IQ4_NL to FP32 (scalar)
/// IQ4_NL: 4-bit indices into lookup table
#[inline(never)]
pub fn dequantize_row_iq4_nl(x: &[BlockIQ4NL], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK / 2 {
            let idx0 = (x[i].qs[j] & 0x0F) as usize;
            let idx1 = ((x[i].qs[j] >> 4) & 0x0F) as usize;
            y[i * QK + j] = KVALUES_IQ4NL[idx0] * d;
            y[i * QK + j + QK / 2] = KVALUES_IQ4NL[idx1] * d;
        }
    }
}

/// Dequantize IQ4_NL to FP32 (NEON)
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_iq4_nl_neon(x: &[BlockIQ4NL], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    unsafe {
        // Process 4 bytes at a time (8 4-bit indices)
        for (i, block) in x.iter().enumerate() {
            let d = fp16_to_fp32(block.d);

            for j in 0..4 {
                let byte_idx = j * 4;
                let qs = vld1_u8(block.qs.as_ptr().add(byte_idx));

                // Unpack low nibbles
                let idx0 = vand_u8(qs, vdup_n_u8(0x0F));
                // Unpack high nibbles
                let idx1 = vshr_n_u8(qs, 4);

                // Unrolled lookup and scale
                y[i * QK + byte_idx + 0] = KVALUES_IQ4NL[vget_lane_u8(idx0, 0) as usize] * d;
                y[i * QK + byte_idx + 1] = KVALUES_IQ4NL[vget_lane_u8(idx0, 1) as usize] * d;
                y[i * QK + byte_idx + 2] = KVALUES_IQ4NL[vget_lane_u8(idx0, 2) as usize] * d;
                y[i * QK + byte_idx + 3] = KVALUES_IQ4NL[vget_lane_u8(idx0, 3) as usize] * d;

                y[i * QK + byte_idx + QK / 2 + 0] = KVALUES_IQ4NL[vget_lane_u8(idx1, 0) as usize] * d;
                y[i * QK + byte_idx + QK / 2 + 1] = KVALUES_IQ4NL[vget_lane_u8(idx1, 1) as usize] * d;
                y[i * QK + byte_idx + QK / 2 + 2] = KVALUES_IQ4NL[vget_lane_u8(idx1, 2) as usize] * d;
                y[i * QK + byte_idx + QK / 2 + 3] = KVALUES_IQ4NL[vget_lane_u8(idx1, 3) as usize] * d;
            }
        }
    }
}
