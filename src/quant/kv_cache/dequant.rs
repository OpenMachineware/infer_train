// Dequantization kernels: Quantized formats -> FP32

use super::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// F16 Dequantization
// ============================================================================

/// Dequantize F16 to FP32 (scalar)
pub fn dequantize_row_f16(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    for i in 0..x.len() {
        y[i] = fp16_to_fp32(x[i]);
    }
}

/// Dequantize F16 to FP32 (NEON)
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_f16_neon(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    assert!(x.len() % 4 == 0);

    let n = x.len() / 4;
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    unsafe {
        for i in 0..n {
            // Load u16 and reinterpret as f16
            let x_u16 = vld1_u16(x_ptr.add(i * 4));
            let x_f16 = vreinterpret_f16_u16(x_u16);
            let y_vec = vcvt_f32_f16(x_f16);
            vst1q_f32(y_ptr.add(i * 4), y_vec);
        }
    }
}

// ============================================================================
// BF16 Dequantization
// ============================================================================

/// Dequantize BF16 to FP32 (scalar)
pub fn dequantize_row_bf16(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    for i in 0..x.len() {
        y[i] = bf16_to_fp32(x[i]);
    }
}

/// Dequantize BF16 to FP32 (NEON)
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_bf16_neon(x: &[u16], y: &mut [f32]) {
    assert!(x.len() == y.len());
    assert!(x.len() % 4 == 0);

    let n = x.len() / 4;
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    unsafe {
        for i in 0..n {
            let x_u16 = vld1_u16(x_ptr.add(i * 4));
            // Extend to 32-bit
            let x_u32 = vmovl_u16(x_u16);
            // Shift left by 16 to reconstruct FP32
            let x_u32_shifted = vshlq_n_u32(x_u32, 16);
            let y_vec = vreinterpretq_f32_u32(x_u32_shifted);
            vst1q_f32(y_ptr.add(i * 4), y_vec);
        }
    }
}

// ============================================================================
// Q8_0 Dequantization
// ============================================================================

/// Dequantize Q8_0 to FP32 (scalar)
pub fn dequantize_row_q8_0(x: &[BlockQ8_0], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    let mut yi = 0;
    for block in x {
        let d = fp16_to_fp32(block.d);
        for j in 0..QK {
            y[yi] = block.qs[j] as f32 * d;
            yi += 1;
        }
    }
}

/// Dequantize Q8_0 to FP32 (NEON)
#[cfg(target_arch = "aarch64")]
pub fn dequantize_row_q8_0_neon(x: &[BlockQ8_0], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    unsafe {
        let y_ptr = y.as_mut_ptr();

        for (i, block) in x.iter().enumerate() {
            let d = fp16_to_fp32(block.d);
            let d_vec = vdupq_n_f32(d);

            // Process 8 x 4 values
            for j in 0..8 {
                // Load 4 int8 values using scalar and convert to int32x4
                let idx = j * 4;
                let qs = [
                    block.qs[idx + 0] as i32,
                    block.qs[idx + 1] as i32,
                    block.qs[idx + 2] as i32,
                    block.qs[idx + 3] as i32,
                ];
                let qs_i32 = vld1q_s32(qs.as_ptr());

                // Convert to float and multiply by scale
                let y_vec = vmulq_f32(vcvtq_f32_s32(qs_i32), d_vec);
                vst1q_f32(y_ptr.add(i * QK + j * 4), y_vec);
            }
        }
    }
}

// ============================================================================
// Q4_0 Dequantization
// ============================================================================

/// Dequantize Q4_0 to FP32 (scalar)
pub fn dequantize_row_q4_0(x: &[BlockQ4_0], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for (i, block) in x.iter().enumerate() {
        let d = fp16_to_fp32(block.d);
        for j in 0..QK / 2 {
            let q = block.qs[j];
            // Unpack 2 4-bit values
            // llama.cpp layout: low nibbles go to positions 0-15, high nibbles to 16-31
            let q0 = (q & 0x0F) as i32 - 8;  // shift back to -8 to 7 range
            let q1 = ((q >> 4) & 0x0F) as i32 - 8;
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
pub fn dequantize_row_q4_1(x: &[BlockQ4_1], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for (i, block) in x.iter().enumerate() {
        let d = fp16_to_fp32(block.d);
        let m = fp16_to_fp32(block.m);
        for j in 0..QK / 2 {
            let q = block.qs[j];
            let q0 = (q & 0x0F) as f32;  // 0-15 range
            let q1 = ((q >> 4) & 0x0F) as f32;
            y[i * QK + j] = d * q0 + m;
            y[i * QK + j + QK / 2] = d * q1 + m;
        }
    }
}

// ============================================================================
// Q5_0 Dequantization
// ============================================================================

/// Dequantize Q5_0 to FP32 (scalar)
/// Q5_0: 5-bit values with extra bits in qh array
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
pub fn dequantize_row_q5_1(x: &[BlockQ5_1], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    for (i, block) in x.iter().enumerate() {
        let d = fp16_to_fp32(block.d);
        let m = fp16_to_fp32(block.m);

        let qh = block.qh[0] as u32
              | ((block.qh[1] as u32) << 8)
              | ((block.qh[2] as u32) << 16)
              | ((block.qh[3] as u32) << 24);

        for j in 0..QK / 2 {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;

            let q0 = ((block.qs[j] & 0x0F) as i32 | xh0 as i32);
            let q1 = (((block.qs[j] >> 4) & 0x0F) as i32 | xh1 as i32);

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
pub fn dequantize_row_iq4_nl(x: &[BlockIQ4NL], y: &mut [f32]) {
    let nb = x.len();
    assert!(y.len() >= nb * QK);

    let mut yi = 0;
    for block in x {
        let d = fp16_to_fp32(block.d);
        for j in 0..QK / 2 {
            let q = block.qs[j];
            // Unpack 2 4-bit indices
            let idx0 = (q & 0x0F) as usize;
            let idx1 = ((q >> 4) & 0x0F) as usize;
            // Lookup and scale
            y[yi] = KVALUES_IQ4NL[idx0] * d;
            y[yi + 1] = KVALUES_IQ4NL[idx1] * d;
            yi += 2;
        }
    }
}
