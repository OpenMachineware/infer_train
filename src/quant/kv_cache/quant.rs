// Quantization kernels: FP32 -> Quantized formats

use super::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// F16 Quantization
// ============================================================================

/// Quantize FP32 to F16 (scalar)
pub fn quantize_row_f16(x: &[f32], y: &mut [u16]) {
    assert!(x.len() == y.len());
    for i in 0..x.len() {
        y[i] = fp32_to_fp16(x[i]);
    }
}

/// Quantize FP32 to F16 (NEON)
/// Uses vcvtnq_f16_f32 which is available on ARMv8.2+
#[cfg(target_arch = "aarch64")]
pub fn quantize_row_f16_neon(x: &[f32], y: &mut [u16]) {
    assert!(x.len() == y.len());
    assert!(x.len() % 4 == 0);

    let n = x.len() / 4;
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    unsafe {
        for i in 0..n {
            let x_vec = vld1q_f32(x_ptr.add(i * 4));
            // Convert F32 to F16 using NEON
            // Note: vcvt_f16_f32 converts to f16 but we need to store as u16
            let y_vec = vcvt_f16_f32(x_vec);
            // Reinterpret as u16 and store
            let y_u16 = vreinterpret_u16_f16(y_vec);
            vst1_u16(y_ptr.add(i * 4), y_u16);
        }
    }
}

// ============================================================================
// BF16 Quantization
// ============================================================================

/// Quantize FP32 to BF16 (scalar)
/// BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
/// Simply truncate the lower 16 bits of FP32
pub fn quantize_row_bf16(x: &[f32], y: &mut [u16]) {
    assert!(x.len() == y.len());
    for i in 0..x.len() {
        y[i] = fp32_to_bf16(x[i]);
    }
}

/// Quantize FP32 to BF16 (NEON)
#[cfg(target_arch = "aarch64")]
pub fn quantize_row_bf16_neon(x: &[f32], y: &mut [u16]) {
    assert!(x.len() == y.len());
    assert!(x.len() % 4 == 0);

    let n = x.len() / 4;
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    unsafe {
        for i in 0..n {
            let x_vec = vld1q_f32(x_ptr.add(i * 4));
            // BF16: shift right by 16 bits to keep upper 16 bits
            let x_u32 = vreinterpretq_u32_f32(x_vec);
            let y_u32 = vshrq_n_u32(x_u32, 16);
            let y_u16 = vmovn_u32(y_u32);
            vst1_u16(y_ptr.add(i * 4), y_u16);
        }
    }
}

// ============================================================================
// Q8_0 Quantization
// ============================================================================

/// Quantize FP32 to Q8_0 (scalar)
/// Q8_0: block of 32 values, each scaled by a single FP16 scale
/// Formula: q = round(x / scale), where scale = max(abs(x)) / 127
pub fn quantize_row_q8_0(x: &[f32], y: &mut [BlockQ8_0]) {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;

    for i in 0..nb {
        let x_block = &x[i * QK..(i + 1) * QK];

        // Find max absolute value
        let mut amax: f32 = 0.0;
        for &val in x_block {
            let abs_val = val.abs();
            if abs_val > amax {
                amax = abs_val;
            }
        }

        // Calculate scale
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Quantize
        y[i].d = fp32_to_fp16(d);
        for j in 0..QK {
            y[i].qs[j] = (x_block[j] * id).round() as i8;
        }
    }
}

/// Quantize FP32 to Q8_0 (NEON)
#[cfg(target_arch = "aarch64")]
pub fn quantize_row_q8_0_neon(x: &[f32], y: &mut [BlockQ8_0]) {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;

    unsafe {
        for i in 0..nb {
            let x_ptr = x.as_ptr().add(i * QK);

            // Load 8 x 4 floats
            let mut srcv = [vdupq_n_f32(0.0); 8];
            let mut asrcv = [vdupq_n_f32(0.0); 8];
            for j in 0..8 {
                srcv[j] = vld1q_f32(x_ptr.add(j * 4));
                asrcv[j] = vabsq_f32(srcv[j]);
            }

            // Find max by reduction
            let mut amaxv = [vdupq_n_f32(0.0); 4];
            for j in 0..4 {
                amaxv[j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
            }
            amaxv[0] = vmaxq_f32(amaxv[0], amaxv[1]);
            amaxv[0] = vmaxq_f32(amaxv[2], amaxv[3]);
            amaxv[0] = vmaxq_f32(amaxv[0], amaxv[0]); // Final max

            let amax = vmaxvq_f32(amaxv[0]);

            // Calculate scale
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };

            y[i].d = fp32_to_fp16(d);

            // Quantize and store
            for j in 0..8 {
                let v = vmulq_n_f32(srcv[j], id);
                let vi = vcvtnq_s32_f32(v);

                y[i].qs[j * 4 + 0] = vgetq_lane_s32(vi, 0) as i8;
                y[i].qs[j * 4 + 1] = vgetq_lane_s32(vi, 1) as i8;
                y[i].qs[j * 4 + 2] = vgetq_lane_s32(vi, 2) as i8;
                y[i].qs[j * 4 + 3] = vgetq_lane_s32(vi, 3) as i8;
            }
        }
    }
}

// ============================================================================
// Q4_0 Quantization
// ============================================================================

/// Quantize FP32 to Q4_0 (scalar)
/// Q4_0: block of 32 values, each stored as 4-bit nibbles
/// llama.cpp formula: d = max_val / -8, where max_val has largest abs
pub fn quantize_row_q4_0(x: &[f32], y: &mut [BlockQ4_0]) {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;

    for i in 0..nb {
        let x_block = &x[i * QK..(i + 1) * QK];

        // Find value with largest absolute value
        let mut amax: f32 = 0.0;
        let mut max_val: f32 = 0.0;
        for &val in x_block {
            let abs_val = val.abs();
            if abs_val > amax {
                amax = abs_val;
                max_val = val;
            }
        }

        // llama.cpp formula: d = max_val / -8
        let d = max_val / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        y[i].d = fp32_to_fp16(d);

        // Pack 2 4-bit values into 1 byte
        // llama.cpp layout: low nibbles from x[0..15], high nibbles from x[16..31]
        for j in 0..QK / 2 {
            let q0 = ((x_block[j] * id + 8.5).min(15.0)) as u32;
            let q1 = ((x_block[j + QK / 2] * id + 8.5).min(15.0)) as u32;
            y[i].qs[j] = ((q0 & 0xF) | ((q1 & 0xF) << 4)) as u8;
        }
    }
}

// ============================================================================
// Q4_1 Quantization
// ============================================================================

/// Quantize FP32 to Q4_1 (scalar)
/// Q4_1: block of 32 values with scale and min offset
/// Formula: q = round((x - min) / scale), where scale = (max - min) / 15
pub fn quantize_row_q4_1(x: &[f32], y: &mut [BlockQ4_1]) {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;

    for i in 0..nb {
        let x_block = &x[i * QK..(i + 1) * QK];

        // Find min and max
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &val in x_block {
            if val < min { min = val; }
            if val > max { max = val; }
        }

        // Calculate scale and offset
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        y[i].d = fp32_to_fp16(d);
        y[i].m = fp32_to_fp16(min);

        // Pack 2 4-bit values into 1 byte
        // llama.cpp layout: low nibbles from x[0..15], high nibbles from x[16..31]
        for j in 0..QK / 2 {
            let q0 = ((x_block[j] - min) * id).round() as i32;
            let q1 = ((x_block[j + QK / 2] - min) * id).round() as i32;
            y[i].qs[j] = ((q0 & 0xF) | ((q1 & 0xF) << 4)) as u8;
        }
    }
}

// ============================================================================
// Q5_0 Quantization
// ============================================================================

/// Quantize FP32 to Q5_0 (scalar)
/// Q5_0: block of 32 values with 5-bit quantization
/// llama.cpp formula: d = max_val / -16, where max_val has largest abs
pub fn quantize_row_q5_0(x: &[f32], y: &mut [BlockQ5_0]) {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;

    for i in 0..nb {
        let x_block = &x[i * QK..(i + 1) * QK];

        // Find value with largest absolute value
        let mut amax: f32 = 0.0;
        let mut max_val: f32 = 0.0;
        for &val in x_block {
            let abs_val = val.abs();
            if abs_val > amax {
                amax = abs_val;
                max_val = val;
            }
        }

        // llama.cpp formula: d = max_val / -16
        let d = max_val / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        y[i].d = fp32_to_fp16(d);

        // Pack 5-bit values: low 4 bits in qs, high bit in qh
        // llama.cpp layout: low nibbles from x[0..15], high nibbles from x[16..31]
        let mut qh = 0u32;
        for j in 0..QK / 2 {
            let q0 = ((x_block[j] * id + 16.5).min(31.0)) as u32;
            let q1 = ((x_block[j + QK / 2] * id + 16.5).min(31.0)) as u32;

            // Low 4 bits in qs
            y[i].qs[j] = ((q0 & 0xF) | ((q1 & 0xF) << 4)) as u8;

            // High bits in qh
            if q0 & 0x10 != 0 { qh |= 1 << j; }
            if q1 & 0x10 != 0 { qh |= 1 << (j + 16); }
        }

        // Pack qh into 4 bytes
        y[i].qh[0] = (qh & 0xFF) as u8;
        y[i].qh[1] = ((qh >> 8) & 0xFF) as u8;
        y[i].qh[2] = ((qh >> 16) & 0xFF) as u8;
        y[i].qh[3] = ((qh >> 24) & 0xFF) as u8;
    }
}

// ============================================================================
// Q5_1 Quantization
// ============================================================================

/// Quantize FP32 to Q5_1 (scalar)
pub fn quantize_row_q5_1(x: &[f32], y: &mut [BlockQ5_1]) {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;

    for i in 0..nb {
        let x_block = &x[i * QK..(i + 1) * QK];

        // Find min and max
        let mut min = f32::MAX;
        let mut max = -f32::MAX;
        for &v in x_block {
            if v < min { min = v; }
            if v > max { max = v; }
        }

        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        y[i].d = fp32_to_fp16(d);
        y[i].m = fp32_to_fp16(min);

        let mut qh: u32 = 0;

        for j in 0..QK / 2 {
            let x0 = (x_block[j] - min) * id;
            let x1 = (x_block[j + QK / 2] - min) * id;

            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            qh |= ((xi0 & 0x10) as u32) >> 4 << j;
            qh |= ((xi1 & 0x10) as u32) >> 4 << (j + 16);
        }

        y[i].qh[0] = qh as u8;
        y[i].qh[1] = (qh >> 8) as u8;
        y[i].qh[2] = (qh >> 16) as u8;
        y[i].qh[3] = (qh >> 24) as u8;
    }
}

/// Quantize FP32 to Q5_1 (NEON)
#[cfg(target_arch = "aarch64")]
pub fn quantize_row_q5_1_neon(x: &[f32], y: &mut [BlockQ5_1]) {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;

    unsafe {
        let x_ptr = x.as_ptr();

        for i in 0..nb {
            // Find min and max using NEON
            let mut min_vec = vdupq_n_f32(f32::MAX);
            let mut max_vec = vdupq_n_f32(f32::MIN);

            for j in 0..8 {
                let xv = vld1q_f32(x_ptr.add(i * QK + j * 4));
                min_vec = vminq_f32(min_vec, xv);
                max_vec = vmaxq_f32(max_vec, xv);
            }

            let min = vminvq_f32(min_vec);
            let max = vmaxvq_f32(max_vec);

            let d = (max - min) / 31.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };

            y[i].d = fp32_to_fp16(d);
            y[i].m = fp32_to_fp16(min);

            let min_vec = vdupq_n_f32(min);
            let id_vec = vdupq_n_f32(id);

            let mut qh = 0u32;
            for j in 0..4 {
                let x0 = vld1q_f32(x_ptr.add(i * QK + j * 4));
                let x1 = vld1q_f32(x_ptr.add(i * QK + 16 + j * 4));

                let q0_v = vmulq_f32(vsubq_f32(x0, min_vec), id_vec);
                let q1_v = vmulq_f32(vsubq_f32(x1, min_vec), id_vec);

                let q0_i = vcvtnq_s32_f32(q0_v);
                let q1_i = vcvtnq_s32_f32(q1_v);

                // Unrolled extraction
                let q0_0 = vgetq_lane_s32(q0_i, 0);
                let q0_1 = vgetq_lane_s32(q0_i, 1);
                let q0_2 = vgetq_lane_s32(q0_i, 2);
                let q0_3 = vgetq_lane_s32(q0_i, 3);

                let q1_0 = vgetq_lane_s32(q1_i, 0);
                let q1_1 = vgetq_lane_s32(q1_i, 1);
                let q1_2 = vgetq_lane_s32(q1_i, 2);
                let q1_3 = vgetq_lane_s32(q1_i, 3);

                let byte_idx = j * 4;
                y[i].qs[byte_idx + 0] = ((q0_0 & 0xF) | ((q1_0 & 0xF) << 4)) as u8;
                y[i].qs[byte_idx + 1] = ((q0_1 & 0xF) | ((q1_1 & 0xF) << 4)) as u8;
                y[i].qs[byte_idx + 2] = ((q0_2 & 0xF) | ((q1_2 & 0xF) << 4)) as u8;
                y[i].qs[byte_idx + 3] = ((q0_3 & 0xF) | ((q1_3 & 0xF) << 4)) as u8;

                if q0_0 & 0x10 != 0 { qh |= 1 << (byte_idx + 0); }
                if q0_1 & 0x10 != 0 { qh |= 1 << (byte_idx + 1); }
                if q0_2 & 0x10 != 0 { qh |= 1 << (byte_idx + 2); }
                if q0_3 & 0x10 != 0 { qh |= 1 << (byte_idx + 3); }
                if q1_0 & 0x10 != 0 { qh |= 1 << (byte_idx + 0 + 16); }
                if q1_1 & 0x10 != 0 { qh |= 1 << (byte_idx + 1 + 16); }
                if q1_2 & 0x10 != 0 { qh |= 1 << (byte_idx + 2 + 16); }
                if q1_3 & 0x10 != 0 { qh |= 1 << (byte_idx + 3 + 16); }
            }

            y[i].qh[0] = (qh & 0xFF) as u8;
            y[i].qh[1] = ((qh >> 8) & 0xFF) as u8;
            y[i].qh[2] = ((qh >> 16) & 0xFF) as u8;
            y[i].qh[3] = ((qh >> 24) & 0xFF) as u8;
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Convert FP32 to FP16
#[inline]
pub fn fp32_to_fp16(f: f32) -> u16 {
    // Use half crate's conversion
    half::f16::from_f32(f).to_bits()
}

/// Convert FP16 to FP32
#[inline]
pub fn fp16_to_fp32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Convert FP32 to BF16 (truncate lower 16 bits)
#[inline]
pub fn fp32_to_bf16(f: f32) -> u16 {
    // BF16 is just the upper 16 bits of FP32
    (f.to_bits() >> 16) as u16
}

/// Convert BF16 to FP32
#[inline]
pub fn bf16_to_fp32(h: u16) -> f32 {
    // Extend BF16 to FP32 by setting lower 16 bits to 0
    f32::from_bits((h as u32) << 16)
}

// ============================================================================
// IQ4_NL Quantization
// ============================================================================

/// Find best index in lookup table (binary search)
fn best_index_iq4nl(x: f32) -> usize {
    let values = &super::KVALUES_IQ4NL;
    if x <= values[0] {
        return 0;
    }
    if x >= values[15] {
        return 15;
    }

    let mut lo = 0usize;
    let mut hi = 15usize;

    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x < values[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    if x - values[lo] < values[hi] - x {
        lo
    } else {
        hi
    }
}

/// Quantize FP32 to IQ4_NL (scalar)
/// IQ4_NL: 4-bit indices into non-linear lookup table
pub fn quantize_row_iq4_nl(x: &[f32], y: &mut [super::BlockIQ4NL]) {
    use super::KVALUES_IQ4NL;

    assert!(x.len() % super::QK == 0);
    let nb = x.len() / super::QK;

    for i in 0..nb {
        let x_block = &x[i * super::QK..(i + 1) * super::QK];

        // Find max absolute value
        let mut amax: f32 = 0.0;
        let mut max_val: f32 = 0.0;
        for &val in x_block {
            let abs_val = val.abs();
            if abs_val > amax {
                amax = abs_val;
                max_val = val;
            }
        }

        // Calculate initial scale
        let d = if amax < 1e-15 {
            0.0
        } else {
            max_val / KVALUES_IQ4NL[0]
        };

        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Find indices and optimize scale
        let mut sumqx: f32 = 0.0;
        let mut sumq2: f32 = 0.0;

        for &val in x_block {
            let al = id * val;
            let l = best_index_iq4nl(al);
            let q = KVALUES_IQ4NL[l];
            sumqx += q * val * val; // weight = x^2
            sumq2 += q * q * val * val;
        }

        let d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        y[i].d = fp32_to_fp16(d);

        // Find final indices and pack
        let mut indices = [0u8; super::QK];
        for (j, &val) in x_block.iter().enumerate() {
            let al = id * val;
            indices[j] = best_index_iq4nl(al) as u8;
        }

        // Pack: low nibbles from indices[0..15], high nibbles from indices[16..31]
        for j in 0..super::QK / 2 {
            y[i].qs[j] = (indices[j] & 0x0F) | ((indices[j + super::QK / 2] & 0x0F) << 4);
        }
    }
}
