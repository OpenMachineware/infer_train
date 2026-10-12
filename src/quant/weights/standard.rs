// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Standard quantization formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q1_0, Q2_0

use crate::quant::types::*;
use half::f16;

// QK = 32 for standard formats
const QK: usize = 32;

/// Quantize FP32 to Q4_0
pub fn quantize_q4_0(x: &[f32]) -> Vec<BlockQ4_0> {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let xb = &x[i * QK..(i + 1) * QK];

        // Find max absolute value
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..QK {
            let v = xb[j].abs();
            if v > amax {
                amax = v;
                max = xb[j];
            }
        }

        // Scale = max / -8 (symmetric quantization)
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        for j in 0..16 {
            // Quantize to [0, 15] range
            let x0 = xb[j] * id;
            let x1 = xb[j + 16] * id;
            let xi0 = ((x0 + 8.5).min(15.0).max(0.0)) as u8;
            let xi1 = ((x1 + 8.5).min(15.0).max(0.0)) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }

        blocks.push(BlockQ4_0 { d: f16::from_f32(d).to_bits(), qs });
    }
    blocks
}

/// Quantize FP32 to Q4_1 (with min)
pub fn quantize_q4_1(x: &[f32]) -> Vec<BlockQ4_1> {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let xb = &x[i * QK..(i + 1) * QK];

        // Find min and max
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for j in 0..QK {
            min = min.min(xb[j]);
            max = max.max(xb[j]);
        }

        // Scale and min
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = (xb[j] - min) * id;
            let x1 = (xb[j + 16] - min) * id;
            let xi0 = (x0 + 0.5).min(15.0).max(0.0) as u8;
            let xi1 = (x1 + 0.5).min(15.0).max(0.0) as u8;
            qs[j] = xi0 | (xi1 << 4);
        }

        blocks.push(BlockQ4_1 {
            d: f16::from_f32(d).to_bits(),
            m: f16::from_f32(min).to_bits(),
            qs,
        });
    }
    blocks
}

/// Quantize FP32 to Q5_0
pub fn quantize_q5_0(x: &[f32]) -> Vec<BlockQ5_0> {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let xb = &x[i * QK..(i + 1) * QK];

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..QK {
            let v = xb[j].abs();
            if v > amax {
                amax = v;
                max = xb[j];
            }
        }

        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];

        for j in 0..16 {
            let x0 = xb[j] * id;
            let x1 = xb[j + 16] * id;

            // 5-bit quantization: use high bit stored in qh
            let xi0 = ((x0 + 16.5).min(31.0).max(0.0)) as u8;
            let xi1 = ((x1 + 16.5).min(31.0).max(0.0)) as u8;

            // Store low 4 bits in qs, high bit in qh
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh[j / 8] |= ((xi0 >> 4) as u8) << (j % 8);
            qh[j / 8 + 2] |= ((xi1 >> 4) as u8) << (j % 8);
        }

        blocks.push(BlockQ5_0 { d: f16::from_f32(d).to_bits(), qh, qs });
    }
    blocks
}

/// Quantize FP32 to Q5_1
pub fn quantize_q5_1(x: &[f32]) -> Vec<BlockQ5_1> {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let xb = &x[i * QK..(i + 1) * QK];

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for j in 0..QK {
            min = min.min(xb[j]);
            max = max.max(xb[j]);
        }

        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];

        for j in 0..16 {
            let x0 = (xb[j] - min) * id;
            let x1 = (xb[j + 16] - min) * id;

            let xi0 = (x0 + 0.5).min(31.0).max(0.0) as u8;
            let xi1 = (x1 + 0.5).min(31.0).max(0.0) as u8;

            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh[j / 8] |= ((xi0 >> 4) as u8) << (j % 8);
            qh[j / 8 + 2] |= ((xi1 >> 4) as u8) << (j % 8);
        }

        blocks.push(BlockQ5_1 {
            d: f16::from_f32(d).to_bits(),
            m: f16::from_f32(min).to_bits(),
            qh,
            qs,
        });
    }
    blocks
}

/// Quantize FP32 to Q8_0
pub fn quantize_q8_0(x: &[f32]) -> Vec<BlockQ8_0> {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let xb = &x[i * QK..(i + 1) * QK];

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for j in 0..QK {
            let v = xb[j].abs();
            if v > amax {
                amax = v;
                max = xb[j];
            }
        }

        let d = max / -128.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0i8; 32];
        for j in 0..32 {
            qs[j] = (xb[j] * id).round().clamp(-128.0, 127.0) as i8;
        }

        blocks.push(BlockQ8_0 { d: f16::from_f32(d).to_bits(), qs });
    }
    blocks
}

/// Quantize FP32 to Q1_0 (1-bit)
pub fn quantize_q1_0(x: &[f32]) -> Vec<BlockQ1_0> {
    assert!(x.len() % QK == 0);
    let nb = x.len() / QK;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let xb = &x[i * QK..(i + 1) * QK];

        let mut amax = 0.0f32;
        for j in 0..QK {
            amax = amax.max(xb[j].abs());
        }

        let d = amax / -1.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 4];
        for j in 0..4 {
            let mut byte = 0u8;
            for k in 0..8 {
                let val = xb[j * 8 + k] * id;
                if val > 0.0 {
                    byte |= 1 << k;
                }
            }
            qs[j] = byte;
        }

        blocks.push(BlockQ1_0 { d: f16::from_f32(d).to_bits(), qs });
    }
    blocks
}

/// Quantize FP32 to Q2_0 (2-bit)
/// Q2_0 has 64 elements per block
pub fn quantize_q2_0(x: &[f32]) -> Vec<BlockQ2_0> {
    const QK_Q2_0: usize = 64;
    assert!(x.len() % QK_Q2_0 == 0);
    let nb = x.len() / QK_Q2_0;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let xb = &x[i * QK_Q2_0..(i + 1) * QK_Q2_0];

        let mut amax = 0.0f32;
        for j in 0..QK_Q2_0 {
            amax = amax.max(xb[j].abs());
        }

        let d = amax / -2.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        for j in 0..16 {
            let mut byte = 0u8;
            for k in 0..4 {
                let val = xb[j * 4 + k] * id;
                let q = ((val + 2.5).clamp(0.0, 3.0) as u8) & 0x03;
                byte |= q << (2 * k);
            }
            qs[j] = byte;
        }

        blocks.push(BlockQ2_0 { d: f16::from_f32(d).to_bits(), qs });
    }
    blocks
}
