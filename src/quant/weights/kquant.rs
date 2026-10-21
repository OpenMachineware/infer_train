// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// K-quant formats: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K

use crate::quant::types::*;
use half::f16;

// QK_K = 256 for K-quant formats
const QK_K: usize = 256;

/// Quantize FP32 to Q2_K
pub fn quantize_q2_k(x: &[f32]) -> Vec<BlockQ2K> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_K..(ib + 1) * QK_K];

        // Find max per 16-element sub-block
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];

        for j in 0..16 {
            let xsub = &xb[j * 16..(j + 1) * 16];
            let mut amax = 0.0f32;
            for k in 0..16 {
                amax = amax.max(xsub[k].abs());
            }

            // 4-bit scale
            let d = amax / 2.0;
            let scale = if d > 0.0 { (d * 64.0).min(255.0) as u8 } else { 0 };
            scales[j] = scale;

            let id = if scale > 0 { 1.0 / (scale as f32 / 64.0) } else { 0.0 };

            // Quantize to 2-bit
            for k in 0..4 {
                let mut byte = 0u8;
                for l in 0..4 {
                    let val = xsub[k * 4 + l] * id;
                    let q = ((val + 2.5).clamp(0.0, 3.0) as u8) & 0x03;
                    byte |= q << (2 * l);
                }
                qs[j * 4 + k] = byte;
            }
        }

        blocks.push(BlockQ2K {
            scales,
            qs,
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
        });
    }
    blocks
}

/// Quantize FP32 to Q3_K
pub fn quantize_q3_k(x: &[f32]) -> Vec<BlockQ3K> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_K..(ib + 1) * QK_K];

        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        let scales = [0u8; 12];

        // Simplified: use basic pattern
        for j in 0..QK_K {
            // High bit mask
            if xb[j] > 0.0 {
                hmask[j / 8] |= 1 << (j % 8);
            }
        }

        // 2-bit quantization
        let mut amax = 0.0f32;
        for j in 0..QK_K {
            amax = amax.max(xb[j].abs());
        }
        let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };

        for j in 0..64 {
            let mut byte = 0u8;
            for k in 0..4 {
                let val = (xb[j * 4 + k] * id + 2.0).clamp(0.0, 3.0) as u8;
                byte |= val << (2 * k);
            }
            qs[j] = byte;
        }

        blocks.push(BlockQ3K {
            d: f16::from_f32(1.0).to_bits(),
            hmask,
            qs,
            scales,
        });
    }
    blocks
}

/// Quantize FP32 to Q4_K
pub fn quantize_q4_k(x: &[f32]) -> Vec<BlockQ4K> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_K..(ib + 1) * QK_K];

        // Find block max
        let mut amax = 0.0f32;
        for j in 0..QK_K {
            amax = amax.max(xb[j].abs());
        }

        // Scale: map to [-8, 8] range, then shift to [0, 16]
        let d = if amax > 0.0 { amax / 8.0 } else { 1.0 };
        let id = 1.0 / d;

        let mut qs = [0u8; 128];
        for j in 0..128 {
            let x0 = xb[j] * id + 8.0; // Shift to [0, 16]
            let x1 = xb[j + 128] * id + 8.0;
            let q0 = (x0.clamp(0.0, 15.999) as u8) & 0x0F;
            let q1 = (x1.clamp(0.0, 15.999) as u8) & 0x0F;
            qs[j] = q0 | (q1 << 4);
        }

        // Set scales and mins for dequantization formula: d * v - dmin * min
        // We want: d * (v - 8) = d * v - d * 8
        // So: sc = 1, min = 8 for all sub-blocks
        // This gives: dl = d * 1 = d, ml = dmin * 8
        // Result: d * v - dmin * 8 = d * (v - 8) when dmin = d
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = 1; // scale = 1 for sub-blocks 0-3
            scales[j + 4] = 8; // min = 8 for sub-blocks 0-3
        }
        // scales[8..12] pack scales and mins for sub-blocks 4-7
        for j in 0..4 {
            // For sub-block j+4: scale = 1, min = 8
            // Packed format: lower 4 bits = scale, upper 4 bits = min
            scales[j + 8] = (8 << 4) | 1; // 0x81 = min=8 in upper nibble, scale=1 in lower
        }

        blocks.push(BlockQ4K {
            d: f16::from_f32(d).to_bits(),
            dmin: f16::from_f32(d).to_bits(), // dmin = d to compute d * 8
            scales,
            qs,
        });
    }
    blocks
}

/// Quantize FP32 to Q5_K (simplified)
pub fn quantize_q5_k(x: &[f32]) -> Vec<BlockQ5K> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_K..(ib + 1) * QK_K];

        // Find block max
        let mut amax = 0.0f32;
        for j in 0..QK_K {
            amax = amax.max(xb[j].abs());
        }
        let d = amax / 16.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];

        for j in 0..QK_K {
            let val = xb[j] * id;
            let q = ((val + 16.5).clamp(0.0, 31.0) as u8) & 0x1F;
            let j8 = j / 2;
            if j % 2 == 0 {
                qs[j8] = q & 0x0F;
                qh[j / 8] |= ((q >> 4) << (j % 8)) as u8;
            } else {
                qs[j8] |= (q & 0x0F) << 4;
                qh[j / 8] |= ((q >> 4) << (j % 8)) as u8;
            }
        }

        blocks.push(BlockQ5K {
            d: f16::from_f32(d).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qh,
            qs,
        });
    }
    blocks
}

/// Quantize FP32 to Q6_K
pub fn quantize_q6_k(x: &[f32]) -> Vec<BlockQ6K> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_K..(ib + 1) * QK_K];

        // Quantize to 6-bit values (0-63, representing -32 to 31)
        let mut l = [0i32; QK_K];
        let mut scales = [0i8; 16];

        let mut max_scale = 0.0f32;
        for j in 0..16 {
            let xsub = &xb[j * 16..(j + 1) * 16];
            let mut sub_amax = 0.0f32;
            for k in 0..16 {
                sub_amax = sub_amax.max(xsub[k].abs());
            }
            let scale = sub_amax / 31.0;
            scales[j] = (scale * 127.0).min(127.0) as i8;
            max_scale = max_scale.max(scale);
        }

        let d = max_scale;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        // Quantize each value
        for j in 0..QK_K {
            let val = xb[j] * id / (scales[j / 16] as f32 / 127.0).max(1e-10);
            let q = (val.clamp(-32.0, 31.0) + 32.0) as i32;
            l[j] = q;
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        // Process in chunks of 128 values
        // Each chunk: ql[0..63] and qh[0..31], then ql[64..127] and qh[32..63]
        for chunk in 0..2 {
            let ql_offset = chunk * 64;
            let qh_offset = chunk * 32;
            let val_offset = chunk * 128;

            for l_idx in 0..32 {
                let q1 = l[val_offset + l_idx + 0] as u8 & 0xF;
                let q2 = l[val_offset + l_idx + 32] as u8 & 0xF;
                let q3 = l[val_offset + l_idx + 64] as u8 & 0xF;
                let q4 = l[val_offset + l_idx + 96] as u8 & 0xF;

                ql[ql_offset + l_idx] = q1 | (q3 << 4);
                ql[ql_offset + 32 + l_idx] = q2 | (q4 << 4);

                qh[qh_offset + l_idx] = ((l[val_offset + l_idx] >> 4) & 0x3)
                    as u8
                    | (((l[val_offset + l_idx + 32] >> 4) & 0x3) << 2) as u8
                    | (((l[val_offset + l_idx + 64] >> 4) & 0x3) << 4) as u8
                    | (((l[val_offset + l_idx + 96] >> 4) & 0x3) << 6) as u8;
            }
        }

        blocks.push(BlockQ6K { d: f16::from_f32(d).to_bits(), ql, qh, scales });
    }
    blocks
}
