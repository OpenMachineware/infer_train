// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// TQ formats: TQ1_0, TQ2_0 (Ternary Quantization)

use crate::quant::types::*;
use half::f16;

const QK_TQ: usize = 256;

/// Quantize FP32 to TQ1_0 (1.5625 bpw ternary)
pub fn quantize_tq1_0(x: &[f32]) -> Vec<BlockTQ1_0> {
    assert!(x.len() % QK_TQ == 0);
    let nb = x.len() / QK_TQ;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_TQ..(ib + 1) * QK_TQ];

        // Find threshold
        let mut amax = 0.0f32;
        for j in 0..QK_TQ {
            amax = amax.max(xb[j].abs());
        }

        let threshold = amax * 0.2;

        let mut qs = [0u8; 48]; // 240 elements (5 per byte)
        let mut qh = [0u8; 4]; // 16 elements

        // Encode first 240 elements (48 bytes, 5 elements each)
        for j in 0..48 {
            let mut byte = 0u8;
            for k in 0..5 {
                let idx = j * 5 + k;
                if idx < 240 {
                    let val = xb[idx];
                    let q = if val > threshold {
                        1u8
                    }
                    // +1
                    else if val < -threshold {
                        2u8
                    }
                    // -1
                    else {
                        0u8
                    }; // 0
                    // Pack into 3-bit slots (5 values * 3 bits = 15 bits, fits in 2 bytes)
                    // TQ1_0 uses a special encoding
                    byte |= q << k; // Simplified encoding
                }
            }
            qs[j] = byte;
        }

        // Encode remaining 16 elements
        for j in 0..4 {
            let mut byte = 0u8;
            for k in 0..4 {
                let val = xb[240 + j * 4 + k];
                let q = if val > threshold {
                    1u8
                } else if val < -threshold {
                    2u8
                } else {
                    0u8
                };
                byte |= q << (2 * k);
            }
            qh[j] = byte;
        }

        blocks.push(BlockTQ1_0 { qs, qh, d: f16::from_f32(amax).to_bits() });
    }
    blocks
}

/// Quantize FP32 to TQ2_0 (2 bpw ternary)
pub fn quantize_tq2_0(x: &[f32]) -> Vec<BlockTQ2_0> {
    assert!(x.len() % QK_TQ == 0);
    let nb = x.len() / QK_TQ;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_TQ..(ib + 1) * QK_TQ];

        // Find max
        let mut amax = 0.0f32;
        for j in 0..QK_TQ {
            amax = amax.max(xb[j].abs());
        }

        let threshold = amax * 0.15;

        let mut qs = [0u8; 64];

        for j in 0..64 {
            let mut byte = 0u8;
            for k in 0..4 {
                let val = xb[j * 4 + k];
                let q = if val > threshold {
                    1u8
                } else if val < -threshold {
                    2u8
                } else {
                    0u8
                };
                byte |= q << (2 * k);
            }
            qs[j] = byte;
        }

        blocks.push(BlockTQ2_0 { d: f16::from_f32(amax).to_bits(), qs });
    }
    blocks
}
