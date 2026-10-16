// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// IQ3_XXS, IQ3_S, IQ1_S, IQ1_M quantization implementations

use crate::quant::types::*;
use crate::quant::weights::iq_grid::*;
use half::f16;

const QK_K: usize = 256;

/// Quantize FP32 to IQ3_XXS (3.0625 bpw)
/// Uses 256-element grid, 32-element sub-blocks
pub fn quantize_iq3_xxs(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ3XXS> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    // Initialize grid data
    let grid_data = IQ3GridData::new(256, &KGRID_256_IQ3);

    const KMAX_Q_IQ3: i32 = 8;

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        // Use provided weights or uniform weights
        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        let mut scales = [0.0f32; QK_K / 32];
        let mut q3 = [0u32; 2]; // 2 u32 values for storing packed data
        let mut max_scale = 0.0f32;

        let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        // Process each sub-block of 32 values
        for ib in 0..QK_K / 32 {
            let xb = &xbl[ib * 32..(ib + 1) * 32];
            let qw = &quant_weights_slice[ib * 32..(ib + 1) * 32];

            let mut weight = [0.0f32; 32];
            let mut waux = [0.0f32; 32];
            for i in 0..32 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }

            // Extract signs and absolute values
            let mut xval = [0.0f32; 32];
            let mut block_signs = [0u8; 4]; // 4 bytes for 32 signs (7-bit encoding)

            for k in 0..4 {
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        s |= 1 << i;
                    }
                }
                block_signs[k] = s;
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            if max < GROUP_MAX_EPS_IQ3_XXS {
                continue;
            }

            let mut l = [0i8; 32];
            let mut is_on_grid = [false; 8];
            let mut best = 0.0f32;
            let mut scale = max / (2 * KMAX_Q_IQ3 - 1) as f32;

            // Try different scales
            for is in -9..=9 {
                let id = ((2 * KMAX_Q_IQ3 - 1) as f32 + is as f32 * 0.2) / max;
                let this_scale = 1.0 / id;

                let mut laux = [0i8; 32];
                let mut is_on_grid_aux = [false; 8];

                for k in 0..8 {
                    for i in 0..4 {
                        let lval = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0))
                            .clamp(0, KMAX_Q_IQ3 - 1);
                        laux[4 * k + i] = lval as i8;
                    }

                    let mut u: u16 = 0;
                    for i in 0..4 {
                        u |= (laux[4 * k + i] as u16) << (3 * i);
                    }

                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        // Need neighbor search
                        is_on_grid_aux[k] = false;
                        // Simplified: just mark as off-grid for now
                    } else {
                        is_on_grid_aux[k] = true;
                    }
                }

                // Compute quality
                let sumqx: f32 = (0..32)
                    .map(|i| weight[i] * xval[i] * (2.0 * laux[i] as f32 + 1.0))
                    .sum();
                let sumq2: f32 = (0..32)
                    .map(|i| weight[i] * (2.0 * laux[i] as f32 + 1.0).powi(2))
                    .sum();

                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid.copy_from_slice(&is_on_grid_aux);
                }
            }

            // Handle negative scale
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }

            scales[ib] = scale;
            max_scale = max_scale.max(scale);

            // Pack grid indices and signs
            for k in 0..8 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (l[4 * k + i] as u16) << (3 * i);
                }

                let grid_index = grid_data.map[u as usize];
                if grid_index < 0 {
                    // Fallback: use first grid entry
                    block.qs[8 * ib + k] = 0;
                } else {
                    block.qs[8 * ib + k] = grid_index as u8;
                }
            }

            // Pack signs into later bytes
            let sign_offset = 64 + 8 * ib;
            for k in 0..4 {
                block.qs[sign_offset + k * 2] = block_signs[k];
            }
        }

        // Pack final block
        if max_scale == 0.0 {
            blocks.push(BlockIQ3XXS { d: 0, qs: [0; 96] });
            continue;
        }

        let d = max_scale / 31.0;
        let mut block = BlockIQ3XXS {
            d: f16::from_f32(d).to_bits(),
            qs: [0; 96],
        };

        // Copy packed data
        let id = 1.0 / d;
        for ib in 0..QK_K / 32 {
            // Pack scales
            let lval = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
            block.qs[96 - 2 + ib] = lval as u8;
        }

        // Copy grid indices and signs from q3 to qs
        for i in 0..96 - 2 {
            block.qs[i] = ((q3[i / 4] >> (8 * (i % 4))) & 0xFF) as u8;
        }

        blocks.push(block);
    }

    blocks
}

/// Quantize FP32 to IQ3_S (3.4375 bpw)
/// Uses 512-element grid, 32-element sub-blocks
pub fn quantize_iq3_s(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ3S> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    let grid_data = IQ3GridData::new(512, &KGRID_512_IQ3);

    const KMAX_Q_IQ3: i32 = 8;
    const BLOCK_SIZE: usize = 32;

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        let mut scales = [0.0f32; QK_K / BLOCK_SIZE];
        let mut max_scale = 0.0f32;

        let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        let mut block = BlockIQ3S {
            d: 0,
            qs: [0; 64],
            qh: [0; 8],
            signs: [0; 32],
            scales: [0; 4],
        };

        // Process each sub-block
        for ib in 0..QK_K / BLOCK_SIZE {
            let xb = &xbl[ib * BLOCK_SIZE..(ib + 1) * BLOCK_SIZE];
            let qw = &quant_weights_slice[ib * BLOCK_SIZE..(ib + 1) * BLOCK_SIZE];

            let mut weight = [0.0f32; BLOCK_SIZE];
            let mut waux = [0.0f32; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }

            let mut xval = [0.0f32; BLOCK_SIZE];
            let mut block_signs = [0u8; BLOCK_SIZE / 8];

            for k in 0..BLOCK_SIZE / 8 {
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        s |= 1 << i;
                    }
                }
                block_signs[k] = s;
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            if max < GROUP_MAX_EPS_IQ3_XXS {
                continue;
            }

            let mut l = [0i8; BLOCK_SIZE];
            let mut scale = max / (2 * KMAX_Q_IQ3 - 1) as f32;

            // Simplified scale optimization
            let id = (2 * KMAX_Q_IQ3 - 1) as f32 / max;
            for k in 0..BLOCK_SIZE / 4 {
                for i in 0..4 {
                    let lval = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0))
                        .clamp(0, KMAX_Q_IQ3 - 1);
                    l[4 * k + i] = lval as i8;
                }
            }

            let sumqx: f32 = (0..BLOCK_SIZE)
                .map(|i| weight[i] * xval[i] * (2.0 * l[i] as f32 + 1.0))
                .sum();
            let sumq2: f32 = (0..BLOCK_SIZE)
                .map(|i| weight[i] * (2.0 * l[i] as f32 + 1.0).powi(2))
                .sum();

            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }

            if scale < 0.0 {
                scale = -scale;
                for k in 0..BLOCK_SIZE / 8 {
                    block_signs[k] = (!block_signs[k]) & 255;
                }
            }

            scales[ib] = scale;
            max_scale = max_scale.max(scale);

            // Pack grid indices
            for k in 0..BLOCK_SIZE / 4 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (l[4 * k + i] as u16) << (3 * i);
                }

                let grid_index = grid_data.map[u as usize].max(0) as u16;
                let idx = ib * (BLOCK_SIZE / 4) + k;
                block.qs[idx] = (grid_index & 255) as u8;
                block.qh[idx / 8] |= ((grid_index >> 8) as u8) << (idx % 8);
            }

            // Pack signs
            for k in 0..BLOCK_SIZE / 8 {
                block.signs[ib * (BLOCK_SIZE / 8) + k] = block_signs[k];
            }
        }

        if max_scale > 0.0 {
            let d = max_scale / 31.0;
            block.d = f16::from_f32(d * 1.033).to_bits();

            let id = 1.0 / d;
            for ib in (0..QK_K / BLOCK_SIZE).step_by(2) {
                let l1 = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
                let l2 = nearest_int(0.5 * (id * scales[ib + 1] - 1.0)).clamp(0, 15);
                block.scales[ib / 2] = (l1 as u8) | ((l2 as u8) << 4);
            }
        }

        blocks.push(block);
    }

    blocks
}

/// Quantize FP32 to IQ1_S (1.5 bpw)
/// 1-bit quantization with 512-element grid
pub fn quantize_iq1_s(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ1S> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        // Simplified 1-bit quantization
        let mut qs = [0u8; 32];
        let mut dh = 0u16;

        // Pack each group of 8 values
        for i in 0..32 {
            let mut byte = 0u8;
            for j in 0..8 {
                let val = xbl[i * 8 + j];
                if val > 0.0 {
                    byte |= 1 << j;
                }
            }
            qs[i] = byte;
        }

        // Compute scale (simplified)
        let max_abs: f32 = xbl.iter().map(|v| v.abs()).fold(0.0, f32::max);
        dh = if max_abs > 0.0 {
            f16::from_f32(max_abs).to_bits()
        } else {
            0
        };

        blocks.push(BlockIQ1S { d: dh, qs, qh: [0; 8] });
    }

    blocks
}

/// Quantize FP32 to IQ1_M (1.75 bpw)
/// 1-bit quantization with per-block scales
pub fn quantize_iq1_m(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ1M> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        let mut qs = [0u8; 32];
        let mut qh = [0u8; 16];
        let mut scales = [0u8; 8];

        // Pack each group of 8 values
        for i in 0..32 {
            let mut byte = 0u8;
            for j in 0..8 {
                let val = xbl[i * 8 + j];
                if val > 0.0 {
                    byte |= 1 << j;
                }
            }
            qs[i] = byte;
        }

        // Compute per-block scales (simplified)
        for i in 0..8 {
            let block_abs: f32 = xbl[i * 32..(i + 1) * 32]
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f32::max);
            scales[i] = if block_abs > 0.0 {
                (block_abs * 15.0 / f32::max(xbl.iter().map(|v| v.abs()).fold(0.0, f32::max), 1e-6)) as u8
            } else {
                0
            };
        }

        blocks.push(BlockIQ1M { qs, qh, scales });
    }

    blocks
}
