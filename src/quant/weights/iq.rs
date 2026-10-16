// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// IQ formats: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS

#![allow(dead_code, non_snake_case)]

use crate::quant::types::*;
use crate::quant::weights::iq_grid::*;
use half::f16;

const QK_K: usize = 256;
const GROUP_MAX_EPS: f32 = 1e-6;

fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

/// Finds optimal quantization for n values with range [0, nmax]
/// Returns scale such that q = scale * L[i]
fn make_qp_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [u8],
    quant_weights: &[f32],
) -> f32 {
    let max = x[0..n].iter().cloned().fold(0.0f32, f32::max);
    if max < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }

    let mut iscale = nmax as f32 / max;
    let scale = 1.0 / iscale;
    let mut best_mse: f32 = (0..n)
        .map(|i| {
            let li = nearest_int(iscale * x[i]).min(nmax);
            let diff = x[i] - scale * li as f32;
            quant_weights[i] * diff * diff
        })
        .sum();

    // Try different scales
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mse: f32 = (0..n)
            .map(|i| {
                let li = nearest_int(iscale_is * x[i]).min(nmax);
                let diff = x[i] - scale_is * li as f32;
                quant_weights[i] * diff * diff
            })
            .sum();

        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    // Compute L values and weighted least squares scale (lines 1113-1122)
    let mut sumlx: f32 = 0.0;
    let mut suml2: f32 = 0.0;
    for i in 0..n {
        let l_int = nearest_int(iscale * x[i]).min(nmax);
        // Store in u8 (may wrap if negative, but we read back as signed for calculations)
        l[i] = l_int as u8;
        // Use the signed value for calculations (matching C behavior)
        let l_signed = if l_int < 0 { l_int } else { l[i] as i32 };
        sumlx += quant_weights[i] * x[i] * l_signed as f32;
        suml2 += quant_weights[i] * l_signed as f32 * l_signed as f32;
    }

    // Iterative refinement (lines 1123-1135)
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            // Read L[i] as signed (matching C behavior where negative values are stored in uint8_t)
            let li = if l[i] > 127 { (l[i] as i32) - 256 } else { l[i] as i32 };
            if li > 0 && li < nmax {
                let w = quant_weights[i];
                let old_mse = w * (x[i] - sumlx / suml2 * li as f32).powi(2);
                let new_mse =
                    w * (x[i] - sumlx / suml2 * (li - 1) as f32).powi(2);
                if new_mse < old_mse {
                    l[i] = (li - 1) as u8;
                    sumlx -= w * x[i];
                    suml2 -= w
                        * (li as f32 * li as f32
                            - (li - 1) as f32 * (li - 1) as f32);
                    n_changed += 1;
                } else {
                    let new_mse =
                        w * (x[i] - sumlx / suml2 * (li + 1) as f32).powi(2);
                    if new_mse < old_mse {
                        l[i] = (li + 1) as u8;
                        sumlx += w * x[i];
                        suml2 += w
                            * ((li + 1) as f32 * (li + 1) as f32
                                - li as f32 * li as f32);
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }

    if suml2 > 0.0 { sumlx / suml2 } else { 1.0 / iscale }
}

/// Quantize FP32 to IQ4_NL (simple 4-bit non-linear)
pub fn quantize_iq4_nl(x: &[f32]) -> Vec<BlockIQ4NL> {
    assert!(x.len() % 32 == 0);
    let nb = x.len() / 32;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * 32..(ib + 1) * 32];

        // Find max
        let mut amax = 0.0f32;
        for j in 0..32 {
            amax = amax.max(xb[j].abs());
        }

        let d = amax / 8.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = xb[j] * id;
            let x1 = xb[j + 16] * id;
            let q0 = ((x0 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
            let q1 = ((x1 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
            qs[j] = q0 | (q1 << 4);
        }

        blocks.push(BlockIQ4NL { d: f16::from_f32(d).to_bits(), qs });
    }
    blocks
}

/// Quantize FP32 to IQ4_XS (4-bit with per-block scales)
pub fn quantize_iq4_xs(x: &[f32]) -> Vec<BlockIQ4XS> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    const KVALUES_IQ4NL: [i8; 16] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    for ib in 0..nb {
        let xb = &x[ib * QK_K..(ib + 1) * QK_K];

        let mut scales = [0.0f32; 8];
        let mut L = [0u8; QK_K];
        let mut max_scale = 0.0f32;

        for sib in 0..8 {
            let sx = &xb[sib * 32..(sib + 1) * 32];

            let mut amax = 0.0f32;
            let mut max_val = 0.0f32;
            for &v in sx {
                let ax = v.abs();
                if ax > amax {
                    amax = ax;
                    max_val = v;
                }
            }

            if amax < GROUP_MAX_EPS {
                scales[sib] = 0.0;
                continue;
            }

            let mut d = max_val / KVALUES_IQ4NL[0] as f32;
            let id = 1.0 / d;

            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            let mut L_block = [0u8; 32];
            for j in 0..32 {
                let al = id * sx[j];
                let mut best_idx = 0usize;
                let mut best_diff = (al - KVALUES_IQ4NL[0] as f32).abs();
                for k in 1..16 {
                    let diff = (al - KVALUES_IQ4NL[k] as f32).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = k;
                    }
                }
                L_block[j] = best_idx as u8;
                let q = KVALUES_IQ4NL[best_idx] as f32;
                let w = sx[j] * sx[j];
                sumqx += w * q * sx[j];
                sumq2 += w * q * q;
            }

            d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };

            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            for j in 0..32 {
                let al = id * sx[j];
                let mut best_idx = 0usize;
                let mut best_diff = (al - KVALUES_IQ4NL[0] as f32).abs();
                for k in 1..16 {
                    let diff = (al - KVALUES_IQ4NL[k] as f32).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = k;
                    }
                }
                L[sib * 32 + j] = best_idx as u8;
            }

            scales[sib] = d;
            let abs_d = d.abs();
            if abs_d > max_scale.abs() {
                max_scale = d;
            }
        }

        let d = -max_scale / 32.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut scales_h = 0u16;
        let mut scales_l = [0u8; 4];

        for sib in 0..8 {
            let mut l = (id * scales[sib]).round() as i32;
            l = l.clamp(-32, 31);

            let dl = d * l as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            for j in 0..32 {
                let al = idl * xb[sib * 32 + j];
                let mut best_idx = 0usize;
                let mut best_diff = (al - KVALUES_IQ4NL[0] as f32).abs();
                for k in 1..16 {
                    let diff = (al - KVALUES_IQ4NL[k] as f32).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = k;
                    }
                }
                L[sib * 32 + j] = best_idx as u8;
            }

            l += 32;
            let l_l = (l & 0xF) as u8;
            let l_h = (l >> 4) as u8;

            if sib % 2 == 0 {
                scales_l[sib / 2] = l_l;
            } else {
                scales_l[sib / 2] |= l_l << 4;
            }
            scales_h |= (l_h as u16) << (2 * (sib % 8));
        }

        let mut qs = [0u8; 128];
        for j in 0..64 {
            let q0 = L[j];
            let q1 = L[j + 64];
            qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        blocks.push(BlockIQ4XS {
            d: f16::from_f32(d).to_bits(),
            scales_h,
            scales_l,
            qs,
        });
    }
    blocks
}

// ============================================================================
// ============================================================================

pub fn quantize_iq2_xxs(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ2XXS> {
    quantize_iq2_xxs_impl(x, quant_weights, None)
}

/// Quantize FP32 to IQ2_XXS with debug dump
pub fn quantize_iq2_xxs_with_dump(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    dump_path: Option<&str>,
) -> Vec<BlockIQ2XXS> {
    quantize_iq2_xxs_impl(x, quant_weights, dump_path)
}

fn quantize_iq2_xxs_impl(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    dump_path: Option<&str>,
) -> Vec<BlockIQ2XXS> {
    const KMAX_Q: i32 = 3;
    const KMAX_Q_IQ2_XS: i32 = 3;

    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    // Open dump file if specified
    let mut dump_file = None;
    if let Some(path) = dump_path {
        use std::fs::File;
        use std::io::Write;
        dump_file =
            Some(File::create(path).expect("Failed to create dump file"));
        if let Some(ref mut f) = dump_file {
            writeln!(f, "=== RUST IQ2_XXS DUMP ===").ok();
        }
    }

    // Initialize grid data
    let grid_data = IQ2GridData::new(256, &KGRID_2BIT_256, 2);

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        // Use provided weights or uniform weights
        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        let mut scales = [0.0f32; QK_K / 32];
        let mut q2 = [0u32; 2 * (QK_K / 32)];
        let mut max_scale = 0.0f32;

        let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
        let sigma2 = sumx2 / QK_K as f32;

        // Dump sigma2
        if ibl == 0 {
            if let Some(ref mut f) = dump_file {
                use std::io::Write;
                writeln!(f, "\n=== Block {} ===", ibl).ok();
                writeln!(f, "sumx2 = {:.6}, sigma2 = {:.6}", sumx2, sigma2)
                    .ok();
            }
        }

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

            // Dump weight and waux
            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(f, "\n=== Sub-block {} ===", ib).ok();
                    write!(f, "weight[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", weight[i]).ok();
                    }
                    writeln!(f).ok();
                    write!(f, "waux[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", waux[i]).ok();
                    }
                    writeln!(f).ok();
                }
            }

            let mut xval = [0.0f32; 32];
            let mut block_signs = [0u8; 4];

            for k in 0..4 {
                let mut nflip = 0;
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                // Ensure odd number of signs per group (parity constraint)
                if nflip % 2 != 0 {
                    let mut min_idx = 0;
                    let mut min_val = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax =
                            weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min_val {
                            min_val = ax;
                            min_idx = i;
                        }
                    }
                    xval[8 * k + min_idx] = -xval[8 * k + min_idx];
                    s ^= 1 << min_idx;
                }
                block_signs[k] = s & 127;
            }

            // Dump sign extraction results
            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(
                        f,
                        "\n=== LLAMA.CPP DUMP ibl={} ib={} ===",
                        ibl, ib
                    )
                    .ok();
                    write!(f, "xval[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", xval[i]).ok();
                    }
                    writeln!(f).ok();
                    writeln!(
                        f,
                        "block_signs: {} {} {} {}",
                        block_signs[0],
                        block_signs[1],
                        block_signs[2],
                        block_signs[3]
                    )
                    .ok();
                }
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue;
            }

            let mut l = [0u8; 32];
            let scale = make_qp_quants(32, KMAX_Q + 1, &xval, &mut l, &weight);
            let eff_max = scale * KMAX_Q as f32;

            // Dump make_qp_quants results
            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(
                        f,
                        "After make_qp_quants: scale={:.6}, eff_max={:.6}",
                        scale, eff_max
                    )
                    .ok();
                    write!(f, "L[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {}", l[i] as i8).ok();
                    }
                    writeln!(f).ok();
                }
            }

            if eff_max <= 0.0 {
                scales[ib] = 0.0;
                continue;
            }

            let mut best = 0.0f32;
            let mut l_final = [0i8; 32];

            for is in -6..=6 {
                let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / eff_max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 32];

                for k in 0..4 {
                    for i in 0..8 {
                        let lval =
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = lval.clamp(0, KMAX_Q - 1) as i8;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as u16) << (2 * i);
                    }
                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        let neighbors_idx =
                            (-grid_data.map[u as usize] - 1) as usize;
                        let neighbors = &grid_data.neighbors[neighbors_idx..];
                        find_best_neighbor_iq2(
                            neighbors,
                            &grid_data.grid,
                            &xval[8 * k..8 * (k + 1)],
                            &waux[8 * k..8 * (k + 1)],
                            this_scale,
                            &mut laux[8 * k..8 * (k + 1)],
                        );
                    }
                }

                let sumqx: f32 = (0..32)
                    .map(|i| weight[i] * xval[i] * (2.0 * laux[i] as f32 + 1.0))
                    .sum();
                let sumq2: f32 = (0..32)
                    .map(|i| weight[i] * (2.0 * laux[i] as f32 + 1.0).powi(2))
                    .sum();

                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    let new_scale = sumqx / sumq2;
                    best = new_scale * sumqx;
                    l_final.copy_from_slice(&laux);
                }
            }

            let mut scale = 0.0f32;
            if best > 0.0 {
                // First compute scale from the optimization loop result
                let sumqx: f32 = (0..32)
                    .map(|i| {
                        weight[i] * xval[i] * (2.0 * l_final[i] as f32 + 1.0)
                    })
                    .sum();
                let sumq2: f32 = (0..32)
                    .map(|i| {
                        weight[i] * (2.0 * l_final[i] as f32 + 1.0).powi(2)
                    })
                    .sum();
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }

                if scale > 0.0 {
                    let id = 1.0 / scale;
                    for k in 0..4 {
                        let mut u: u16 = 0;
                        for i in 0..8 {
                            let l =
                                nearest_int(0.5 * (id * xval[8 * k + i] - 1.0))
                                    .clamp(0, KMAX_Q - 1)
                                    as i8;
                            l_final[8 * k + i] = l as i8;
                            u |= (l as u16) << (2 * i);
                        }
                        let grid_index = grid_data.map[u as usize];
                        if grid_index < 0 {
                            let neighbors_idx =
                                (-grid_data.map[u as usize] - 1) as usize;
                            let neighbors =
                                &grid_data.neighbors[neighbors_idx..];
                            find_best_neighbor_iq2(
                                neighbors,
                                &grid_data.grid,
                                &xval[8 * k..8 * (k + 1)],
                                &waux[8 * k..8 * (k + 1)],
                                scale,
                                &mut l_final[8 * k..8 * (k + 1)],
                            );
                        }
                    }

                    let sumqx: f32 = (0..32)
                        .map(|i| {
                            weight[i]
                                * xval[i]
                                * (2.0 * l_final[i] as f32 + 1.0)
                        })
                        .sum();
                    let sumq2: f32 = (0..32)
                        .map(|i| {
                            weight[i] * (2.0 * l_final[i] as f32 + 1.0).powi(2)
                        })
                        .sum();
                    if sumq2 > 0.0 {
                        scale = sumqx / sumq2;
                    }
                }
            }

            // Dump grid refinement results
            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(f, "After grid refinement: scale={:.6}", scale)
                        .ok();
                    write!(f, "L[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {}", l_final[i]).ok();
                    }
                    writeln!(f).ok();
                }
            }

            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }

            for k in 0..4 {
                let mut u: u16 = 0;
                for i in 0..8 {
                    u |= (l_final[8 * k + i] as u16) << (2 * i);
                }
                let grid_index = grid_data.map[u as usize];
                if grid_index < 0 {
                    panic!("Oops: found point {} not on grid", u);
                }
                q2[2 * ib] |= (grid_index as u32) << (8 * k);
                q2[2 * ib + 1] |= (block_signs[k] as u32) << (7 * k);
            }

            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            let qs = [0u16; 32];
            blocks.push(BlockIQ2XXS { d: f16::from_f32(0.0).to_bits(), qs });
            continue;
        }

        let d = max_scale / 31.0;
        let id = 1.0 / d;
        for ib in 0..QK_K / 32 {
            let lval =
                nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u32;
            q2[2 * ib + 1] |= lval << 28;
        }

        // Convert u32 array to u16 array
        let mut qs = [0u16; 32];
        for i in 0..16 {
            qs[2 * i] = (q2[i] & 0xFFFF) as u16;
            qs[2 * i + 1] = ((q2[i] >> 16) & 0xFFFF) as u16;
        }

        blocks.push(BlockIQ2XXS { d: f16::from_f32(d).to_bits(), qs });
    }
    blocks
}

fn find_best_neighbor_iq2(
    neighbors: &[u16],
    grid: &[u64],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) {
    let num_neighbors = neighbors[0] as usize;
    let mut best_d2 = f32::MAX;
    let mut best_idx = 0usize;

    for &nj in &neighbors[1..=num_neighbors] {
        let pg = grid[nj as usize].to_le_bytes();
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = pg[i] as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best_idx = nj as usize;
        }
    }

    let pg = grid[best_idx].to_le_bytes();
    for i in 0..8 {
        l[i] = ((pg[i] as i16 - 1) / 2) as i8;
    }
}

/// Quantize FP32 to IQ2_S
pub fn quantize_iq2_s(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ2S> {
    quantize_iq2_s_impl(x, quant_weights, None)
}

/// Quantize FP32 to IQ2_S with debug dump
pub fn quantize_iq2_s_with_dump(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    dump_path: Option<&str>,
) -> Vec<BlockIQ2S> {
    quantize_iq2_s_impl(x, quant_weights, dump_path)
}

fn quantize_iq2_s_impl(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    dump_path: Option<&str>,
) -> Vec<BlockIQ2S> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    let mut dump_file = None;
    if let Some(path) = dump_path {
        use std::fs::File;
        use std::io::Write;
        dump_file =
            Some(File::create(path).expect("Failed to create dump file"));
        if let Some(ref mut f) = dump_file {
            writeln!(f, "=== RUST IQ2_S DUMP ===").ok();
        }
    }

    let grid_data = IQ2GridData::new(512, &KGRID_2BIT_512, 2);

    const KMAX_Q_IQ2_S: i32 = 3;

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        let mut scales = [0.0f32; QK_K / 16];
        let mut max_scale = 0.0f32;

        let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        if ibl == 0 {
            if let Some(ref mut f) = dump_file {
                use std::io::Write;
                writeln!(f, "\n=== Block {} ===", ibl).ok();
                writeln!(f, "sumx2 = {:.6}, sigma2 = {:.6}", sumx2, sigma2)
                    .ok();
            }
        }

        // Process each sub-block of 16 values
        for ib in 0..QK_K / 16 {
            let xb = &xbl[ib * 16..(ib + 1) * 16];
            let qw = &quant_weights_slice[ib * 16..(ib + 1) * 16];

            let mut weight = [0.0f32; 16];
            let mut waux = [0.0f32; 16];
            for i in 0..16 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }

            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(f, "\n=== Sub-block {} ===", ib).ok();
                    write!(f, "weight[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", weight[i]).ok();
                    }
                    writeln!(f).ok();
                    write!(f, "waux[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", waux[i]).ok();
                    }
                    writeln!(f).ok();
                }
            }

            // Extract signs and absolute values (no parity constraint for IQ2_S)
            let mut xval = [0.0f32; 16];
            let mut block_signs = [0u8; 2];

            for k in 0..2 {
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

            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(
                        f,
                        "\n=== LLAMA.CPP DUMP ibl={} ib={} ===",
                        ibl, ib
                    )
                    .ok();
                    write!(f, "xval[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", xval[i]).ok();
                    }
                    writeln!(f).ok();
                    writeln!(
                        f,
                        "block_signs: {} {}",
                        block_signs[0], block_signs[1]
                    )
                    .ok();
                }
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            let mut l = [0i8; 16];
            if max < GROUP_MAX_EPS_IQ2_S {
                scales[ib] = 0.0;
                continue;
            }

            // Scale optimization loop
            let mut best = 0.0f32;
            let mut scale = max / (2 * KMAX_Q_IQ2_S - 1) as f32;
            let mut is_on_grid = [true; 2];

            for is in -9..=9 {
                let id =
                    ((2 * KMAX_Q_IQ2_S - 1) as f32 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 16];
                let mut is_on_grid_aux = [true; 2];

                for k in 0..2 {
                    for i in 0..8 {
                        let lval =
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] = lval.clamp(0, KMAX_Q_IQ2_S - 1) as i8;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as u16) << (2 * i);
                    }
                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let neighbors_idx =
                            (-grid_data.map[u as usize] - 1) as usize;
                        let neighbors = &grid_data.neighbors[neighbors_idx..];
                        find_best_neighbor_iq2(
                            neighbors,
                            &grid_data.grid,
                            &xval[8 * k..8 * (k + 1)],
                            &waux[8 * k..8 * (k + 1)],
                            this_scale,
                            &mut laux[8 * k..8 * (k + 1)],
                        );
                    }
                }

                let sumqx: f32 = (0..16)
                    .map(|i| weight[i] * xval[i] * (2.0 * laux[i] as f32 + 1.0))
                    .sum();
                let sumq2: f32 = (0..16)
                    .map(|i| weight[i] * (2.0 * laux[i] as f32 + 1.0).powi(2))
                    .sum();

                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid.copy_from_slice(&is_on_grid_aux);
                }
            }

            // Off-grid refinement
            let mut n_not_ongrid = 0;
            for k in 0..2 {
                if !is_on_grid[k] {
                    n_not_ongrid += 1;
                }
            }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let lval =
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0))
                                .clamp(0, KMAX_Q_IQ2_S - 1);
                        u |= (lval as u16) << (2 * i);
                        l[8 * k + i] = lval as i8;
                    }
                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        let neighbors_idx =
                            (-grid_data.map[u as usize] - 1) as usize;
                        let neighbors = &grid_data.neighbors[neighbors_idx..];
                        find_best_neighbor_iq2(
                            neighbors,
                            &grid_data.grid,
                            &xval[8 * k..8 * (k + 1)],
                            &waux[8 * k..8 * (k + 1)],
                            scale,
                            &mut l[8 * k..8 * (k + 1)],
                        );
                    }
                }
                let sumqx: f32 = (0..16)
                    .map(|i| weight[i] * xval[i] * (2.0 * l[i] as f32 + 1.0))
                    .sum();
                let sumq2: f32 = (0..16)
                    .map(|i| weight[i] * (2.0 * l[i] as f32 + 1.0).powi(2))
                    .sum();
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }

            // Handle negative scale
            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 {
                    block_signs[k] = (!block_signs[k]) & 255;
                }
            }

            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        // Pack final block
        if max_scale == 0.0 {
            blocks.push(BlockIQ2S {
                d: 0,
                qs: [0; 64],
                qh: [0; 8],
                scales: [0; 8],
            });
            continue;
        }

        let d = max_scale / 31.0;
        let id = 1.0 / d;

        let mut block = BlockIQ2S {
            d: f16::from_f32(d * 0.9875).to_bits(),
            qs: [0; 64],
            qh: [0; 8],
            scales: {
                let mut arr = [0u8; 8];
                for ib in 0..QK_K / 16 {
                    let lval =
                        nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
                    if ib % 2 == 0 {
                        arr[ib / 2] = lval as u8;
                    } else {
                        arr[ib / 2] |= (lval as u8) << 4;
                    }
                }
                arr
            },
        };

        // Now pack grid indices and signs - need to recompute
        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        for ib in 0..QK_K / 16 {
            let xb = &xbl[ib * 16..(ib + 1) * 16];
            let qw = &quant_weights_slice[ib * 16..(ib + 1) * 16];

            let mut weight = [0.0f32; 16];
            let mut waux = [0.0f32; 16];
            for i in 0..16 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }

            let mut xval = [0.0f32; 16];
            let mut block_signs = [0u8; 2];

            for k in 0..2 {
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
            if max < GROUP_MAX_EPS_IQ2_S {
                continue;
            }

            let mut l = [0i8; 16];
            let scale = scales[ib];
            if scale > 0.0 {
                let id_local = 1.0 / scale;
                for k in 0..2 {
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let lval = nearest_int(
                            0.5 * (id_local * xval[8 * k + i] - 1.0),
                        )
                        .clamp(0, KMAX_Q_IQ2_S - 1);
                        u |= (lval as u16) << (2 * i);
                        l[8 * k + i] = lval as i8;
                    }
                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        let neighbors_idx =
                            (-grid_data.map[u as usize] - 1) as usize;
                        let neighbors = &grid_data.neighbors[neighbors_idx..];
                        find_best_neighbor_iq2(
                            neighbors,
                            &grid_data.grid,
                            &xval[8 * k..8 * (k + 1)],
                            &waux[8 * k..8 * (k + 1)],
                            scale,
                            &mut l[8 * k..8 * (k + 1)],
                        );
                    }
                }

                // Handle negative scale
                let mut final_signs = block_signs;
                if scale < 0.0 {
                    for k in 0..2 {
                        final_signs[k] = (!final_signs[k]) & 255;
                    }
                }

                // Pack grid indices and signs
                for k in 0..2 {
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        u |= (l[8 * k + i] as u16) << (2 * i);
                    }
                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        panic!(
                            "Oops: found point {} not on grid: {:?}",
                            u,
                            &l[8 * k..8 * (k + 1)]
                        );
                    }
                    let i8_idx = 2 * ib + k;
                    block.qs[i8_idx] = (grid_index as u16 & 255) as u8;
                    block.qh[i8_idx / 4] |=
                        ((grid_index as u16 >> 8) as u8) << (2 * (i8_idx % 4));
                    block.qs[QK_K / 8 + i8_idx] = final_signs[k];
                }
            }
        }

        blocks.push(block);
    }

    blocks
}

/// Quantize FP32 to IQ2_XS
pub fn quantize_iq2_xs(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ2XS> {
    quantize_iq2_xs_impl(x, quant_weights, None)
}

/// Quantize FP32 to IQ2_XS with debug dump
pub fn quantize_iq2_xs_with_dump(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    dump_path: Option<&str>,
) -> Vec<BlockIQ2XS> {
    quantize_iq2_xs_impl(x, quant_weights, dump_path)
}

fn quantize_iq2_xs_impl(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    dump_path: Option<&str>,
) -> Vec<BlockIQ2XS> {
    const KMAX_Q_IQ2_XS: i32 = 3;

    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    // Open dump file if specified
    let mut dump_file = None;
    if let Some(path) = dump_path {
        use std::fs::File;
        use std::io::Write;
        dump_file =
            Some(File::create(path).expect("Failed to create dump file"));
        if let Some(ref mut f) = dump_file {
            writeln!(f, "=== RUST IQ2_XS DUMP ===").ok();
        }
    }

    // Initialize grid data
    let grid_data = IQ2GridData::new(512, &KGRID_2BIT_512, 2);

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        // Use provided weights or uniform weights
        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        let mut scales = [0.0f32; QK_K / 16]; // 16-element sub-blocks
        let mut q2 = [0u16; 2 * (QK_K / 16)]; // 2 values per 16-element sub-block
        let mut max_scale = 0.0f32;

        let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
        let sigma2 = sumx2 / QK_K as f32;

        // Dump sigma2
        if ibl == 0 {
            if let Some(ref mut f) = dump_file {
                use std::io::Write;
                writeln!(f, "\n=== Block {} ===", ibl).ok();
                writeln!(f, "sumx2 = {:.6}, sigma2 = {:.6}", sumx2, sigma2)
                    .ok();
            }
        }

        for ib in 0..QK_K / 16 {
            let xb = &xbl[ib * 16..(ib + 1) * 16];
            let qw = &quant_weights_slice[ib * 16..(ib + 1) * 16];

            let mut weight = [0.0f32; 16];
            let mut waux = [0.0f32; 16];
            for i in 0..16 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }

            // Dump weight and waux
            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(f, "\n=== Sub-block {} ===", ib).ok();
                    write!(f, "weight[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", weight[i]).ok();
                    }
                    writeln!(f).ok();
                    write!(f, "waux[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", waux[i]).ok();
                    }
                    writeln!(f).ok();
                }
            }

            let mut xval = [0.0f32; 16];
            let mut block_signs = [0u8; 2];

            for k in 0..2 {
                let mut nflip = 0;
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                // Ensure odd number of signs per group (parity constraint)
                if nflip % 2 != 0 {
                    let mut min_idx = 0;
                    let mut min_val = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax =
                            weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min_val {
                            min_val = ax;
                            min_idx = i;
                        }
                    }
                    xval[8 * k + min_idx] = -xval[8 * k + min_idx];
                    s ^= 1 << min_idx;
                }
                block_signs[k] = s & 127;
            }

            // Dump sign extraction results
            if ibl == 0 && ib == 0 {
                if let Some(ref mut f) = dump_file {
                    use std::io::Write;
                    writeln!(
                        f,
                        "\n=== LLAMA.CPP DUMP ibl={} ib={} ===",
                        ibl, ib
                    )
                    .ok();
                    write!(f, "xval[0:8]:").ok();
                    for i in 0..8 {
                        write!(f, " {:.6}", xval[i]).ok();
                    }
                    writeln!(f).ok();
                    writeln!(
                        f,
                        "block_signs: {} {}",
                        block_signs[0], block_signs[1]
                    )
                    .ok();
                }
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            let mut l = [0i8; 16];
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue;
            }

            let mut best = 0.0f32;
            let mut scale = max / (2 * KMAX_Q_IQ2_XS - 1) as f32;
            let mut is_on_grid = [true; 2];

            for is in -9..=9 {
                let id =
                    ((2 * KMAX_Q_IQ2_XS - 1) as f32 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 16];
                let mut is_on_grid_aux = [true; 2];

                for k in 0..2 {
                    for i in 0..8 {
                        let lval =
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                        laux[8 * k + i] =
                            lval.clamp(0, KMAX_Q_IQ2_XS - 1) as i8;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as u16) << (2 * i);
                    }
                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        is_on_grid_aux[k] = false;
                        let neighbors_idx =
                            (-grid_data.map[u as usize] - 1) as usize;
                        let neighbors = &grid_data.neighbors[neighbors_idx..];
                        find_best_neighbor_iq2(
                            neighbors,
                            &grid_data.grid,
                            &xval[8 * k..8 * (k + 1)],
                            &waux[8 * k..8 * (k + 1)],
                            this_scale,
                            &mut laux[8 * k..8 * (k + 1)],
                        );
                    }
                }

                let sumqx: f32 = (0..16)
                    .map(|i| weight[i] * xval[i] * (2.0 * laux[i] as f32 + 1.0))
                    .sum();
                let sumq2: f32 = (0..16)
                    .map(|i| weight[i] * (2.0 * laux[i] as f32 + 1.0).powi(2))
                    .sum();

                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid.copy_from_slice(&is_on_grid_aux);
                }
            }

            let mut n_not_ongrid = 0;
            for k in 0..2 {
                if !is_on_grid[k] {
                    n_not_ongrid += 1;
                }
            }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let lval =
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0))
                                .clamp(0, KMAX_Q_IQ2_XS - 1);
                        u |= (lval as u16) << (2 * i);
                        l[8 * k + i] = lval as i8;
                    }
                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        let neighbors_idx =
                            (-grid_data.map[u as usize] - 1) as usize;
                        let neighbors = &grid_data.neighbors[neighbors_idx..];
                        find_best_neighbor_iq2(
                            neighbors,
                            &grid_data.grid,
                            &xval[8 * k..8 * (k + 1)],
                            &waux[8 * k..8 * (k + 1)],
                            scale,
                            &mut l[8 * k..8 * (k + 1)],
                        );
                    }
                }
                let sumqx: f32 = (0..16)
                    .map(|i| weight[i] * xval[i] * (2.0 * l[i] as f32 + 1.0))
                    .sum();
                let sumq2: f32 = (0..16)
                    .map(|i| weight[i] * (2.0 * l[i] as f32 + 1.0).powi(2))
                    .sum();
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }

            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }

            for k in 0..2 {
                let mut u: u16 = 0;
                for i in 0..8 {
                    u |= (l[8 * k + i] as u16) << (2 * i);
                }
                let grid_index = grid_data.map[u as usize];
                if grid_index < 0 {
                    panic!(
                        "Oops: found point {} not on grid: {:?}",
                        u,
                        &l[8 * k..8 * (k + 1)]
                    );
                }
                q2[2 * ib + k] =
                    (grid_index as u16) | ((block_signs[k] as u16) << 9);
            }

            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            let block = BlockIQ2XS { d: 0, qs: [0; 32], scales: [0; 8] };
            blocks.push(block);
            continue;
        }

        let d = max_scale / 31.0;
        let block = BlockIQ2XS {
            d: f16::from_f32(d).to_bits(),
            qs: {
                let mut arr = [0u16; 32];
                for i in 0..q2.len() {
                    arr[i] = q2[i];
                }
                arr
            },
            scales: {
                let id = 1.0 / d;
                let mut arr = [0u8; 8];
                for ib in 0..QK_K / 16 {
                    let lval =
                        nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
                    if ib % 2 == 0 {
                        arr[ib / 2] = lval as u8;
                    } else {
                        arr[ib / 2] |= (lval as u8) << 4;
                    }
                }
                arr
            },
        };
        blocks.push(block);
    }

    blocks
}

// ============================================================================
// ============================================================================

pub fn quantize_iq3_xxs(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ3XXS> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    // Initialize grid data once
    let grid_data = IQ3GridData::new(256, &KGRID_256_IQ3);

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        // Use provided weights or uniform weights
        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            vec![1.0; QK_K]
        };

        let mut scales = [0.0f32; QK_K / 32];
        let mut q3 = [0u8; 3 * (QK_K / 8) + QK_K / 32]; // 96 + 8 = 104 bytes
        let mut max_scale = 0.0f32;

        // Compute sigma2 for the block
        let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
        let sigma2 = sumx2 / QK_K as f32;

        // Process each sub-block of 32 values
        for ib in 0..QK_K / 32 {
            let xb = &xbl[ib * 32..(ib + 1) * 32];
            let qw = &quant_weights_slice[ib * 32..(ib + 1) * 32];

            // Compute quantization weights
            let mut weight = [0.0f32; 32];
            let mut waux = [0.0f32; 32];
            for i in 0..32 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }

            // Compute signs and absolute values
            let mut xval = [0.0f32; 32];
            let mut block_signs = [0u8; 4];

            for k in 0..4 {
                let mut nflip = 0;
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                // Ensure even parity
                if nflip % 2 == 1 {
                    let mut imin = 0;
                    let mut min = weight[8 * k + imin]
                        * xb[8 * k + imin]
                        * xb[8 * k + imin];
                    for i in 1..8 {
                        let ax =
                            weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }

            // Find max absolute value
            let max_val = xval.iter().fold(0.0f32, |a, &b| a.max(b));
            let mut l = [0i8; 32];

            if max_val < GROUP_MAX_EPS_IQ3_XXS {
                scales[ib] = 0.0;
                continue;
            }

            // Search for optimal scale
            let mut best = 0.0f32;
            let mut scale = max_val / (2 * 8 - 1) as f32;
            let mut is_on_grid = [true; 8];

            for is in -15..=15 {
                let id = ((2 * 8 - 1) as f32 + is as f32 * 0.2) / max_val;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 32];
                let mut is_on_grid_aux = [true; 8];

                for k in 0..8 {
                    for i in 0..4 {
                        let lval =
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        laux[4 * k + i] = lval.clamp(0, 7) as i8;
                    }

                    let mut u: u16 = 0;
                    for i in 0..4 {
                        u |= (laux[4 * k + i] as u16) << (3 * i);
                    }

                    let grid_index = grid_data.map[u as usize];
                    is_on_grid_aux[k] = grid_index >= 0;

                    if grid_index < 0 {
                        // Find best neighbor
                        grid_data.find_best_neighbor(
                            u,
                            &xval[4 * k..],
                            &waux[4 * k..],
                            this_scale,
                            &mut laux[4 * k..],
                        );
                    }
                }

                // Compute MSE
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }

                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid.copy_from_slice(&is_on_grid_aux);
                }
            }

            // Refine off-grid points
            let n_not_ongrid = is_on_grid.iter().filter(|&&x| !x).count();
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..8 {
                    if is_on_grid[k] {
                        continue;
                    }

                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let lval =
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        let lval = lval.clamp(0, 7) as u16;
                        u |= lval << (3 * i);
                    }

                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        grid_data.find_best_neighbor(
                            u,
                            &xval[4 * k..],
                            &waux[4 * k..],
                            scale,
                            &mut l[4 * k..],
                        );
                    } else {
                        let pg =
                            grid_data.grid[grid_index as usize].to_le_bytes();
                        for i in 0..4 {
                            l[4 * k + i] = ((pg[i] as i16 - 4) / 8) as i8;
                        }
                    }
                }

                // Recompute scale
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }

            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }

            // Pack grid indices
            for k in 0..8 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (l[4 * k + i] as u16) << (3 * i);
                }

                let grid_index = grid_data.map[u as usize];
                if grid_index < 0 {
                    panic!("Grid point {} not found", u);
                }
                q3[8 * ib + k] = grid_index as u8;
            }

            // Pack signs and scale
            let signs_u32 = block_signs[0] as u32
                | (block_signs[1] as u32) << 7
                | (block_signs[2] as u32) << 14
                | (block_signs[3] as u32) << 21;

            scales[ib] = scale;
            max_scale = max_scale.max(scale);

            // Store signs in q3 buffer
            let scales_and_signs = unsafe {
                &mut *(&mut q3[QK_K / 4..] as *mut [u8] as *mut [u32])
            };
            scales_and_signs[ib] = signs_u32;
        }

        // Pack final block
        if max_scale == 0.0 {
            blocks.push(BlockIQ3XXS { d: 0, qs: [0; 96] });
            continue;
        }

        let d = max_scale / 31.0;
        let id = 1.0 / d;
        let mut block =
            BlockIQ3XXS { d: f16::from_f32(d).to_bits(), qs: [0; 96] };

        // Pack scales
        for ib in 0..QK_K / 32 {
            let lval =
                nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
            let scales_and_signs = unsafe {
                &mut *(&mut q3[QK_K / 4..] as *mut [u8] as *mut [u32])
            };
            scales_and_signs[ib] |= (lval as u32) << 28;
        }

        // Copy packed data
        for i in 0..96 {
            block.qs[i] = q3[i];
        }

        blocks.push(block);
    }

    blocks
}

/// Uses 512-element grid, 32-element sub-blocks
/// Block layout: d(2) + qs(64) + qh(8) + signs(32) + scales(4) = 110 bytes
pub fn quantize_iq3_s(
    x: &[f32],
    quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ3S> {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    // Initialize grid data for IQ3_S (512-element grid)
    let grid_data = IQ3GridData::new(512, &KGRID_256_IQ3);

    const KMAX_Q_IQ3: i32 = 8;
    const BLOCK_SIZE: usize = 32;

    for ibl in 0..nb {
        let xbl = &x[ibl * QK_K..(ibl + 1) * QK_K];

        // Use provided weights or compute from x^2
        let quant_weights_slice: Vec<f32> = if let Some(qw) = quant_weights {
            qw[ibl * QK_K..(ibl + 1) * QK_K].to_vec()
        } else {
            xbl.iter().map(|v| v * v).collect()
        };

        let mut scales = [0.0f32; QK_K / BLOCK_SIZE];
        let mut max_scale = 0.0f32;

        // Initialize block
        let mut block = BlockIQ3S {
            d: f16::from_f32(0.0).to_bits(),
            qs: [0; 64],
            qh: [0; 8],
            signs: [0; 32],
            scales: [0; 4],
        };

        // Compute sigma2
        let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        // Process each sub-block of 32 values
        for ib in 0..QK_K / BLOCK_SIZE {
            let xb = &xbl[ib * BLOCK_SIZE..(ib + 1) * BLOCK_SIZE];
            let qw =
                &quant_weights_slice[ib * BLOCK_SIZE..(ib + 1) * BLOCK_SIZE];

            // Compute quantization weights
            let mut weight = [0.0f32; BLOCK_SIZE];
            let mut waux = [0.0f32; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }

            // Extract signs and absolute values
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

            let max_val = xval.iter().fold(0.0f32, |a, &b| a.max(b));
            let mut l = [0i8; BLOCK_SIZE];

            if max_val == 0.0 {
                scales[ib] = 0.0;
                continue;
            }

            // Search for optimal scale
            let mut best = 0.0f32;
            let mut scale = max_val / (2 * KMAX_Q_IQ3 - 1) as f32;
            let mut is_on_grid = [true; BLOCK_SIZE / 4];

            for is in -9..=9 {
                let id =
                    ((2 * KMAX_Q_IQ3 - 1) as f32 + is as f32 * 0.2) / max_val;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; BLOCK_SIZE];
                let mut is_on_grid_aux = [true; BLOCK_SIZE / 4];

                for k in 0..BLOCK_SIZE / 4 {
                    for i in 0..4 {
                        let lval =
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        laux[4 * k + i] = lval.clamp(0, KMAX_Q_IQ3 - 1) as i8;
                    }

                    let mut u: u16 = 0;
                    for i in 0..4 {
                        u |= (laux[4 * k + i] as u16) << (3 * i);
                    }

                    let grid_index = grid_data.map[u as usize];
                    is_on_grid_aux[k] = grid_index >= 0;

                    if grid_index < 0 {
                        // Find best neighbor
                        grid_data.find_best_neighbor(
                            u,
                            &xval[4 * k..],
                            &waux[4 * k..],
                            this_scale,
                            &mut laux[4 * k..],
                        );
                    }
                }

                // Compute MSE
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..BLOCK_SIZE {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }

                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                    is_on_grid.copy_from_slice(&is_on_grid_aux);
                }
            }

            // Refine off-grid points
            let n_not_ongrid = is_on_grid.iter().filter(|&&x| !x).count();
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..BLOCK_SIZE / 4 {
                    if is_on_grid[k] {
                        continue;
                    }

                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let lval =
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0));
                        let lval = lval.clamp(0, KMAX_Q_IQ3 - 1) as u16;
                        u |= lval << (3 * i);
                    }

                    let grid_index = grid_data.map[u as usize];
                    if grid_index < 0 {
                        grid_data.find_best_neighbor(
                            u,
                            &xval[4 * k..],
                            &waux[4 * k..],
                            scale,
                            &mut l[4 * k..],
                        );
                    } else {
                        let pg =
                            grid_data.grid[grid_index as usize].to_le_bytes();
                        for i in 0..4 {
                            l[4 * k + i] = ((pg[i] as i16 - 1) / 2) as i8;
                        }
                    }
                }

                // Recompute scale
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..BLOCK_SIZE {
                    let w = weight[i];
                    let q = 2.0 * l[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }

            if scale < 0.0 {
                scale = -scale;
                for k in 0..BLOCK_SIZE / 8 {
                    block_signs[k] = !block_signs[k];
                }
            }

            // Pack grid indices (low 8 bits in qs, high bit in qh)
            for k in 0..BLOCK_SIZE / 4 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (l[4 * k + i] as u16) << (3 * i);
                }

                let grid_index = grid_data.map[u as usize];
                if grid_index < 0 {
                    panic!("Grid point {} not found", u);
                }

                let idx = ib * (BLOCK_SIZE / 4) + k;
                block.qs[idx] = (grid_index & 255) as u8;
                block.qh[idx / 8] |= ((grid_index >> 8) as u8) << (idx % 8);
            }

            // Pack signs
            for k in 0..BLOCK_SIZE / 8 {
                block.signs[ib * (BLOCK_SIZE / 8) + k] = block_signs[k];
            }

            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            blocks.push(block);
            continue;
        }

        // Pack final block
        let d = max_scale / 31.0;
        block.d = f16::from_f32(d).to_bits();
        let id = 1.0 / d;

        // Pack scales (2 scales per byte, 4 bits each)
        for ib in 0..QK_K / BLOCK_SIZE / 2 {
            let l1 = nearest_int(0.5 * (id * scales[2 * ib] - 1.0)).clamp(0, 15)
                as u8;
            let l2 = nearest_int(0.5 * (id * scales[2 * ib + 1] - 1.0))
                .clamp(0, 15) as u8;
            block.scales[ib] = l1 | (l2 << 4);
        }

        blocks.push(block);
    }

    blocks
}

/// Quantize FP32 to IQ1_S (simplified implementation for GPU kernel verification)
pub fn quantize_iq1_s(
    _x: &[f32],
    _quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ1S> {
    // TODO: Implement full IQ1_S quantization algorithm
    // For now, return empty block for testing
    vec![BlockIQ1S { d: 0, qs: [0; 32], qh: [0; 8] }]
}

/// Quantize FP32 to IQ1_M (simplified implementation for GPU kernel verification)
pub fn quantize_iq1_m(
    _x: &[f32],
    _quant_weights: Option<&[f32]>,
) -> Vec<BlockIQ1M> {
    // TODO: Implement full IQ1_M quantization algorithm
    // For now, return empty block for testing
    vec![BlockIQ1M { qs: [0; 32], qh: [0; 16], scales: [0; 8] }]
}
