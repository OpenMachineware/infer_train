// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trace scale calculation in IQ2_XXS quantization

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;

fn main() {
    println!("=== IQ2_XXS Scale Trace ===\n");

    // Use simple input: all 1.0
    let data = vec![1.0f32; 256];

    // Manually trace the quantization
    let nb = 1;
    let ibl = 0;
    let xbl = &data[0..256];

    // Compute sigma2
    let sumx2: f32 = xbl.iter().map(|v| v * v).sum();
    let sigma2 = sumx2 / 256.0;
    println!("sigma2 = {:.6}", sigma2);

    // Process sub-block 0
    let ib = 0;
    let xb = &xbl[ib * 32..(ib + 1) * 32];

    // Weight calculation
    let qw: Vec<f32> = vec![1.0; 32];
    let mut weight = [0.0f32; 32];
    for i in 0..32 {
        weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
    }
    println!("\nweight[0..4] = {:?}", &weight[0..4]);

    // Extract signs - all positive
    let mut xval = [0.0f32; 32];
    let mut block_signs = [0u8; 4];

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
        block_signs[k] = s & 127;
    }

    println!("xval[0..4] = {:?}", &xval[0..4]);
    println!("block_signs = {:?}", block_signs);

    // Find max
    let max = xval.iter().cloned().fold(0.0f32, f32::max);
    println!("\nmax = {:.6}", max);

    // Initial scale using make_qp_quants
    const KMAX_Q: i32 = 3;
    let mut l = [0u8; 32];
    let scale = make_qp_quants_test(32, KMAX_Q + 1, &xval, &mut l, &weight);
    let eff_max = scale * KMAX_Q as f32;

    println!("\nAfter make_qp_quants:");
    println!("  scale = {:.6}", scale);
    println!("  eff_max = {:.6}", eff_max);
    println!("  l[0..8] = {:?}", &l[0..8]);

    // Now trace the full quantization
    let blocks = quantize_iq2_xxs(&data, None);
    let block = &blocks[0];
    let d = f16::from_bits(block.d).to_f32();

    println!("\nFinal block:");
    println!("  d = {:.6}", d);

    // Decode scale from bits 28-31
    let aux32_metal: u32 = (block.qs[2] as u32) | ((block.qs[3] as u32) << 16);
    let scale_bits = aux32_metal >> 28;
    println!("  scale_bits = {}", scale_bits);

    // What should scale be?
    // scales[ib] = scale from optimization
    // d = max_scale / 31.0
    // lval = 0.5 * (id * scales[ib] - 1.0) where id = 1.0 / d
    // So scales[ib] = d * (2 * lval + 1)

    let decoded_scale = d * (2.0 * scale_bits as f32 + 1.0);
    println!(
        "  decoded_scale = d * (2 * {} + 1) = {:.6}",
        scale_bits, decoded_scale
    );

    // Alternative decoding (Metal formula)
    let d_sub = d * (0.5 + scale_bits as f32) * 0.25;
    println!(
        "  d_sub (Metal) = d * (0.5 + {}) * 0.25 = {:.6}",
        scale_bits, d_sub
    );
}

fn make_qp_quants_test(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [u8],
    quant_weights: &[f32],
) -> f32 {
    let max = x[0..n].iter().cloned().fold(0.0f32, f32::max);
    if max < 1e-6 {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }

    let mut iscale = nmax as f32 / max;
    let scale = 1.0 / iscale;
    let mut best_mse: f32 = (0..n)
        .map(|i| {
            let li = (iscale * x[i]).round() as i32;
            let li = li.min(nmax) as u8;
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
                let li = (iscale_is * x[i]).round() as i32;
                let li = li.min(nmax) as u8;
                let diff = x[i] - scale_is * li as f32;
                quant_weights[i] * diff * diff
            })
            .sum();

        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    // Compute L values and weighted least squares scale
    let mut sumlx: f32 = 0.0;
    let mut suml2: f32 = 0.0;
    for i in 0..n {
        let li = (iscale * x[i]).round() as i32;
        let li = li.min(nmax) as u8;
        l[i] = li;
        sumlx += quant_weights[i] * x[i] * li as f32;
        suml2 += quant_weights[i] * li as f32 * li as f32;
    }

    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        1.0 / iscale
    }
}
