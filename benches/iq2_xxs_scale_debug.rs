// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Debug IQ2_XXS quantization - trace scale calculation

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;

fn main() {
    println!("=== IQ2_XXS Scale Debug ===\n");

    // Create simple test data: all values are 1.0
    let data: Vec<f32> = vec![1.0; 256];

    println!("Input: 256 values, all 1.0\n");

    // Trace the quantization manually
    const QK_K: usize = 256;
    const KMAX_Q: i32 = 3;

    // Compute sigma2
    let sumx2: f32 = data.iter().map(|v| v * v).sum();
    let sigma2 = sumx2 / QK_K as f32;
    println!("sigma2 = {} (from sumx2 = {})", sigma2, sumx2);

    // Process sub-block 0
    let ib = 0;
    let xb = &data[ib * 32..(ib + 1) * 32];
    let quant_weights: Vec<f32> = vec![1.0; QK_K];
    let qw = &quant_weights[ib * 32..(ib + 1) * 32];

    // Compute weight and waux
    let mut weight = [0.0f32; 32];
    let mut waux = [0.0f32; 32];
    for i in 0..32 {
        weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
        waux[i] = weight[i].sqrt();
    }
    println!("weight[0] = {} (should be ~1.414)", weight[0]);
    println!("waux[0] = {} (should be ~1.189)", waux[0]);

    // Extract signs and xval
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
        // Parity constraint
        if nflip % 2 != 0 {
            let mut min_idx = 0;
            let mut min_val = weight[8 * k] * xb[8 * k] * xb[8 * k];
            for i in 1..8 {
                let val = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                if val < min_val {
                    min_val = val;
                    min_idx = i;
                }
            }
            xval[8 * k + min_idx] = -xval[8 * k + min_idx];
            s ^= 1 << min_idx;
        }
        block_signs[k] = s & 127;
    }
    println!("xval[0..4] = {:?}", &xval[0..4]);
    println!("block_signs = {:?}", block_signs);

    // Find max
    let max = xval.iter().cloned().fold(0.0f32, f32::max);
    println!("max = {}", max);

    // Initial scale using make_qp_quants
    let mut l = [0u8; 32];
    let scale = make_qp_quants_debug(32, KMAX_Q + 1, &xval, &mut l, &weight);
    println!("\nmake_qp_quants returned scale = {}", scale);
    println!("l[0..8] = {:?}", &l[0..8]);

    let eff_max = scale * KMAX_Q as f32;
    println!("eff_max = scale * {} = {}", KMAX_Q, eff_max);

    // Main optimization loop
    println!("\n=== Optimization loop ===");
    let mut best = 0.0f32;
    let mut l_final = [0i8; 32];

    for is in -6..=6 {
        let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / eff_max;
        let this_scale = 1.0 / id;

        let mut laux = [0i8; 32];
        for k in 0..4 {
            for i in 0..8 {
                let lval = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0));
                laux[8 * k + i] = lval.clamp(0, KMAX_Q - 1) as i8;
            }
        }

        let sumqx: f32 = (0..32)
            .map(|i| weight[i] * xval[i] * (2.0 * laux[i] as f32 + 1.0))
            .sum();
        let sumq2: f32 = (0..32)
            .map(|i| weight[i] * (2.0 * laux[i] as f32 + 1.0).powi(2))
            .sum();

        let new_scale = sumqx / sumq2;
        let score = new_scale * sumqx;

        if is <= 2 {
            println!(
                "is={}, id={:.4}, this_scale={:.4}, sumqx={:.4}, sumq2={:.4}, new_scale={:.4}, score={:.4}",
                is, id, this_scale, sumqx, sumq2, new_scale, score
            );
        }

        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            best = score;
            l_final.copy_from_slice(&laux);
        }
    }

    println!("\nbest score = {}", best);
    println!("l_final[0..8] = {:?}", &l_final[0..8]);

    // Compute final scale
    let sumqx: f32 = (0..32)
        .map(|i| weight[i] * xval[i] * (2.0 * l_final[i] as f32 + 1.0))
        .sum();
    let sumq2: f32 = (0..32)
        .map(|i| weight[i] * (2.0 * l_final[i] as f32 + 1.0).powi(2))
        .sum();
    let final_scale = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };

    println!("\nFinal scale = {}", final_scale);
}

fn make_qp_quants_debug(
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

    println!(
        "make_qp_quants: max={}, initial iscale={}, scale={}",
        max, iscale, scale
    );

    let mut best_mse: f32 = (0..n)
        .map(|i| {
            let li = nearest_int(iscale * x[i]).min(nmax) as u8;
            let diff = x[i] - scale * li as f32;
            quant_weights[i] * diff * diff
        })
        .sum();

    println!("Initial MSE = {}", best_mse);

    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mse: f32 = (0..n)
            .map(|i| {
                let li = nearest_int(iscale_is * x[i]).min(nmax) as u8;
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
        let li = nearest_int(iscale * x[i]).min(nmax) as u8;
        l[i] = li;
        sumlx += quant_weights[i] * x[i] * li as f32;
        suml2 += quant_weights[i] * li as f32 * li as f32;
    }

    println!(
        "After optimization: iscale={}, sumlx={}, suml2={}",
        iscale, sumlx, suml2
    );

    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        1.0 / iscale
    }
}

fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}
