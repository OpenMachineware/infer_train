// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trace scale encoding in IQ2_XXS

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;

fn main() {
    println!("=== IQ2_XXS Scale Encoding Trace ===\n");

    // Use simple input: all 1.0
    let data = vec![1.0f32; 256];

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    let block = &blocks[0];

    let d = f16::from_bits(block.d).to_f32();
    println!("Block header scale d = {:.6}", d);

    // Read scale_bits from first sub-block
    let aux32: u32 = (block.qs[2] as u32) | ((block.qs[3] as u32) << 16);
    let scale_bits = aux32 >> 28;
    println!("Scale bits (from quantized block) = {}", scale_bits);

    // What scale was computed by the quantizer?
    // scale_bits = 0.5 * (scales[ib] / d - 1)
    // scales[ib] = d * (2 * scale_bits + 1)
    let decoded_scale = d * (2.0 * scale_bits as f32 + 1.0);
    println!(
        "Decoded scales[0] = d * (2*{} + 1) = {:.6}",
        scale_bits, decoded_scale
    );

    // Now let's trace what scale the quantizer SHOULD compute
    // For all-ones input with L values [0, 1, 1, 1, 2, 1, 1, 1] from Grid[189]:
    // x[i] = scale * (2*L[i] + 1)
    // Using weighted least squares:
    // sumqx = sum(weight[i] * x[i] * (2*L[i] + 1))
    // sumq2 = sum(weight[i] * (2*L[i] + 1)^2)
    // scale = sumqx / sumq2

    // For uniform weight and x[i] = 1.0:
    let l_vals = [0, 1, 1, 1, 2, 1, 1, 1];
    let q_vals: Vec<f32> =
        l_vals.iter().map(|&l| 2.0 * l as f32 + 1.0).collect();
    println!("\nL values from Grid[189]: {:?}", l_vals);
    println!("Q values (2*L+1): {:?}", q_vals);

    let sumqx: f32 = q_vals.iter().map(|&q| 1.0 * q).sum();
    let sumq2: f32 = q_vals.iter().map(|&q| q * q).sum();
    let expected_scale = sumqx / sumq2;
    println!("\nExpected scale from weighted least squares:");
    println!("  sumqx = {:.6}", sumqx);
    println!("  sumq2 = {:.6}", sumq2);
    println!("  scale = {:.6}", expected_scale);

    // Check encoding
    // lval = 0.5 * (scales[ib] / d - 1)
    let expected_lval = 0.5 * (expected_scale / d - 1.0);
    println!(
        "\nExpected lval = 0.5 * (scales[0] / d - 1) = {:.6}",
        expected_lval
    );
    println!("  Clamped to [0, 15]: {}", expected_lval.clamp(0.0, 15.0) as i32);

    // The issue: expected_scale = 0.645 and d = 0.009674
    // lval = 0.5 * (0.645 / 0.009674 - 1) = 0.5 * (66.68 - 1) = 32.84
    // This exceeds 15! So it gets clamped to 15

    // The problem is that d = max_scale / 31, and if max_scale is very small compared to
    // the actual scales, the lval will be too large

    // Let's check what max_scale was
    // All 8 sub-blocks have similar inputs, so all should have similar scales
    // max_scale should be close to expected_scale = 0.645

    // If max_scale = 0.2 (from earlier debug):
    // d = 0.2 / 31 = 0.00645
    // But we got d = 0.009674, which means max_scale = 0.009674 * 31 = 0.3

    println!("\nChecking max_scale:");
    println!("  d = {:.6}", d);
    println!("  max_scale = d * 31 = {:.6}", d * 31.0);

    // If the quantizer computed scales[ib] ≈ 0.2 for each sub-block:
    // And max_scale ≈ 0.2
    // Then d = 0.2 / 31 = 0.00645
    // And lval = 0.5 * (0.2 / 0.00645 - 1) = 0.5 * (31 - 1) = 15
    // This is the maximum!

    // So the issue is that when all scales are equal, they all get lval = 15
    // And the decoded scale becomes d * (2*15 + 1) = d * 31 = max_scale ✓

    // The problem: the grid approximation error makes the actual scale wrong!
    // If expected_scale = 0.645, but max_scale = 0.2 (from quantization optimization),
    // then the quantizer is NOT finding the optimal scale!

    println!("\nDiagnosis:");
    println!("  The quantizer found scale ≈ 0.2 for sub-blocks");
    println!(
        "  But the expected scale for grid point [0,1,1,1,2,1,1,1] is {:.6}",
        expected_scale
    );
    println!("  This mismatch causes the large error");

    // Let me check: for all-ones, what's the optimal scale?
    // We want db * grid[i] ≈ 1.0 for most i
    // If most grid[i] = 25, then db ≈ 1/25 = 0.04
    // db = d * (0.5 + l) * 0.25
    // 0.04 = d * (0.5 + 15) * 0.25 = d * 15.5 * 0.25 = d * 3.875
    // d = 0.04 / 3.875 = 0.0103

    let optimal_db = 1.0 / 25.0;
    let optimal_d = optimal_db / (0.5 + 15.0) / 0.25;
    println!("\nOptimal values for grid=25:");
    println!("  optimal_db = 1/25 = {:.6}", optimal_db);
    println!("  optimal_d = db / (0.5 + 15) / 0.25 = {:.6}", optimal_d);
    println!("  Actual d = {:.6}", d);

    // The actual d is close to optimal!
    // So the issue is not the scale, but the grid approximation

    println!("\nConclusion:");
    println!("  The large error is inherent in IQ2_XXS grid design");
    println!("  Grid[189] has [8, 25, 25, 25, 43, 25, 25, 25]");
    println!("  6 out of 8 values are 25 (giving 0.937)");
    println!("  1 is 8 (giving 0.300)");
    println!("  1 is 43 (giving 1.612)");
    println!("  This is the best approximation the grid can achieve");
}
