// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trace L value selection in IQ2_XXS quantization

use infer_train::quant::weights::iq_grid::*;

fn main() {
    println!("=== IQ2_XXS L Value Selection ===\n");

    // Check what L values correspond to Grid[189]
    let grid189 = IQ2XXS_GRID[189].to_le_bytes();
    println!("Grid[189] = {:?}", grid189);

    // L = (grid[i] - 8) / 17
    println!("\nL values from Grid[189]:");
    for (i, &g) in grid189.iter().enumerate() {
        let l = if g >= 8 { (g - 8) / 17 } else { 0 };
        let q = 2 * l + 1;
        println!("  [{}]: grid={}, L={}, q=2*L+1={}", i, g, l, q);
    }

    // The quantization should choose L values such that scale * (2*L+1) ≈ 1.0
    // For all-ones, all L values should be the same

    // Check what L values Grid[0] has
    let grid0 = IQ2XXS_GRID[0].to_le_bytes();
    println!("\nGrid[0] = {:?}", grid0);
    println!("All 8s, L=0, q=1");

    // For all-ones input, what scale should we use?
    // If L=0 for all values: scale * 1 = 1.0 => scale = 1.0
    // If L=1 for all values: scale * 3 = 1.0 => scale = 0.333
    // If L=2 for all values: scale * 5 = 1.0 => scale = 0.2

    // But there's no grid entry for uniform L=1 or L=2!
    // Grid[0] is the only uniform entry, and it's L=0

    // So the quantizer has to choose a non-uniform grid
    // For input=1.0, what's the best approximation?

    // If we use Grid[189] = [8, 25, 25, 25, 43, 25, 25, 25]:
    // dequant = db * grid * 0.125 (if db is the sub-block scale)
    // For grid=8: val = db * 8 * 0.125 = db
    // For grid=25: val = db * 25 * 0.125 = db * 3.125
    // For grid=43: val = db * 43 * 0.125 = db * 5.375

    // With db = 0.037487:
    // grid=8: 0.300
    // grid=25: 0.937 (close to 1.0!)
    // grid=43: 1.612 (too high)

    // So the quantizer chose a grid that gives mostly 0.937, with some outliers

    // Check if there's a grid with all 25s (uniform L=1)
    println!("\nSearching for uniform grids:");
    let mut found_uniform = false;
    for idx in 0..256 {
        let grid = IQ2XXS_GRID[idx].to_le_bytes();
        if grid.iter().all(|&v| v == grid[0]) {
            let l = if grid[0] >= 8 { (grid[0] - 8) / 17 } else { 0 };
            println!("  Grid[{}] = all {}, L={}", idx, grid[0], l);
            found_uniform = true;
        }
    }
    if !found_uniform {
        println!("  No uniform grids found except Grid[0]");
    }

    // This confirms: IQ2_XXS grid does NOT have uniform entries for L=1 or L=2
    // The quantizer MUST choose a mixed-L grid, which introduces quantization error

    // Let's check what the actual quantization does
    println!("\nExpected quantization for all-ones:");
    println!("  No uniform L=1 or L=2 grid exists");
    println!("  Best grid will have most values with L=1 (grid=25)");
    println!("  But some positions will have L=0 (grid=8) or L=2 (grid=43)");
    println!("  This is inherent in the IQ2_XXS grid design");

    // Compute expected MSE for this quantization
    println!("\nExpected MSE for Grid[189] with db=0.037487:");
    let db = 0.037487f32;
    let expected_mse: f32 = grid189
        .iter()
        .map(|&g| {
            let val = db * g as f32 * 0.125; // wait, should use Metal formula
            let val_metal = db * g as f32; // Metal: no 0.125 factor here
            let err = val_metal - 1.0;
            err * err
        })
        .sum::<f32>()
        / 8.0;
    println!("  Using Metal formula (no 0.125): MSE = {:.6}", expected_mse);

    // Actually Metal uses db * grid directly, then multiplies sum by 0.25 at the end
    // Let me recompute
    let vals: Vec<f32> =
        grid189.iter().map(|&g| db * g as f32 * 0.25).collect();
    let mse_correct: f32 =
        vals.iter().map(|&v| (v - 1.0).powi(2)).sum::<f32>() / 8.0;
    println!("  Using db * grid * 0.25: MSE = {:.6}", mse_correct);
    println!("  Values: {:?}", vals);
}
