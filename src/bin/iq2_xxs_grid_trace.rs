// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Check the relationship between quantization and dequantization grids

use infer_train::quant::weights::iq_grid::*;

fn main() {
    println!("=== IQ2_XXS Grid Relationship ===\n");

    // KGRID_2BIT_256 stores L values as 2-bit fields
    // Each entry encodes 8 L values, each in [0, 2]
    // The quantization grid converts these to q = 2*L + 1 = 1, 3, 5

    // IQ2XXS_GRID stores dequant values: 8, 25, 43

    println!("L values and corresponding grid values:");
    println!("L=0: quant = 2*0+1 = 1, dequant = 8, ratio = {}", 8.0 / 1.0);
    println!("L=1: quant = 2*1+1 = 3, dequant = 25, ratio = {}", 25.0 / 3.0);
    println!("L=2: quant = 2*2+1 = 5, dequant = 43, ratio = {}", 43.0 / 5.0);

    println!("\nDequant grid formula: g = 8 + 17*L");
    println!("  L=0: g = 8 + 17*0 = 8");
    println!("  L=1: g = 8 + 17*1 = 25");
    println!("  L=2: g = 8 + 17*2 = 42 (but grid has 43?)");

    // Check actual grid values
    println!("\nActual grid entries:");
    for idx in [0, 189, 255] {
        let grid = IQ2XXS_GRID[idx].to_le_bytes();
        println!("  Grid[{}] = {:?}", idx, grid);
    }

    // Check KGRID_2BIT_256
    println!("\nKGRID_2BIT_256[0] = 0x{:04X}", KGRID_2BIT_256[0]);
    // Decode L values from KGRID
    let kgrid0 = KGRID_2BIT_256[0];
    let mut l_vals = [0u8; 8];
    for i in 0..8 {
        l_vals[i] = ((kgrid0 >> (2 * i)) & 0x3) as u8;
    }
    println!("  L values: {:?}", l_vals);

    // The quantization scale formula: x[i] ≈ scale * (2*L[i] + 1)
    // The dequantization scale formula: db = d * (0.5 + scale_bits) * 0.25
    // The dequantization grid formula: y[i] = db * grid[i] * sign

    // If x[i] = scale * (2*L[i] + 1) and grid[i] ≈ 8*(2*L[i] + 1), then:
    // y[i] = db * 8 * (2*L[i] + 1)
    // For this to equal x[i], we need: db * 8 = scale
    // So: d * (0.5 + scale_bits) * 0.25 * 8 = scale
    //     d * (0.5 + scale_bits) * 2 = scale
    //     d * (scale_bits + 0.5) * 2 = scale
    //     d * (2*scale_bits + 1) = scale

    // YES! This matches the quantization encoding:
    // scales[ib] = d * (2*l + 1) where l is the scale_bits

    println!("\nScale relationship:");
    println!("  Quantization: scales[ib] = d * (2*l + 1)");
    println!("  Dequantization: db = d * (0.5 + l) * 0.25");
    println!();
    println!("  If db * 8 = scale, then:");
    println!("    d * (0.5 + l) * 0.25 * 8 = d * (2*l + 1)");
    println!("    d * (0.5 + l) * 2 = d * (2*l + 1)");
    println!("    d * (l + 0.5) * 2 = d * (2*l + 1)");
    println!("    d * (2*l + 1) = d * (2*l + 1)  ✓");

    // But the grid is not exactly 8x, it varies from 8 to 8.6
    // So there's inherent approximation error!
    println!("\nApproximation error:");
    println!("  Grid ratio varies from 8.0 to 8.6");
    println!(
        "  This introduces ~7.5% quantization error even with perfect scale"
    );
}
