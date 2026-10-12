// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Debug IQ2_XXS quantization with known values

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;

fn main() {
    println!("=== IQ2_XXS Debug Test ===\n");

    // Create simple test data: all values are 1.0
    let data: Vec<f32> = vec![1.0; 256];

    println!("Input: 256 values, all 1.0");

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    assert_eq!(blocks.len(), 1);

    let block = &blocks[0];
    let d = f16::from_bits(block.d).to_f32();

    println!("Block d = {:.4}", d);
    println!("qs[0..8] = {:?}", &block.qs[0..8]);

    // Print the first 32 values of qs as u32
    let q2: Vec<u32> = (0..8)
        .map(|i| {
            (block.qs[2 * i] as u32) | ((block.qs[2 * i + 1] as u32) << 16)
        })
        .collect();

    println!("\nReconstructed q2[0..8] as u32:");
    for i in 0..8 {
        println!("q2[{}] = 0x{:08X}", i, q2[i]);
    }

    // For sub-block 0, decode:
    // q2[0] has grid indices (bits 0-31)
    // q2[1] has signs (bits 0-27) + scale (bits 28-31)

    println!("\n=== Sub-block 0 ===");
    let aux32_0 = q2[0];
    let aux32_1 = q2[1];

    println!("aux32[0] = 0x{:08X} (grid indices)", aux32_0);
    println!("aux32[1] = 0x{:08X} (signs + scale)", aux32_1);

    let scale_bits = aux32_1 >> 28;
    println!("Scale bits = {} (from bits 28-31)", scale_bits);

    let db = d * (0.5 + scale_bits as f32) * 0.25;
    println!("db = d * (0.5 + {}) * 0.25 = {:.4}", scale_bits, db);

    // Grid indices from aux32[0] bytes
    let aux8: [u8; 4] = aux32_0.to_le_bytes();
    println!("\nGrid indices (from aux32[0] bytes):");
    for l in 0..4 {
        println!("  aux8[{}] = {} (grid index)", l, aux8[l]);
    }

    // Signs from aux32[1]
    println!("\nSigns from aux32[1]:");
    for l in 0..4 {
        let signs = (aux32_1 >> (7 * l)) & 0x7F;
        println!("  Signs[{}] = 0x{:02X}", l, signs);
    }

    // Now dequantize the first 8 values (l=0)
    use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;

    println!("\n=== Dequantizing first 8 values (l=0) ===");
    let grid_idx = aux8[0];
    let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
    println!("Grid[{}] = {:?}", grid_idx, grid);

    let signs = ((aux32_1 >> 0) & 0x7F) as u8;
    println!("Signs = 0x{:02X}", signs);

    println!("\nDequantized values:");
    for j in 0..8 {
        let q = grid[j] as f32 * 0.125; // grid values 8,25,43 -> divide by 8
        let sign = if (signs >> j) & 1 != 0 { -1.0f32 } else { 1.0f32 };
        let y = sign * db * q;
        println!(
            "  y[{}] = {:.4} (grid={}, sign={:.0}, db={:.4})",
            j, y, grid[j], sign, db
        );
    }

    println!("\nExpected: 8 values of approximately 1.0");
}
