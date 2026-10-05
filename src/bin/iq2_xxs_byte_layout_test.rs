// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trace byte layout for IQ2_XXS to match Metal kernel

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;

fn main() {
    println!("=== IQ2_XXS Byte Layout Test ===\n");

    // Use simple input: all 1.0
    let data = vec![1.0f32; 256];

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    assert_eq!(blocks.len(), 1);

    let block = &blocks[0];

    // Print block header
    let d = half::f16::from_bits(block.d).to_f32();
    println!("Block scale d = {:.6}", d);

    // Print first 8 u16 values
    println!("\nqs[0..8]:");
    for i in 0..8 {
        println!("  qs[{}] = 0x{:04X}", i, block.qs[i]);
    }

    // Now trace what Metal kernel would read for ib=0
    println!("\n=== Metal kernel read for ib=0 ===");

    // Metal: q2 = qs + 4*ib = qs + 0
    // q2[0] = qs[0], q2[1] = qs[1], q2[2] = qs[2], q2[3] = qs[3]
    // aux8 = (uint8_t*)q2
    // aux32 = q2[2] | (q2[3] << 16)

    let aux32_metal: u32 = (block.qs[2] as u32) | ((block.qs[3] as u32) << 16);
    println!("aux32 (Metal) = 0x{:08X}", aux32_metal);

    // aux8 bytes (little-endian)
    // Read first 4 u16 values and convert to bytes
    let aux8_metal: [u8; 8] = {
        let mut bytes = [0u8; 8];
        for i in 0..4 {
            let u16_bytes = block.qs[i].to_le_bytes();
            bytes[2 * i] = u16_bytes[0];
            bytes[2 * i + 1] = u16_bytes[1];
        }
        bytes
    };
    println!("aux8[0..8]: {:?}", aux8_metal);

    // Scale from bits 28-31
    let scale_bits = aux32_metal >> 28;
    println!("scale_bits (bits 28-31) = {}", scale_bits);

    // Grid indices from aux8[0..4]
    println!("\nGrid indices:");
    for l in 0..4 {
        println!("  l={} grid_idx = {}", l, aux8_metal[l]);
    }

    // Compare with what we expect from quantization
    println!("\n=== From Rust quantization ===");
    println!("Expected: all values should quantize to same grid point");
    println!("Grid[0] = all 8s (uniform L=0)");
    println!("If L values are all 0, grid index should be 0");

    // Let's manually compute what should be in q2[0] and q2[1]
    // For ib=0, k=0..3, each group of 8 values
    // L values: all should be the same since input is uniform
    // The make_qp_quants should find optimal L values

    // Actually let me check what the L values are
    // For uniform input, we'd expect:
    // - xval = [1.0; 32] (all positive)
    // - make_qp_quants finds scale such that L[i] ≈ xval[i]/scale
    // - With scale optimization, L should be near 2 (max) for max precision

    // Check first grid entry
    let grid0 = IQ2XXS_GRID[0].to_le_bytes();
    println!("\nGrid[0] values: {:?}", grid0);

    // Dequantize using Metal formula
    println!("\n=== Dequantization using Metal formula ===");

    let d_sub = d * (0.5 + scale_bits as f32) * 0.25;
    println!("d_sub = d * (0.5 + {}) * 0.25 = {:.6}", scale_bits, d_sub);

    // Signs from aux32
    // Metal: signs = ssigns[(aux32 >> 7*l) & 127]
    const KSIGNS_IQ2XS: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
    ];
    const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    for l in 0..4 {
        let grid_idx = aux8_metal[l];
        let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
        let signs = KSIGNS_IQ2XS[(aux32_metal >> (7 * l)) as usize & 127];

        print!("  Group {}: ", l);
        for j in 0..8 {
            let q = grid[j] as f32;
            let sign =
                if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            let val = sign * d_sub * q;
            print!("{:.3} ", val);
        }
        println!();
    }
}
