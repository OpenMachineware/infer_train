// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Direct comparison between Rust quantizer and llama.cpp output
// Uses hardcoded test data and compares the block output

use half::f16;
use infer_train::quant::types::BlockIQ2XXS;
use infer_train::quant::weights::iq::quantize_iq2_xxs;

fn main() {
    println!("=== IQ2_XXS Direct Comparison ===\n");

    // Hardcoded test data (must be 256 elements)
    let mut test_data: Vec<f32> = vec![
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
        0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
        -0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
        0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
        0.567890, -0.234567,
    ];
    test_data.extend(vec![0.0f32; 256 - 32]);

    // Quantize using Rust
    let quant_weights: Vec<f32> = vec![1.0; 256];
    let blocks = quantize_iq2_xxs(&test_data, Some(&quant_weights));

    println!("Rust output:");
    println!(
        "  d = {:.6} (f16 bits: 0x{:04x})",
        f16::from_bits(blocks[0].d).to_f32(),
        blocks[0].d
    );
    println!("  qs =");
    for i in 0..8 {
        println!("    qs[{}] = 0x{:04x}", i, blocks[0].qs[i]);
    }

    // Decode and print values
    println!("\nDecoded values from Rust:");
    let block = &blocks[0];
    let d = f16::from_bits(block.d).to_f32();
    let id = 1.0 / d;

    for ib in 0..8 {
        let q2_0 =
            (block.qs[2 * ib] as u32) | ((block.qs[2 * ib + 1] as u32) << 16);
        let grid_idx = (q2_0 >> (8 * (ib % 4))) & 0xFF;
        let signs = (q2_0 >> (7 * (ib % 4))) & 0x7F;
        let scale_bits = (q2_0 >> 28) & 0xF;
        let scale = d * (2.0 * scale_bits as f32 + 1.0);

        println!(
            "  ib={}: grid_idx={}, signs=0x{:02x}, scale={:.6}",
            ib, grid_idx, signs, scale
        );
    }

    // Compare with expected llama.cpp output
    // The exact output depends on llama.cpp's grid implementation
    println!("\nExpected from llama.cpp:");
    println!("  (Run iq2_xxs_llama_test.c to get actual values)");
}
