// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare IQ2_XXS quantization step by step with llama.cpp
// Generate test data and compare quantized outputs

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

const QK_K: usize = 256;

fn main() {
    println!("=== IQ2_XXS Quantization Comparison ===\n");

    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate simple test data - uniform weights
    let x: Vec<f32> =
        (0..QK_K).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let qw: Vec<f32> = vec![1.0; QK_K];

    // Quantize
    let blocks = quantize_iq2_xxs(&x, Some(&qw));

    println!("Generated {} blocks", blocks.len());
    println!("\nBlock 0:");
    println!("  d = {:.6}", f16::from_bits(blocks[0].d).to_f32());

    // Show first few qs values
    println!("  qs[0..8]: {:?}", &blocks[0].qs[0..8]);

    // Decode the qs values to see what they represent
    println!("\nDecoding qs[0] and qs[1] (first 32 values):");
    let qs0 = blocks[0].qs[0];
    let qs1 = blocks[0].qs[1];

    // Reconstruct uint32_t
    let q2_0 = (qs0 as u32) | ((qs1 as u32) << 16);
    println!("  q2[0] = 0x{:08X}", q2_0);

    // Extract grid indices (8 bits each)
    for k in 0..4 {
        let grid_idx = (q2_0 >> (8 * k)) & 0xFF;
        println!("    grid_index[{}] = {}", k, grid_idx);
    }

    println!("\n  qs[2] and qs[3] (first 32 values, signs+scales):");
    let qs2 = blocks[0].qs[2];
    let qs3 = blocks[0].qs[3];
    let q2_1 = (qs2 as u32) | ((qs3 as u32) << 16);
    println!("  q2[1] = 0x{:08X}", q2_1);

    // Extract signs (7 bits each) and scale (4 bits)
    for k in 0..4 {
        let signs = (q2_1 >> (7 * k)) & 0x7F;
        println!("    block_signs[{}] = 0x{:02X} ({:07b})", k, signs, signs);
    }
    let scale_bits = (q2_1 >> 28) & 0xF;
    println!("    scale_bits = {}", scale_bits);

    // Show expected range
    let d = f16::from_bits(blocks[0].d).to_f32();
    println!("\nExpected values:");
    println!("  Global scale d = {}", d);
    println!(
        "  Sub-block scale = d * (2*{}+1) = {}",
        scale_bits,
        d * (2.0 * scale_bits as f32 + 1.0)
    );
}
