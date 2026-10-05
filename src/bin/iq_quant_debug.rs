// Debug IQ quantization step by step
// Compare Rust vs llama.cpp output for a single block

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

const QK_K: usize = 256;

fn main() {
    println!("=== IQ Quantization Debug ===\n");

    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate simple test data
    let x: Vec<f32> =
        (0..QK_K).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let qw: Vec<f32> = vec![1.0; QK_K];

    // Test IQ1_S
    println!("Testing IQ1_S quantization...");
    let blocks = quantize_iq1_s(&x, Some(&qw));
    println!("Generated {} blocks", blocks.len());
    println!("First block:");
    println!("  d = {} (f16)", f16::from_bits(blocks[0].d).to_f32());
    println!("  First 4 qs bytes: {:?}", &blocks[0].qs[0..4]);
    println!("  First 2 qh values: {:?}", &blocks[0].qh[0..2]);

    // Test IQ2_XXS
    println!("\nTesting IQ2_XXS quantization...");
    let blocks = quantize_iq2_xxs(&x, Some(&qw));
    println!("Generated {} blocks", blocks.len());
    println!("First block:");
    println!("  d = {} (f16)", f16::from_bits(blocks[0].d).to_f32());
    println!("  First 4 qs bytes: {:?}", &blocks[0].qs[0..4]);

    // Test IQ4_NL (should pass)
    println!("\nTesting IQ4_NL quantization...");
    let blocks = quantize_iq4_nl(&x);
    println!("Generated {} blocks", blocks.len());
    println!("First block:");
    println!("  d = {} (f16)", f16::from_bits(blocks[0].d).to_f32());
    println!("  First 4 qs bytes: {:?}", &blocks[0].qs[0..4]);

    // Compare with expected ranges
    println!("\n=== Expected Value Ranges ===");
    println!("Scale (d) should be ~0.01-0.1 for normalized weights");
    println!("qs/qh should be non-zero for non-trivial input");
}
