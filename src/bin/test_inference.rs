// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// End-to-end inference test

use infer_train::inference::{dequant_to_f32, InferenceContext};
use infer_train::model::tensor::DataType;
use infer_train::ops::sampling::SamplingStrategy;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    // Load model
    let mut ctx = match InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    ) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ Model loaded");

    let config = ctx.config.clone();
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Vocab size: {}", config.vocab_size);

    // Test embedding dequantization
    println!("\n=== Testing Embedding ===");
    if let Some(ref embed) = ctx.model.tok_embeddings {
        let dtype = embed.dtype;
        println!("  Embedding dtype: {:?}", dtype);
        println!("  Embedding shape: {:?}", embed.shape);

        let bytes = embed.bytes();
        println!("  Total bytes: {}", bytes.len());

        // Calculate bytes per row based on shape
        // Shape [1536, 151936] means 151936 rows of 1536 elements each
        let hidden_size = config.hidden_size;
        let rows = if embed.shape[0] == hidden_size {
            embed.shape[1]
        } else {
            embed.shape[0]
        };
        let cols = if embed.shape[0] == hidden_size {
            hidden_size
        } else {
            embed.shape[1]
        };

        println!("  Interpretation: {} rows x {} cols", rows, cols);

        let bytes_per_row = dtype.bytes_for_elems(cols);
        println!("  Bytes per row: {}", bytes_per_row);

        // Expected: rows * bytes_per_row
        let expected_bytes = rows * bytes_per_row;
        println!("  Expected total bytes: {}", expected_bytes);
        println!("  Actual total bytes: {}", bytes.len());

        // Print first few bytes of first row
        println!("\n  First 20 bytes of row 0:");
        for i in 0..20.min(bytes.len()) {
            print!("{:02x} ", bytes[i]);
        }
        println!();

        if dtype == DataType::Q5_K {
            // Try to manually interpret the first block
            println!("\n  Manual Q5_K block interpretation:");

            // BlockQ5K: d: u16, dmin: u16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]
            // For 1536 elements, we need ceil(1536/256) = 6 blocks

            let blocks_needed = cols.div_ceil(256);
            println!("    Blocks needed per row: {}", blocks_needed);

            // Read first block
            let block_bytes = &bytes[0..176];

            // d and dmin
            let d_bits = u16::from_le_bytes([block_bytes[0], block_bytes[1]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            let dmin_bits =
                u16::from_le_bytes([block_bytes[2], block_bytes[3]]);
            let dmin = half::f16::from_bits(dmin_bits).to_f32();

            println!("    Block 0: d={}, dmin={}", d, dmin);
            println!("    Scales: {:02x?}", &block_bytes[4..16]);

            // Dequantize first row
            let data = &bytes[0..bytes_per_row];
            let dequant = dequant_to_f32(data, dtype, cols);

            let max_val =
                dequant.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = dequant.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_val: f32 =
                dequant.iter().sum::<f32>() / dequant.len() as f32;

            println!("\n  Dequantified row 0:");
            println!(
                "    Min: {:.4}, Max: {:.4}, Mean: {:.4}",
                min_val, max_val, mean_val
            );
            println!("    First 10: {:?}", &dequant[..10.min(dequant.len())]);
        }
    }

    // Test generation
    println!("\n=== Testing Generation ===");
    let generated =
        ctx.generate(&[1, 2, 3, 4, 5], 5, &SamplingStrategy::Greedy);
    println!("✓ Generated {} tokens: {:?}", generated.len(), generated);

    println!("\n=== Done ===");
}
