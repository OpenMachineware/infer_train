// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GEMM template with batch > 1

use infer_train::quant::types::BlockQ4K;
use metal::Device;

fn main() {
    let _device = Device::system_default().expect("No Metal device found");

    // Test dimensions similar to the model
    let m = 4096; // q_out_size
    let k = 5120; // hidden_size
    let n = 3; // batch_size

    println!("Testing GEMM: M={}, N={}, K={}", m, n, k);

    // Create mock Q4_K weights
    let num_blocks = k / 256;
    let mut weights = vec![0u8; num_blocks * std::mem::size_of::<BlockQ4K>()];

    // Initialize weights with simple values
    for i in 0..num_blocks {
        let offset = i * std::mem::size_of::<BlockQ4K>();
        // Set d = 0.01 in FP16 = 0x2E9D
        weights[offset] = 0x9D;
        weights[offset + 1] = 0x2E;
        // Set dmin = 0
        weights[offset + 2] = 0x00;
        weights[offset + 3] = 0x00;
        // Set scales and qs
        for j in 4..144 {
            weights[offset + j] = 0;
        }
    }

    // Create input: 3 tokens, each with hidden_size=5120
    let input: Vec<f32> = vec![0.1; k * n];

    println!(
        "Input: {} values, range=[{:.4}, {:.4}]",
        input.len(),
        input.iter().cloned().fold(f32::INFINITY, f32::min),
        input.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Create Metal context and run GEMM
    let ctx = infer_train::quant::vec_dot::gpu::metal::MetalContext::new()
        .expect("Failed to create Metal context");

    let weights_slice: &[BlockQ4K] = bytemuck::cast_slice(&weights);
    let result = ctx.gemm_template_q4_k_f32(m, n, k, weights_slice, &input);

    match result {
        Ok(output) => {
            println!("Output: {} values", output.len());
            println!(
                "  nan: {}",
                output.iter().filter(|&&v| v.is_nan()).count()
            );
            println!(
                "  range: [{:.4}, {:.4}]",
                output.iter().cloned().fold(f32::INFINITY, f32::min),
                output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );

            // Check first few values
            println!(
                "  First 10 values: {:?}",
                &output[0..10.min(output.len())]
            );
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}
