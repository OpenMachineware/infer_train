// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug GEMM NaN issue

use infer_train::quant::types::{BlockQ4K, BlockQ6K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    println!("=== GEMM Debug ===\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");

        // Test 1: Simple Q6_K GEMM
        let m = 2048; // Output size
        let n = 2; // Batch size
        let k = 1024; // Input size

        println!("Test 1: Q6_K GEMM (M={}, N={}, K={})", m, n, k);

        // Create random input
        let input: Vec<f32> =
            (0..n * k).map(|i| (i as f32 * 0.001 - 1.0)).collect();

        // Create Q6_K weights (all zeros for simplicity)
        let n_blocks = (k / 256) * m;
        let mut weights: Vec<BlockQ6K> =
            vec![unsafe { std::mem::zeroed() }; n_blocks];

        // Set scale to 1.0 for first few blocks
        for block in &mut weights {
            block.d = half::f16::from_f32(0.1).to_bits();
            block.scales[0] = 1;
        }

        println!(
            "  Input range: [{:.3}, {:.3}]",
            input.iter().cloned().fold(f32::INFINITY, f32::min),
            input.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );

        match gpu_ctx.gemm_template_q6_k_f32(m, n, k, &weights, &input) {
            Ok(output) => {
                let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
                let num_inf =
                    output.iter().filter(|&&v| v.is_infinite()).count();
                let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
                let max =
                    output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                println!("  Output: {} values", output.len());
                println!("  NaN: {}, Inf: {}", num_nan, num_inf);
                println!("  Range: [{:.3}, {:.3}]", min, max);
                println!("  First 5: {:?}", &output[..5.min(output.len())]);
            }
            Err(e) => {
                println!("  Error: {}", e);
            }
        }

        // Test 2: Smaller dimensions
        println!("\nTest 2: Q6_K GEMM (M=64, N=1, K=256)");
        let m2 = 64;
        let n2 = 1;
        let k2 = 256;

        let input2: Vec<f32> = vec![0.1; n2 * k2];
        let weights2: Vec<BlockQ6K> =
            vec![unsafe { std::mem::zeroed() }; (k2 / 256) * m2];

        match gpu_ctx.gemm_template_q6_k_f32(m2, n2, k2, &weights2, &input2) {
            Ok(output) => {
                let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
                println!(
                    "  NaN: {}, first 3: {:?}",
                    num_nan,
                    &output[..3.min(output.len())]
                );
            }
            Err(e) => println!("  Error: {}", e),
        }

        // Test 3: Q4_K
        println!("\nTest 3: Q4_K GEMM (M=2048, N=2, K=1024)");
        let weights_q4: Vec<BlockQ4K> =
            vec![unsafe { std::mem::zeroed() }; (k / 256) * m];

        match gpu_ctx.gemm_template_q4_k_f32(m, n, k, &weights_q4, &input) {
            Ok(output) => {
                let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
                println!("  NaN: {}", num_nan);
            }
            Err(e) => println!("  Error: {}", e),
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU support not available");
    }
}
