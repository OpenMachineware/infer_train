// SPDX-License-Identifier: AGPL-3.0-or-later
// Test to compare gemv_impl vs gemv_parallel_row

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use infer_train::model::LlamaModel;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model from {}...", model_path);

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    println!("Model config:");
    println!("  Hidden size: {}", ctx.config.hidden_size);
    println!("  Layers: {}", ctx.config.n_layers);
    println!("  Heads: {}", ctx.config.n_heads);

    // Get first layer
    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;

    // Create a test input
    let test_input: Vec<f32> =
        (0..hidden_size).map(|i| (i as f32 * 0.01).sin()).collect();

    // Test Q projection
    if let Some(ref wq) = layer.wq {
        println!("\n=== Testing Q projection ===");
        println!("Weight dtype: {:?}", wq.dtype);
        println!("Weight shape: {:?}", wq.shape);
        println!("Input size: {}", test_input.len());
        println!("Output size: {}", wq.shape[1]);

        let out_size = wq.shape[1];

        // Test gemv_impl
        let result_impl = ctx.model.gemv_impl(&test_input, wq, out_size);
        if let Some(ref r) = result_impl {
            println!("\ngemv_impl result:");
            println!("  Len: {}", r.len());
            println!("  First 5: {:?}", &r[..5.min(r.len())]);
            println!("  Last 5: {:?}", &r[r.len().saturating_sub(5)..]);
            println!("  Non-zero: {}", r.iter().filter(|&&x| x != 0.0).count());
        } else {
            println!("gemv_impl returned None!");
        }

        // Test gemv_parallel_row (single thread for comparison)
        let mut result_parallel = vec![0.0f32; out_size];
        ctx.model.gemv_parallel_row(
            0,
            1,
            &test_input,
            wq,
            &mut result_parallel,
            test_input.len(),
            out_size,
        );

        println!("\ngemv_parallel_row result (single thread):");
        println!("  Len: {}", result_parallel.len());
        println!(
            "  First 5: {:?}",
            &result_parallel[..5.min(result_parallel.len())]
        );
        println!(
            "  Last 5: {:?}",
            &result_parallel[result_parallel.len().saturating_sub(5)..]
        );
        println!(
            "  Non-zero: {}",
            result_parallel.iter().filter(|&&x| x != 0.0).count()
        );

        // Compare
        if let Some(impl_result) = result_impl {
            let mut max_diff = 0.0f32;
            let mut avg_diff = 0.0f32;
            let mut num_diffs = 0;
            for i in 0..impl_result.len().min(result_parallel.len()) {
                let diff = (impl_result[i] - result_parallel[i]).abs();
                if diff > 0.0 {
                    max_diff = max_diff.max(diff);
                    avg_diff += diff;
                    num_diffs += 1;
                }
            }
            if num_diffs > 0 {
                avg_diff /= num_diffs as f32;
            }

            println!("\nComparison:");
            println!("  Max diff: {}", max_diff);
            println!("  Avg diff (non-zero): {}", avg_diff);
            println!(
                "  Number of differences: {}/{}",
                num_diffs,
                impl_result.len()
            );

            if max_diff > 0.001 {
                println!("  ✗ Results differ significantly!");

                // Find first difference
                for i in 0..impl_result.len().min(result_parallel.len()).min(20)
                {
                    if (impl_result[i] - result_parallel[i]).abs() > 0.001 {
                        println!(
                            "    [{}] impl={} parallel={}",
                            i, impl_result[i], result_parallel[i]
                        );
                    }
                }
            } else {
                println!("  ✓ Results match!");
            }
        }
    }
}
