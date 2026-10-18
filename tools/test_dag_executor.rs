// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test DAG executor with single command buffer optimization

use infer_train::inference::{Device, InferenceContext};
use std::time::Instant;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::dag::DagExecutor;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    let tokens = vec![1u32];

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // Test: Compare old vs new implementation
        println!("\n=== Test: Old Implementation ===\n");

        // 1. Old implementation (per-layer command buffers)
        println!("1. Old implementation (forward_gpu_native_batched):");
        let start = Instant::now();
        let result_old = ctx.model.forward_gpu_native_batched(
            &tokens,
            ctx.gpu_kv_cache.as_mut().unwrap(),
            ctx.gpu_ctx.as_ref().unwrap(),
        );
        let old_duration = start.elapsed();
        println!(
            "   Result: {} logits, time: {:?}",
            result_old.as_ref().map(|r| r.len()).unwrap_or(0),
            old_duration
        );

        // Check for NaN in old result
        if let Some(ref old) = result_old {
            let nan_count = old.iter().filter(|&&v| v.is_nan()).count();
            println!("   NaN count: {}", nan_count);
        }

        // Create fresh context for new implementation
        println!("\n=== Test: New Implementation (DagExecutor) ===\n");
        let mut ctx2 = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");

        // 2. New implementation
        println!("2. New implementation (DagExecutor::forward_gpu_graph):");
        let start = Instant::now();
        let result_new = DagExecutor::forward_gpu_graph(
            &ctx2.model,
            &tokens,
            ctx2.gpu_kv_cache.as_mut().unwrap(),
            ctx2.gpu_ctx.as_ref().unwrap(),
        );
        let new_duration = start.elapsed();
        println!(
            "   Result: {} logits, time: {:?}",
            result_new.as_ref().map(|r| r.len()).unwrap_or(0),
            new_duration
        );

        // Compare
        println!("\n=== Performance Comparison ===");
        if old_duration.as_nanos() > 0 {
            let speedup =
                old_duration.as_secs_f64() / new_duration.as_secs_f64();
            println!(
                "Speedup: {:.2}x (old: {:?}, new: {:?})",
                speedup, old_duration, new_duration
            );
        }

        // Check correctness
        if let (Some(old), Some(new)) = (&result_old, &result_new) {
            // Check for NaN in new result
            let nan_count = new.iter().filter(|&&v| v.is_nan()).count();
            println!("NaN count in new result: {}", nan_count);

            if nan_count > 0 {
                println!("ERROR: New implementation produces NaN!");
                println!("First 5 old values: {:?}", &old[..5.min(old.len())]);
                println!("First 5 new values: {:?}", &new[..5.min(new.len())]);
                return;
            }

            let max_diff = old
                .iter()
                .zip(new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("Max difference: {}", max_diff);

            // Check first few values
            println!("\nFirst 5 logits comparison:");
            for i in 0..5.min(old.len()) {
                println!("  [{}] old: {:.4}, new: {:.4}", i, old[i], new[i]);
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU tests require aarch64 with metal feature");
    }
}
