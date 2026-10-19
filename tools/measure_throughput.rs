// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Measure actual throughput (tokens per second)
// Uses shared MetalContext to avoid GPU state corruption

use std::time::Instant;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        // Create ONE shared MetalContext for all tests
        let shared_gpu_ctx = MetalContext::new().unwrap();
        let device = shared_gpu_ctx.device();

        // Test 1: Single token prefill
        println!("\n=== Single Token Prefill ===");
        let ctx1 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            shared_gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx1.model.config.clone();

        let mut kv_cache1 = KVCacheGPU::new(device, &config, 2048);
        let tokens = vec![1u32];
        let start = Instant::now();
        let result = ctx1.model.forward_gpu_native_batched(
            &tokens,
            &mut kv_cache1,
            &shared_gpu_ctx,
        );
        let elapsed = start.elapsed();
        println!(
            "Result: {} logits",
            result.as_ref().map(|r| r.len()).unwrap_or(0)
        );
        println!("Time: {:?}", elapsed);
        println!("Throughput: {:.1} t/s", 1000.0 / elapsed.as_millis() as f64);

        // Drop context but keep shared MetalContext
        drop(ctx1);
        drop(kv_cache1);

        // Test 2: Prefill 10 tokens (new context, same MetalContext)
        println!("\n=== Prefill 10 Tokens ===");
        let ctx2 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            shared_gpu_ctx.clone(),
        )
        .unwrap();
        let mut kv_cache2 = KVCacheGPU::new(device, &config, 2048);

        let tokens: Vec<u32> = (0..10).collect();
        let start = Instant::now();
        let result = ctx2.model.forward_gpu_native_batched(
            &tokens,
            &mut kv_cache2,
            &shared_gpu_ctx,
        );
        let elapsed = start.elapsed();
        println!(
            "Result: {} logits",
            result.as_ref().map(|r| r.len()).unwrap_or(0)
        );
        println!("Time: {:?}", elapsed);
        println!(
            "Throughput: {:.1} t/s",
            10.0 * 1000.0 / elapsed.as_millis() as f64
        );

        drop(ctx2);
        drop(kv_cache2);

        // Test 3: Decode (new context for clean state)
        println!("\n=== Decode (Position 1) ===");
        let ctx3 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            shared_gpu_ctx.clone(),
        )
        .unwrap();
        let mut kv_cache3 = KVCacheGPU::new(device, &config, 2048);

        // Prefill first
        ctx3.model.forward_gpu_native_batched(
            &[1u32],
            &mut kv_cache3,
            &shared_gpu_ctx,
        );

        // Decode
        let tokens = vec![2u32];
        let start = Instant::now();
        let result = ctx3.model.forward_gpu_native_batched(
            &tokens,
            &mut kv_cache3,
            &shared_gpu_ctx,
        );
        let elapsed = start.elapsed();
        println!(
            "Result: {} logits",
            result.as_ref().map(|r| r.len()).unwrap_or(0)
        );
        println!("Time: {:?}", elapsed);
        println!("Throughput: {:.1} t/s", 1000.0 / elapsed.as_millis() as f64);

        drop(ctx3);
        drop(kv_cache3);

        // Test 4: Multiple decode steps
        println!("\n=== Decode 5 Steps ===");
        let ctx4 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            shared_gpu_ctx.clone(),
        )
        .unwrap();
        let mut kv_cache4 = KVCacheGPU::new(device, &config, 2048);

        // Prefill first
        ctx4.model.forward_gpu_native_batched(
            &[1u32],
            &mut kv_cache4,
            &shared_gpu_ctx,
        );

        let start = Instant::now();
        for i in 0..5 {
            ctx4.model.forward_gpu_native_batched(
                &[(i + 2) as u32],
                &mut kv_cache4,
                &shared_gpu_ctx,
            );
        }
        let elapsed = start.elapsed();
        println!("Time: {:?}", elapsed);
        println!(
            "Throughput: {:.1} t/s",
            5.0 * 1000.0 / elapsed.as_millis() as f64
        );

        println!("\n=== Summary ===");
        println!("llama.cpp target (from roadmap):");
        println!("  Prefill: 347.2 t/s (2.88 ms/token)");
        println!("  Decode: 117.7 t/s (8.49 ms/token)");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
