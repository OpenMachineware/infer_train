// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test using the same InferenceContext twice (no drop/recreate)

use std::time::Instant;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();

        // Create context ONCE
        println!("=== Creating InferenceContext ===");
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();

        // Test 1: First prefill
        println!("\n=== Test 1: First Prefill ===");
        let mut kv1 = KVCacheGPU::new(device, &config, 2048);
        let result1 =
            ctx.model.forward_gpu_native_batched(&[1u32], &mut kv1, &gpu_ctx);
        println!(
            "Test 1 result: {} NaN",
            result1
                .as_ref()
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(0)
        );

        // Test 2: Second prefill with SAME context, NEW KV cache
        println!(
            "\n=== Test 2: Second Prefill (same context, new KV cache) ==="
        );
        let mut kv2 = KVCacheGPU::new(device, &config, 2048);
        let result2 =
            ctx.model.forward_gpu_native_batched(&[1u32], &mut kv2, &gpu_ctx);
        println!(
            "Test 2 result: {} NaN",
            result2
                .as_ref()
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(0)
        );

        // Test 3: Third prefill with SAME context and SAME KV cache (continuing)
        println!(
            "\n=== Test 3: Third Prefill (same context, same KV cache - continuing) ==="
        );
        let result3 =
            ctx.model.forward_gpu_native_batched(&[2u32], &mut kv2, &gpu_ctx);
        println!(
            "Test 3 result: {} NaN",
            result3
                .as_ref()
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(0)
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
