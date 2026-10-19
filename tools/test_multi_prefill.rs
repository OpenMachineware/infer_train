// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test multiple prefill calls

use std::time::Instant;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();

        // Test 1: First prefill
        println!("=== Test 1: First Prefill ===");
        let ctx1 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx1.model.config.clone();
        let mut kv1 = KVCacheGPU::new(device, &config, 2048);

        let result1 =
            ctx1.model.forward_gpu_native_batched(&[1u32], &mut kv1, &gpu_ctx);
        println!(
            "Test 1 result: {} NaN",
            result1
                .as_ref()
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(0)
        );

        drop(ctx1);
        drop(kv1);

        // Test 2: Second prefill (new context)
        println!("\n=== Test 2: Second Prefill ===");
        let ctx2 = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let mut kv2 = KVCacheGPU::new(device, &config, 2048);

        // Check output tensor state
        if let Some(output) = &ctx2.model.output {
            println!(
                "Test 2 output tensor: dtype={:?}, shape={:?}",
                output.dtype, output.shape
            );
        }

        let result2 =
            ctx2.model.forward_gpu_native_batched(&[1u32], &mut kv2, &gpu_ctx);
        println!(
            "Test 2 result: {} NaN",
            result2
                .as_ref()
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(0)
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
