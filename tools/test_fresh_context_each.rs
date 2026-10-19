// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test creating fresh InferenceContext for each forward pass (same thread)

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        // Test 1: First forward pass with fresh context
        println!("=== Test 1: Fresh InferenceContext ===");
        {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(model_path, gpu_ctx.clone()).unwrap();
            let config = ctx.model.config.clone();
            let mut kv = KVCacheGPU::new(device, &config, 2048);

            let result = ctx.model.forward_gpu_native_batched(&[1u32], &mut kv, &gpu_ctx);
            println!("Test 1 result: {} NaN", result.as_ref().map(|r| r.iter().filter(|v| v.is_nan()).count()).unwrap_or(0));
        }

        // Test 2: Second forward pass with NEW InferenceContext (same thread)
        println!("\n=== Test 2: Fresh InferenceContext (same thread) ===");
        {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(model_path, gpu_ctx.clone()).unwrap();
            let config = ctx.model.config.clone();
            let mut kv = KVCacheGPU::new(device, &config, 2048);

            let result = ctx.model.forward_gpu_native_batched(&[1u32], &mut kv, &gpu_ctx);
            println!("Test 2 result: {} NaN", result.as_ref().map(|r| r.iter().filter(|v| v.is_nan()).count()).unwrap_or(0));
        }

        // Test 3: Third forward pass with NEW InferenceContext (same thread)
        println!("\n=== Test 3: Fresh InferenceContext (same thread) ===");
        {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(model_path, gpu_ctx.clone()).unwrap();
            let config = ctx.model.config.clone();
            let mut kv = KVCacheGPU::new(device, &config, 2048);

            let result = ctx.model.forward_gpu_native_batched(&[1u32], &mut kv, &gpu_ctx);
            println!("Test 3 result: {} NaN", result.as_ref().map(|r| r.iter().filter(|v| v.is_nan()).count()).unwrap_or(0));
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal"))]
    println!("GPU tests require aarch64+metal");
}
