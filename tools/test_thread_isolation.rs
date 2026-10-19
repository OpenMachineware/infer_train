// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test loading fresh model instances for each forward pass

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::thread;

        // Test 1: First process in its own thread
        println!("=== Test 1: Thread 1 ===");
        let h1 = thread::spawn(move || {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            let config = ctx.model.config.clone();
            let mut kv = KVCacheGPU::new(device, &config, 2048);
            let result = ctx.model.forward_gpu_native_batched(
                &[1u32],
                &mut kv,
                &gpu_ctx,
            );
            result
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(999999)
        });
        let r1 = h1.join().unwrap();
        println!("Test 1 result: {} NaN", r1);

        // Give some time for cleanup
        thread::sleep(std::time::Duration::from_millis(500));

        // Test 2: Second process in its own thread
        println!("\n=== Test 2: Thread 2 ===");
        let h2 = thread::spawn(move || {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            let config = ctx.model.config.clone();
            let mut kv = KVCacheGPU::new(device, &config, 2048);
            let result = ctx.model.forward_gpu_native_batched(
                &[1u32],
                &mut kv,
                &gpu_ctx,
            );
            result
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(999999)
        });
        let r2 = h2.join().unwrap();
        println!("Test 2 result: {} NaN", r2);

        // Give some time for cleanup
        thread::sleep(std::time::Duration::from_millis(500));

        // Test 3: Third process in its own thread
        println!("\n=== Test 3: Thread 3 ===");
        let h3 = thread::spawn(move || {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            let config = ctx.model.config.clone();
            let mut kv = KVCacheGPU::new(device, &config, 2048);
            let result = ctx.model.forward_gpu_native_batched(
                &[1u32],
                &mut kv,
                &gpu_ctx,
            );
            result
                .map(|r| r.iter().filter(|v| v.is_nan()).count())
                .unwrap_or(999999)
        });
        let r3 = h3.join().unwrap();
        println!("Test 3 result: {} NaN", r3);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
