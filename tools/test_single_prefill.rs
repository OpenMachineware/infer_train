// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Single prefill test to debug NaN

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

        println!("=== Single Token Prefill ===");
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();

        let mut kv_cache = KVCacheGPU::new(device, &config, 2048);
        let tokens = vec![1u32];

        let start = Instant::now();
        let result = ctx.model.forward_gpu_native_batched(
            &tokens,
            &mut kv_cache,
            &gpu_ctx,
        );
        let elapsed = start.elapsed();

        match result {
            Some(logits) => {
                let nan_count = logits.iter().filter(|v| v.is_nan()).count();
                println!("Result: {} logits, {} NaN", logits.len(), nan_count);
                println!("First 5: {:?}", &logits[..5.min(logits.len())]);
            }
            None => println!("Forward pass failed"),
        }
        println!("Time: {:?}", elapsed);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
