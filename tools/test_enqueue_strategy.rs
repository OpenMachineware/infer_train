// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test different Metal execution strategies

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).expect("Failed to load model");
        let gpu_ctx = MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();

        let mut kv_cache = KVCacheGPU::new(&device, &ctx.config, ctx.config.max_position_embeddings);

        println!("=== Testing Metal Execution Strategies ===\n");

        // Prefill
        println!("Prefilling...");
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(&prefill_tokens, &mut kv_cache, &gpu_ctx);

        println!("\n=== Strategy 1: Current (enqueue + commit + wait) ===");
        let iterations = 30;
        let mut times: Vec<f64> = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
            times.push(start.elapsed().as_micros() as f64);
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[iterations / 2];
        let tps = 1_000_000.0 / median;

        println!("Median time: {:.1} µs", median);
        println!("Throughput: {:.1} t/s", tps);

        println!("\n=== Analysis ===");
        println!("Current strategy: enqueue() + commit() + wait_until_completed()");
        println!("llama.cpp strategy: enqueue() + commit() + NO wait (async)");
        println!("\nHypothesis:");
        println!("- Removing wait_until_completed() may allow CPU-GPU overlap");
        println!("- But we need to ensure KV cache advance happens after GPU completes");
        println!("\nNext steps:");
        println!("1. Modify DagExecutor to save command buffer reference");
        println!("2. Wait on previous command buffer at start of next forward");
        println!("3. This allows CPU to continue while GPU executes");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}