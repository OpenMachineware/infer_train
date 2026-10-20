// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test decode performance with DAG executor

use std::time::Instant;

fn main() {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    println!("=== Decode Performance with DAG Executor ===\n");
    println!("Model: {}", model_path);

    // Run in isolated thread for Metal thread safety
    let result = std::thread::spawn(move || {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device().clone();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(&model_path, gpu_ctx.clone()).unwrap();

        println!(
            "Layers: {}, Hidden: {}",
            ctx.model.layers.len(),
            ctx.model.config.hidden_size
        );

        // Create KV cache
        let mut kv = KVCacheGPU::new(&device, &ctx.model.config, 2048);

        // Prefill first
        println!("\n=== Prefill 64 tokens ===");
        let prefill_tokens: Vec<u32> = (1..=64).collect();
        let start = Instant::now();
        let _ = ctx.model.forward_gpu_native_batched(&prefill_tokens, &mut kv, &gpu_ctx);
        println!("Prefill time: {:?}", start.elapsed());

        // Test decode
        println!("\n=== Decode Performance ===");
        let num_tokens = 50;
        let mut total_time = std::time::Duration::ZERO;

        for i in 0..num_tokens {
            let token = (65 + i) as u32;
            let start = Instant::now();
            let result = ctx.model.forward_gpu_native_batched(&[token], &mut kv, &gpu_ctx);
            let elapsed = start.elapsed();
            total_time += elapsed;

            let has_nan = result
                .as_ref()
                .map(|logits| logits.iter().any(|&x| x.is_nan() || x.is_infinite()))
                .unwrap_or(true);

            if i < 5 || i >= num_tokens - 5 {
                println!("Token {}: {:?} (nan: {})", i + 1, elapsed, has_nan);
            }
        }

        let avg_time = total_time / num_tokens;
        let tps = num_tokens as f64 / total_time.as_secs_f64();

        println!("\n=== Summary ===");
        println!("Total time: {:?}", total_time);
        println!("Average per token: {:?}", avg_time);
        println!("Throughput: {:.1} t/s", tps);

        tps
    })
    .join()
    .unwrap();

    println!("\nFinal: {:.1} t/s", result);
}
