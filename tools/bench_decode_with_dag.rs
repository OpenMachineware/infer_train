// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark decode performance with DAG fusion

use std::time::Instant;

fn main() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string()
    });

    println!("=== Decode Performance with DAG Fusion ===\n");
    println!("Model: {}", model_path);

    // Run in isolated thread for Metal thread safety
    let result = std::thread::spawn(move || {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device().clone();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        println!(
            "Layers: {}, Hidden: {}",
            ctx.model.layers.len(),
            ctx.model.config.hidden_size
        );

        // Check if DAG is built
        if ctx.model.optimized_dag.is_some() {
            println!("DAG: Enabled (fusion active)");
        } else {
            println!("DAG: Not available");
        }

        // Create KV cache
        let mut kv = KVCacheGPU::new(&device, &ctx.model.config, 2048);

        // Prefill first
        println!("\n=== Prefill 64 tokens ===");
        let prefill_tokens: Vec<u32> = (1..=64).collect();
        let start = Instant::now();
        let result = ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv,
            &gpu_ctx,
        );
        let prefill_time = start.elapsed();
        let has_nan = result
            .as_ref()
            .map(|logits| logits.iter().any(|&x| x.is_nan()))
            .unwrap_or(true);
        println!("Prefill time: {:?}", prefill_time);
        println!("Prefill t/s: {:.1}", 64.0 / prefill_time.as_secs_f64());
        println!("NaN in output: {}", has_nan);

        // Test decode
        println!("\n=== Decode Performance ===");
        let num_tokens = 50;
        let mut total_time = std::time::Duration::ZERO;
        let mut nan_count = 0;

        for i in 0..num_tokens {
            let token = (65 + i) as u32;
            let start = Instant::now();
            let result = ctx.model.forward_gpu_native_batched(
                &[token],
                &mut kv,
                &gpu_ctx,
            );
            let elapsed = start.elapsed();
            total_time += elapsed;

            let has_nan = result
                .as_ref()
                .map(|logits| {
                    logits.iter().any(|&x| x.is_nan() || x.is_infinite())
                })
                .unwrap_or(true);

            if has_nan {
                nan_count += 1;
            }

            if i < 3 || i >= num_tokens - 3 {
                println!("Token {}: {:?} (nan: {})", i + 1, elapsed, has_nan);
            }
        }

        let avg_time = total_time / num_tokens;
        let tps = num_tokens as f64 / total_time.as_secs_f64();

        println!("\n=== Summary ===");
        println!("Total time: {:?}", total_time);
        println!("Average per token: {:?}", avg_time);
        println!("Throughput: {:.1} t/s", tps);
        println!("NaN outputs: {}/{}", nan_count, num_tokens);

        tps
    })
    .join()
    .unwrap();

    println!("\n=== Final: {:.1} t/s ===", result);
}
