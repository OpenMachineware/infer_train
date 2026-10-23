// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed timing comparison between our implementation and llama.cpp

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("=== Detailed Decode Timing Analysis ===\n");
        println!("Model: {}\n", model_path);

        // Load model and create context
        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device().clone();
        let mut ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        println!("Model config:");
        println!("  Layers: {}", ctx.model.layers.len());
        println!("  Hidden size: {}", ctx.model.config.hidden_size);
        println!("  Heads: {}", ctx.model.config.n_heads);
        println!("  KV heads: {}", ctx.model.config.n_kv_heads);
        println!();

        // Create KV cache
        let mut kv = KVCacheGPU::new(&device, &ctx.model.config, 2048);

        // Prefill
        println!("=== Prefill ===");
        let prefill_tokens: Vec<u32> = (1..=10).collect();
        let start = Instant::now();
        ctx.model
            .forward_gpu_native_batched(&prefill_tokens, &mut kv, &gpu_ctx)
            .unwrap();
        let prefill_time = start.elapsed();
        println!("Time: {:?}", prefill_time);
        println!("Tokens/s: {:.1}\n", 10.0 / prefill_time.as_secs_f64());

        // Decode with detailed timing
        println!("=== Decode Timing ===");
        let mut total_time = std::time::Duration::ZERO;
        let mut min_time = std::time::Duration::MAX;
        let mut max_time = std::time::Duration::ZERO;

        for i in 0..20 {
            let token = (65 + i) as u32;
            let start = Instant::now();
            ctx.model
                .forward_gpu_native_batched(&[token], &mut kv, &gpu_ctx)
                .unwrap();
            let elapsed = start.elapsed();
            total_time += elapsed;
            min_time = min_time.min(elapsed);
            max_time = max_time.max(elapsed);

            if i < 5 || i >= 15 {
                println!(
                    "Token {}: {:?} ({:.1} t/s)",
                    i + 1,
                    elapsed,
                    1.0 / elapsed.as_secs_f64()
                );
            } else if i == 5 {
                println!("...");
            }
        }

        let avg_time = total_time / 20;
        let tps = 20.0 / total_time.as_secs_f64();

        println!("\n=== Summary ===");
        println!("Min: {:?}", min_time);
        println!("Max: {:?}", max_time);
        println!("Avg: {:?}", avg_time);
        println!("Tokens/s: {:.1}", tps);
        println!("Variance: {:?}", max_time - min_time);
        println!("\nCompare with llama.cpp benchmark results");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
