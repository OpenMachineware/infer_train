// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile decode performance breakdown by operation

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        println!("=== Decode Performance Breakdown ===");
        println!("Model: {}\n", model_path);

        let result = std::thread::spawn(move || {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                &model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            let config = ctx.model.config.clone();
            let n_layers = config.n_layers;
            let hidden_size = config.hidden_size;

            println!("Layers: {}, Hidden size: {}", n_layers, hidden_size);

            let mut kv = KVCacheGPU::new(device, &config, 2048);

            // Prefill first
            println!("\n=== Prefill 10 tokens ===");
            let prefill_tokens: Vec<u32> = (1..=10).collect();
            let start = Instant::now();
            let _ = ctx.model.forward_gpu_native_batched(
                &prefill_tokens,
                &mut kv,
                &gpu_ctx,
            );
            println!("Prefill time: {:?}", start.elapsed());

            // Profile decode tokens
            println!("\n=== Decode tokens (detailed timing) ===");
            for i in 0..5 {
                let token = (11 + i) as u32;
                println!("\n--- Token {} ---", i + 1);

                let start = Instant::now();
                let result = ctx.model.forward_gpu_native_batched(
                    &[token],
                    &mut kv,
                    &gpu_ctx,
                );
                let elapsed = start.elapsed();
                let has_nan = result.is_none();

                println!("  Total: {:?} (nan: {})", elapsed, has_nan);
            }

            // Compare with llama.cpp-like decode
            println!("\n=== Throughput Summary ===");
            let mut total_time = std::time::Duration::ZERO;
            let num_tokens = 20;

            for i in 0..num_tokens {
                let token = (11 + i) as u32;
                let start = Instant::now();
                let _ = ctx.model.forward_gpu_native_batched(
                    &[token],
                    &mut kv,
                    &gpu_ctx,
                );
                total_time += start.elapsed();
            }

            let avg_time = total_time / num_tokens;
            let tps = 1.0 / avg_time.as_secs_f64();
            println!("Average decode time: {:?}", avg_time);
            println!("Throughput: {:.1} t/s", tps);

            tps
        })
        .join()
        .unwrap();

        println!("\n=== Final: {:.1} t/s ===", result);
    }
}
