// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark GPU inference throughput with thread isolation

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [num_tokens]", args[0]);
        eprintln!("  num_tokens: number of tokens for prefill (default: 128)");
        std::process::exit(1);
    }
    let model_path = args[1].clone();
    let num_tokens: usize =
        args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        println!("=== GPU Inference Benchmark ===");
        println!("Model: {}", model_path);
        println!("Prefill tokens: {}", num_tokens);
        println!();

        // Warmup
        println!("Warmup...");
        {
            let model_path = model_path.clone();
            std::thread::spawn(move || {
                let gpu_ctx = MetalContext::new().unwrap();
                let device = gpu_ctx.device();
                let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                    &model_path,
                    gpu_ctx.clone(),
                )
                .unwrap();
                let config = ctx.model.config.clone();
                let mut kv = KVCacheGPU::new(device, &config, 2048);
                let tokens: Vec<u32> = (1..=10).collect();
                let _ = ctx
                    .model
                    .forward_gpu_native_batched(&tokens, &mut kv, &gpu_ctx);
            })
            .join()
            .unwrap();
        }

        // Prefill benchmark - measure forward time only (model pre-loaded)
        println!("\n=== Prefill ({} tokens) ===", num_tokens);
        let tokens: Vec<u32> = (1..=num_tokens as u32).collect();

        // Pre-load model to avoid measuring loading time
        let gpu_ctx_warm = MetalContext::new().unwrap();
        let device_warm = gpu_ctx_warm.device();
        let ctx_warm = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx_warm.clone(),
        )
        .unwrap();
        let config_warm = ctx_warm.model.config.clone();
        drop(ctx_warm);
        drop(gpu_ctx_warm);

        let mut prefill_times = Vec::new();
        for i in 0..5 {
            let model_path = model_path.clone();
            let tokens = tokens.clone();
            let start = std::time::Instant::now();
            let result = std::thread::spawn(move || {
                let gpu_ctx = MetalContext::new().unwrap();
                let device = gpu_ctx.device();
                let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                    &model_path,
                    gpu_ctx.clone(),
                )
                .unwrap();
                let config = ctx.model.config.clone();
                let mut kv = KVCacheGPU::new(device, &config, 2048);
                ctx.model.forward_gpu_native_batched(&tokens, &mut kv, &gpu_ctx)
            })
            .join()
            .unwrap();
            let elapsed = start.elapsed();
            prefill_times.push(elapsed);
            println!(
                "  Run {}: {:?} (has_nan: {})",
                i + 1,
                elapsed,
                result.is_none()
            );
        }

        let avg_prefill = prefill_times.iter().sum::<std::time::Duration>()
            / prefill_times.len() as u32;
        let prefill_tps = num_tokens as f64 / avg_prefill.as_secs_f64();
        println!("\n  Average: {:?}", avg_prefill);
        println!("  Throughput: {:.1} tokens/sec", prefill_tps);

        // Decode benchmark (single token)
        println!("\n=== Decode (single token) ===");
        let mut decode_times = Vec::new();
        for i in 0..20 {
            let model_path = model_path.clone();
            let start = std::time::Instant::now();
            std::thread::spawn(move || {
                let gpu_ctx = MetalContext::new().unwrap();
                let device = gpu_ctx.device();
                let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                    &model_path,
                    gpu_ctx.clone(),
                )
                .unwrap();
                let config = ctx.model.config.clone();
                let mut kv = KVCacheGPU::new(device, &config, 2048);
                // Prefill first
                let _ = ctx.model.forward_gpu_native_batched(
                    &[1u32],
                    &mut kv,
                    &gpu_ctx,
                );
                // Then decode
                ctx.model.forward_gpu_native_batched(&[2u32], &mut kv, &gpu_ctx)
            })
            .join()
            .unwrap();
            let elapsed = start.elapsed();
            decode_times.push(elapsed);
            if i < 5 || i >= 15 {
                println!("  Run {}: {:?}", i + 1, elapsed);
            }
        }

        let avg_decode = decode_times.iter().sum::<std::time::Duration>()
            / decode_times.len() as u32;
        let decode_tps = 1.0 / avg_decode.as_secs_f64();
        println!("\n  Average: {:?}", avg_decode);
        println!("  Throughput: {:.1} tokens/sec", decode_tps);

        println!("\n=== Summary ===");
        println!(
            "Prefill: {:.1} tokens/sec ({} tokens in {:?})",
            prefill_tps, num_tokens, avg_prefill
        );
        println!(
            "Decode:  {:.1} tokens/sec (1 token in {:?})",
            decode_tps, avg_decode
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU benchmark requires aarch64+metal");
    }
}
