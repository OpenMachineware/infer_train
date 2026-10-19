// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark GPU inference with breakdown of model loading vs forward time

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

        println!("=== GPU Inference Benchmark ===");
        println!("Model: {}", model_path);
        println!();

        // 1. Measure model loading time
        println!("=== Model Loading Time ===");
        let mut load_times = Vec::new();
        for i in 0..3 {
            let model_path = model_path.clone();
            let start = std::time::Instant::now();
            std::thread::spawn(move || {
                let gpu_ctx = MetalContext::new().unwrap();
                let _ctx = InferenceContext::from_gguf_with_gpu_ctx(
                    &model_path,
                    gpu_ctx.clone(),
                )
                .unwrap();
            })
            .join()
            .unwrap();
            let elapsed = start.elapsed();
            load_times.push(elapsed);
            println!("  Run {}: {:?}", i + 1, elapsed);
        }
        let avg_load = load_times.iter().sum::<std::time::Duration>()
            / load_times.len() as u32;
        println!("\n  Average load time: {:?}", avg_load);

        // 2. Measure forward-only time (single token decode with fresh context)
        println!("\n=== Forward Pass Time (Decode, 1 token) ===");
        let mut forward_times = Vec::new();
        for i in 0..10 {
            let model_path = model_path.clone();
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
                // Just decode (after minimal prefill)
                ctx.model.forward_gpu_native_batched(
                    &[1u32, 2u32],
                    &mut kv,
                    &gpu_ctx,
                )
            })
            .join()
            .unwrap();
            let elapsed = start.elapsed();
            forward_times.push(elapsed);
            let has_nan = result.is_none();
            println!("  Run {}: {:?} (nan: {})", i + 1, elapsed, has_nan);
        }
        let avg_forward = forward_times.iter().sum::<std::time::Duration>()
            / forward_times.len() as u32;
        let forward_only = avg_forward - avg_load;
        println!("\n  Average total: {:?}", avg_forward);
        println!("  Estimated forward-only: {:?}", forward_only);
        println!(
            "  Throughput: {:.2} tokens/sec",
            1.0 / forward_only.as_secs_f64()
        );

        // 3. Summary
        println!("\n=== Breakdown ===");
        println!(
            "Model loading: {:?} ({:.1}%)",
            avg_load,
            avg_load.as_secs_f64() / avg_forward.as_secs_f64() * 100.0
        );
        println!(
            "Forward pass:  {:?} ({:.1}%)",
            forward_only,
            forward_only.as_secs_f64() / avg_forward.as_secs_f64() * 100.0
        );
        println!("Total:         {:?}", avg_forward);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU benchmark requires aarch64+metal");
    }
}
