// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark pure decode throughput (prefill once, decode many times in same thread)

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

        println!("=== Pure Decode Benchmark ===");
        println!("Model: {}", model_path);

        // Run in isolated thread to ensure correctness
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

            // Prefill once
            println!("\nPrefilling with 10 tokens...");
            let prefill_tokens: Vec<u32> = (1..=10).collect();
            let start = std::time::Instant::now();
            let _ = ctx.model.forward_gpu_native_batched(
                &prefill_tokens,
                &mut kv,
                &gpu_ctx,
            );
            println!("Prefill time: {:?}", start.elapsed());

            // Decode many times in same thread
            println!("\nDecoding 20 tokens...");
            let mut decode_times = Vec::new();
            for i in 0..20 {
                let token = (i + 11) as u32;
                let start = std::time::Instant::now();
                let result = ctx.model.forward_gpu_native_batched(
                    &[token],
                    &mut kv,
                    &gpu_ctx,
                );
                let elapsed = start.elapsed();
                let has_nan = result.is_none();
                decode_times.push(elapsed);
                if i < 5 || i >= 15 {
                    println!(
                        "  Token {}: {:?} (nan: {})",
                        i + 1,
                        elapsed,
                        has_nan
                    );
                }
            }

            let avg_decode = decode_times.iter().sum::<std::time::Duration>()
                / decode_times.len() as u32;
            let tps = 1.0 / avg_decode.as_secs_f64();
            println!("\nAverage decode time: {:?}", avg_decode);
            println!("Throughput: {:.1} tokens/sec", tps);
            tps
        })
        .join()
        .unwrap();

        println!("\n=== Final Result: {:.1} tokens/sec ===", result);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU benchmark requires aarch64+metal");
    }
}
