// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test enqueue mechanism for decode performance
// Compare: per-layer CB commit vs single CB with enqueue

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

        println!("=== Enqueue Decode Test ===");
        println!("Model: {}", model_path);

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

            // Test 1: Current approach (per-layer CB commit)
            println!("\n=== Test 1: Per-layer CB commit ===");
            let mut times = Vec::new();
            for i in 0..10 {
                let token = (i + 11) as u32;
                let start = std::time::Instant::now();
                let _ = ctx.model.forward_gpu_native_batched(
                    &[token],
                    &mut kv,
                    &gpu_ctx,
                );
                times.push(start.elapsed());
            }
            let avg: std::time::Duration =
                times.iter().sum::<std::time::Duration>() / times.len() as u32;
            println!("Average: {:?} ({:.1} t/s)", avg, 1.0 / avg.as_secs_f64());

            // Test 2: Try single CB with enqueue (if possible)
            // This requires modifying the DAG executor

            avg
        })
        .join()
        .unwrap();

        println!("\n=== Final Result: {:.1} t/s ===", 1.0 / result.as_secs_f64());
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU test requires aarch64+metal");
    }
}
