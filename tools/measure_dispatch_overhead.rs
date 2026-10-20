// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Measure dispatch overhead vs actual GPU compute time

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

        println!("=== Dispatch Overhead Measurement ===");
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
            let prefill_tokens: Vec<u32> = (1..=10).collect();
            let _ = ctx.model.forward_gpu_native_batched(
                &prefill_tokens,
                &mut kv,
                &gpu_ctx,
            );

            // Warm up
            for _ in 0..5 {
                let _ = ctx.model.forward_gpu_native_batched(
                    &[11],
                    &mut kv,
                    &gpu_ctx,
                );
            }

            // Measure single token decode
            let mut times = Vec::new();
            for i in 0..20 {
                let token = (i + 20) as u32;
                let start = std::time::Instant::now();
                let _ = ctx.model.forward_gpu_native_batched(
                    &[token],
                    &mut kv,
                    &gpu_ctx,
                );
                times.push(start.elapsed());
            }

            let avg_time: std::time::Duration =
                times.iter().sum::<std::time::Duration>() / times.len() as u32;

            println!("\nAverage decode time: {:?}", avg_time);

            // Calculate theoretical minimum
            let n_layers = config.n_layer;
            let hidden_size = config.hidden_size;
            let intermediate_size = config.intermediate_size;
            let n_heads = config.n_head;
            let head_dim = hidden_size / n_heads;
            let vocab_size = config.vocab_size;

            // Per-layer operations
            // QKV projections: 3 * hidden_size * hidden_size
            // FFN up: hidden_size * intermediate_size
            // FFN gate: hidden_size * intermediate_size
            // FFN down: intermediate_size * hidden_size
            let ops_per_layer = 3 * hidden_size * hidden_size
                + 2 * hidden_size * intermediate_size
                + intermediate_size * hidden_size;

            // Attention operations (simplified)
            // Flash attention: O(n_heads * seq_len * head_dim)
            // KV cache update: O(2 * n_heads * head_dim)

            let total_ops = ops_per_layer * n_layers;
            let gflops = total_ops as f64 / 1e9;

            println!("Approximate FLOPs: {:.2} GFLOPS", gflops);
            println!("Time: {:?}", avg_time);
            println!("Throughput: {:.1} t/s", 1.0 / avg_time.as_secs_f64());

            // If we had zero dispatch overhead
            let compute_time = gflops / 500.0; // Assume 500 GFLOPS peak
            let overhead = avg_time.as_secs_f64() - compute_time;
            println!(
                "\nEstimated compute time: {:.3} ms",
                compute_time * 1000.0
            );
            println!(
                "Estimated dispatch overhead: {:.3} ms ({:.1}%)",
                overhead * 1000.0,
                (overhead / avg_time.as_secs_f64()) * 100.0
            );

            avg_time
        })
        .join()
        .unwrap();
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Requires aarch64+metal");
    }
}
