// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile single token decode to identify bottlenecks

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

        println!("=== Single Token Decode Profiling ===");
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

            // Enable timing
            std::env::set_var("INFER_TIMING", "1");

            // Decode single token
            let start = std::time::Instant::now();
            let _ = ctx.model.forward_gpu_native_batched(&[11], &mut kv, &gpu_ctx);
            let total_time = start.elapsed();

            println!("\nTotal decode time: {:?}", total_time);

            // Calculate FLOPs
            let n_layers = config.n_layer;
            let hidden_size = config.hidden_size;
            let intermediate_size = config.intermediate_size;

            // Per-layer FLOPs (decode, single token):
            // QKV: 3 * 2 * hidden_size^2 (GEMV)
            // FFN up/gate: 2 * 2 * hidden_size * intermediate_size
            // FFN down: 2 * intermediate_size * hidden_size
            let qkv_flops = 3 * 2 * hidden_size * hidden_size;
            let ffn_flops = 2 * 2 * hidden_size * intermediate_size
                + 2 * intermediate_size * hidden_size;
            let per_layer = qkv_flops + ffn_flops;
            let total_flops = per_layer * n_layers;

            println!("Total FLOPs: {:.2} M", total_flops as f64 / 1e6);
            println!("Effective throughput: {:.1} GFLOPS",
                (total_flops as f64 / 1e9) / total_time.as_secs_f64()
            );

            // Calculate theoretical minimum
            // M3 GPU: ~500 GFLOPS for F32, ~1000 for F16
            // Our Q5_K GEMV: ~8 GFLOPS measured
            let theoretical_time = total_flops as f64 / (8e9); // 8 GFLOPS
            println!("Theoretical minimum (8 GFLOPS): {:.3} ms", theoretical_time * 1000.0);

            let overhead = total_time.as_secs_f64() - theoretical_time;
            println!("Dispatch + other overhead: {:.3} ms ({:.1}%)",
                overhead * 1000.0,
                (overhead / total_time.as_secs_f64()) * 100.0
            );

            total_time
        })
        .join()
        .unwrap();
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Requires aarch64+metal");
    }
}
