// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile bandwidth difference between our implementation and llama.cpp

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();

        let mut kv_cache = KVCacheGPU::new(
            &device,
            &ctx.config,
            ctx.config.max_position_embeddings,
        );

        println!("=== Bandwidth Gap Analysis ===\n");

        // Prefill
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        // Profile decode in detail
        println!("=== Profiling 30 decode iterations ===\n");

        let mut decode_times: Vec<f64> = Vec::new();
        let iterations = 30;

        for _ in 0..iterations {
            let start = Instant::now();
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
            decode_times.push(start.elapsed().as_micros() as f64);
        }

        decode_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = decode_times[iterations / 2];
        let avg_us: f64 = decode_times.iter().sum::<f64>() / iterations as f64;

        println!("Our decode performance:");
        println!("  Median: {:.1} µs", median_us);
        println!("  Average: {:.1} µs", avg_us);
        println!("  Throughput: {:.1} t/s", 1_000_000.0 / median_us);

        // Calculate bandwidth
        let total_weights_mb = 258.95;
        let bandwidth_gbs = total_weights_mb / (median_us / 1000.0);
        let utilization = (bandwidth_gbs / 100.0) * 100.0;

        println!("\nOur bandwidth:");
        println!("  Total weights: {:.2} MB", total_weights_mb);
        println!("  Bandwidth: {:.1} GB/s", bandwidth_gbs);
        println!("  Utilization: {:.1}%", utilization);

        // Compare with llama.cpp
        println!("\n=== Comparison with llama.cpp ===\n");

        let llama_tps = 235.35;
        let llama_time_us = 1_000_000.0 / llama_tps;
        let llama_bandwidth = total_weights_mb / (llama_time_us / 1000.0);
        let llama_util = (llama_bandwidth / 100.0) * 100.0;

        println!("llama.cpp decode performance:");
        println!("  Time: {:.1} µs", llama_time_us);
        println!("  Throughput: {:.1} t/s", llama_tps);
        println!("  Bandwidth: {:.1} GB/s", llama_bandwidth);
        println!("  Utilization: {:.1}%", llama_util);

        println!("\n=== The Gap ===");
        println!(
            "  Bandwidth gap: {:.1} GB/s vs {:.1} GB/s",
            llama_bandwidth, bandwidth_gbs
        );
        println!(
            "  Utilization gap: {:.1}% vs {:.1}%",
            llama_util, utilization
        );
        println!(
            "  Performance gap: {:.2}x",
            llama_tps / (1_000_000.0 / median_us)
        );

        // Analysis
        println!("\n=== Root Cause Analysis ===");
        println!("\nPossible causes of bandwidth gap:");
        println!(
            "1. Dispatch overhead (our 564 dispatches vs llama.cpp's 589)"
        );
        println!("2. Kernel overlap efficiency");
        println!("3. Memory access patterns");
        println!("4. Metal encoder overhead");
        println!("5. Command buffer management");

        println!("\nDispatch overhead calculation:");
        let dispatch_count = 564;
        let dispatch_overhead_us = 6.5; // From previous profiling
        let total_overhead_ms =
            dispatch_count as f64 * dispatch_overhead_us / 1000.0;

        println!("  Dispatches: {}", dispatch_count);
        println!("  Overhead per dispatch: {:.1} µs", dispatch_overhead_us);
        println!("  Total overhead: {:.2} ms", total_overhead_ms);
        println!(
            "  %% of total time: {:.1}%%",
            (total_overhead_ms * 1000.0 / median_us) * 100.0
        );

        let kernel_time_us = median_us - total_overhead_ms * 1000.0;
        println!("  Kernel execution time: {:.1} µs", kernel_time_us);

        println!("\nIf llama.cpp has same dispatch count:");
        println!("  Their overhead would be {:.2} ms", total_overhead_ms);
        println!(
            "  Their kernel time would be {:.1} µs",
            llama_time_us - total_overhead_ms * 1000.0
        );
        println!(
            "  Gap in kernel execution: {:.2}x",
            kernel_time_us / (llama_time_us - total_overhead_ms * 1000.0)
        );

        println!("\n=== Hypothesis ===");
        println!("llama.cpp achieves better bandwidth through:");
        println!("1. More efficient kernel scheduling");
        println!("2. Better memory prefetching");
        println!("3. Optimized Metal encoding");
        println!("4. Lower dispatch overhead");

        println!("\n=== Next Steps ===");
        println!("1. Profile Metal encoder time");
        println!("2. Compare dispatch strategies");
        println!("3. Analyze kernel overlap");
        println!("4. Study llama.cpp Metal implementation");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
