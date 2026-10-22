// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple KV Cache Update performance test

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== KV Cache Update Performance Test ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();
        let queue = ctx.queue();

        // Qwen3-0.6B config
        let n_kv_heads = 8;
        let head_dim = 128;
        let max_seq = 128;
        let pos = 0; // First position

        println!(
            "Config: n_kv_heads={}, head_dim={}, max_seq={}, pos={}",
            n_kv_heads, head_dim, max_seq, pos
        );

        // Create test data for K and V
        let k_size = n_kv_heads * head_dim; // Single position
        let v_size = n_kv_heads * head_dim;

        let k: Vec<f32> = (0..k_size).map(|i| (i % 10) as f32 * 0.1).collect();
        let v: Vec<f32> = (0..v_size).map(|i| (i % 10) as f32 * 0.1).collect();

        let k_buf = device.new_buffer_with_data(
            k.as_ptr() as *const std::ffi::c_void,
            (k_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_buf = device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // KV cache (F16) - larger buffer for all positions
        let k_cache_size = max_seq * n_kv_heads * head_dim;
        let k_cache = device.new_buffer(
            (k_cache_size * 2) as u64, // F16 = 2 bytes
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_cache = device.new_buffer(
            (k_cache_size * 2) as u64, // F16 = 2 bytes
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        println!("Warming up...");
        for _ in 0..10 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            // Update K cache
            ctx.kv_update_k_encode(
                &encoder, &k_buf, &k_cache, pos, max_seq, n_kv_heads, head_dim,
            );

            // Update V cache
            ctx.kv_update_v_encode(
                &encoder, &v_buf, &v_cache, pos, max_seq, n_kv_heads, head_dim,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Benchmark
        println!("Running benchmark (100 iterations)...\n");
        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            // Update K cache
            ctx.kv_update_k_encode(
                &encoder, &k_buf, &k_cache, pos, max_seq, n_kv_heads, head_dim,
            );

            // Update V cache
            ctx.kv_update_v_encode(
                &encoder, &v_buf, &v_cache, pos, max_seq, n_kv_heads, head_dim,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;

        println!("=== Results ===");
        println!("KV Cache Update (K + V, pos={}): {:.2} μs", pos, avg_time_us);
        println!("\nFor comparison:");
        println!("  - Flash Attention: ~855 μs");
        println!("  - RoPE: ~257 μs");
        println!("  - 7x GEMV: ~2450 μs");
        println!("  - Other ops: ~1365 μs");
        println!("  - Total layer time: ~9500 μs (from bench-decode)");
        println!("\nKV Cache Update占比: {:.1}%", avg_time_us / 9500.0 * 100.0);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}
