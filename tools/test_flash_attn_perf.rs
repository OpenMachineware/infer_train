// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple Flash Attention performance test

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Flash Attention Performance Test ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();
        let queue = ctx.queue();

        // Qwen3-0.6B config
        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 128;
        let seq_k = 128; // Current KV cache size
        let seq_q = 1; // Decode: single token

        println!(
            "Config: n_heads={}, n_kv_heads={}, head_dim={}, seq_k={}, seq_q={}",
            n_heads, n_kv_heads, head_dim, seq_k, seq_q
        );

        // Create test data
        let q_size = seq_q * n_heads * head_dim;
        let k_size = seq_k * n_kv_heads * head_dim;
        let v_size = seq_k * n_kv_heads * head_dim;

        let q: Vec<f32> = (0..q_size).map(|i| (i % 10) as f32 * 0.1).collect();
        let k: Vec<f32> = (0..k_size).map(|i| (i % 10) as f32 * 0.1).collect();
        let v: Vec<f32> = (0..v_size).map(|i| (i % 10) as f32 * 0.1).collect();

        let q_buf = device.new_buffer_with_data(
            q.as_ptr() as *const std::ffi::c_void,
            (q_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
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

        let output_size = seq_q * n_heads * head_dim;
        let output_buf = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        println!("\nWarming up...");
        for _ in 0..10 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.flash_attn_mma_encode(
                &encoder,
                n_heads,
                n_kv_heads,
                seq_k,
                seq_k * n_kv_heads * head_dim, // kv_stride
                &q_buf,
                &k_buf,
                &v_buf,
                &output_buf,
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
            ctx.flash_attn_mma_encode(
                &encoder,
                n_heads,
                n_kv_heads,
                seq_k,
                seq_k * n_kv_heads * head_dim, // kv_stride
                &q_buf,
                &k_buf,
                &v_buf,
                &output_buf,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;

        println!("=== Results ===");
        println!(
            "Flash Attention (decode, seq_k={}): {:.2} μs",
            seq_k, avg_time_us
        );
        println!("\nFor comparison:");
        println!("  - 7x GEMV: ~2450 μs");
        println!("  - Other ops: ~1365 μs");
        println!("  - Total layer time: ~9500 μs (from bench-decode)");
        println!("\nFlash Attention占比: {:.1}%", avg_time_us / 9500.0 * 100.0);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}
