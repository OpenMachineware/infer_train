// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Flash Attention MMA Batched kernel performance

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Flash Attention MMA Batched Performance Test ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();
        let queue = ctx.queue();

        // Qwen3-0.6B config
        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 128;
        let seq_k = 128;

        println!(
            "Config: n_heads={}, n_kv_heads={}, head_dim={}, seq_k={}",
            n_heads, n_kv_heads, head_dim, seq_k
        );

        // Create test data
        let q_size = n_heads * head_dim;
        let k_size = seq_k * n_kv_heads * head_dim;
        let v_size = seq_k * n_kv_heads * head_dim;

        let q: Vec<half::f16> = (0..q_size)
            .map(|i| half::f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let k: Vec<half::f16> = (0..k_size)
            .map(|i| half::f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let v: Vec<half::f16> = (0..v_size)
            .map(|i| half::f16::from_f32((i % 10) as f32 * 0.1))
            .collect();

        let q_buf = device.new_buffer_with_data(
            q.as_ptr() as *const std::ffi::c_void,
            (q_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buf = device.new_buffer_with_data(
            k.as_ptr() as *const std::ffi::c_void,
            (k_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_buf = device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_size = n_heads * head_dim;
        let output_buf = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        println!("Warming up...");
        for _ in 0..10 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.flash_attn_mma_batched_encode(
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
            ctx.flash_attn_mma_batched_encode(
                &encoder,
                n_heads,
                n_kv_heads,
                seq_k,
                seq_k * n_kv_heads * head_dim,
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
            "Flash Attention MMA Batched (seq_k={}): {:.2} μs",
            seq_k, avg_time_us
        );
        println!("\nFor comparison:");
        println!("  - Old MMA kernel: ~855 μs");
        println!("  - Expected speedup: 2-3x");
        println!("  - Achieved speedup: {:.1}x", 855.0 / avg_time_us);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}
