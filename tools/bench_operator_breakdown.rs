// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test each operator performance breakdown

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::*;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Operator Performance Breakdown ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();
        let queue = ctx.queue();

        let hidden_size = 1024; // Qwen3-0.6B
        let head_dim = 128;
        let n_heads = 16;
        let n_kv_heads = 8;
        let seq_len = 128; // Current position during decode

        // Test each operator
        let iterations = 100;

        println!(
            "Testing with hidden_size={}, head_dim={}, n_heads={}",
            hidden_size, head_dim, n_heads
        );
        println!("Iterations: {}\n", iterations);

        // 1. RMS Norm
        println!("1. RMS Norm (1024 elements):");
        let input: Vec<f32> =
            (0..hidden_size).map(|i| (i % 10) as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..hidden_size).map(|i| 1.0).collect();
        let input_buf = device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buf = device.new_buffer_with_data(
            weight.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        for _ in 0..10 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.rms_norm_encode(
                &encoder,
                &input_buf,
                &weight_buf,
                &output_buf,
                1,
                hidden_size,
                1e-5,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.rms_norm_encode(
                &encoder,
                &input_buf,
                &weight_buf,
                &output_buf,
                1,
                hidden_size,
                1e-5,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let rms_norm_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("   Average: {:.2} μs\n", rms_norm_time);

        // 2. SiLU
        println!("3. SiLU (1024 elements):");
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.silu_encode(&encoder, &input_buf, &output_buf, hidden_size);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let silu_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("   Average: {:.2} μs\n", silu_time);

        // 4. Mul (element-wise)
        println!("4. Element-wise Mul (1024 elements):");
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.mul_encode(
                &encoder,
                &input_buf,
                &input_buf,
                &output_buf,
                hidden_size,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let mul_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("   Average: {:.2} μs\n", mul_time);

        // 5. Add (residual)
        println!("5. Add/Residual (1024 elements):");
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.add_encode(
                &encoder,
                &input_buf,
                &input_buf,
                &output_buf,
                hidden_size,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let add_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("   Average: {:.2} μs\n", add_time);

        // Summary
        println!("=== Summary ===");
        println!("Per-layer operations:");
        println!("  2x RMS Norm:      {:.2} μs", 2.0 * rms_norm_time);
        println!("  1x SiLU:          {:.2} μs", silu_time);
        println!("  1x Mul:           {:.2} μs", mul_time);
        println!("  2x Add:           {:.2} μs", 2.0 * add_time);
        println!(
            "  Total non-GEMV:   {:.2} μs",
            2.0 * rms_norm_time + silu_time + mul_time + 2.0 * add_time
        );
        println!(
            "\n  7x GEMV:          ~2500 μs (estimated from earlier data)"
        );
        println!("  1x Flash Attn:    ??? μs (need to test separately)");
        println!("  2x RoPE:          ??? μs (need to test separately)");
        println!("  2x KV Cache:      ??? μs (need to test separately)");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}
