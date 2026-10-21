// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test F16 GEMV kernel performance across different K values

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::model::tensor::DataType;
use std::time::Instant;

fn main() {
    let gpu_ctx = MetalContext::new().unwrap();
    let device = gpu_ctx.device();

    // Test different K sizes
    let k_sizes = [128, 256, 512, 1024, 2048, 4096];
    let m = 1024; // Fixed M (output rows)
    let num_iterations = 100;

    println!("=== F16 GEMV Performance Test ===");
    println!("M = {}, iterations = {}\n", m, num_iterations);

    for &k in &k_sizes.iter() {
        // Create test data
        let weights: Vec<half::f16> = vec![half::f16::from_f32(0.5); m * k];
        let input: Vec<f32> = vec![1.0; k];
        let mut output = vec![0.0f32; m];

        // Upload to GPU
        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (m * k * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device.new_buffer_with_data(
            output.as_mut_ptr() as *const std::ffi::c_void,
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        for _ in 0..10 {
            let queue = gpu_ctx.queue();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.mv_encode(
                encoder,
                &weights_buffer,
                &input_buffer,
                &output_buffer,
                m,
                k,
                infer_train::model::tensor::DataType::F16,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..num_iterations {
            let queue = gpu_ctx.queue();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.mv_encode(
                encoder,
                &weights_buffer,
                &input_buffer,
                &output_buffer,
                m,
                k,
                infer_train::model::tensor::DataType::F16,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let elapsed = start.elapsed();
        let time_per_iter = elapsed.as_nanos() as f64 / num_iterations as f64;

        // Calculate GFLOPS
        let flops = 2.0 * m as f64 * k as f64; // multiply-add = 2 ops
        let gflops = flops / time_per_iter;

        println!(
            "K={:5}: {:.2} µs/iter, {:.1} GFLOPS",
            k,
            time_per_iter / 1000.0,
            gflops
        );
    }
}
