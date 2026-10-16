// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark F16 GEMV adaptive kernels against llama.cpp
// Test each K range: short (<256), medium (256-2048), large (>=2048)

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = MetalContext::new().expect("Failed to create MetalContext");

        // Test different K values across the three ranges
        let k_values = vec![
            32, 64, 128, 256, // Short range
            256, 512, 1024, 2048, // Medium range
            2048, 4096, 8192, 16384, // Large range
        ];

        // M values (rows) to test
        let m_values = vec![128, 1024, 4096, 16384];

        println!("F16 GEMV Adaptive Kernel Performance Test");
        println!("==========================================");
        println!();

        for &k in &k_values {
            for &m in &m_values {
                // Generate test data
                let weights_f16: Vec<u16> = (0..m * k)
                    .map(|i| ((i as f32 * 0.1).sin() * 100.0) as u16)
                    .collect();
                let input_f32: Vec<f32> =
                    (0..k).map(|i| (i as f32 * 0.05).cos()).collect();

                // Create buffers
                let weights_buffer = ctx.device().new_buffer_with_data(
                    weights_f16.as_ptr() as *const std::ffi::c_void,
                    (m * k * 2) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let input_buffer = ctx.device().new_buffer_with_data(
                    input_f32.as_ptr() as *const std::ffi::c_void,
                    (k * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                // Warm up
                let _ = ctx.mv_f16_f32_gpu_weights(
                    m,
                    k,
                    &weights_buffer,
                    &input_buffer,
                );

                // Benchmark
                let iterations = 100;
                let start = std::time::Instant::now();
                for _ in 0..iterations {
                    let _ = ctx.mv_f16_f32_gpu_weights(
                        m,
                        k,
                        &weights_buffer,
                        &input_buffer,
                    );
                }
                let elapsed = start.elapsed();
                let avg_time_us =
                    elapsed.as_micros() as f64 / iterations as f64;

                // Calculate performance
                let flops = 2.0 * m as f64 * k as f64;
                let gflops = flops / (avg_time_us * 1e-6) / 1e9;

                // Calculate bandwidth (bytes read: m*k*2 (weights) + k*4 (input), bytes written: m*4)
                let bytes_read = m as f64 * k as f64 * 2.0 + k as f64 * 4.0;
                let bytes_written = m as f64 * 4.0;
                let total_bytes = bytes_read + bytes_written;
                let bandwidth_gbps = total_bytes / (avg_time_us * 1e-6) / 1e9;

                // Determine kernel type
                let kernel_type = if k < 256 {
                    "short"
                } else if k < 2048 {
                    "medium"
                } else {
                    "large"
                };

                println!(
                    "M={:>5} K={:>5} ({:>6}): {:>8.2} us, {:>6.1} GFLOPS, {:>6.1} GB/s",
                    m, k, kernel_type, avg_time_us, gflops, bandwidth_gbps
                );
            }
            println!();
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support on aarch64");
    }
}
