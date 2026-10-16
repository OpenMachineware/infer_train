// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark F16 GEMV kernel vs llama.cpp

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== F16 GEMV Performance Benchmark ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();

        // Test sizes: (M, K) pairs
        // M = output dimension (vocab size typically)
        // K = input dimension (hidden size typically)
        let test_sizes = vec![
            (32000, 1536),  // Small model output layer
            (151936, 1536), // Qwen 1.5B output layer
            (128256, 4096), // Llama 7B output layer
            (256000, 5120), // Large model output layer
            // Different aspect ratios
            (1024, 4096),
            (4096, 4096),
            (8192, 4096),
            (16384, 4096),
            (32000, 4096),
        ];

        println!("Testing {} different sizes\n", test_sizes.len());
        println!(
            "{:<20} {:<15} {:<15} {:<15}",
            "Size (M×K)", "Time (ms)", "GFLOPS", "GB/s"
        );
        println!("{}", "-".repeat(70));

        for (m, k) in test_sizes {
            // Create random F16 weights
            let total_weights = m * k;
            let mut weights_f16 = Vec::with_capacity(total_weights);
            for i in 0..total_weights {
                // Use simple pseudo-random values
                let f32_val = ((i as f32 * 0.12345) % 2.0) - 1.0;
                let f16_val = half::f16::from_f32(f32_val);
                weights_f16.push(f16_val.to_bits());
            }

            // Create random F32 input
            let mut input_f32 = Vec::with_capacity(k);
            for i in 0..k {
                input_f32.push(((i as f32 * 0.98765) % 2.0) - 1.0);
            }

            // Create GPU buffers
            let weights_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                weights_f16.as_ptr() as *const std::ffi::c_void,
                (total_weights * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                input_f32.as_ptr() as *const std::ffi::c_void,
                (k * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Warmup
            for _ in 0..3 {
                let _ = ctx.mv_f16_f32_gpu_weights(
                    m,
                    k,
                    &weights_buffer,
                    &input_buffer,
                );
            }

            // Benchmark
            let n_iters = 10;
            let start = std::time::Instant::now();

            for _ in 0..n_iters {
                let _ = ctx.mv_f16_f32_gpu_weights(
                    m,
                    k,
                    &weights_buffer,
                    &input_buffer,
                );
            }

            let elapsed = start.elapsed();
            let avg_time = elapsed.as_secs_f64() / n_iters as f64;

            // Calculate metrics
            let flops = 2.0 * m as f64 * k as f64; // Multiply-add per element
            let gflops = flops / (avg_time * 1e9);

            // Memory bandwidth: read weights (M*K*2 bytes) + read input (K*4 bytes) + write output (M*4 bytes)
            let bytes_read = (m * k * 2 + k * 4) as f64;
            let bytes_written = (m * 4) as f64;
            let total_bytes = bytes_read + bytes_written;
            let bandwidth_gbs = total_bytes / (avg_time * 1e9);

            println!(
                "{:<20} {:<15.3} {:<15.2} {:<15.2}",
                format!("{}×{}", m, k),
                avg_time * 1000.0,
                gflops,
                bandwidth_gbs
            );
        }

        println!("\n=== Test Complete ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("F16 GEMV benchmark requires Metal support (macOS ARM64)");
    }
}
