// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test F16 GEMV for K=1024 specifically (Qwen3-0.6B FFN critical path)

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== F16 GEMV K=1024 Test (Qwen3-0.6B FFN) ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();

        // Test sizes matching Qwen3-0.6B FFN:
        // w1/w3: M=intermediate_size, K=hidden_size=1024
        // w2: M=hidden_size=1024, K=intermediate_size

        let test_configs = vec![
            ("w1/w3 (small)", 2048, 1024), // Small intermediate
            ("w1/w3 (medium)", 4096, 1024), // Medium intermediate
            ("w1/w3 (large)", 8192, 1024), // Large intermediate
            ("w2 (small)", 1024, 2048),
            ("w2 (medium)", 1024, 4096),
            ("w2 (large)", 1024, 8192),
        ];

        println!(
            "{:<20} {:<15} {:<15} {:<15}",
            "Config", "Time (ms)", "GFLOPS", "GB/s"
        );
        println!("{}", "-".repeat(70));

        for (name, m, k) in test_configs {
            // Create random F16 weights
            let total_weights = m * k;
            let mut weights_f16 = Vec::with_capacity(total_weights);
            for i in 0..total_weights {
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

            // Benchmark 100 iterations
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
            let avg_time_ms =
                elapsed.as_secs_f64() * 1000.0 / iterations as f64;

            let gflops =
                (2.0 * m as f64 * k as f64) / (avg_time_ms * 1e-3) / 1e9;
            let gbps = gflops * 6.0; // F16 (2 bytes) * 2 (read + write) + F32 (4 bytes) * 2

            println!(
                "{:<20} {:<15.3} {:<15.2} {:<15.2}",
                name, avg_time_ms, gflops, gbps
            );
        }

        println!("\n=== Test Complete ===");
    }
}
