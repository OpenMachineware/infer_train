// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare Q4_K GEMV with llama.cpp

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::types::BlockQ4K;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Q4_K GEMV: Our Implementation vs llama.cpp ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();

        // Test sizes matching Qwen3-0.6B
        let test_configs = vec![
            ("wq/wk/wv (1024x2048)", 2048, 1024),
            ("gate/up (1024x4096)", 4096, 1024), // intermediate_size may vary
            ("wo (2048x1024)", 1024, 2048),
            ("down (4096x1024)", 1024, 4096),
        ];

        println!(
            "{:<25} {:<15} {:<15} {:<15}",
            "Config", "Time (μs)", "GFLOPS", "GB/s"
        );
        println!("{}", "-".repeat(70));

        for (name, m, k) in test_configs {
            let nb = k / 256;
            let total_blocks = m * nb;

            // Create Q4_K weights
            let mut weights_q4k = Vec::with_capacity(total_blocks);
            for i in 0..total_blocks {
                let mut qs = [0u8; 128];
                let mut scales = [0u8; 12];
                for j in 0..128 {
                    qs[j] = ((i * 23 + j * 17) % 256) as u8;
                }
                for j in 0..12 {
                    scales[j] = ((i * 17 + j * 13) % 64) as u8;
                }
                weights_q4k.push(BlockQ4K {
                    d: half::f16::from_f32(0.5 + (i % 10) as f32 * 0.1)
                        .to_bits(),
                    dmin: half::f16::from_f32(0.0).to_bits(),
                    scales,
                    qs,
                });
            }

            // Create F32 input
            let mut input_f32 = Vec::with_capacity(k);
            for i in 0..k {
                input_f32.push(((i % 10) as f32) * 0.1);
            }

            // Create GPU buffers
            let weights_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                weights_q4k.as_ptr() as *const std::ffi::c_void,
                (total_blocks * std::mem::size_of::<BlockQ4K>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                input_f32.as_ptr() as *const std::ffi::c_void,
                (k * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Warmup
            for _ in 0..3 {
                let _ = ctx.mv_q4_k_f32_gpu_weights(
                    m,
                    k,
                    &weights_buffer,
                    &input_buffer,
                );
            }

            // Benchmark
            let iterations = 100;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = ctx.mv_q4_k_f32_gpu_weights(
                    m,
                    k,
                    &weights_buffer,
                    &input_buffer,
                );
            }
            let elapsed = start.elapsed();
            let avg_time_us = elapsed.as_secs_f64() * 1e6 / iterations as f64;

            // Calculate performance
            // Q4_K: 4-bit weights = 0.5 bytes per weight
            // Operation: M * K * 2 FLOPs (multiply-add)
            let gflops =
                (2.0 * m as f64 * k as f64) / (avg_time_us * 1e-6) / 1e9;

            // Bandwidth: read weights (M * K * 0.5) + read input (K * 4) + write output (M * 4)
            let gbps =
                (m as f64 * k as f64 * 0.5 + k as f64 * 4.0 + m as f64 * 4.0)
                    / (avg_time_us * 1e-6)
                    / 1e9;

            println!(
                "{:<25} {:<15.2} {:<15.2} {:<15.2}",
                name, avg_time_us, gflops, gbps
            );
        }

        println!("\n=== Test Complete ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}
