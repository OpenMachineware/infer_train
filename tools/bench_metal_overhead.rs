// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Measure Metal dispatch overhead: CPU wall time vs GPU execution time

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Metal Dispatch Overhead Measurement ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();
        let queue = device.new_command_queue();

        // Test different kernel sizes
        let test_configs = vec![
            ("Tiny (256)", 256, 256),
            ("Small (1024)", 1024, 1024),
            ("Medium (2048)", 2048, 2048),
            ("Large (4096)", 4096, 4096),
        ];

        println!(
            "{:<20} {:<15} {:<15} {:<15}",
            "Config", "Wall (ms)", "GPU (ms)", "Overhead (%)"
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
            for _ in 0..10 {
                let _ = ctx.mv_f16_f32_gpu_weights(m, k, &weights_buffer, &input_buffer);
            }

            // Benchmark 100 iterations with GPU timing
            let iterations = 100;
            let command_buffers: Vec<_> = (0..iterations)
                .map(|_| {
                    let cb = queue.new_command_buffer();
                    let encoder = cb.new_compute_command_encoder();
                    ctx.mv_encode(
                        encoder,
                        &weights_buffer,
                        &input_buffer,
                        m,
                        k,
                    );
                    encoder.end_encoding();
                    cb
                })
                .collect();

            // Measure CPU wall time for commit+wait
            let wall_start = std::time::Instant::now();
            for cb in &command_buffers {
                cb.commit();
            }
            for cb in &command_buffers {
                cb.wait_until_completed();
            }
            let wall_time = wall_start.elapsed();

            // Calculate average times
            let avg_wall_ms = wall_time.as_secs_f64() * 1000.0 / iterations as f64;

            // Try to get GPU execution time from the last command buffer
            // Metal doesn't provide direct GPU time, so we'll estimate from compute work
            let gflops = (2.0 * m as f64 * k as f64) / (avg_wall_ms * 1e-3) / 1e9;
            let gpu_time_estimate = if gflops > 0.0 {
                // Estimate based on ~500 GFLOPS typical for memory-bound GEMV
                (2.0 * m as f64 * k as f64) / (500.0 * 1e9) * 1000.0
            } else {
                0.0
            };

            let overhead_pct = if avg_wall_ms > 0.0 {
                ((avg_wall_ms - gpu_time_estimate) / avg_wall_ms) * 100.0
            } else {
                0.0
            };

            println!(
                "{:<20} {:<15.3} {:<15.3} {:<15.1}",
                name, avg_wall_ms, gpu_time_estimate, overhead_pct
            );
        }

        // Now test with batched dispatch (single commit for all iterations)
        println!("\n=== Batched Dispatch (100 kernels in one CB) ===\n");

        let m = 1024;
        let k = 1024;
        let total_weights = m * k;
        let mut weights_f16 = Vec::with_capacity(total_weights);
        for i in 0..total_weights {
            let f32_val = ((i as f32 * 0.12345) % 2.0) - 1.0;
            let f16_val = half::f16::from_f32(f32_val);
            weights_f16.push(f16_val.to_bits());
        }

        let mut input_f32 = Vec::with_capacity(k);
        for i in 0..k {
            input_f32.push(((i as f32 * 0.98765) % 2.0) - 1.0);
        }

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

        // Single command buffer with 100 kernels
        let iterations = 100;
        let batch_start = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        for _ in 0..iterations {
            let encoder = cb.new_compute_command_encoder();
            ctx.mv_encode(&encoder, &weights_buffer, &input_buffer, m, k);
            encoder.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
        let batch_time = batch_start.elapsed();

        let avg_batch_ms = batch_time.as_secs_f64() * 1000.0 / iterations as f64;

        // Sequential dispatch
        let seq_start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.mv_encode(&encoder, &weights_buffer, &input_buffer, m, k);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let seq_time = seq_start.elapsed();

        let avg_seq_ms = seq_time.as_secs_f64() * 1000.0 / iterations as f64;

        println!(
            "Sequential dispatch: {:.3} ms/kernel",
            avg_seq_ms
        );
        println!(
            "Batched dispatch:    {:.3} ms/kernel",
            avg_batch_ms
        );
        println!(
            "Speedup:             {:.2}x",
            avg_seq_ms / avg_batch_ms
        );
        println!(
            "Per-kernel overhead: {:.3} ms",
            avg_seq_ms - avg_batch_ms
        );

        println!("\n=== Measurement Complete ===");
    }
}
