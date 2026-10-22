// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Measure Metal dispatch overhead by comparing sequential vs batched dispatch

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Metal Dispatch Overhead Measurement ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();
        let queue = device.new_command_queue();

        // Test with typical decode sizes (M=1536, K=1536 for Qwen1.5B)
        let m = 1536;
        let k = 1536;

        // Create test data
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
        for _ in 0..5 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.mv_encode(&encoder, &weights_buffer, &input_buffer, m, k);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let iterations = 100;

        // Sequential dispatch (one CB per kernel)
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

        // Batched dispatch (all kernels in one CB)
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

        // Calculate overhead
        let avg_seq_ms = seq_time.as_secs_f64() * 1000.0 / iterations as f64;
        let avg_batch_ms =
            batch_time.as_secs_f64() * 1000.0 / iterations as f64;
        let overhead_ms = avg_seq_ms - avg_batch_ms;
        let overhead_pct = (overhead_ms / avg_seq_ms) * 100.0;

        println!("Problem size: {}x{}", m, k);
        println!("Sequential dispatch: {:.3} ms/kernel", avg_seq_ms);
        println!("Batched dispatch:    {:.3} ms/kernel", avg_batch_ms);
        println!(
            "Dispatch overhead:   {:.3} ms ({:.1}%)",
            overhead_ms, overhead_pct
        );
        println!(
            "GPU compute time:    {:.3} ms ({:.1}%)",
            avg_batch_ms,
            100.0 - overhead_pct
        );

        // Estimate per-dispatch overhead
        let dispatch_overhead = overhead_ms / iterations as f64;
        println!("\nPer-kernel dispatch overhead: {:.3} ms", dispatch_overhead);

        println!("\n=== Measurement Complete ===");
    }
}
