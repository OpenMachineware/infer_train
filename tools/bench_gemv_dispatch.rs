// Measure dispatch overhead for a single GEMV kernel
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let m = 1024; // Output size
        let k = 1024; // Input size

        // Create input and weight buffers
        let input = vec![0.1f32; k];
        let input_buffer = std::rc::Rc::new(ctx.device().new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Create weight buffer (Q4_0 quantized simulation - just use F32 for simplicity)
        let weight_buffer = std::rc::Rc::new(ctx.device().new_buffer(
            (m * k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Benchmark: dispatch overhead vs computation
        println!("Measuring single GEMV dispatch overhead");
        println!("M={}, K={}", m, k);

        let n_iterations = 1000;
        let mut times = Vec::with_capacity(n_iterations);

        for _ in 0..n_iterations {
            let start = std::time::Instant::now();
            let _ = ctx.mv_q4_0_f32_gpu_weights(
                m,
                k,
                &weight_buffer,
                &input_buffer,
            );
            times.push(start.elapsed());
        }

        // Analyze
        let total_ns: u128 = times.iter().map(|t| t.as_nanos()).sum();
        let avg_us = total_ns as f64 / n_iterations as f64 / 1000.0;

        let min_us = times.iter().map(|t| t.as_micros()).min().unwrap() as f64;
        let max_us = times.iter().map(|t| t.as_micros()).max().unwrap() as f64;

        println!("  Average: {:.2} us", avg_us);
        println!("  Min: {:.2} us", min_us);
        println!("  Max: {:.2} us", max_us);

        // Estimate dispatch overhead vs computation
        // FLOPS for GEMV: 2 * M * K = 2M FLOPS
        // At 100 GFLOPS, should take: 2M / 100G = 20us
        // So if we measure 400us, overhead is ~380us
        let compute_time_us =
            2.0 * m as f64 * k as f64 / 100_000_000_000.0 * 1_000_000.0;
        let overhead_us = avg_us - compute_time_us;

        println!(
            "\n  Estimated compute time: {:.2} us (at 100 GFLOPS)",
            compute_time_us
        );
        println!("  Estimated dispatch overhead: {:.2} us", overhead_us);
        println!("  Overhead ratio: {:.1}%", overhead_us / avg_us * 100.0);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
