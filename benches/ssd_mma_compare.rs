// SSD MMA performance comparison with correct measurement
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod bench {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::*;
    use std::time::Instant;

    fn compute_ssd_gflops(n_tok: usize, d_state: usize, time_ns: f64) -> f64 {
        // Approximate FLOPs: n_tok * d_state * 5 (softplus + state update + output)
        let flops = n_tok as f64 * d_state as f64 * 5.0 * 64.0; // *64 for head_dim
        flops / time_ns * 1000.0
    }

    pub fn bench_ssd_comparison() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let d_state = 64;
        let head_dim = 64;
        let n_tok = 512; // Match llama.cpp benchmark

        let x_size = n_tok * head_dim;
        let bc_size = n_tok * d_state;

        let x: Vec<f32> =
            (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
        let dt: Vec<f32> =
            (0..n_tok).map(|i| 0.001 * (i as f32 % 5.0 + 1.0)).collect();
        let a = vec![-0.1f32];
        let b: Vec<f32> =
            (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
        let c: Vec<f32> =
            (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();

        // Create buffers once
        let x_buffer = ctx.device().new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            (x_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let dt_buffer = ctx.device().new_buffer_with_data(
            dt.as_ptr() as *const std::ffi::c_void,
            (n_tok * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let a_buffer = ctx.device().new_buffer_with_data(
            a.as_ptr() as *const std::ffi::c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buffer = ctx.device().new_buffer_with_data(
            b.as_ptr() as *const std::ffi::c_void,
            (bc_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buffer = ctx.device().new_buffer_with_data(
            c.as_ptr() as *const std::ffi::c_void,
            (bc_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let state_buffer = ctx.device().new_buffer(
            (head_dim * d_state * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buffer = ctx.device().new_buffer(
            (n_tok * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Warm up
        for _ in 0..10 {
            let _ = ctx.ssm_scan_ssd_mma(
                &x, &dt, &a, &b, &c, None, head_dim, n_tok, d_state, 1,
            );
        }

        // Measure with multiple kernels per command buffer
        let iterations = 50;
        let kernels_per_cb = 5;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            // Run multiple SSD scans
            for _ in 0..kernels_per_cb {
                let _ = ctx.ssm_scan_ssd_mma(
                    &x, &dt, &a, &b, &c, None, head_dim, n_tok, d_state, 1,
                );
            }
            times.push(
                start.elapsed().as_nanos() as f64 / kernels_per_cb as f64,
            );
        }

        let mean_ns = times.iter().sum::<f64>() / times.len() as f64;
        let mean_us = mean_ns / 1000.0;
        let gflops = compute_ssd_gflops(n_tok, d_state, mean_ns);

        println!(
            "\n=== SSD MMA Performance (buffer reuse, batched measurement) ==="
        );
        println!("d_state={}, head_dim={}, n_tok={}", d_state, head_dim, n_tok);
        println!("time: {:.2} us", mean_us);
        println!("GFLOPS: {:.2}", gflops);
        println!("\nllama.cpp reference: ~8-10 TFLOPS for SSD");
        println!("Gap: {:.1}x", 10000.0 / gflops);
    }
}

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        bench::bench_ssd_comparison();
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support on aarch64");
    }
}
