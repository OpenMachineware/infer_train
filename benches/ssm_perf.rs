// SSM GPU operator performance benchmark with correct measurement
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod bench {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::*;
    use std::time::Instant;

    // Shared memory size for SSD kernel (large enough for CS=128)
    const SSD_SHARED_SIZE: usize = 32768;

    fn compute_gflops(
        d_inner: usize,
        n_tok: usize,
        d_conv: usize,
        time_ns: f64,
    ) -> f64 {
        let flops = d_inner as f64 * n_tok as f64 * (d_conv * 2) as f64;
        flops / time_ns * 1000.0
    }

    fn compute_scan_gflops(
        d_inner: usize,
        n_tok: usize,
        d_state: usize,
        time_ns: f64,
    ) -> f64 {
        let flops =
            d_inner as f64 * n_tok as f64 * (10.0 + d_state as f64 * 5.0);
        flops / time_ns * 1000.0
    }

    /// Benchmark SSM Conv1D with batched command buffer
    pub fn bench_ssm_conv1d() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let d_conv = 4;
        let d_inner = 3328; // Match llama.cpp test
        let n_tok = 512;

        let x_size = d_inner * (n_tok + d_conv - 1);
        let w_size = d_inner * d_conv;

        let x: Vec<f32> =
            (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
        let w: Vec<f32> =
            (0..w_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
        let bias: Vec<f32> =
            (0..d_inner).map(|i| 0.01 * (i as f32 % 5.0)).collect();

        // Create buffers once
        let x_buffer = ctx.device().new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            (x_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w_buffer = ctx.device().new_buffer_with_data(
            w.as_ptr() as *const std::ffi::c_void,
            (w_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let bias_buffer = ctx.device().new_buffer_with_data(
            bias.as_ptr() as *const std::ffi::c_void,
            (d_inner * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buffer = ctx.device().new_buffer(
            (d_inner * n_tok * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Warm up
        for _ in 0..10 {
            run_conv1d_kernel(
                &ctx,
                &x_buffer,
                &w_buffer,
                &bias_buffer,
                &y_buffer,
                d_inner,
                n_tok,
                5,
            );
        }

        // Measure with batched command buffer
        let iterations = 50;
        let kernels_per_cb = 10;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            run_conv1d_kernel(
                &ctx,
                &x_buffer,
                &w_buffer,
                &bias_buffer,
                &y_buffer,
                d_inner,
                n_tok,
                kernels_per_cb,
            );
            times.push(
                start.elapsed().as_nanos() as f64 / kernels_per_cb as f64,
            );
        }

        let mean_ns = times.iter().sum::<f64>() / times.len() as f64;
        let mean_us = mean_ns / 1000.0;
        let gflops = compute_gflops(d_inner, n_tok, d_conv, mean_ns);

        println!(
            "\n=== Conv1D (d_conv=4, d_inner={}, n_tok={}) ===",
            d_inner, n_tok
        );
        println!("time: {:.2} us", mean_us);
        println!("GFLOPS: {:.2}", gflops);
        println!("llama.cpp: 111.51 us, 122.24 GFLOPS");
        println!("Ratio: {:.2}x", mean_us / 111.51);
    }

    fn run_conv1d_kernel(
        ctx: &MetalContext,
        x_buffer: &Buffer,
        w_buffer: &Buffer,
        bias_buffer: &Buffer,
        y_buffer: &Buffer,
        d_inner: usize,
        n_tok: usize,
        count: usize,
    ) {
        let batch_size = if n_tok > 128 {
            256usize
        } else if n_tok > 64 {
            128
        } else if n_tok > 32 {
            64
        } else if n_tok > 16 {
            32
        } else if n_tok > 8 {
            16
        } else if n_tok > 4 {
            8
        } else {
            2
        };
        let n_token_batches = (n_tok + batch_size - 1) / batch_size;

        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(
            ctx.ssm_conv1d_d4_batched_pipeline(batch_size),
        );
        encoder.set_buffer(0, Some(x_buffer), 0);
        encoder.set_buffer(1, Some(w_buffer), 0);
        encoder.set_buffer(2, Some(bias_buffer), 0);
        encoder.set_buffer(3, Some(y_buffer), 0);
        let d_inner_u32 = d_inner as u32;
        let n_tok_u32 = n_tok as u32;
        encoder.set_bytes(
            4,
            4,
            &d_inner_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &n_tok_u32 as *const u32 as *const std::ffi::c_void,
        );

        let grid_size = MTLSize {
            width: d_inner as u64,
            height: n_token_batches as u64,
            depth: 1,
        };
        let threadgroup_size =
            MTLSize { width: batch_size as u64, height: 1, depth: 1 };

        for _ in 0..count {
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Benchmark SSM Scan (Mamba-1, d_state=16)
    pub fn bench_ssm_scan() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let d_state = 16;
        let d_inner = 4096;
        let n_tok = 512;

        let x_size = d_inner * n_tok;
        let a_size = d_inner * d_state;
        let bc_size = d_state * n_tok;

        let x: Vec<f32> =
            (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
        let dt: Vec<f32> = (0..d_inner * n_tok)
            .map(|i| 0.001 * (i as f32 % 5.0 + 1.0))
            .collect();
        let a: Vec<f32> =
            (0..a_size).map(|i| -0.1 * ((i as f32 % 10.0) / 10.0)).collect();
        let b: Vec<f32> =
            (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
        let c: Vec<f32> =
            (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();

        // Warm up and measure
        for _ in 0..10 {
            let _ = ctx.ssm_scan_d16(&x, &dt, &a, &b, &c, None, d_inner, n_tok);
        }

        let iterations = 50;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = ctx.ssm_scan_d16(&x, &dt, &a, &b, &c, None, d_inner, n_tok);
            times.push(start.elapsed().as_nanos() as f64);
        }

        let mean_ns = times.iter().sum::<f64>() / times.len() as f64;
        let mean_us = mean_ns / 1000.0;
        let gflops = compute_scan_gflops(d_inner, n_tok, d_state, mean_ns);

        println!(
            "\n=== SSM Scan (d_state=16, d_inner={}, n_tok={}) ===",
            d_inner, n_tok
        );
        println!("time: {:.2} us", mean_us);
        println!("GFLOPS: {:.2}", gflops);
        println!(
            "Note: No direct llama.cpp benchmark available for d_state=16"
        );
    }

    /// Benchmark SSD Scan (Mamba-2, d_state=64) with buffer reuse
    pub fn bench_ssd_scan() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let d_state = 64;
        let head_dim = 64;
        let n_tok = 512;
        let n_head = 1;

        let x_size = n_tok * head_dim;
        let bc_size = n_tok * d_state;
        let state_size = head_dim * d_state;
        let y_size = n_tok * n_head * head_dim;

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
        let s0_buffer = ctx.device().new_buffer(
            (state_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
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
        let output_buffer = ctx.device().new_buffer(
            ((y_size + state_size) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Threadgroup memory for CS=64
        let threadgroup_mem = (64 * 3 + 64 * 64 + 4 * 8 * 64 + 4 * 2 * 64) * 4;

        // Warm up
        for _ in 0..10 {
            run_ssd_kernel(
                &ctx,
                &s0_buffer,
                &x_buffer,
                &dt_buffer,
                &a_buffer,
                &b_buffer,
                &c_buffer,
                &output_buffer,
                head_dim,
                n_tok,
                d_state,
                n_head,
                threadgroup_mem,
                3,
            );
        }

        // Measure with batched command buffer
        let iterations = 50;
        let kernels_per_cb = 5;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            run_ssd_kernel(
                &ctx,
                &s0_buffer,
                &x_buffer,
                &dt_buffer,
                &a_buffer,
                &b_buffer,
                &c_buffer,
                &output_buffer,
                head_dim,
                n_tok,
                d_state,
                n_head,
                threadgroup_mem,
                kernels_per_cb,
            );
            times.push(
                start.elapsed().as_nanos() as f64 / kernels_per_cb as f64,
            );
        }

        let mean_ns = times.iter().sum::<f64>() / times.len() as f64;
        let mean_us = mean_ns / 1000.0;
        let gflops = compute_scan_gflops(head_dim, n_tok, d_state, mean_ns);

        println!(
            "\n=== SSD (d_state=64, head_dim={}, n_tok={}) ===",
            head_dim, n_tok
        );
        println!("time: {:.2} us", mean_us);
        println!("GFLOPS: {:.2}", gflops);
        println!("llama.cpp: ~8000-10000 GFLOPS");
        println!("Gap: {:.1}x", 10000.0 / gflops);
    }

    fn run_ssd_kernel(
        ctx: &MetalContext,
        s0_buffer: &Buffer,
        x_buffer: &Buffer,
        dt_buffer: &Buffer,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        c_buffer: &Buffer,
        output_buffer: &Buffer,
        head_dim: usize,
        n_tok: usize,
        d_state: usize,
        n_head: usize,
        threadgroup_mem: usize,
        count: usize,
    ) {
        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.ssm_scan_ssd_mma_pipeline());
        encoder.set_buffer(0, Some(s0_buffer), 0);
        encoder.set_buffer(1, Some(x_buffer), 0);
        encoder.set_buffer(2, Some(dt_buffer), 0);
        encoder.set_buffer(3, Some(a_buffer), 0);
        encoder.set_buffer(4, Some(b_buffer), 0);
        encoder.set_buffer(5, Some(c_buffer), 0);
        encoder.set_buffer(7, Some(output_buffer), 0);

        let d_state_u32 = d_state as u32;
        let head_dim_u32 = head_dim as u32;
        let n_head_u32 = n_head as u32;
        let n_tok_u32 = n_tok as u32;
        let n_seqs_u32 = 1u32;

        encoder.set_bytes(
            8,
            4,
            &d_state_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            9,
            4,
            &head_dim_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            10,
            4,
            &n_head_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            11,
            4,
            &n_tok_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            12,
            4,
            &n_seqs_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, threadgroup_mem as u64);

        // Grid: one threadgroup per head, 128 threads per threadgroup
        let grid_size = MTLSize { width: 1, height: n_head as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: 128, height: 1, depth: 1 };

        for _ in 0..count {
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    pub fn run_benchmarks() {
        println!("\n========================================");
        println!("SSM GPU Performance (Correct Measurement)");
        println!("========================================");

        bench_ssm_conv1d();
        bench_ssm_scan();
        bench_ssd_scan();
    }
}

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        bench::run_benchmarks();
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support on aarch64");
    }
}
