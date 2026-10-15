// Direct Conv1D comparison with buffer reuse
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod tests {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::*;
    use std::time::Instant;

    pub fn test_conv1d_comparison() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        // Match llama.cpp test parameters
        let d_conv = 4;
        let d_inner = 3328; // Match llama.cpp
        let n_tok = 512;

        let x_size = d_inner * (n_tok + d_conv - 1);
        let w_size = d_inner * d_conv;

        let x: Vec<f32> =
            (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
        let w: Vec<f32> =
            (0..w_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
        let bias: Vec<f32> =
            (0..d_inner).map(|i| 0.01 * (i as f32 % 5.0)).collect();

        // Create buffers ONCE and reuse
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
            let _ = run_conv1d_kernels(
                &ctx,
                &x_buffer,
                &w_buffer,
                &bias_buffer,
                &y_buffer,
                d_inner,
                n_tok,
                1,
            );
        }

        // Measure: run 10 kernels per command buffer to amortize sync overhead
        let iterations = 100;
        let kernels_per_cb = 10;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            run_conv1d_kernels(
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

        println!(
            "iterations: {}, kernels per CB: {}",
            iterations, kernels_per_cb
        );
        println!("mean time: {:.2} us per kernel", mean_us);

        let flops = d_inner * n_tok * d_conv * 2;
        let gflops = flops as f64 / mean_ns * 1000.0;

        // Memory bandwidth
        let bytes = (x_size + w_size + d_inner + d_inner * n_tok) * 4;
        let gbps = bytes as f64 / mean_ns * 1000.0;

        println!(
            "\n=== Conv1D Direct Comparison (buffer reuse, batched CB) ==="
        );
        println!("d_inner={}, n_tok={}", d_inner, n_tok);
        println!("time: {:.2} us", mean_us);
        println!("GFLOPS: {:.2}", gflops);
        println!("GB/s: {:.2}", gbps);
        println!("\nllama.cpp result: 111.51 us, 122.24 GFLOPS, 123 GB/s");
        let gap = mean_us / 111.51;
        println!("Gap: {:.2}x", gap);
    }

    fn run_conv1d_kernels(
        ctx: &MetalContext,
        x_buffer: &Buffer,
        w_buffer: &Buffer,
        bias_buffer: &Buffer,
        y_buffer: &Buffer,
        d_inner: usize,
        n_tok: usize,
        count: usize,
    ) {
        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Batched dispatch
        let batch_size = if n_tok > 128 {
            256usize
        } else if n_tok > 64 {
            128usize
        } else if n_tok > 32 {
            64usize
        } else if n_tok > 16 {
            32usize
        } else if n_tok > 8 {
            16usize
        } else if n_tok > 4 {
            8usize
        } else {
            2usize
        };
        let n_token_batches = (n_tok + batch_size - 1) / batch_size;

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

        // Dispatch multiple times in the same command buffer
        for _ in 0..count {
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        tests::test_conv1d_comparison();
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This test requires Metal support on aarch64");
    }
}
