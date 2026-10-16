// Test if batching multiple kernels into one command buffer reduces overhead
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let n = 1024;
        let num_kernels = 50;

        // Create buffers
        let input = vec![0.1f32; n];
        let input_buffer = ctx.device().new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (n * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Approach 1: Individual dispatches (current approach)
        println!("Testing {} individual dispatches", num_kernels);
        let mut individual_times = Vec::with_capacity(100);
        for _ in 0..100 {
            let start = std::time::Instant::now();
            for _ in 0..num_kernels {
                let output_buffer = ctx.device().new_buffer(
                    (n * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let cmd_buf = ctx.queue().new_command_buffer();
                let encoder = cmd_buf.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&ctx.add_pipeline());
                encoder.set_buffer(0, Some(&input_buffer), 0);
                encoder.set_buffer(1, Some(&input_buffer), 0);
                encoder.set_buffer(2, Some(&output_buffer), 0);
                encoder.dispatch_thread_groups(
                    metal::MTLSize {
                        width: (n as u64 + 255) / 256,
                        height: 1,
                        depth: 1,
                    },
                    metal::MTLSize { width: 256, height: 1, depth: 1 },
                );
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }
            individual_times.push(start.elapsed());
        }

        let avg_individual =
            individual_times.iter().map(|t| t.as_micros() as f64).sum::<f64>()
                / individual_times.len() as f64;
        println!("  Average: {:.2} us", avg_individual);
        println!("  Per kernel: {:.2} us", avg_individual / num_kernels as f64);

        // Approach 2: Batched dispatches (one command buffer for all kernels)
        println!(
            "\nTesting {} batched dispatches in one command buffer",
            num_kernels
        );
        let mut batched_times = Vec::with_capacity(100);
        for _ in 0..100 {
            let start = std::time::Instant::now();

            // Create output buffers upfront
            let output_buffers: Vec<_> = (0..num_kernels)
                .map(|_| {
                    ctx.device().new_buffer(
                        (n * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    )
                })
                .collect();

            // Create one command buffer
            let cmd_buf = ctx.queue().new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();

            // Encode all kernels
            for i in 0..num_kernels {
                encoder.set_compute_pipeline_state(&ctx.add_pipeline());
                encoder.set_buffer(0, Some(&input_buffer), 0);
                encoder.set_buffer(1, Some(&input_buffer), 0);
                encoder.set_buffer(2, Some(&output_buffers[i]), 0);
                encoder.dispatch_thread_groups(
                    metal::MTLSize {
                        width: (n as u64 + 255) / 256,
                        height: 1,
                        depth: 1,
                    },
                    metal::MTLSize { width: 256, height: 1, depth: 1 },
                );
            }

            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            batched_times.push(start.elapsed());
        }

        let avg_batched =
            batched_times.iter().map(|t| t.as_micros() as f64).sum::<f64>()
                / batched_times.len() as f64;
        println!("  Average: {:.2} us", avg_batched);
        println!("  Per kernel: {:.2} us", avg_batched / num_kernels as f64);

        println!("\n  Speedup: {:.2}x", avg_individual / avg_batched);
    }
}
