// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Measure per-dispatch overhead with actual kernel execution

use std::time::Instant;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::MTLResourceOptions;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLSize;

        let ctx = MetalContext::new().expect("Failed to create Metal context");
        let device = ctx.device();
        let queue = ctx.queue();

        // Create buffers that match GEMV sizes
        let input_size = 1024; // Typical hidden_size
        let output_size = 1024;
        let input_buf = device.new_buffer(
            (input_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buf = device.new_buffer(
            (input_size * output_size * 4 / 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ); // F16 weights
        let output_buf = device.new_buffer(
            (output_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Get F16 GEMV pipeline
        let pipeline = &ctx.mv_f16_basic_pipeline;

        // Measure dispatch overhead with minimal computation
        let n_dispatches = 10; // Small number to keep time reasonable
        let iterations = 20;

        let mut total_times: Vec<u64> = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            // Dispatch n times
            for _ in 0..n_dispatches {
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(&weight_buf), 0);
                encoder.set_buffer(1, Some(&input_buf), 0);
                encoder.set_buffer(2, Some(&output_buf), 0);

                let grid_size = MTLSize::new(1, 1, 1);
                let thread_size = MTLSize::new(32, 1, 1);
                encoder.dispatch_thread_groups(grid_size, thread_size);
            }

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let elapsed = start.elapsed().as_micros() as u64;
            total_times.push(elapsed);
        }

        println!("=== Dispatch Overhead with Kernel Execution ===");
        println!("Dispatches per iteration: {}", n_dispatches);
        println!("Grid size: 1x1 (minimal computation)");
        println!("\nTotal time (µs):");
        println!("  Min: {}", total_times.iter().min().unwrap());
        println!("  Max: {}", total_times.iter().max().unwrap());
        println!(
            "  Avg: {}",
            total_times.iter().sum::<u64>() / total_times.len() as u64
        );

        let avg_total =
            total_times.iter().sum::<u64>() / total_times.len() as u64;
        let per_dispatch = avg_total as f64 / n_dispatches as f64;
        println!("\nPer dispatch: {:.1} µs", per_dispatch);

        // Extrapolate to full decode
        println!("\n=== Extrapolation to Decode ===");
        let our_dispatches = 564;
        let actual_decode_us = 8910;
        let estimated_overhead = our_dispatches as f64 * per_dispatch;
        println!("Our dispatches: {}", our_dispatches);
        println!("Our decode time: {} µs", actual_decode_us);
        println!("Estimated dispatch overhead: {:.0} µs", estimated_overhead);
        println!(
            "Overhead percentage: {:.1}%",
            estimated_overhead / actual_decode_us as f64 * 100.0
        );

        // Hypothetical reduction
        for reduction in [2, 3, 4].iter() {
            let reduced_dispatches = our_dispatches / reduction;
            let reduced_overhead = reduced_dispatches as f64 * per_dispatch;
            let time_saved = estimated_overhead - reduced_overhead;
            println!(
                "\nIf dispatches reduced by {}x ({} dispatches):",
                reduction, reduced_dispatches
            );
            println!("  Estimated overhead: {:.0} µs", reduced_overhead);
            println!(
                "  Time saved: {:.0} µs ({:.1}%)",
                time_saved,
                time_saved / actual_decode_us as f64 * 100.0
            );
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
