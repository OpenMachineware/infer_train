// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Simple Metal dispatch overhead measurement
//!
//! Usage: cargo run --release --bin simple-dispatch-timing

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::*;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Metal Dispatch Overhead Measurement ===\n");

        let device = Device::system_default().expect("No Metal device");
        let queue = device.new_command_queue();

        // Create a minimal compute pipeline (no-op kernel)
        let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void noop_kernel(device float *output [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
            if (idx == 0) {
                output[0] = 0.0;
            }
        }
        "#;

        let library = device.new_library_with_source(source, &CompileOptions::new())
            .expect("Failed to compile library");
        let kernel = library.get_function("noop_kernel", None).expect("Failed to get kernel");
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline");

        let output_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        let iterations = 100;

        // Warmup
        for _ in 0..10 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&output_buffer), 0);
            encoder.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Measure single dispatch overhead
        println!("Measuring single dispatch timing...");
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&output_buffer), 0);
            encoder.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let single_time = start.elapsed();

        // Measure batched dispatch
        println!("Measuring batched dispatch timing...");
        let start = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        for _ in 0..iterations {
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&output_buffer), 0);
            encoder.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
        let batched_time = start.elapsed();

        // Calculate overhead
        let single_us = single_time.as_micros() as f64 / iterations as f64;
        let batched_us = batched_time.as_micros() as f64 / iterations as f64;
        let overhead_us = single_us - batched_us;

        println!("\nResults:");
        println!("  Single dispatch: {:.2} µs/dispatch", single_us);
        println!("  Batched dispatch: {:.2} µs/dispatch", batched_us);
        println!("  Dispatch overhead: {:.2} µs ({:.1}%)", overhead_us, (overhead_us / single_us) * 100.0);
        println!("  Kernel time: {:.2} µs ({:.1}%)", batched_us, (batched_us / single_us) * 100.0);

        println!("\n=== Measurement Complete ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}