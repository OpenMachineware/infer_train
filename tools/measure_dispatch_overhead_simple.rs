// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Measure Metal dispatch overhead - minimal version
//!
//! Usage: cargo run --release --bin measure-dispatch-overhead

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::*;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Metal Dispatch Overhead Measurement ===\n");

        let device = Device::system_default().expect("No Metal device");
        let queue = device.new_command_queue();

        // Minimal test: just dispatch empty command buffers
        let iterations = 100;

        // Warmup
        for _ in 0..10 {
            let cb = queue.new_command_buffer();
            cb.commit();
            cb.wait_until_completed();
        }

        // Sequential dispatch (one CB, wait each time)
        println!("Measuring sequential dispatch...");
        let seq_start = std::time::Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            // No actual kernel, just measuring overhead
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let seq_time = seq_start.elapsed();

        // Calculate overhead
        let avg_seq_us = seq_time.as_micros() as f64 / iterations as f64;

        println!("Problem size: Empty kernel (overhead only)");
        println!("Iterations: {}", iterations);
        println!("Average time per dispatch: {:.2} µs", avg_seq_us);
        println!();

        // Now measure with actual kernel - Q4_K GEMV
        println!("To measure actual kernel time with Q4_K:");
        println!("  1. Use existing benchmark tools");
        println!("  2. Run bench-q4k-gemv-vs-llama");
        println!("  3. Compare single kernel timing");

        println!("\n=== Measurement Complete ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}