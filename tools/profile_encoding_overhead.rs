// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile Metal encoding overhead - how long does it take to encode 564 kernel dispatches?

use std::time::Instant;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::MTLResourceOptions;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let ctx = MetalContext::new().expect("Failed to create Metal context");
        let device = ctx.device();
        let queue = ctx.queue();

        // Create dummy buffers
        let buf_size = 1024 * 1024; // 1 MB
        let buf_a = device.new_buffer(buf_size as u64, MTLResourceOptions::StorageModeShared);
        let buf_b = device.new_buffer(buf_size as u64, MTLResourceOptions::StorageModeShared);
        let buf_c = device.new_buffer(buf_size as u64, MTLResourceOptions::StorageModeShared);

        // Test encoding 564 dummy kernel dispatches
        let n_dispatches = 564;
        let iterations = 10;

        let mut encode_times: Vec<u64> = Vec::new();
        let mut commit_times: Vec<u64> = Vec::new();
        let mut total_times: Vec<u64> = Vec::new();

        for _ in 0..iterations {
            // Measure encoding time
            let encode_start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            for _ in 0..n_dispatches {
                // Dummy kernel dispatch - just set buffer references
                encoder.set_buffer(0, Some(&buf_a), 0);
                encoder.set_buffer(1, Some(&buf_b), 0);
                encoder.set_buffer(2, Some(&buf_c), 0);
            }

            let encode_time = encode_start.elapsed().as_micros() as u64;

            // Measure commit + wait time
            let commit_start = Instant::now();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let commit_time = commit_start.elapsed().as_micros() as u64;

            encode_times.push(encode_time);
            commit_times.push(commit_time);
            total_times.push(encode_time + commit_time);
        }

        println!("=== Encoding Overhead Analysis ===");
        println!("Dispatches: {}", n_dispatches);
        println!("\nEncoding time (µs):");
        println!("  Min: {}", encode_times.iter().min().unwrap());
        println!("  Max: {}", encode_times.iter().max().unwrap());
        println!("  Avg: {}", encode_times.iter().sum::<u64>() / encode_times.len() as u64);
        println!("  Per dispatch: {:.2} µs", encode_times.iter().sum::<u64>() as f64 / encode_times.len() as f64 / n_dispatches as f64);

        println!("\nCommit + wait time (µs):");
        println!("  Min: {}", commit_times.iter().min().unwrap());
        println!("  Max: {}", commit_times.iter().max().unwrap());
        println!("  Avg: {}", commit_times.iter().sum::<u64>() / commit_times.len() as u64);

        println!("\nTotal time (µs):");
        println!("  Min: {}", total_times.iter().min().unwrap());
        println!("  Max: {}", total_times.iter().max().unwrap());
        println!("  Avg: {}", total_times.iter().sum::<u64>() / total_times.len() as u64);

        // Compare with actual decode time
        let actual_decode_us = 8219; // From our profiler
        println!("\n=== Comparison ===");
        println!("Encoding overhead: {} µs", encode_times.iter().sum::<u64>() / encode_times.len() as u64);
        println!("Actual decode time: {} µs", actual_decode_us);
        println!("Overhead percentage: {:.1}%", (encode_times.iter().sum::<u64>() / encode_times.len() as u64) as f64 / actual_decode_us as f64 * 100.0);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
