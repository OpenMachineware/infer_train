// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile individual kernel execution times
//!
//! Usage: cargo run --release --bin profile-kernel-times

#[cfg(feature = "metal")]
fn main() {
    use metal::*;
    use std::time::Instant;

    println!("Kernel Execution Time Profiler");
    println!("===============================");
    println!();

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Create a simple test: Q4_K GEMV with 1024 rows, 2048 cols, batch=1
    // This mimics a typical decode projection

    let m = 1024; // output rows
    let k = 2048; // input dimension
    let block_size = 144; // Q4_K block size

    // Create test data
    let weights_size = (m * block_size) as u64;
    let input_size = (k * std::mem::size_of::<f32>()) as u64;
    let output_size = (m * std::mem::size_of::<f32>()) as u64;

    let weights = device.new_buffer(weights_size, MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer(input_size, MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer(output_size, MTLResourceOptions::StorageModeShared);

    // Initialize with test data
    unsafe {
        let weights_ptr = weights.contents() as *mut u8;
        for i in 0..weights_size as usize {
            *weights_ptr.add(i) = (i % 256) as u8;
        }

        let input_ptr = input.contents() as *mut f32;
        for i in 0..k {
            *input_ptr.add(i) = (i as f32) / 1000.0;
        }
    }

    println!("Test Configuration:");
    println!("  Weights: {} x Q4_K blocks ({} bytes)", m, weights_size);
    println!("  Input: {} x f32 ({} bytes)", k, input_size);
    println!("  Output: {} x f32 ({} bytes)", m, output_size);
    println!();

    // Measure dispatch overhead
    println!("Measuring dispatch overhead...");

    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        // Minimal dispatch - just to measure overhead
        drop(encoder);
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    let elapsed = start.elapsed();
    let overhead_per_dispatch = elapsed.as_micros() as f64 / iterations as f64;

    println!("  Total time for {} dispatches: {:?}", iterations, elapsed);
    println!("  Average dispatch overhead: {:.2} µs", overhead_per_dispatch);
    println!();

    // Now measure actual kernel execution
    // We need to load our Metal library
    println!("To measure actual kernel time, run:");
    println!("  cargo run --release --bin bench-decode -- --model Qwen3-0.6B-UD-Q4_K_XL.gguf");
}

#[cfg(not(feature = "metal"))]
fn main() {
    println!("This tool requires Metal support");
}