// Measure Metal dispatch overhead
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::{Device, MTLSize};
use std::rc::Rc;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Simple kernel that does minimal work
    let source = r#"
        kernel void dummy(
            device float* out [[buffer(0)]],
            uint idx [[thread_position_in_grid]])
        {
            out[idx] = float(idx);
        }
    "#;

    let compile_options = metal::CompileOptions::new();
    let library =
        device.new_library_with_source(source, &compile_options).unwrap();
    let function = library.get_function("dummy", None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&function).unwrap();

    let buffer = Rc::new(
        device
            .new_buffer(1024 * 4, metal::MTLResourceOptions::StorageModeShared),
    );

    println!("Metal Dispatch Overhead Analysis");
    println!("Device: {}\n", device.name());

    // Test 1: Single dispatch
    let iterations = 1000;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.dispatch_thread_groups(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 1, height: 1, depth: 1 },
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    let single_time = start.elapsed();
    let single_us = single_time.as_micros() as f64 / iterations as f64;

    println!("Single dispatch:");
    println!("  Total: {:?}", single_time);
    println!("  Per dispatch: {:.2} us", single_us);

    // Test 2: Multiple dispatches in one command buffer
    let start = std::time::Instant::now();
    let dispatches_per_cb = 32; // Match n_heads=32

    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        for _ in 0..dispatches_per_cb {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
            encoder.end_encoding();
        }
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    let batched_time = start.elapsed();
    let batched_us = batched_time.as_micros() as f64 / iterations as f64;
    let per_dispatch_us = batched_us / dispatches_per_cb as f64;

    println!("\nBatched dispatch ({} per CB):", dispatches_per_cb);
    println!("  Total: {:?}", batched_time);
    println!("  Per CB: {:.2} us", batched_us);
    println!("  Per dispatch: {:.2} us", per_dispatch_us);

    // Test 3: Single dispatch with more threads
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.dispatch_thread_groups(
            MTLSize { width: 32, height: 1, depth: 1 }, // 32x more threads
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    let multi_thread_time = start.elapsed();
    let multi_thread_us =
        multi_thread_time.as_micros() as f64 / iterations as f64;

    println!("\nSingle dispatch (32x threads):");
    println!("  Total: {:?}", multi_thread_time);
    println!("  Per dispatch: {:.2} us", multi_thread_us);

    // Overhead analysis
    println!("\nOverhead analysis:");
    println!("  CB commit + wait overhead: ~{:.0} us", single_us);
    println!("  Flash Attention kernel time @ seq_k=1024: ~620 us");
    println!("  Overhead percentage: ~{:.0}%", single_us / 620.0 * 100.0);
}
