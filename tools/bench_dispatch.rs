// Benchmark Metal dispatch overhead
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device,
    Library, MTLSize,
};

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Simple kernel that does nothing
    let source = r#"
        kernel void dummy(
            device float* out [[buffer(0)]],
            uint idx [[thread_position_in_grid]])
        {
            out[idx] = float(idx);
        }
    "#;

    let compile_options = CompileOptions::new();
    let library =
        device.new_library_with_source(source, &compile_options).unwrap();
    let function = library.get_function("dummy", None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&function).unwrap();

    let buffer = device
        .new_buffer(1024 * 4, metal::MTLResourceOptions::StorageModeShared);

    // Test 1: Individual commits (what we're doing)
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

    let individual_time = start.elapsed();
    println!("Individual commits: {:?}", individual_time);
    println!(
        "  Per operation: {:.3}µs",
        individual_time.as_micros() as f64 / iterations as f64
    );

    // Test 2: Batched commits (what llama.cpp does)
    let start = std::time::Instant::now();

    let cmd_buf = queue.new_command_buffer();
    for _ in 0..iterations {
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

    let batched_time = start.elapsed();
    println!("\nBatched commits: {:?}", batched_time);
    println!(
        "  Per operation: {:.3}µs",
        batched_time.as_micros() as f64 / iterations as f64
    );

    println!(
        "\nOverhead ratio: {:.1}x",
        individual_time.as_micros() as f64 / batched_time.as_micros() as f64
    );
}
