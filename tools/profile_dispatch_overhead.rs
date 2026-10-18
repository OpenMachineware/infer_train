// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Measure: Command buffer dispatch overhead

use metal::{Device, MTLResourceOptions};

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("\n=== Command Buffer Dispatch Overhead ===\n");

    // Create a tiny buffer for minimal kernel
    let buffer = device.new_buffer(16, MTLResourceOptions::StorageModeShared);

    // Create a minimal compute pipeline (just copy)
    let source = r#"
        kernel void dummy(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            if (id < 4) out[id] = 0.0;
        }
    "#;

    let library = device
        .new_library_with_source(source, &metal::CompileOptions::new())
        .unwrap();
    let function = library.get_function("dummy", None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&function).unwrap();

    // Test 1: Single kernel, single command buffer
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize { width: 1, height: 1, depth: 1 },
            metal::MTLSize { width: 1, height: 1, depth: 1 },
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let single_time = start.elapsed();

    println!("1000 single kernel dispatches: {:?}", single_time);
    println!("Per dispatch: {:?}", single_time / 1000);

    // Test 2: 18 kernels per command buffer (simulating one layer)
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let cb = queue.new_command_buffer();
        for _ in 0..18 {
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&buffer), 0);
            encoder.dispatch_thread_groups(
                metal::MTLSize { width: 1, height: 1, depth: 1 },
                metal::MTLSize { width: 1, height: 1, depth: 1 },
            );
            encoder.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
    }
    let batched_time = start.elapsed();

    println!("\n1000 batched (18 kernels per CB): {:?}", batched_time);
    println!("Per batch: {:?}", batched_time / 1000);
    println!("Per kernel in batch: {:?}", batched_time / 1000 / 18);

    // Test 3: 504 kernels in single command buffer (simulating all 28 layers)
    let start = std::time::Instant::now();
    let cb = queue.new_command_buffer();
    for _ in 0..504 {
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize { width: 1, height: 1, depth: 1 },
            metal::MTLSize { width: 1, height: 1, depth: 1 },
        );
        encoder.end_encoding();
    }
    cb.commit();
    cb.wait_until_completed();
    let all_in_one = start.elapsed();

    println!("\n504 kernels in single CB: {:?}", all_in_one);
    println!("Per kernel: {:?}", all_in_one / 504);

    println!("\n=== Summary ===");
    println!("Separate CBs for 504 kernels: {:?}", single_time / 1000 * 504);
    println!("Single CB for 504 kernels: {:?}", all_in_one);
    println!(
        "Overhead ratio: {:.1}x",
        (single_time / 1000 * 504).as_secs_f64() / all_in_one.as_secs_f64()
    );
}
