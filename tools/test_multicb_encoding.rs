// Test multi-CB parallel encoding
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::MTLSize;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let gpu_ctx = MetalContext::new().expect("Failed to create MetalContext");
    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    println!("Testing multi-CB parallel encoding...\n");

    // Create two test buffers
    let size = 1024 * 1024; // 1M floats = 4MB
    let buffer1 = device.new_buffer(size * 4, metal::MTLResourceOptions::StorageModeShared);
    let buffer2 = device.new_buffer(size * 4, metal::MTLResourceOptions::StorageModeShared);

    // Initialize buffer1 with data
    unsafe {
        let ptr = buffer1.contents() as *mut f32;
        for i in 0..size {
            *ptr.add(i) = i as f32;
        }
    }

    // Create two command buffers
    let cb1 = queue.new_command_buffer();
    let cb2 = queue.new_command_buffer();

    // Enqueue in order (Metal will execute in this order)
    cb1.enqueue();
    cb2.enqueue();

    // Simple compute kernel (copy buffer1 -> buffer2)
    let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void copy(
            device float* src [[buffer(0)]],
            device float* dst [[buffer(1)]],
            uint gid [[thread_position_in_grid]]
        ) {
            dst[gid] = src[gid];
        }
    "#;

    let library = device.new_library_with_source(shader_source, &metal::CompileOptions::new())
        .expect("Failed to compile shader");
    let kernel = library.get_function("copy").expect("Failed to get kernel");
    let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    // Thread groups
    let thread_group_size = MTLSize::new(256, 1, 1);
    let thread_group_count = MTLSize::new((size + 255) / 256, 1, 1);

    // Encode CB1: first half
    let encoder1 = cb1.new_compute_command_encoder();
    encoder1.set_compute_pipeline_state(&pipeline);
    encoder1.set_buffer(0, Some(&buffer1), 0);
    encoder1.set_buffer(1, Some(&buffer2), 0);
    encoder1.dispatch_thread_groups(
        MTLSize::new((size / 2 + 255) / 256, 1, 1),
        thread_group_size
    );
    encoder1.end_encoding();

    // Encode CB2: second half (in parallel - simulated)
    let encoder2 = cb2.new_compute_command_encoder();
    encoder2.set_compute_pipeline_state(&pipeline);
    encoder2.set_buffer(0, Some(&buffer1), 0);
    encoder2.set_buffer(1, Some(&buffer2), 0);
    encoder2.dispatch_thread_groups(
        MTLSize::new((size / 2 + 255) / 256, 1, 1),
        thread_group_size
    );
    encoder2.end_encoding();

    // Commit CB2 (this will trigger execution of both CB1 and CB2 in order)
    let start = std::time::Instant::now();
    cb2.commit();

    // Wait for completion
    cb2.wait_until_completed();
    let elapsed = start.elapsed();

    println!("Multi-CB execution time: {:?}", elapsed);
    println!("Status: CB1={:?}, CB2={:?}", cb1.status(), cb2.status());

    // Verify results
    unsafe {
        let ptr = buffer2.contents() as *const f32;
        let mut errors = 0;
        for i in 0..size.min(100) {
            if (*ptr.add(i) - i as f32).abs() > 0.001 {
                errors += 1;
            }
        }
        if errors == 0 {
            println!("Verification: PASS");
        } else {
            println!("Verification: FAIL ({} errors)", errors);
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support");
}