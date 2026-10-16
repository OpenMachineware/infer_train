// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Performance test: Our F16 GEMV basic kernel vs llama.cpp native kernel

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::{Device, MTLSize};

    let ctx = MetalContext::new().expect("Failed to create MetalContext");
    let device = ctx.device();

    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Load our kernel (reference implementation)
    let our_source =
        include_str!("../src/ops/metal/gemv/mul_mv_f16_kernels.metal");
    let our_library = device
        .new_library_with_source(our_source, &compile_options)
        .expect("Failed to compile library");
    let our_kernel = our_library
        .get_function("kernel_mul_mv_f16_f32", None)
        .expect("Failed to get kernel");
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(&our_kernel)
        .expect("Failed to create pipeline");

    // Test sizes for basic kernel (K >= 32, K % 4 != 0)
    let test_sizes = vec![(33, 4096), (65, 4096), (129, 4096), (257, 4096)];
    let iterations = 500;

    println!("F16 GEMV Basic Kernel Performance Test (K >= 32, K % 4 != 0)");
    println!(
        "===============================================================\n"
    );

    for (k, m) in test_sizes {
        // Generate test data
        let weights_f16: Vec<u16> = (0..m * k)
            .map(|i| ((i as f32 * 0.1).sin() * 100.0) as u16)
            .collect();
        let input_f32: Vec<f32> =
            (0..k).map(|i| (i as f32 * 0.05).cos()).collect();

        // Create buffers
        let weights_buffer = device.new_buffer_with_data(
            weights_f16.as_ptr() as *const std::ffi::c_void,
            (m * k * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input_f32.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let mut our_times = Vec::new();
        let mut ref_times = Vec::new();

        // Alternating A-B-A-B pattern
        for i in 0..iterations {
            if i % 2 == 0 {
                // Test MetalContext API
                let start = std::time::Instant::now();
                let _ = ctx.mv_f16_f32_gpu_weights(
                    m,
                    k,
                    &weights_buffer,
                    &input_buffer,
                );
                our_times.push(start.elapsed().as_nanos() as f64);
            } else {
                // Test direct kernel call
                let output_buffer = device.new_buffer(
                    (m * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let queue = device.new_command_queue();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&our_pipeline);

                #[repr(C)]
                #[derive(Clone, Copy)]
                struct MvArgs {
                    ne00: i32,
                    ne01: i32,
                    ne02: i32,
                    nb00: u64,
                    nb01: u64,
                    nb02: u64,
                    nb03: u64,
                    ne10: i32,
                    ne11: i32,
                    ne12: i32,
                    nb10: u64,
                    nb11: u64,
                    nb12: u64,
                    nb13: u64,
                    ne0: i32,
                    ne1: i32,
                    nr0: i32,
                    r2: i16,
                    r3: i16,
                    nsg: i16,
                    ne12_batch: i16,
                }

                let args = MvArgs {
                    ne00: k as i32,
                    ne01: m as i32,
                    ne02: 1,
                    nb00: 2,
                    nb01: (k * 2) as u64,
                    nb02: 0,
                    nb03: 0,
                    ne10: k as i32,
                    ne11: 1,
                    ne12: 1,
                    nb10: 4,
                    nb11: (k * 4) as u64,
                    nb12: 0,
                    nb13: 0,
                    ne0: m as i32,
                    ne1: 1,
                    nr0: 2,
                    r2: 1,
                    r3: 1,
                    nsg: 1,
                    ne12_batch: 1,
                };

                enc.set_bytes(
                    0,
                    std::mem::size_of::<MvArgs>() as u64,
                    &args as *const MvArgs as *const std::ffi::c_void,
                );
                enc.set_buffer(1, Some(&weights_buffer), 0);
                enc.set_buffer(2, Some(&input_buffer), 0);
                enc.set_buffer(3, Some(&output_buffer), 0);

                let thread_group_size =
                    MTLSize { width: 32, height: 1, depth: 1 };
                let thread_group_count = MTLSize {
                    width: ((m + 1) / 2) as u64,
                    height: 1,
                    depth: 1,
                };

                let start = std::time::Instant::now();
                enc.dispatch_thread_groups(
                    thread_group_count,
                    thread_group_size,
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                ref_times.push(start.elapsed().as_nanos() as f64);
            }
        }

        // Calculate averages
        let our_avg: f64 =
            our_times.iter().sum::<f64>() / our_times.len() as f64;
        let ref_avg: f64 =
            ref_times.iter().sum::<f64>() / ref_times.len() as f64;

        let our_us = our_avg / 1000.0;
        let ref_us = ref_avg / 1000.0;
        let speedup = ref_avg / our_avg;

        println!("K={}, M={}", k, m);
        println!("  API:  {:.3}µs", our_us);
        println!("  Direct: {:.3}µs", ref_us);
        println!("  Speedup: {:.2}x", speedup);

        if speedup > 1.0 {
            println!("  ✓ {:.1}% faster\n", (speedup - 1.0) * 100.0);
        } else {
            println!("  ✗ {:.1}% slower\n", (1.0 - speedup) * 100.0);
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support on aarch64");
}
