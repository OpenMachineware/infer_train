// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Correctness test for F16 GEMV short kernel (K < 32)

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(false); // Disable fast math for accuracy

    let source = include_str!("../src/ops/metal/gemv/mul_mv_f16_kernels.metal");
    let library = match device.new_library_with_source(source, &compile_options)
    {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("Failed to compile Metal library: {}", e);
            std::process::exit(1);
        }
    };

    let kernel = match library.get_function("kernel_mul_mv_f16_f32_short", None)
    {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Failed to get kernel: {}", e);
            std::process::exit(1);
        }
    };
    let pipeline =
        match device.new_compute_pipeline_state_with_function(&kernel) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to create pipeline: {}", e);
                std::process::exit(1);
            }
        };

    println!("Metal kernel compiled successfully\n");

    // MvArgs structure matching metal kernel
    #[repr(C)]
    struct MvArgs {
        ne00: i32, // K
        ne01: i32, // M
        ne02: i32,
        nb00: u64, // sizeof(half) = 2
        nb01: u64, // row stride = K * 2
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        nb10: u64, // sizeof(float) = 4
        nb11: u64, // input stride = K * 4
        nb12: u64,
        nb13: u64,
        ne0: i32, // M (output rows)
        ne1: i32, // 1 (output cols)
        nr0: i32, // 32 for short kernel
        r2: i16,
        r3: i16,
        nsg: i16, // 1 for short kernel
        ne12_batch: i16,
    }

    // Test various K values < 32
    let test_k_sizes = [1, 2, 4, 8, 16, 24, 31];
    let m = 64; // Test with 64 rows

    println!("F16 GEMV Short Kernel Correctness Test (K < 32)");
    println!("================================================\n");

    for k in test_k_sizes {
        // Generate test data
        let weights_f16: Vec<u16> = (0..m * k)
            .map(|i| {
                half::f16::from_f32((i % 100) as f32 * 0.1 - 5.0).to_bits()
            })
            .collect();
        let input_f32: Vec<f32> =
            (0..k).map(|i| (i % 50) as f32 * 0.05 - 1.25).collect();

        // CPU reference computation
        let mut expected = vec![0.0f32; m];
        for row in 0..m {
            for col in 0..k {
                let w =
                    half::f16::from_bits(weights_f16[row * k + col]).to_f32();
                expected[row] += w * input_f32[col];
            }
        }

        // GPU computation
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
        let output_buffer = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

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
            nr0: 32,
            r2: 1,
            r3: 1,
            nsg: 1,
            ne12_batch: 1,
        };

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        // Note: args is the first parameter (buffer 0), then src0/src1/dst
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: ((m + 31) / 32) as u64, height: 1, depth: 1 };
        enc.dispatch_thread_groups(thread_group_count, thread_group_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
        };

        // Calculate errors
        let mut max_error = 0.0f32;
        let mut sum_error = 0.0f32;
        let mut error_count = 0;

        for i in 0..m {
            let error = (expected[i] - result[i]).abs();
            if error > max_error {
                max_error = error;
            }
            if error > 1e-4 {
                error_count += 1;
                sum_error += error;
            }
        }

        // Print first few results for inspection
        println!("K={:2}  M={}", k, m);
        println!("  First 5 rows:");
        for i in 0..5.min(m) {
            println!(
                "    [{:2}] Expected: {:10.4}, GPU: {:10.4}, Error: {:.6}",
                i,
                expected[i],
                result[i],
                (expected[i] - result[i]).abs()
            );
        }

        if max_error > 1e-4 {
            println!(
                "  ⚠️  Max error: {:.6}, Errors > 1e-4: {}/{}",
                max_error, error_count, m
            );
        } else {
            println!("  ✓ Max error: {:.6} (PASS)", max_error);
        }
        println!();
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support on aarch64");
}
