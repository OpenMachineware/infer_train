// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Correctness test for F16 GEMV basic kernel (K >= 32, K % 4 != 0)

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(false); // Disable fast math for accuracy

    let source = include_str!("../src/ops/metal/gemv/mul_mv_f16_kernels.metal");
    let library = device
        .new_library_with_source(source, &compile_options)
        .expect("Failed to compile");

    let kernel = library
        .get_function("kernel_mul_mv_f16_f32", None)
        .expect("Failed to get kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    // Test K values that are >= 32 and not divisible by 4
    // Also test K values that are divisible by 4 (since _4 kernel is disabled)
    let test_sizes = [
        33, 34, 35, 64, 65, 66, 67, 128, 129, 130, 131, 256, 512, 1024, 2048,
        4096,
    ];
    let m = 4;

    for k in test_sizes {
        // Generate test data
        let weights_f16: Vec<u16> = (0..m * k)
            .map(|i| half::f16::from_f32(i as f32 * 0.1).to_bits())
            .collect();
        let input_f32: Vec<f32> = (0..k).map(|i| i as f32 * 0.01).collect();

        // CPU reference
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

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);

        // Set args struct
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

        let nsg = std::cmp::min(4, (k + 127) / 128);
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
            nsg: nsg as i16,
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

        // Dispatch: threadgroup size = 32 * nsg, thread groups = (m+1)/2
        let thread_group_size =
            MTLSize { width: 32, height: nsg as u64, depth: 1 };
        let thread_group_count =
            MTLSize { width: ((m + 1) / 2) as u64, height: 1, depth: 1 };

        // Allocate threadgroup memory: NW * NR0 * sizeof(float) = 32 * 2 * 4 = 256 bytes
        let shmem_size = 32 * 2 * 4;
        enc.set_threadgroup_memory_length(0, shmem_size);

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

        let max_error = expected
            .iter()
            .zip(result.iter())
            .map(|(e, r)| (e - r).abs())
            .fold(0.0f32, |a, b| a.max(b));

        println!(
            "K={:3} Expected: [{:.2}, {:.2}, {:.2}, {:.2}]",
            k, expected[0], expected[1], expected[2], expected[3]
        );
        println!(
            "K={:3} GPU:      [{:.2}, {:.2}, {:.2}, {:.2}]  error: {:.6}  {}",
            k,
            result[0],
            result[1],
            result[2],
            result[3],
            max_error,
            if max_error < 0.1 { "✓ PASS" } else { "✗ FAIL" }
        );
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support on aarch64");
}
