// Test basic kernel for K % 4 != 0
// For these K, llama.cpp uses short kernel, we use basic kernel

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = metal::CompileOptions::new();

    // Our basic kernel
    let our_source =
        include_str!("../src/ops/metal/gemv/mul_mv_f16_kernels.metal");
    let our_lib =
        device.new_library_with_source(our_source, &compile_options).unwrap();
    let our_kernel =
        our_lib.get_function("kernel_mul_mv_f16_f32", None).unwrap();
    let our_pipeline =
        device.new_compute_pipeline_state_with_function(&our_kernel).unwrap();

    // Our short kernel
    let our_short_kernel =
        our_lib.get_function("kernel_mul_mv_f16_f32_short", None).unwrap();
    let our_short_pipeline = device
        .new_compute_pipeline_state_with_function(&our_short_kernel)
        .unwrap();

    println!("=== Basic Kernel (K%4!=0) vs Short Kernel ===\n");

    // Test K values not divisible by 4
    let test_ks: Vec<usize> = vec![33, 65, 129, 257, 513, 1025];
    let m = 4096;
    let iterations = 300;

    for k in test_ks {
        let weights: Vec<u16> = (0..m * k)
            .map(|i| half::f16::from_f32((i % 100) as f32 * 0.1).to_bits())
            .collect();
        let input: Vec<f32> = (0..k).map(|i| (i % 50) as f32 * 0.01).collect();

        let w_buf = device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (m * k * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let i_buf = device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let o_buf = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        #[repr(C)]
        struct Args {
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

        let args = Args {
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

        // Test basic kernel
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&our_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<Args>() as u64,
                &args as *const Args as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&i_buf), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize { width: ((m + 1) / 2) as u64, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&our_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<Args>() as u64,
                &args as *const Args as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&i_buf), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize { width: ((m + 1) / 2) as u64, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let basic_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Test short kernel (for comparison)
        let args_short = Args { nr0: 32, ..args };

        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&our_short_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<Args>() as u64,
                &args_short as *const Args as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&i_buf), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize { width: ((m + 31) / 32) as u64, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&our_short_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<Args>() as u64,
                &args_short as *const Args as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&i_buf), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize { width: ((m + 31) / 32) as u64, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let short_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let ratio = basic_us / short_us * 100.0;
        println!(
            "K={:4}  Basic: {:6.2}μs  Short: {:6.2}μs  Ratio: {:5.1}%  {}",
            k,
            basic_us,
            short_us,
            ratio,
            if ratio < 100.0 { "✓ Basic faster" } else { "✗ Basic slower" }
        );
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("Requires Metal");
}
