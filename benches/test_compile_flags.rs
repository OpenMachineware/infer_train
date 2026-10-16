#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test different compile options
    let test_opts = [
        ("fast_math=true", {
            let opts = metal::CompileOptions::new();
            opts.set_fast_math_enabled(true);
            opts
        }),
        ("fast_math=false", {
            let opts = metal::CompileOptions::new();
            opts.set_fast_math_enabled(false);
            opts
        }),
    ];

    let source =
        include_str!("../src/ops/metal/gemv/mul_mv_f16_adaptive.metal");

    let k = 128usize;
    let m = 4096usize;
    let iterations = 100;

    let weights_f16: Vec<u16> =
        (0..m * k).map(|i| ((i as f32 * 0.1).sin() * 100.0) as u16).collect();
    let input_f32: Vec<f32> = (0..k).map(|i| (i as f32 * 0.05).cos()).collect();

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

    for (name, opts) in test_opts.iter() {
        let library = device
            .new_library_with_source(source, opts)
            .expect("Failed to compile");
        let kernel = library
            .get_function("kernel_mul_mv_f16_f32_short", None)
            .expect("Failed to get kernel");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline");

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&weights_buffer), 0);
            enc.set_buffer(1, Some(&input_buffer), 0);
            enc.set_buffer(2, Some(&output_buffer), 0);
            enc.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize { width: (m + 31) as u64 / 32, height: 1, depth: 1 },
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
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&weights_buffer), 0);
            enc.set_buffer(1, Some(&input_buffer), 0);
            enc.set_buffer(2, Some(&output_buffer), 0);
            enc.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize { width: (m + 31) as u64 / 32, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let time = start.elapsed().as_micros() as f64 / iterations as f64;

        println!("{}: {:.2} μs", name, time);
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {}
