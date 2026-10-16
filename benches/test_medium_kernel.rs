#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    let source =
        include_str!("../src/ops/metal/gemv/mul_mv_f16_adaptive.metal");
    let library = device
        .new_library_with_source(source, &compile_options)
        .expect("Failed to compile");

    // Test medium kernel for K=4096
    let kernel = library
        .get_function("kernel_mul_mv_f16_f32_medium", None)
        .expect("Failed to get kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    let k = 4096usize;
    let m = 4usize;

    let weights_f16: Vec<u16> = (0..m * k)
        .map(|i| half::f16::from_f32(i as f32 * 0.1).to_bits())
        .collect();
    let input_f32: Vec<f32> = (0..k).map(|i| i as f32 * 0.01).collect();

    // CPU reference
    let mut expected = vec![0.0f32; m];
    for row in 0..m {
        for col in 0..k {
            let w = half::f16::from_bits(weights_f16[row * k + col]).to_f32();
            expected[row] += w * input_f32[col];
        }
    }

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
    enc.set_buffer(0, Some(&weights_buffer), 0);
    enc.set_buffer(1, Some(&input_buffer), 0);
    enc.set_buffer(2, Some(&output_buffer), 0);
    enc.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        MTLSize { width: 2, height: 1, depth: 1 }, // (m+1)/2
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, m)
    };

    println!("Medium kernel K=4096:");
    println!("Expected: {:?}", expected);
    println!("GPU:      {:?}", result);
    let max_error = expected
        .iter()
        .zip(result.iter())
        .map(|(e, r)| (e - r).abs())
        .fold(0.0f32, |a, b| a.max(b));
    println!("Max error: {}", max_error);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {}
