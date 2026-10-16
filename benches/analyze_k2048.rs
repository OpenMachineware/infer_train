#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Test only K=2048 with different kernel variants
    let k = 2048usize;
    let m = 4096usize;

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

    // Test 1: Our medium kernel
    let our_source =
        include_str!("../src/ops/metal/gemv/mul_mv_f16_adaptive.metal");
    let our_library = device
        .new_library_with_source(our_source, &compile_options)
        .expect("Failed to compile");
    let our_kernel = our_library
        .get_function("kernel_mul_mv_f16_f32_medium", None)
        .expect("Failed to get kernel");
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(&our_kernel)
        .expect("Failed to create pipeline");

    // Test 2: Simple float4 vectorized kernel
    let simple_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kernel_simple_vec4(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x;
    if (row >= m) return;

    device const half4* w4 = (device const half4*)(weights + row * k);
    device const float4* x4 = (device const float4*)input;

    float4 sum4 = 0.0f;
    const int n4 = k / 4;

    for (int i = tiisg; i < n4; i += 32) {
        sum4 += float4(w4[i]) * x4[i];
    }

    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    sum = simd_sum(sum);

    if (tiisg == 0) {
        output[row] = sum;
    }
}
"#;
    let simple_library = device
        .new_library_with_source(simple_source, &compile_options)
        .expect("Failed to compile simple");
    let simple_kernel = simple_library
        .get_function("kernel_simple_vec4", None)
        .expect("Failed to get simple kernel");
    let simple_pipeline = device
        .new_compute_pipeline_state_with_function(&simple_kernel)
        .expect("Failed to create simple pipeline");

    // Benchmark our medium
    let iterations = 100;
    for _ in 0..10 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&our_pipeline);
        enc.set_buffer(0, Some(&weights_buffer), 0);
        enc.set_buffer(1, Some(&input_buffer), 0);
        enc.set_buffer(2, Some(&output_buffer), 0);
        enc.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize { width: (m + 1) as u64 / 2, height: 1, depth: 1 },
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
        enc.set_buffer(0, Some(&weights_buffer), 0);
        enc.set_buffer(1, Some(&input_buffer), 0);
        enc.set_buffer(2, Some(&output_buffer), 0);
        enc.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize { width: (m + 1) as u64 / 2, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let our_time = start.elapsed().as_micros() as f64 / iterations as f64;

    // Benchmark simple
    for _ in 0..10 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&simple_pipeline);
        enc.set_buffer(0, Some(&weights_buffer), 0);
        enc.set_buffer(1, Some(&input_buffer), 0);
        enc.set_buffer(2, Some(&output_buffer), 0);
        enc.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
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
        enc.set_compute_pipeline_state(&simple_pipeline);
        enc.set_buffer(0, Some(&weights_buffer), 0);
        enc.set_buffer(1, Some(&input_buffer), 0);
        enc.set_buffer(2, Some(&output_buffer), 0);
        enc.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize { width: m as u64, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let simple_time = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("K=2048, M=4096");
    println!("Our medium kernel: {:.2} μs", our_time);
    println!("Simple vec4 kernel: {:.2} μs", simple_time);
    println!(
        "Improvement: {:.1}%",
        (our_time - simple_time) / our_time * 100.0
    );
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {}
