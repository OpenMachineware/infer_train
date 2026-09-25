// Simple test for templated GEMM kernel
// Uses zero inputs to verify kernel runs without errors

use metal::{Device, MTLSize};

fn main() {
    println!("=== Simple Templated GEMM Test ===\n");

    let device = Device::system_default().expect("No Metal device found");

    // Load shader
    let shader_source = include_str!("../../shaders/gemm_llama_cpp.metal");
    let compile_options = metal::CompileOptions::new();
    let library = device.new_library_with_source(shader_source, &compile_options)
        .expect("Failed to compile shader");
    println!("✓ Shader compiled");

    let kernel = library.get_function("kernel_mul_mm_q4_K_f32", None)
        .expect("Failed to get kernel");
    let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");
    println!("✓ Pipeline created");

    // Test case: small dimensions for debugging
    let m = 64;   // rows
    let k = 256;  // K dimension (1 block)
    let n = 32;   // columns

    println!("\nTest dimensions: M={}, K={}, N={}", m, k, n);

    // Create zero inputs
    let num_blocks = k / 256;
    let weights_size = m * num_blocks * 144; // BlockQ4K size = 144 bytes
    let input_size = k * n * 2; // FP16
    let output_size = m * n * 4; // FP32

    println!("Buffer sizes: weights={}, input={}, output={}",
             weights_size, input_size, output_size);

    let weights_buffer = device.new_buffer(weights_size as u64, metal::MTLResourceOptions::StorageModeManaged);
    let input_buffer = device.new_buffer(input_size as u64, metal::MTLResourceOptions::StorageModeManaged);
    let output_buffer = device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeManaged);

    println!("✓ Buffers allocated");

    // Set args - CRITICAL: match llama.cpp's layout
    // A is [M, K] (row-major), B is [K, N] (column-major), C is [M, N] (row-major)
    #[repr(C)]
    #[derive(Debug)]
    struct Args {
        ne00: i32, ne01: i32, ne02: i32, ne03: i32,  // src0 dimensions
        nb00: u64, nb01: u64, nb02: u64, nb03: u64,  // src0 strides
        ne10: i32, ne11: i32, ne12: i32, ne13: i32,  // src1 dimensions
        nb10: u64, nb11: u64, nb12: u64, nb13: u64,  // src1 strides
        ne0: i32, ne1: i32, ne2: i32, ne3: i32,      // dst dimensions
        nb0: u64, nb1: u64, nb2: u64, nb3: u64,      // dst strides
    }

    let args = Args {
        // src0 (A): [M, K] in Q4_K
        ne00: k as i32, ne01: m as i32, ne02: 1, ne03: 1,
        nb00: 144,                    // BlockQ4K size
        nb01: (num_blocks * 144) as u64, // Row stride in bytes
        nb02: 0, nb03: 0,
        // src1 (B): [K, N] in FP16 (column-major)
        ne10: k as i32, ne11: n as i32, ne12: 1, ne13: 1,
        nb10: 2,                      // Element size (FP16)
        nb11: (k * 2) as u64,         // Column stride in bytes
        nb12: 0, nb13: 0,
        // dst (C): [M, N] in FP32
        ne0: m as i32, ne1: n as i32, ne2: 1, ne3: 1,
        nb0: 4,                       // Element size (FP32)
        nb1: (n * 4) as u64,          // Row stride in bytes
        nb2: 0, nb3: 0,
    };

    println!("\nArgs: {:?}", args);

    let args_buffer = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    // Dispatch kernel
    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&args_buffer), 0);
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);

    // Threadgroup memory (8KB for fallback kernel)
    encoder.set_threadgroup_memory_length(0, 8192);

    // Grid: ceil(M/64) x ceil(N/32) threadgroups
    let grid_size = MTLSize::new(
        ((n + 31) / 32) as u64,  // N dimension
        ((m + 63) / 64) as u64,  // M dimension
        1,
    );

    // Threadgroup: 128 threads (matches llama.cpp fallback)
    let threadgroup_size = MTLSize::new(128, 1, 1);

    println!("\nGrid: {} x {} threadgroups", grid_size.width, grid_size.height);
    println!("Threadgroup size: {}", threadgroup_size.width);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    println!("✓ Dispatched");
    command_buffer.commit();
    command_buffer.wait_until_completed();
    println!("✓ Completed");

    // Check output (should be all zeros)
    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m * n) };

    let has_nan = output.iter().any(|x| x.is_nan());
    let has_inf = output.iter().any(|x| x.is_infinite());
    let non_zero = output.iter().filter(|x| **x != 0.0).count();

    println!("\nResults:");
    println!("  Has NaN: {}", has_nan);
    println!("  Has Inf: {}", has_inf);
    println!("  Non-zero values: {}/{}", non_zero, m * n);

    if !has_nan && !has_inf {
        println!("\n✓ Test PASSED - kernel runs without NaN/Inf");
    } else {
        println!("\n✗ Test FAILED - kernel produces NaN or Inf");
    }
}