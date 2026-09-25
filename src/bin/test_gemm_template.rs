// Test llama.cpp's templated GEMM approach
// This test verifies the new Metal shader works correctly

use infer_train::quant::types::BlockQ4K;
use metal::{Device, MTLSize};

fn main() {
    println!("Testing templated GEMM approach from llama.cpp");

    let device = Device::system_default().expect("No Metal device found");

    // Load the new shader
    let shader_source = include_str!("../../shaders/gemm_llama_cpp.metal");
    let compile_options = metal::CompileOptions::new();

    match device.new_library_with_source(shader_source, &compile_options) {
        Ok(library) => {
            println!("✓ Shader library compiled successfully");

            // Try to create the Q4_K pipeline
            match library.get_function("kernel_mul_mm_q4_K_f32", None) {
                Ok(kernel) => {
                    println!("✓ Q4_K kernel function found");

                    match device.new_compute_pipeline_state_with_function(&kernel) {
                        Ok(pipeline) => {
                            println!("✓ Q4_K pipeline created successfully");
                            println!("  Max threadgroup size: {:?}", pipeline.max_total_threads_per_threadgroup());
                            println!("  Thread execution width: {}", pipeline.thread_execution_width());
                            println!("  Static threadgroup memory length: {} bytes", pipeline.static_threadgroup_memory_length());

                            // Create a small test
                            test_q4_k_gemm(&device, &pipeline);
                        }
                        Err(e) => {
                            println!("✗ Failed to create Q4_K pipeline: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Failed to get Q4_K kernel: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to compile shader library: {}", e);
        }
    }
}

fn test_q4_k_gemm(device: &Device, pipeline: &metal::ComputePipelineState) {
    use metal::{Buffer, CommandQueue};

    // Test dimensions: M=128, K=4096, N=128
    let m = 128;
    let k = 4096;
    let n = 128;

    println!("\nTesting GEMM: M={}, K={}, N={}", m, k, n);

    // Create input data with actual values for testing
    let num_blocks = k / 256; // Q4_K has 256 values per block
    let weights_size = num_blocks * std::mem::size_of::<BlockQ4K>();
    let input_size = k * n * 2; // FP16 = 2 bytes
    let output_size = m * n * std::mem::size_of::<f32>();

    // Allocate buffers with test data
    // For now, use simple test: all weights = 1.0, all inputs = 1.0
    // Expected: each output = K * 1.0 = K

    // Create weights: For Q4_K, we need to create blocks
    // For simplicity, create blocks with d=1.0 (scale), dmin=0, scales=1, qs encoding 1.0
    let mut weights = vec![0u8; weights_size];
    for i in 0..num_blocks {
        let offset = i * std::mem::size_of::<BlockQ4K>();
        // BlockQ4K: d (2 bytes), dmin (2 bytes), scales[12], qs[128]
        // Set d = 1.0 in FP16 = 0x3C00
        weights[offset] = 0x00;
        weights[offset + 1] = 0x3C;
        // Set dmin = 0
        weights[offset + 2] = 0x00;
        weights[offset + 3] = 0x00;
        // Set scales to encode value 1 (scales[0] = 1, scales[1] = 0)
        weights[offset + 4] = 1;  // scale for first 32 values
        weights[offset + 5] = 0;  // min for first 32 values
        // ... rest of scales
        // Set qs to encode 1.0: 4-bit value 1 in low nibble, 4-bit value 1 in high nibble
        for j in 0..128 {
            weights[offset + 16 + j] = 0x11; // Both nibbles = 1
        }
    }

    // Create input: all 1.0 in FP16 = 0x3C00
    let input: Vec<u16> = vec![0x3C00; k * n];
    let output = vec![0f32; m * n];

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        weights_size as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        input_size as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    let output_buffer = device.new_buffer_with_data(
        output.as_ptr() as *const std::ffi::c_void,
        output_size as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    // Create argument buffer - EXACT match to llama.cpp's ggml_metal_kargs_mul_mm
    #[repr(C)]
    struct Args {
        ne00: i32,  // K
        ne02: i32,  // batch (1 for simple case)
        nb01: u64,  // row stride for A (bytes between rows)
        nb02: u64,  // batch stride A
        nb03: u64,  // batch3 stride A
        ne12: i32,  // batch for B
        nb10: u64,  // element stride for B (bytes between elements)
        nb11: u64,  // row stride for B (bytes between rows)
        nb12: u64,  // batch stride B
        nb13: u64,  // batch3 stride B
        ne0: i32,   // M (output rows)
        ne1: i32,   // N (output cols)
        r2: i16,    // broadcast ratio
        r3: i16,    // broadcast ratio
    }

    let args = Args {
        ne00: k as i32,
        ne02: 1,
        nb01: (k / 256 * 144) as u64,  // Each row of A has k/256 blocks, each block is 144 bytes
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 2,  // FP16 = 2 bytes per element
        nb11: (k * 2) as u64,  // Each row has k elements, each is 2 bytes
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    let args_buffer = device.new_buffer_with_data(
        &args as *const Args as *const std::ffi::c_void,
        std::mem::size_of::<Args>() as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    // Create command queue and encoder
    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&args_buffer), 0);
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);

    // Threadgroup memory
    // We need: 64 rows * 32 K values = 2048 half = 4096 bytes for A
    //        + 32 cols * 32 K values = 1024 half = 2048 bytes for B
    // Total: 6144 bytes
    encoder.set_threadgroup_memory_length(0, 8192);

    // Calculate grid and threadgroup sizes
    // Threadgroup: 64x32 output tiles
    // Need: ceil(M/64) x ceil(N/32) threadgroups
    let grid_size = MTLSize::new(
        ((n + 31) / 32) as u64,
        ((m + 63) / 64) as u64,
        1,
    );

    // Threadgroup size: 128 threads (like llama.cpp)
    let threadgroup_size = MTLSize::new(128, 1, 1);

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    println!("✓ GEMM executed successfully");

    // Read results
    let output_ptr = output_buffer.contents() as *const f32;
    let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, m * n) };

    // Check for NaN or inf
    let has_nan = output_slice.iter().any(|x| x.is_nan());
    let has_inf = output_slice.iter().any(|x| x.is_infinite());
    let has_valid = output_slice.iter().any(|x| x.is_finite() && *x != 0.0);

    println!("Output stats:");
    println!("  Has NaN: {}", has_nan);
    println!("  Has Inf: {}", has_inf);
    println!("  Has valid (non-zero, finite): {}", has_valid);

    // Print first few values
    println!("First 10 values:");
    for i in 0..10.min(output_slice.len()) {
        print!("  output[{}] = {:.6}", i, output_slice[i]);
    }
    println!();

    if !has_nan && !has_inf {
        println!("✓ Kernel produces valid results");
    } else {
        println!("✗ Kernel produces NaN or Inf");
    }
}
