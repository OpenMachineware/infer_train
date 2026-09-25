// Verify template kernel correctness by comparing with llama.cpp directly
// Run: cargo run --release --bin verify_vs_llama_cpp

use infer_train::quant::types::BlockQ4K;
use metal::{Device, MTLSize};
use half::f16;
use std::time::Instant;

fn main() {
    println!("=== Template Kernel Correctness Verification ===\n");

    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // Test dimensions matching llama.cpp benchmark
    let m = 4096;
    let k = 4096;
    let n = 128;

    // Create identical test data as llama.cpp
    let num_blocks = k / 256;
    let mut weights_data = vec![0u8; m * num_blocks * std::mem::size_of::<BlockQ4K>()];
    let weights_ptr = weights_data.as_mut_ptr() as *mut BlockQ4K;

    unsafe {
        for i in 0..m * num_blocks {
            let block = weights_ptr.add(i);
            // Match llama.cpp test data: (i * 17 + 13) % 256
            (*block).d = half::f16::from_f32(1.0).to_bits();
            (*block).dmin = half::f16::from_f32(0.0).to_bits();
            for j in 0..12 {
                (*block).scales[j] = ((i * 17 + 13 + j) % 256) as u8;
            }
            for j in 0..128 {
                (*block).qs[j] = ((i * 17 + 13 + j) % 256) as u8;
            }
        }
    }
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr, m * num_blocks) };

    // Create FP16 input matching llama.cpp: (i % 10) as FP16
    let input: Vec<u16> = (0..(k * n))
        .map(|i| half::f16::from_f32((i % 10) as f32).to_bits())
        .collect();

    // Pre-allocate buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        (weights.len() * std::mem::size_of::<BlockQ4K>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (input.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (m * n * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Compile template kernel
    let shader_source = include_str!("../../shaders/mul_mm_standalone.metal");
    let compile_options = metal::CompileOptions::new();
    let library = device.new_library_with_source(shader_source, &compile_options)
        .expect("Failed to compile shader");
    let kernel = library.get_function("kernel_mul_mm_q4_K_f32", None)
        .expect("Failed to get kernel");
    let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline");

    // Args
    #[repr(C)]
    struct GemmArgs {
        ne00: i32, ne02: i32, nb01: u64, nb02: u64, nb03: u64,
        ne12: i32, nb10: u64, nb11: u64, nb12: u64, nb13: u64,
        ne0: i32, ne1: i32, r2: i16, r3: i16,
    }

    let args = GemmArgs {
        ne00: k as i32, ne02: 1,
        nb01: (num_blocks * std::mem::size_of::<BlockQ4K>()) as u64,
        nb02: 0, nb03: 0,
        ne12: 1, nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0,
        ne0: m as i32, ne1: n as i32, r2: 1, r3: 1,
    };

    let args_buffer = device.new_buffer_with_data(
        &args as *const GemmArgs as *const std::ffi::c_void,
        std::mem::size_of::<GemmArgs>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Run kernel
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&args_buffer), 0);
    encoder.set_buffer(1, Some(&weights_buffer), 0);
    encoder.set_buffer(2, Some(&input_buffer), 0);
    encoder.set_buffer(3, Some(&output_buffer), 0);
    encoder.set_threadgroup_memory_length(0, 8192);

    let grid = MTLSize::new(16, 4, 1);
    let threadgroup = MTLSize::new(32, 4, 1);
    encoder.dispatch_thread_groups(grid, threadgroup);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Get output
    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m * n) };

    // Print sample outputs
    println!("Output matrix size: {} x {} = {} values", m, n, m * n);
    println!("\nFirst 10 output values:");
    for i in 0..10 {
        println!("  C[{}] = {:.4}", i, output[i]);
    }

    println!("\nOutput at key positions (column-major: row + col*M):");
    println!("  C[0,0] = output[0] = {:.4}", output[0]);
    println!("  C[1,0] = output[1] = {:.4}", output[1]);
    println!("  C[0,1] = output[M] = {:.4}", output[m]);
    println!("  C[{},{}, at {}] = {:.4}", m-1, n-1, (n-1)*m + m-1, output[(n-1)*m + m-1]);

    // Sample some row-column pairs
    println!("\nSample C[row,col] = output[row + col*M]:");
    for col in 0..4 {
        for row in 0..5 {
            let idx = row + col * m;
            print!("  C[{row},{col}]={:.2}", output[idx]);
        }
        println!();
    }

    // Check for NaN/Inf
    let nan_count = output.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = output.iter().filter(|&&x| x.is_infinite()).count();
    let zero_count = output.iter().filter(|&&x| x == 0.0).count();

    println!("\nValue statistics:");
    println!("  NaN count: {}", nan_count);
    println!("  Inf count: {}", inf_count);
    println!("  Zero count: {} ({:.1}%)", zero_count, zero_count as f64 / output.len() as f64 * 100.0);

    // Compute statistics
    let sum: f64 = output.iter().map(|&x| x as f64).sum();
    let mean = sum / output.len() as f64;
    let variance: f64 = output.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / output.len() as f64;
    let std_dev = variance.sqrt();

    println!("\nOutput statistics:");
    println!("  Mean: {:.4}", mean);
    println!("  Std dev: {:.4}", std_dev);
    println!("  Min: {:.4}", output.iter().cloned().fold(f32::INFINITY, f32::min));
    println!("  Max: {:.4}", output.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    if nan_count > 0 || inf_count > 0 {
        println!("\n❌ FAILED: Output contains NaN or Inf values");
    } else {
        println!("\n✅ PASSED: Output is finite");
    }
}