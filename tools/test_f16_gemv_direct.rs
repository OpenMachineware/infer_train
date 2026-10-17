// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test F16 GEMV directly

use half::f16;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::MTLSize;
    use std::rc::Rc;

    let gpu_ctx = MetalContext::new().expect("Failed to create MetalContext");

    // Test dimensions matching output layer: M=151936, K=1024
    let m = 151936usize;
    let k = 1024usize;

    // Create F16 weights (all 1.0 for simplicity)
    let weights: Vec<u16> = vec![f16::from_f32(0.01).to_bits(); m * k];
    let weights_buffer = Rc::new(gpu_ctx.device().new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        (weights.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Create F32 input (all 1.0)
    let input: Vec<f32> = vec![1.0; k];
    let input_buffer = Rc::new(gpu_ctx.device().new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (input.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    println!("Testing F16 GEMV: M={}, K={}", m, k);
    println!(
        "  Weights: {} elements, {} bytes",
        weights.len(),
        weights.len() * 2
    );
    println!("  Input: {} elements, {} bytes", input.len(), input.len() * 4);

    // Run GPU GEMV
    let result =
        gpu_ctx.mv_f16_f32_gpu_weights(m, k, &weights_buffer, &input_buffer);

    match result {
        Ok(output_buffer) => {
            // Read output
            let output: &[f32] = unsafe {
                let ptr = output_buffer.contents() as *const f32;
                std::slice::from_raw_parts(ptr, m)
            };

            let nan_count = output.iter().filter(|&&v| v.is_nan()).count();
            let inf_count = output.iter().filter(|&&v| v.is_infinite()).count();
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);

            println!("\nGPU Output:");
            println!("  NaN count: {}", nan_count);
            println!("  Inf count: {}", inf_count);
            println!("  Range: [{}, {}]", min, max);
            println!("  Expected: {}", 0.01 * k as f32); // sum of 1024 * 0.01 * 1.0
            println!("  First 10: {:?}", &output[..10]);
        }
        Err(e) => {
            println!("ERROR: {}", e);
        }
    }

    // Test with smaller dimensions
    println!("\n\n=== Test with smaller dimensions ===");
    let m2 = 1000usize;
    let k2 = 1024usize;

    let weights2: Vec<u16> = vec![f16::from_f32(0.01).to_bits(); m2 * k2];
    let weights_buffer2 = Rc::new(gpu_ctx.device().new_buffer_with_data(
        weights2.as_ptr() as *const std::ffi::c_void,
        (weights2.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let input2: Vec<f32> = vec![1.0; k2];
    let input_buffer2 = Rc::new(gpu_ctx.device().new_buffer_with_data(
        input2.as_ptr() as *const std::ffi::c_void,
        (input2.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let result2 = gpu_ctx.mv_f16_f32_gpu_weights(
        m2,
        k2,
        &weights_buffer2,
        &input_buffer2,
    );

    match result2 {
        Ok(output_buffer) => {
            let output: &[f32] = unsafe {
                let ptr = output_buffer.contents() as *const f32;
                std::slice::from_raw_parts(ptr, m2)
            };

            let nan_count = output.iter().filter(|&&v| v.is_nan()).count();
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);

            println!("GPU Output (M=1000, K=1024):");
            println!("  NaN count: {}", nan_count);
            println!("  Range: [{}, {}]", min, max);
            println!("  Expected: {}", 0.01 * k2 as f32);
            println!("  First 10: {:?}", &output[..10]);
        }
        Err(e) => {
            println!("ERROR: {}", e);
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("Metal not available on this platform");
}
