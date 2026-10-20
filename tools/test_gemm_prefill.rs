// Test GEMM for batched prefill
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use bytemuck::Zeroable;
use infer_train::quant::types::BlockQ5K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    println!("=== GEMM Prefill Test ===\n");

    // Create Metal context
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let device = ctx.device();

    // Test dimensions (similar to first layer of 1.5B model)
    let m = 1536; // hidden_size (output rows)
    let k = 1536; // hidden_size (reduction dim)
    let n = 16; // batch size (number of tokens)

    println!("Testing GEMM: {}x{} x {}x{}", m, k, k, n);

    // Create Q5_K weights (simplified - just random data)
    let num_blocks = (m * k) / 256;
    let mut weights = vec![BlockQ5K::zeroed(); num_blocks];

    // Initialize with simple pattern
    for (i, block) in weights.iter_mut().enumerate() {
        block.d = 100; // scale (FP16)
        block.dmin = 10;
        // Fill with pattern
        for j in 0..12 {
            block.scales[j] = 16; // typical scale
        }
        for j in 0..128 {
            block.qs[j] = (i as u8).wrapping_add(j as u8);
        }
        for j in 0..32 {
            block.qh[j] = 0;
        }
    }

    // Create F16 input (batch_size x hidden_size)
    let input_size = n * k;
    let input_f32: Vec<f32> =
        (0..input_size).map(|i| (i % 100) as f32 / 100.0).collect();
    let input_f16: Vec<half::f16> =
        input_f32.iter().map(|&x| half::f16::from_f32(x)).collect();

    // Create GPU buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        (weights.len() * std::mem::size_of::<BlockQ5K>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const std::ffi::c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (m * n * 4) as u64, // F32 output
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Run GEMM
    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    ctx.mul_mm_q5_k_encode(
        &encoder,
        &weights_buffer,
        &input_buffer,
        &output_buffer,
        m,
        n,
        k,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Check output
    let output_ptr = output_buffer.contents() as *const f32;
    let output: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() };

    println!("\nOutput stats:");
    println!("  Size: {} elements", output.len());

    let sum: f32 = output.iter().sum();
    let mean = sum / output.len() as f32;
    let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("  Sum: {}", sum);
    println!("  Mean: {}", mean);
    println!("  Min: {}", min);
    println!("  Max: {}", max);

    // Check for NaN/Inf
    let has_nan = output.iter().any(|&x| x.is_nan());
    let has_inf = output.iter().any(|&x| x.is_infinite());

    if has_nan || has_inf {
        println!("\n❌ ERROR: Output contains NaN or Inf!");
    } else {
        println!("\n✓ Output is valid (no NaN/Inf)");
    }

    // Show first few output elements
    println!("\nFirst 10 output values:");
    for (i, val) in output.iter().take(10).enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }

    println!("\n=== Test Complete ===");
}
