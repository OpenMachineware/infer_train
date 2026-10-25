// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test F16 GEMM with different sizes

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::*;

fn test_f16_gemm(gpu_ctx: &MetalContext, m: usize, n: usize, k: usize) -> bool {
    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    // Create test data
    let weight_data: Vec<u16> = (0..(m * k))
        .map(|i| half::f16::from_f32((i as f32 % 10.0) * 0.1).to_bits())
        .collect();
    let input_data: Vec<u16> = (0..(n * k))
        .map(|i| half::f16::from_f32((i as f32 % 5.0) * 0.2).to_bits())
        .collect();

    let weight_buffer = device.new_buffer_with_data(
        weight_data.as_ptr() as *const std::ffi::c_void,
        (m * k * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (n * k * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (n * m * 4) as u64, // F32 output
        MTLResourceOptions::StorageModeShared,
    );

    // Run F16 GEMM
    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();

    gpu_ctx.mul_mm_f16_encode(
        encoder,
        &weight_buffer,
        &input_buffer,
        &output_buffer,
        m,
        n,
        k,
    );

    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    // Check output
    let output_ptr = output_buffer.contents() as *const f32;
    let mut nan_count = 0;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for i in 0..(n * m) {
        let val = unsafe { *output_ptr.add(i) };
        if val.is_nan() {
            nan_count += 1;
        } else if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    println!(
        "F16 GEMM M={} N={} K={}: total={}, NaN={}, range=[{:.6}, {:.6}]",
        m,
        n,
        k,
        n * m,
        nan_count,
        min_val,
        max_val
    );

    nan_count == 0
}

fn main() {
    let gpu_ctx = MetalContext::new().expect("Failed to create Metal context");

    println!("=== Testing F16 GEMM with different sizes ===\n");

    // Test with sizes similar to DeepSeek (working)
    test_f16_gemm(&gpu_ctx, 151936, 1, 1536);

    // Test with sizes similar to Devstral (failing)
    test_f16_gemm(&gpu_ctx, 131072, 1, 5120);

    // Test with smaller sizes to find threshold
    test_f16_gemm(&gpu_ctx, 10000, 1, 4096);
    test_f16_gemm(&gpu_ctx, 50000, 1, 4096);
    test_f16_gemm(&gpu_ctx, 100000, 1, 4096);
    test_f16_gemm(&gpu_ctx, 131072, 1, 4096);
}
