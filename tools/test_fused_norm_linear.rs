// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test FusedNormLinear kernel correctness
//!
//! Compares fused RMSNorm+Linear kernel against separate RMSNorm + GEMV

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::MTLResourceOptions;
use std::rc::Rc;

fn reference_rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let k = input.len();
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / k as f32;
    let rms = (mean_sq + epsilon).sqrt();

    input.iter().zip(weight.iter()).map(|(x, w)| x / rms * w).collect()
}

fn reference_gemv_f16(
    weights: &[half::f16],
    input: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += weights[i * k + j].to_f32() * input[j];
        }
        output[i] = sum;
    }
    output
}

fn reference_fused_norm_linear(
    input: &[f32],
    norm_weight: &[f32],
    weights: &[half::f16],
    m: usize,
    k: usize,
    epsilon: f32,
) -> Vec<f32> {
    // First, apply RMSNorm
    let normalized = reference_rms_norm(input, norm_weight, epsilon);

    // Then, apply GEMV
    reference_gemv_f16(weights, &normalized, m, k)
}

fn test_fused_norm_linear(m: usize, k: usize, epsilon: f32) -> bool {
    let gpu_ctx = MetalContext::new().expect("Failed to create MetalContext");
    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    // Generate test data
    let mut input_data: Vec<f32> =
        (0..k).map(|i| ((i as f32 * 0.1) - 5.0).sin()).collect();
    let mut norm_weight_data: Vec<f32> =
        (0..k).map(|i| (i as f32 * 0.05 + 0.5).cos()).collect();
    let weights_data: Vec<half::f16> = (0..m * k)
        .map(|i| half::f16::from_f32((i as f32 * 0.01 - 2.0).sin()))
        .collect();

    // Compute reference output
    let expected = reference_fused_norm_linear(
        &input_data,
        &norm_weight_data,
        &weights_data,
        m,
        k,
        epsilon,
    );

    // Create GPU buffers
    let input_buffer = Rc::new(device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    let norm_weight_buffer = Rc::new(device.new_buffer_with_data(
        norm_weight_data.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    let weights_buffer = Rc::new(device.new_buffer_with_data(
        weights_data.as_ptr() as *const std::ffi::c_void,
        (m * k * 2) as u64, // F16 = 2 bytes
        MTLResourceOptions::StorageModeShared,
    ));

    let output_buffer = Rc::new(
        device
            .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared),
    );

    // Encode and dispatch fused kernel
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    gpu_ctx.fused_norm_linear_f16_encode(
        encoder,
        &input_buffer,
        &norm_weight_buffer,
        &weights_buffer,
        &output_buffer,
        m,
        k,
        epsilon,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read output
    let output_ptr = output_buffer.contents() as *const f32;
    let mut output_data = vec![0.0f32; m];
    unsafe {
        std::ptr::copy_nonoverlapping(output_ptr, output_data.as_mut_ptr(), m);
    }

    // Compare
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut nan_count = 0;

    for i in 0..m {
        let got = output_data[i];
        let want = expected[i];

        if got.is_nan() || got.is_infinite() {
            nan_count += 1;
            continue;
        }

        let abs_err = (got - want).abs();
        let rel_err = if want.abs() > 1e-10 {
            abs_err / want.abs() * 100.0
        } else {
            abs_err * 100.0
        };

        max_abs_error = max_abs_error.max(abs_err);
        max_rel_error = max_rel_error.max(rel_err);
    }

    println!("  Max abs error: {:.10}", max_abs_error);
    println!("  Max rel error: {:.10}%", max_rel_error);
    println!("  NaN count: {}", nan_count);

    if max_rel_error > 1.0 || nan_count > 0 {
        println!("  ✗ FAILED");
        return false;
    }

    println!("  ✓ PASSED");
    true
}

fn main() {
    println!("=== FusedNormLinear Kernel Test ===");
    println!();

    let epsilon = 1e-5;

    // Test various sizes
    let test_cases = vec![
        (64, 64),      // Small
        (256, 256),    // Medium
        (512, 1024),   // Common LLM sizes
        (1024, 4096),  // LLaMA hidden size
        (4096, 4096),  // Large FFN
        (14336, 4096), // LLaMA intermediate size
    ];

    let mut all_passed = true;
    for (m, k) in test_cases {
        println!("Testing M={} K={}", m, k);
        if !test_fused_norm_linear(m, k, epsilon) {
            all_passed = false;
        }
        println!();
    }

    println!("=== Test Complete ===");

    if !all_passed {
        std::process::exit(1);
    }
}
