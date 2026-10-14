// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Correctness test for self-written K-quant MV kernels

use bytemuck::Zeroable;
use infer_train::quant::types::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, QK_K,
};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::{MTLResourceOptions, NSRange};

fn main() {
    println!("=== K-quant MV Kernel Correctness Test ===\n");

    let m = 16; // Small number of rows
    let k = 256; // One block

    // Create Metal context
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Create random input vector
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.1).sin()).collect();

    // Test each K-quant format
    test_q4_k(&ctx, m, k, &input);
    test_q2_k(&ctx, m, k, &input);
    test_q6_k(&ctx, m, k, &input);
    test_q3_k(&ctx, m, k, &input);
    test_q5_k(&ctx, m, k, &input);

    println!("\n=== All tests completed ===");
}

fn test_q4_k(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q4_K...");

    // Create weights with known pattern
    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ4K::zeroed(); m * n_blocks];

    // Fill with simple pattern: all zeros
    // For Q4_K: d=1.0, scales=1, quants=0 should give output near 0
    for block in &mut weights {
        // Set super-block scales to 1.0
        block.d = half::f16::from_f32(1.0).to_bits();
        block.dmin = half::f16::from_f32(0.0).to_bits();
        // scales[12] = all zeros is fine
        // qs[128] = all zeros is fine
    }

    // Create GPU buffers
    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ4K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ4K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    // Run kernel
    let output_buffer = ctx
        .mv_q4_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q4_K MV failed");

    // Read results
    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    // With all-zero quants, output should be near zero
    let max_abs = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
    println!("  Max absolute output: {}", max_abs);
    if max_abs < 1e-3 {
        println!("  ✓ Q4_K output is near zero (expected for zero weights)");
    } else {
        println!(
            "  ✗ Q4_K output is NOT zero! Values: {:?}",
            &output[..4.min(m)]
        );
    }
}

fn test_q2_k(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q2_K...");

    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ2K::zeroed(); m * n_blocks];

    // Fill with zeros
    for block in &mut weights {
        block.d = half::f16::from_f32(1.0).to_bits();
        block.dmin = half::f16::from_f32(0.0).to_bits();
    }

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ2K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ2K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q2_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q2_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let max_abs = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
    println!("  Max absolute output: {}", max_abs);
    if max_abs < 1e-3 {
        println!("  ✓ Q2_K output is near zero");
    } else {
        println!(
            "  ✗ Q2_K output is NOT zero! Values: {:?}",
            &output[..4.min(m)]
        );
    }
}

fn test_q6_k(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q6_K...");

    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ6K::zeroed(); m * n_blocks];

    // Fill with zeros
    for block in &mut weights {
        block.d = half::f16::from_f32(1.0).to_bits();
    }

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ6K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ6K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q6_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q6_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let max_abs = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
    println!("  Max absolute output: {}", max_abs);
    if max_abs < 1e-3 {
        println!("  ✓ Q6_K output is near zero");
    } else {
        println!(
            "  ✗ Q6_K output is NOT zero! Values: {:?}",
            &output[..4.min(m)]
        );
    }
}

fn test_q3_k(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q3_K...");

    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ3K::zeroed(); m * n_blocks];

    // For Q3_K, d=0 ensures all weights are zero
    // (scales are offset by 32, so scale=0 gives dl = d*(0-32) = 0 when d=0)
    for block in &mut weights {
        block.d = half::f16::from_f32(0.0).to_bits();
    }

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ3K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ3K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q3_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q3_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let max_abs = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
    println!("  Max absolute output: {}", max_abs);
    if max_abs < 1e-3 {
        println!("  ✓ Q3_K output is near zero");
    } else {
        println!(
            "  ✗ Q3_K output is NOT zero! Values: {:?}",
            &output[..4.min(m)]
        );
    }
}

fn test_q5_k(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q5_K...");

    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ5K::zeroed(); m * n_blocks];

    // Fill with zeros
    for block in &mut weights {
        block.d = half::f16::from_f32(1.0).to_bits();
        block.dmin = half::f16::from_f32(0.0).to_bits();
    }

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ5K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ5K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q5_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let max_abs = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
    println!("  Max absolute output: {}", max_abs);
    if max_abs < 1e-3 {
        println!("  ✓ Q5_K output is near zero");
    } else {
        println!(
            "  ✗ Q5_K output is NOT zero! Values: {:?}",
            &output[..4.min(m)]
        );
    }
}
