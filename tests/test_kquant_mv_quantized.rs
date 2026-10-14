// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Correctness test for self-written K-quant MV kernels with quantized data

use bytemuck::Zeroable;
use infer_train::inference::dequant::{
    dequant_q2_k, dequant_q3_k, dequant_q4_k, dequant_q5_k, dequant_q6_k,
};
use infer_train::quant::types::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, QK_K,
};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::{MTLResourceOptions, NSRange};

fn main() {
    println!(
        "=== K-quant MV Kernel Correctness Test with Quantized Data ===\n"
    );

    let m = 32; // More rows
    let k = 1024; // Multiple blocks

    // Create Metal context
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Create input vector
    let input: Vec<f32> =
        (0..k).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

    // Test each K-quant format
    test_q4_k_quantized(&ctx, m, k, &input);
    test_q2_k_quantized(&ctx, m, k, &input);
    test_q6_k_quantized(&ctx, m, k, &input);
    test_q3_k_quantized(&ctx, m, k, &input);
    test_q5_k_quantized(&ctx, m, k, &input);

    println!("\n=== All tests completed ===");
}

fn test_q4_k_quantized(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q4_K with quantized data...");

    let n_blocks = m * k / QK_K;
    let mut weights = vec![BlockQ4K::zeroed(); n_blocks];

    // Create simple quantized weights with known pattern
    // Each block has d=1.0, dmin=0, scales=1.0
    for (i, block) in weights.iter_mut().enumerate() {
        block.d = half::f16::from_f32(1.0).to_bits();
        block.dmin = half::f16::from_f32(0.0).to_bits();
        // Set scales to simple pattern
        for j in 0..12 {
            block.scales[j] = 32; // Scale = 32, so sc - 32 = 0 when dequantizing
        }
        // Set quants to known values (alternating pattern)
        for j in 0..128 {
            block.qs[j] = ((i * 128 + j) % 256) as u8;
        }
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

    // Run GPU kernel
    let output_buffer = ctx
        .mv_q4_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q4_K MV failed");

    // Read GPU results
    let output_ptr = output_buffer.contents() as *const f32;
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    // Compute CPU reference using dequant function
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let block_offset = row * (k / QK_K);
        let row_weights = &weights[block_offset..block_offset + (k / QK_K)];

        // Dequantize weights to f32
        let mut dequant_weights = vec![0.0f32; k];
        for (ib, block) in row_weights.iter().enumerate() {
            let block_data = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ4K as *const u8,
                    std::mem::size_of::<BlockQ4K>(),
                )
            };
            dequant_q4_k(block_data, &mut dequant_weights[ib * QK_K..], QK_K);
        }

        // Compute dot product
        for i in 0..k {
            cpu_output[row] += dequant_weights[i] * input[i];
        }
    }

    // Compare results
    println!("  GPU output (first 4): {:?}", &gpu_output[..4.min(m)]);
    println!("  CPU output (first 4): {:?}", &cpu_output[..4.min(m)]);

    let mut max_diff = 0.0f32;
    for i in 0..m {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("  Max difference: {}", max_diff);

    if max_diff < 0.01 {
        println!("  ✓ Q4_K matches CPU reference");
    } else {
        println!("  ✗ Q4_K MISMATCH with CPU reference!");
    }
}

fn test_q2_k_quantized(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q2_K with quantized data...");

    let n_blocks = m * k / QK_K;
    let mut weights = vec![BlockQ2K::zeroed(); n_blocks];

    for (i, block) in weights.iter_mut().enumerate() {
        block.d = half::f16::from_f32(1.0).to_bits();
        block.dmin = half::f16::from_f32(0.0).to_bits();
        for j in 0..16 {
            block.scales[j] = 16;
        }
        for j in 0..64 {
            block.qs[j] = ((i * 64 + j) % 256) as u8;
        }
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
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    // CPU reference
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let block_offset = row * (k / QK_K);
        let row_weights = &weights[block_offset..block_offset + (k / QK_K)];

        let mut dequant_weights = vec![0.0f32; k];
        for (ib, block) in row_weights.iter().enumerate() {
            let block_data = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ2K as *const u8,
                    std::mem::size_of::<BlockQ2K>(),
                )
            };
            dequant_q2_k(block_data, &mut dequant_weights[ib * QK_K..], QK_K);
        }

        for i in 0..k {
            cpu_output[row] += dequant_weights[i] * input[i];
        }
    }

    let mut max_diff = 0.0f32;
    for i in 0..m {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("  Max difference: {}", max_diff);
    if max_diff < 0.01 {
        println!("  ✓ Q2_K matches CPU reference");
    } else {
        println!(
            "  ✗ Q2_K MISMATCH! GPU: {:?}, CPU: {:?}",
            &gpu_output[..4.min(m)],
            &cpu_output[..4.min(m)]
        );
    }
}

fn test_q6_k_quantized(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q6_K with quantized data...");

    let n_blocks = m * k / QK_K;
    let mut weights = vec![BlockQ6K::zeroed(); n_blocks];

    for (i, block) in weights.iter_mut().enumerate() {
        block.d = half::f16::from_f32(1.0).to_bits();
        for j in 0..16 {
            block.scales[j] = 1;
        }
        for j in 0..128 {
            block.ql[j] = ((i * 128 + j) % 256) as u8;
        }
        for j in 0..64 {
            block.qh[j] = ((i * 64 + j) % 256) as u8;
        }
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
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    // CPU reference
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let block_offset = row * (k / QK_K);
        let row_weights = &weights[block_offset..block_offset + (k / QK_K)];

        let mut dequant_weights = vec![0.0f32; k];
        for (ib, block) in row_weights.iter().enumerate() {
            let block_data = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ6K as *const u8,
                    std::mem::size_of::<BlockQ6K>(),
                )
            };
            dequant_q6_k(block_data, &mut dequant_weights[ib * QK_K..], QK_K);
        }

        for i in 0..k {
            cpu_output[row] += dequant_weights[i] * input[i];
        }
    }

    let mut max_diff = 0.0f32;
    for i in 0..m {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("  Max difference: {}", max_diff);
    if max_diff < 0.01 {
        println!("  ✓ Q6_K matches CPU reference");
    } else {
        println!(
            "  ✗ Q6_K MISMATCH! GPU: {:?}, CPU: {:?}",
            &gpu_output[..4.min(m)],
            &cpu_output[..4.min(m)]
        );
    }
}

fn test_q3_k_quantized(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q3_K with quantized data...");

    let n_blocks = m * k / QK_K;
    let mut weights = vec![BlockQ3K::zeroed(); n_blocks];

    for (i, block) in weights.iter_mut().enumerate() {
        block.d = half::f16::from_f32(1.0).to_bits();
        for j in 0..12 {
            block.scales[j] = 32; // Scale offset by 32
        }
        for j in 0..64 {
            block.qs[j] = ((i * 64 + j) % 256) as u8;
        }
        for j in 0..32 {
            block.hmask[j] = ((i * 32 + j) % 256) as u8;
        }
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
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    // CPU reference
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let block_offset = row * (k / QK_K);
        let row_weights = &weights[block_offset..block_offset + (k / QK_K)];

        let mut dequant_weights = vec![0.0f32; k];
        for (ib, block) in row_weights.iter().enumerate() {
            let block_data = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ3K as *const u8,
                    std::mem::size_of::<BlockQ3K>(),
                )
            };
            dequant_q3_k(block_data, &mut dequant_weights[ib * QK_K..], QK_K);
        }

        for i in 0..k {
            cpu_output[row] += dequant_weights[i] * input[i];
        }
    }

    let mut max_diff = 0.0f32;
    for i in 0..m {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("  Max difference: {}", max_diff);
    if max_diff < 0.01 {
        println!("  ✓ Q3_K matches CPU reference");
    } else {
        println!(
            "  ✗ Q3_K MISMATCH! GPU: {:?}, CPU: {:?}",
            &gpu_output[..4.min(m)],
            &cpu_output[..4.min(m)]
        );
    }
}

fn test_q5_k_quantized(ctx: &MetalContext, m: usize, k: usize, input: &[f32]) {
    println!("Testing Q5_K with quantized data...");

    let n_blocks = m * k / QK_K;
    let mut weights = vec![BlockQ5K::zeroed(); n_blocks];

    for (i, block) in weights.iter_mut().enumerate() {
        block.d = half::f16::from_f32(1.0).to_bits();
        block.dmin = half::f16::from_f32(0.0).to_bits();
        for j in 0..12 {
            block.scales[j] = 32;
        }
        for j in 0..128 {
            block.qs[j] = ((i * 128 + j) % 256) as u8;
        }
        for j in 0..32 {
            block.qh[j] = ((i * 32 + j) % 256) as u8;
        }
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
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    // CPU reference
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let block_offset = row * (k / QK_K);
        let row_weights = &weights[block_offset..block_offset + (k / QK_K)];

        let mut dequant_weights = vec![0.0f32; k];
        for (ib, block) in row_weights.iter().enumerate() {
            let block_data = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ5K as *const u8,
                    std::mem::size_of::<BlockQ5K>(),
                )
            };
            dequant_q5_k(block_data, &mut dequant_weights[ib * QK_K..], QK_K);
        }

        for i in 0..k {
            cpu_output[row] += dequant_weights[i] * input[i];
        }
    }

    let mut max_diff = 0.0f32;
    for i in 0..m {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("  Max difference: {}", max_diff);
    if max_diff < 0.01 {
        println!("  ✓ Q5_K matches CPU reference");
    } else {
        println!(
            "  ✗ Q5_K MISMATCH! GPU: {:?}, CPU: {:?}",
            &gpu_output[..4.min(m)],
            &cpu_output[..4.min(m)]
        );
    }
}
