// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Debug test for K-quant MV kernel

use bytemuck::Zeroable;
use infer_train::quant::types::{BlockQ4K, QK_K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::{MTLResourceOptions, NSRange};

fn main() {
    println!("=== K-quant MV Debug Test ===\n");

    let m = 1; // Single row
    let k = 256; // Single block

    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Create input vector with all 1.0
    let input: Vec<f32> = vec![1.0; k];

    // Create Q4_K weights
    let mut weights = vec![BlockQ4K::zeroed(); 1];
    let block = &mut weights[0];

    // Set d = 1.0, dmin = 0
    block.d = half::f16::from_f32(1.0).to_bits();
    block.dmin = half::f16::from_f32(0.0).to_bits();

    // Set all scales to 1.0 (scales are 6-bit values)
    // For sub-block 0, scale is scales[0] & 63
    block.scales[0] = 1; // scale for sub-block 0
    block.scales[4] = 0; // min for sub-block 0
    // Set all other scales to 0
    for j in 1..12 {
        block.scales[j] = 0;
    }

    // Set all quants to 8 (which should give dequantized value = d * sc * 8 - dmin * min)
    for j in 0..128 {
        block.qs[j] = 0x88; // Both nibbles are 8
    }

    // Expected output:
    // Each sub-block 0 has sc=1, min=0
    // dl = d * sc = 1.0 * 1 = 1.0
    // ml = dmin * min = 0
    // Each element = dl * q - ml = 1.0 * 8 = 8.0
    // With all scales=1, each element = 8.0
    // Sum of 32 elements in sub-block 0: 32 * 8 = 256
    // With input all 1.0: sum = 256

    println!("Weights: d={}, dmin={}", 1.0, 0.0);
    println!("scales: {:?}", &block.scales[..12]);
    println!("qs[0..4]: {:?}", &block.qs[..4]);

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

    println!("GPU output: {}", output[0]);
    println!("Expected: ~256.0 (sub-block 0 with 32 elements * 8.0 each)");

    // CPU reference
    use infer_train::inference::dequant::dequant_q4_k;
    let block_data = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            std::mem::size_of::<BlockQ4K>(),
        )
    };
    let mut dequant_weights = vec![0.0f32; k];
    dequant_q4_k(block_data, &mut dequant_weights, k);

    let cpu_sum: f32 =
        dequant_weights.iter().zip(input.iter()).map(|(w, i)| w * i).sum();
    println!("CPU output: {}", cpu_sum);
    println!("Dequant weights[0..10]: {:?}", &dequant_weights[..10]);
}
