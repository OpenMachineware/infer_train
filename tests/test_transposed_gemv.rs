// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test for transposed GEMV kernel

use bytemuck::Zeroable;
use infer_train::inference::dequant::dequant_q6_k;
use infer_train::quant::types::{BlockQ6K, QK_K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    println!("=== Transposed Q6_K GEMV Test ===\n");

    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test dimensions (vocab_size x hidden_size for typical output layer)
    let vocab_size = 4096; // M (output)
    let hidden_size = 1024; // K (input)

    // Generate transposed weights: stored as [K, M] instead of [M, K]
    // For output layer: each row of W is a column of the output projection
    let n_blocks = hidden_size * (vocab_size / QK_K);
    let mut weights = vec![BlockQ6K::zeroed(); n_blocks];

    // Fill with deterministic pattern
    for (i, block) in weights.iter_mut().enumerate() {
        for j in 0..128 {
            block.ql[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        for j in 0..64 {
            block.qh[j] = ((i + j) % 256) as u8;
        }
        for j in 0..16 {
            block.scales[j] = ((i + j) % 64 + 1) as i8;
        }
        block.d = 0x3C00; // 1.0 in f16
    }

    // Generate input vector
    let input: Vec<f32> =
        (0..hidden_size).map(|i| (i % 10) as f32 * 0.1).collect();

    // Run GPU transposed MV
    let gpu_output = ctx
        .mv_q6_k_f32_transposed(vocab_size, hidden_size, &weights, &input)
        .expect("Transposed MV failed");

    // Compute CPU reference
    // For transposed access: output[j] = sum over i of W[i,j] * input[i]
    // W is stored as [K, M], so W[i,j] is in block i * (M/QK_K) + j/QK_K
    let mut cpu_output = vec![0.0f32; vocab_size];

    let blocks_per_row = vocab_size / QK_K;

    for j in 0..vocab_size {
        let j_block = j / QK_K;
        let j_elem = j % QK_K;

        for i in 0..hidden_size {
            let block_idx = i * blocks_per_row + j_block;
            let block = &weights[block_idx];

            // Dequantize element j_elem from block
            let mut block_data = [0.0f32; QK_K];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ6K as *const u8,
                    std::mem::size_of::<BlockQ6K>(),
                )
            };
            dequant_q6_k(block_bytes, &mut block_data, QK_K);

            cpu_output[j] += block_data[j_elem] * input[i];
        }
    }

    // Compare results
    println!("GPU output (first 4): {:?}", &gpu_output[..4.min(vocab_size)]);
    println!("CPU output (first 4): {:?}", &cpu_output[..4.min(vocab_size)]);

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..vocab_size {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("\nMax difference: {} at index {}", max_diff, max_idx);

    if max_diff < 1.0 {
        println!("✓ Transposed Q6_K GEMV matches CPU reference");
    } else {
        println!("✗ Transposed Q6_K GEMV MISMATCH!");
    }
}
