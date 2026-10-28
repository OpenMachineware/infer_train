// Test GEMV cache behavior with different thread assignment strategies
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon;
use std::time::Instant;

fn make_block_q4k(seed: usize) -> BlockQ4K {
    let mut block: BlockQ4K = unsafe { std::mem::zeroed() };
    block.d = half::f16::from_f32(0.5).to_bits();
    block.dmin = half::f16::from_f32(0.0).to_bits();
    for j in 0..128 {
        block.qs[j] = ((seed + j) % 256) as u8;
    }
    block.scales.fill(((seed % 16) + 1) as u8);
    block
}

fn make_block_q8k(seed: usize) -> BlockQ8K {
    let mut block: BlockQ8K = unsafe { std::mem::zeroed() };
    block.d = 0.5;
    for j in 0..256 {
        block.qs[j] = ((seed + j) % 256) as i8;
    }
    // Compute bsums
    for i in 0..16 {
        let mut sum = 0i16;
        for j in 0..16 {
            sum += block.qs[i * 16 + j] as i16;
        }
        block.bsums[i] = sum;
    }
    block
}

/// Rayon parallel approach (work-stealing)
fn matmul_rayon(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use rayon::prelude::*;
    dst.par_iter_mut().enumerate().for_each(|(i, out)| unsafe {
        *out = vec_dot_q4_k_q8_k_neon(k, &weights[i * nb..(i + 1) * nb], x);
    });
}

/// Static thread assignment approach (like llama.cpp)
fn matmul_static(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    _m: usize,
) {
    let nb = k / QK_K;
    let m = dst.len();
    let nth =
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let dr = (m + nth - 1) / nth;

    use rayon::prelude::ParallelSliceMut;
    use rayon::prelude::*;

    dst.par_chunks_mut(dr).enumerate().for_each(|(chunk_idx, dst_chunk)| {
        let ir0 = chunk_idx * dr;
        for (idx, out) in dst_chunk.iter_mut().enumerate() {
            let i = ir0 + idx;
            if i < m {
                unsafe {
                    *out = vec_dot_q4_k_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
    });
}

fn main() {
    println!("=== GEMV Cache Behavior Test ===\n");

    // Test dimensions matching a typical layer
    let k = 2048; // Input dimension
    let m = 4096; // Output dimension (e.g., for gate projection)
    let nb = k / QK_K;

    // Create weights - large enough to exceed L2 cache
    println!(
        "Creating {} MB of weights...",
        (m * nb * std::mem::size_of::<BlockQ4K>()) / 1024 / 1024
    );
    let weights: Vec<BlockQ4K> =
        (0..(m * nb)).map(|i| make_block_q4k(i)).collect();

    // Create input
    let x: Vec<BlockQ8K> = (0..nb).map(|i| make_block_q8k(i)).collect();

    let n_iter = 50;

    // Test 1: Rayon work-stealing
    println!("\n=== Test 1: Rayon work-stealing ===");
    let mut dst1 = vec![0.0f32; m];
    let mut times1 = Vec::new();
    for _ in 0..n_iter {
        let start = Instant::now();
        matmul_rayon(&weights, &x, &mut dst1, k, m);
        times1.push(start.elapsed().as_micros() as f64);
    }
    let avg1: f64 = times1.iter().sum::<f64>() / n_iter as f64;
    println!("Average time: {:.2} µs", avg1);

    // Test 2: Static thread assignment
    println!("\n=== Test 2: Static thread assignment ===");
    let mut dst2 = vec![0.0f32; m];
    let mut times2 = Vec::new();
    for _iter in 0..n_iter {
        let start = Instant::now();
        matmul_static(&weights, &x, &mut dst2, k, m);
        times2.push(start.elapsed().as_micros() as f64);
    }
    let avg2: f64 = times2.iter().sum::<f64>() / n_iter as f64;
    println!("Average time: {:.2} µs", avg2);

    // Compare
    println!("\n=== Comparison ===");
    println!("Rayon:  {:.2} µs", avg1);
    println!("Static: {:.2} µs", avg2);
    println!("Speedup: {:.2}x", avg1 / avg2);

    // Verify correctness
    let max_diff = dst1
        .iter()
        .zip(dst2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {:.6}", max_diff);
}
