// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test tool for interleaved Q4_K GEMV correctness and performance
//!
//! Usage: cargo run --bin test-interleaved-gemv

use infer_train::quant::kv_cache::quant::fp32_to_fp16;
use infer_train::quant::kv_cache::quant_q8k::quantize_row_q8_k_neon;
use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::{
    BlockQ4Kx8, repack_q4_k_batch, vec_dot_q4_k_q8_k_interleaved,
    vec_dot_q4_k_q8_k_neon,
};
use std::time::Instant;

fn main() {
    println!("=== Interleaved Q4_K GEMV Test ===\n");

    // Test dimensions similar to typical LLM layer
    let k = 4096; // Input dimension (K dimension)
    let m = 4096; // Output dimension (M dimension - number of rows)
    let nb = k / QK_K; // Number of blocks

    println!("Dimensions: K={}, M={}, blocks_per_row={}", k, m, nb);

    // Generate random test data
    println!("\nGenerating test data...");
    let mut rng = rand::thread_rng();

    // Generate random FP32 input
    let x_f32: Vec<f32> =
        (0..k).map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0)).collect();

    // Quantize input to Q8_K
    let mut x_q8k = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_row_q8_k_neon(&x_f32, &mut x_q8k);

    // Generate random Q4_K weights
    let num_weights = m * nb;
    let mut weights_q4k: Vec<BlockQ4K> = Vec::with_capacity(num_weights);
    for _ in 0..num_weights {
        let mut block: BlockQ4K = unsafe { std::mem::zeroed() };

        // Random scales and quants
        let d_f32: f32 = rand::Rng::gen_range(&mut rng, 0.1..1.0);
        let dmin_f32: f32 = rand::Rng::gen_range(&mut rng, 0.0..0.5);

        // Convert to FP16
        block.d = fp32_to_fp16(d_f32);
        block.dmin = fp32_to_fp16(dmin_f32);

        // Random scales
        for i in 0..12 {
            block.scales[i] = rand::Rng::gen_range(&mut rng, 0..64) as u8;
        }

        // Random quants
        for i in 0..128 {
            block.qs[i] = rand::Rng::gen_range(&mut rng, 0..255);
        }

        weights_q4k.push(block);
    }

    println!("Generated {} Q4_K weight blocks", weights_q4k.len());

    // Repack weights to interleaved format
    println!("\nRepacking weights to interleaved format...");
    let start = Instant::now();
    let weights_interleaved = repack_q4_k_batch(&weights_q4k, m, nb);
    let repack_time = start.elapsed();
    println!("Repack time: {:?}", repack_time);
    println!("Interleaved blocks: {}", weights_interleaved.len());

    // Compute reference output using standard GEMV
    println!("\nComputing reference output (standard GEMV)...");
    let mut dst_ref = vec![0.0f32; m];

    let start = Instant::now();
    for row in 0..m {
        let row_weights: &[BlockQ4K] = &weights_q4k[row * nb..(row + 1) * nb];
        unsafe {
            dst_ref[row] = vec_dot_q4_k_q8_k_neon(k, row_weights, &x_q8k);
        }
    }
    let standard_time = start.elapsed();
    println!("Standard GEMV time: {:?}", standard_time);
    println!(
        "Standard GEMV throughput: {:.2} M elements/s",
        (m as f64 * k as f64) / standard_time.as_secs_f64() / 1e6
    );

    // Compute output using interleaved GEMV
    println!("\nComputing output (interleaved GEMV)...");
    let mut dst_interleaved = vec![0.0f32; m];

    let start = Instant::now();
    unsafe {
        vec_dot_q4_k_q8_k_interleaved(
            &weights_interleaved,
            &x_q8k,
            &mut dst_interleaved,
            k,
            m,
        );
    }
    let interleaved_time = start.elapsed();
    println!("Interleaved GEMV time: {:?}", interleaved_time);
    println!(
        "Interleaved GEMV throughput: {:.2} M elements/s",
        (m as f64 * k as f64) / interleaved_time.as_secs_f64() / 1e6
    );

    // Compare results
    println!("\n=== Correctness Check ===");
    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    let mut total_rel_err = 0.0f32;

    for i in 0..m {
        let abs_err = (dst_ref[i] - dst_interleaved[i]).abs();
        let rel_err = if dst_ref[i].abs() > 1e-6 {
            abs_err / dst_ref[i].abs()
        } else {
            abs_err
        };

        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
        total_rel_err += rel_err;
    }

    println!("Max absolute error: {:.6e}", max_abs_err);
    println!(
        "Max relative error: {:.6e} ({:.4}%)",
        max_rel_err,
        max_rel_err * 100.0
    );
    println!(
        "Avg relative error: {:.6e} ({:.4}%)",
        total_rel_err / m as f32,
        (total_rel_err / m as f32) * 100.0
    );

    // Performance comparison
    println!("\n=== Performance Comparison ===");
    let speedup = standard_time.as_secs_f64() / interleaved_time.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);

    // Memory bandwidth estimation
    let weights_bytes = (m * k / 2) as f64; // Q4_K is 4 bits per element
    let input_bytes = (k * std::mem::size_of::<f32>()) as f64;
    let output_bytes = (m * std::mem::size_of::<f32>()) as f64;
    let total_bytes = weights_bytes + input_bytes + output_bytes;

    println!("\n=== Memory Bandwidth ===");
    println!("Weights: {:.2} MB", weights_bytes / 1e6);
    println!("Total data: {:.2} MB", total_bytes / 1e6);
    println!(
        "Standard GEMV bandwidth: {:.2} GB/s",
        total_bytes / standard_time.as_secs_f64() / 1e9
    );
    println!(
        "Interleaved GEMV bandwidth: {:.2} GB/s",
        total_bytes / interleaved_time.as_secs_f64() / 1e9
    );

    // Check if correctness is acceptable
    if max_rel_err < 0.001 {
        println!("\n✓ Correctness: PASS (rel err < 0.1%)");
    } else {
        println!("\n✗ Correctness: FAIL (rel err >= 0.1%)");
    }

    // Check if we achieved the target
    let target_speedup = 22.0; // Need 22x to match llama.cpp
    if speedup >= target_speedup {
        println!("✓ Performance: PASS (speedup >= 22x)");
    } else {
        println!("✗ Performance: Current speedup {:.2}x, target 22x", speedup);
    }
}
