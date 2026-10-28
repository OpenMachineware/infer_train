// Test quantization overhead
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::types::BlockQ8K;
use infer_train::quant::types::QK_K;
use std::time::Instant;

fn main() {
    println!("=== Quantization Overhead Test ===\n");

    // Test dimensions matching a typical hidden state
    let hidden_size = 2048; // Typical hidden dimension
    let nb = hidden_size / QK_K;

    // Create test input
    let x: Vec<f32> =
        (0..hidden_size).map(|i| (i as f32 * 0.01).sin()).collect();

    // Pre-allocate output buffer
    let mut x_q8k: Vec<BlockQ8K> = vec![unsafe { std::mem::zeroed() }; nb];

    let n_iter = 1000;

    // Warm up
    for _ in 0..10 {
        unsafe {
            quantize_row_q8_k_neon(&x, &mut x_q8k);
        }
    }

    // Measure quantization time
    let start = Instant::now();
    for _ in 0..n_iter {
        unsafe {
            quantize_row_q8_k_neon(&x, &mut x_q8k);
        }
    }
    let quant_time_us = start.elapsed().as_micros() as f64 / n_iter as f64;

    println!("Hidden size: {}", hidden_size);
    println!("Quantization time: {:.2} µs", quant_time_us);
    println!(
        "Throughput: {:.2} GB/s",
        (hidden_size * 4) as f64 / quant_time_us / 1000.0
    );

    // Compare with memory bandwidth
    // M1 Pro has ~400 GB/s bandwidth, so 8 KB should take ~20 ns
    // If quantization takes 10 µs, we're 500x slower than memory bandwidth!

    // For a 32-layer model with 5 redundant quantizations per layer:
    let wasted_time_us = quant_time_us * 5.0 * 32.0;
    println!("\n=== Impact Analysis ===");
    println!("Redundant quantizations per layer: 5 (3 QKV + 2 FFN)");
    println!("Layers: 32");
    println!(
        "Total wasted time: {:.2} µs = {:.2} ms per forward pass",
        wasted_time_us,
        wasted_time_us / 1000.0
    );

    // At 4.33 t/s, each forward pass takes ~231 ms
    // Wasted quantization time could be significant!
}
