// Measure actual memory bandwidth during GEMV

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    println!("=== Memory Bandwidth Measurement ===\n");

    // Simulate Q4_K GEMV memory access pattern
    let k = 1024;
    let m = 2048;

    // Weight size in bytes (Q4_K packs 4 bits per weight, plus scales)
    // Each BlockQ4K is 144 bytes for 256 elements
    // So for 1024 elements: 4 blocks = 576 bytes
    // For 2048 rows: 576 × 2048 = 1.18 MB
    let weight_bytes = (k / 256) * 144 * m;

    // Input size (Q8_K: 32 bytes per block for 256 elements)
    // For 1024 elements: 4 blocks = 128 bytes
    let input_bytes = (k / 256) * 32;

    // Output size (F32: 4 bytes per element)
    let output_bytes = m * 4;

    let total_memory_bytes = weight_bytes + input_bytes + output_bytes;

    println!("Weight matrix: {:.2} MB", weight_bytes as f64 / 1_000_000.0);
    println!("Input vector: {} bytes", input_bytes);
    println!("Output vector: {} bytes", output_bytes);
    println!(
        "Total memory accessed: {:.2} MB",
        total_memory_bytes as f64 / 1_000_000.0
    );
    println!();

    // Create test data
    let weights: Vec<u8> = vec![0; weight_bytes];
    let input: Vec<u8> = vec![0; input_bytes];
    let mut output: Vec<f32> = vec![0.0; m];

    // Measure memory bandwidth
    let n_iter = 100;
    let start = Instant::now();

    for _ in 0..n_iter {
        // Simulate reading all weight data
        let mut sum = 0u64;
        for &byte in &weights {
            sum += byte as u64;
        }

        // Simulate reading input
        for &byte in &input {
            sum += byte as u64;
        }

        // Simulate writing output
        for out in &mut output {
            *out = sum as f32;
        }

        std::hint::black_box(sum);
    }

    let time_us = start.elapsed().as_micros() as f64 / n_iter as f64;
    let bandwidth_gb_s =
        (total_memory_bytes as f64 / 1_000_000_000.0) / (time_us / 1_000_000.0);

    println!("Sequential memory scan:");
    println!("  Time: {:.2} µs", time_us);
    println!("  Bandwidth: {:.2} GB/s", bandwidth_gb_s);
    println!();

    // Parallel version
    let start = Instant::now();

    for _ in 0..n_iter {
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            // Simulate reading one row of weights
            let row_start = i * (k / 256) * 144;
            let row_end = row_start + (k / 256) * 144;

            let mut sum = 0u64;
            for &byte in &weights[row_start..row_end] {
                sum += byte as u64;
            }

            for &byte in &input {
                sum += byte as u64;
            }

            *out = sum as f32;
        });
    }

    let time_us = start.elapsed().as_micros() as f64 / n_iter as f64;
    let bandwidth_gb_s =
        (total_memory_bytes as f64 / 1_000_000_000.0) / (time_us / 1_000_000.0);

    println!("Parallel memory scan:");
    println!("  Time: {:.2} µs", time_us);
    println!("  Bandwidth: {:.2} GB/s", bandwidth_gb_s);
    println!();

    // Compare with typical M-series memory bandwidth
    // M1: ~100 GB/s, M2: ~100 GB/s, M3: ~150 GB/s
    println!("Typical M-series CPU memory bandwidth: 100-150 GB/s");
    println!("Our utilization: {:.1}%", bandwidth_gb_s / 100.0 * 100.0);
}
