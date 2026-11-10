// Test interleaved GEMV performance vs standard vec_dot

use infer_train::quant::matmul::k_quant::matmul_q4_k_q8_k_single_thread;
use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon;
use std::time::Instant;

fn main() {
    println!("=== Interleaved GEMV Performance Test ===\n");

    // Test dimensions matching typical model
    let k = 4096; // Hidden size (QK_K blocks)
    let m = 4096; // Output dimension (must be multiple of 8 for interleaved)

    let nb = k / QK_K;

    // Generate random weights and input
    let mut weights: Vec<BlockQ4K> = Vec::with_capacity(m * nb);
    for _ in 0..(m * nb) {
        weights.push(unsafe { std::mem::zeroed() });
    }

    let mut input: Vec<BlockQ8K> = Vec::with_capacity(nb);
    for _ in 0..nb {
        input.push(unsafe { std::mem::zeroed() });
    }

    let mut output = vec![0.0f32; m];

    // Warm up
    unsafe {
        matmul_q4_k_q8_k_single_thread(&weights, &input, &mut output, k, m);
    }

    // Test interleaved path (m % 8 == 0)
    println!("Testing interleaved path (m = {}):", m);
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        unsafe {
            matmul_q4_k_q8_k_single_thread(&weights, &input, &mut output, k, m);
        }
    }
    let interleaved_time = start.elapsed() / iterations;
    println!("  Average time: {:?}", interleaved_time);
    println!(
        "  Throughput: {:.2} GFLOPS",
        (2.0 * k as f64 * m as f64) / interleaved_time.as_secs_f64() / 1e9
    );

    // Test standard vec_dot for comparison
    println!("\nTesting standard vec_dot (single row):");
    let start = Instant::now();
    let single_iterations = 1000;
    for _ in 0..single_iterations {
        unsafe {
            output[0] = vec_dot_q4_k_q8_k_neon(k, &weights[0..nb], &input);
        }
    }
    let vec_dot_time = start.elapsed() / single_iterations;
    println!("  Average time: {:?}", vec_dot_time);
    println!(
        "  Throughput: {:.2} GFLOPS",
        (2.0 * k as f64) / vec_dot_time.as_secs_f64() / 1e9
    );

    // Calculate expected improvement
    let expected_single = vec_dot_time * m as u32;
    println!("\n=== Analysis ===");
    println!("Standard vec_dot for m rows: {:?}", expected_single);
    println!("Interleaved actual: {:?}", interleaved_time);
    println!(
        "Speedup: {:.2}x",
        expected_single.as_secs_f64() / interleaved_time.as_secs_f64()
    );
}
