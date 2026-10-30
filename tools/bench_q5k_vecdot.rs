// Benchmark Q5_K vec_dot kernel directly vs llama.cpp

use infer_train::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use std::time::Instant;

fn main() {
    // Test with actual Qwen1.5B dimensions
    // w1: [1536, 8960] - hidden_size x intermediate_size
    // Each vec_dot processes 1536 elements = 6 blocks
    // Total: 8960 vec_dot calls

    let hidden_size = 1536;
    let intermediate_size = 8960;
    let nb = hidden_size / QK_K; // 6 blocks per row

    // Create test data
    let n_weights = intermediate_size * nb;
    let mut weights: Vec<BlockQ5K> = Vec::with_capacity(n_weights);
    let mut x: Vec<BlockQ8K> = Vec::with_capacity(nb);

    for _ in 0..n_weights {
        let mut block: BlockQ5K = unsafe { std::mem::zeroed() };
        block.d = 0x3C00; // FP16 1.0
        weights.push(block);
    }

    for _ in 0..nb {
        let mut block: BlockQ8K = unsafe { std::mem::zeroed() };
        block.d = 1.0;
        x.push(block);
    }

    let mut out: Vec<f32> = vec![0.0; intermediate_size];

    let n_iterations = 100;

    // Warm up
    for _ in 0..10 {
        unsafe {
            infer_train::quant::matmul::k_quant::matmul_q5_k_q8_k(
                &weights,
                &x,
                &mut out,
                hidden_size,
                intermediate_size,
            );
        }
    }

    // Benchmark multi-thread
    let start = Instant::now();
    for _ in 0..n_iterations {
        unsafe {
            infer_train::quant::matmul::k_quant::matmul_q5_k_q8_k(
                &weights,
                &x,
                &mut out,
                hidden_size,
                intermediate_size,
            );
        }
    }
    let mt_time = start.elapsed();

    // Benchmark single-thread
    let start = Instant::now();
    for _ in 0..n_iterations {
        unsafe {
            infer_train::quant::matmul::k_quant::matmul_q5_k_q8_k_single_thread(
                &weights,
                &x,
                &mut out,
                hidden_size,
                intermediate_size,
            );
        }
    }
    let st_time = start.elapsed();

    println!("=== Q5_K GEMV Benchmark (w1 dimensions) ===");
    println!("Matrix: {}x{}", hidden_size, intermediate_size);
    println!("Blocks per row: {}", nb);
    println!("Output rows: {}", intermediate_size);
    println!();
    println!(
        "Multi-thread: {:.2} µs per GEMV",
        mt_time.as_micros() as f64 / n_iterations as f64
    );
    println!(
        "Single-thread: {:.2} µs per GEMV",
        st_time.as_micros() as f64 / n_iterations as f64
    );
    println!("Speedup: {:.2}x", st_time.as_secs_f64() / mt_time.as_secs_f64());
}
