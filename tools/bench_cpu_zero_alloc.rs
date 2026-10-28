// Benchmark zero-allocation CPU GEMV vs allocating version

use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::matmul::k_quant::matmul_q4_k_q8_k;
use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use std::time::Instant;

fn main() {
    // Test dimensions matching typical LLM layers
    let hidden_size = 4096;
    let intermediate_size = 11008;
    let n_layers = 32;

    let nb = hidden_size / QK_K;
    let n_iterations = 100;

    println!("=== CPU GEMV Zero-Allocation Benchmark ===");
    println!(
        "hidden_size: {}, intermediate_size: {}",
        hidden_size, intermediate_size
    );
    println!("iterations: {}\n", n_iterations);

    // Create test data
    let mut x = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        x[i] = (i as f32 % 10.0) - 5.0;
    }

    // Create Q4_K weights (intermediate_size rows, hidden_size cols)
    let weights: Vec<BlockQ4K> =
        vec![unsafe { std::mem::zeroed() }; intermediate_size * nb];

    // Pre-allocated buffers
    let mut x_q8k: Vec<u8> = vec![0u8; nb * std::mem::size_of::<BlockQ8K>()];
    let mut x_padded: Vec<f32> = vec![0.0f32; nb * QK_K];
    let mut out: Vec<f32> = vec![0.0f32; intermediate_size];

    // Benchmark 1: Allocating version (current gemv_into)
    let mut total_alloc_time = 0u64;
    for _ in 0..n_iterations {
        let start = Instant::now();

        // Simulate current gemv_into allocation pattern
        let mut x_q8k_alloc: Vec<BlockQ8K> =
            vec![unsafe { std::mem::zeroed() }; nb];
        quantize_row_q8_k_neon(&x, &mut x_q8k_alloc);

        unsafe {
            matmul_q4_k_q8_k(
                &weights,
                &x_q8k_alloc,
                &mut out,
                hidden_size,
                intermediate_size,
            );
        }

        total_alloc_time += start.elapsed().as_nanos() as u64;
    }

    // Benchmark 2: Zero-allocation version
    let mut total_zero_alloc_time = 0u64;
    for _ in 0..n_iterations {
        let start = Instant::now();

        let x_q8k_blocks = unsafe {
            std::slice::from_raw_parts_mut(
                x_q8k.as_mut_ptr() as *mut BlockQ8K,
                nb,
            )
        };
        quantize_row_q8_k_neon(&x, x_q8k_blocks);

        unsafe {
            matmul_q4_k_q8_k(
                &weights,
                x_q8k_blocks,
                &mut out,
                hidden_size,
                intermediate_size,
            );
        }

        total_zero_alloc_time += start.elapsed().as_nanos() as u64;
    }

    // Results
    println!("Allocating version:");
    println!("  Total time: {:.2} ms", total_alloc_time as f64 / 1_000_000.0);
    println!(
        "  Per iteration: {:.2} µs",
        total_alloc_time as f64 / n_iterations as f64 / 1000.0
    );
    println!(
        "  Per layer estimate (32 layers): {:.2} ms",
        (total_alloc_time as f64 / n_iterations as f64) * n_layers as f64
            / 1_000_000.0
    );

    println!("\nZero-allocation version:");
    println!(
        "  Total time: {:.2} ms",
        total_zero_alloc_time as f64 / 1_000_000.0
    );
    println!(
        "  Per iteration: {:.2} µs",
        total_zero_alloc_time as f64 / n_iterations as f64 / 1000.0
    );
    println!(
        "  Per layer estimate (32 layers): {:.2} ms",
        (total_zero_alloc_time as f64 / n_iterations as f64) * n_layers as f64
            / 1_000_000.0
    );

    let speedup = total_alloc_time as f64 / total_zero_alloc_time as f64;
    println!("\nSpeedup: {:.2}x", speedup);
}
