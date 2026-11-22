// Benchmark vec_dot performance
use infer_train::backend::arm::vec_dot::*;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    let n_blocks = 1024; // Typical for 256K element vector
    let iterations = 1000;

    // Create test data
    let mut q5k_weights: Vec<BlockQ5_K> = Vec::with_capacity(n_blocks);
    let mut q4k_weights: Vec<BlockQ4_K> = Vec::with_capacity(n_blocks);
    let mut q6k_weights: Vec<BlockQ6_K> = Vec::with_capacity(n_blocks);
    let mut q8k_input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);

    for i in 0..n_blocks {
        q5k_weights.push(unsafe { std::mem::zeroed() });
        q4k_weights.push(unsafe { std::mem::zeroed() });
        q6k_weights.push(unsafe { std::mem::zeroed() });
        q8k_input.push(unsafe { std::mem::zeroed() });
    }

    // Warm up
    for _ in 0..10 {
        let _ = vec_dot_q5_k_q8_k(&q5k_weights, &q8k_input);
        let _ = vec_dot_q4_k_q8_k(&q4k_weights, &q8k_input);
        let _ = vec_dot_q6_k_q8_k(&q6k_weights, &q8k_input);
    }

    // Benchmark Q5_K
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = vec_dot_q5_k_q8_k(&q5k_weights, &q8k_input);
    }
    let q5k_time = start.elapsed();

    // Benchmark Q4_K
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = vec_dot_q4_k_q8_k(&q4k_weights, &q8k_input);
    }
    let q4k_time = start.elapsed();

    // Benchmark Q6_K
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = vec_dot_q6_k_q8_k(&q6k_weights, &q8k_input);
    }
    let q6k_time = start.elapsed();

    let elements_per_vec = n_blocks * QK_K;
    let total_bytes_q5k = n_blocks * std::mem::size_of::<BlockQ5_K>();
    let total_bytes_q4k = n_blocks * std::mem::size_of::<BlockQ4_K>();
    let total_bytes_q6k = n_blocks * std::mem::size_of::<BlockQ6_K>();
    let total_bytes_q8k = n_blocks * std::mem::size_of::<BlockQ8_K>();

    println!("=== vec_dot Benchmark ===");
    println!("n_blocks: {}, elements: {}", n_blocks, elements_per_vec);
    println!("iterations: {}", iterations);
    println!();

    println!("Q5_K vec_dot:");
    println!("  Time: {:?}", q5k_time);
    println!(
        "  Per call: {:.2} µs",
        q5k_time.as_secs_f64() * 1e6 / iterations as f64
    );
    println!(
        "  Throughput: {:.2} GB/s",
        (total_bytes_q5k + total_bytes_q8k) as f64 * iterations as f64
            / q5k_time.as_secs_f64()
            / 1e9
    );

    println!("\nQ4_K vec_dot:");
    println!("  Time: {:?}", q4k_time);
    println!(
        "  Per call: {:.2} µs",
        q4k_time.as_secs_f64() * 1e6 / iterations as f64
    );
    println!(
        "  Throughput: {:.2} GB/s",
        (total_bytes_q4k + total_bytes_q8k) as f64 * iterations as f64
            / q4k_time.as_secs_f64()
            / 1e9
    );

    println!("\nQ6_K vec_dot:");
    println!("  Time: {:?}", q6k_time);
    println!(
        "  Per call: {:.2} µs",
        q6k_time.as_secs_f64() * 1e6 / iterations as f64
    );
    println!(
        "  Throughput: {:.2} GB/s",
        (total_bytes_q6k + total_bytes_q8k) as f64 * iterations as f64
            / q6k_time.as_secs_f64()
            / 1e9
    );
}
