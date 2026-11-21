use infer_train::backend::arm::quant;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    let size = 1024; // hidden_size
    let n_blocks = (size + 255) / 256;

    // Create test input
    let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut output: Vec<BlockQ8_K> =
        vec![unsafe { std::mem::zeroed() }; n_blocks];

    // Warmup
    for _ in 0..100 {
        quant::quantize_q8_k(&input, &mut output);
    }

    // Measure
    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        quant::quantize_q8_k(&input, &mut output);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!("=== Quantization Benchmark ===");
    println!("Input size: {} elements", size);
    println!("Output: {} blocks (Q8_K)", n_blocks);
    println!("Average time: {:.2} ns", avg_ns);
    println!("Throughput: {:.2} M elements/s", size as f64 / avg_ns / 1e3);

    // Compare to vec_dot
    println!("\n=== Comparison ===");
    println!("vec_dot Q5_K: ~40 ns for 1024 elements");
    println!("quantize Q8_K: {:.2} ns for {} elements", avg_ns, size);
    println!("Ratio: {:.2}x", avg_ns / 40.0);
}
