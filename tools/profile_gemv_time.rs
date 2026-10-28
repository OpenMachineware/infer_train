// Compare GEMV performance between our implementation and llama.cpp

use std::time::Instant;

fn main() {
    println!("=== GEMV Performance Comparison ===\n");

    // Qwen3-0.6B dimensions
    let hidden = 1024;
    let intermediate = 2816; // FFN intermediate size

    // Simulate Q4_K × Q8_K GEMV
    println!("Testing Q4_K GEMV with dimensions:");
    println!("  hidden_size = {}", hidden);
    println!("  intermediate = {}", intermediate);

    // Our GEMV: hidden x intermediate matrix (gate/up/down projections)
    // Each GEMV: k = hidden, m = intermediate

    let k = hidden;
    let m = intermediate;
    let nb = k / 256; // QK_K = 256

    // Create dummy input and weights
    use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};

    let mut weights: Vec<BlockQ4K> = Vec::with_capacity(m * nb);
    for _ in 0..(m * nb) {
        weights.push(BlockQ4K {
            d: half::f16::from_f32(0.1).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [0x11u8; 12], // Simplified
            qs: [0u8; 128],
        });
    }

    let mut x: Vec<BlockQ8K> = Vec::with_capacity(nb);
    for _ in 0..nb {
        x.push(BlockQ8K { d: 0.1, bsums: [0i16; 16], qs: [0i8; 256] });
    }

    let mut dst = vec![0.0f32; m];

    // Warmup
    println!("\nWarmup...");
    for _ in 0..3 {
        unsafe {
            infer_train::quant::matmul::k_quant::matmul_q4_k_q8_k(
                &weights, &x, &mut dst, k, m,
            );
        }
    }

    // Benchmark
    println!("\nBenchmark (100 iterations)...");
    let mut times = Vec::new();

    for _ in 0..100 {
        let t0 = Instant::now();
        unsafe {
            infer_train::quant::matmul::k_quant::matmul_q4_k_q8_k(
                &weights, &x, &mut dst, k, m,
            );
        }
        times.push(t0.elapsed().as_nanos());
    }

    let avg_ns: u128 = times.iter().sum::<u128>() / 100;
    let min_ns = *times.iter().min().unwrap();

    println!("  Average: {:.2} ms", avg_ns as f64 / 1_000_000.0);
    println!("  Min:     {:.2} ms", min_ns as f64 / 1_000_000.0);

    // Calculate GOPS
    // Q4_K × Q8_K: each output element requires k multiplications
    // But with quantization, each block of 256 is compressed
    // Operations per output: k multiplications + dequantization
    let ops_per_output = k as f64;
    let total_ops = m as f64 * ops_per_output;
    let gops = total_ops / (avg_ns as f64 / 1_000_000_000.0) / 1_000_000_000.0;

    println!("  GOPS:    {:.2}", gops);

    println!("\n=== Model Layer Analysis ===");
    let n_layers = 28;

    // Estimated GEMV time in that layer
    // Actually, we have RMS norms, RoPE, attention, residual adds too
    // Let's estimate GEMV is 80% of layer time
    let gemv_time_per_layer_ms = 6.17; // From profile
    let gemv_time_est = gemv_time_per_layer_ms * 0.8;

    println!("Per layer time: {:.2} ms", gemv_time_per_layer_ms);
    println!("Estimated GEMV time: {:.2} ms", gemv_time_est);

    let total_gemv_time_ms = gemv_time_est * n_layers as f64;
    println!("Total GEMV time (28 layers): {:.2} ms", total_gemv_time_ms);

    println!("\n=== llama.cpp comparison ===");
    println!("llama.cpp decode: 32.1 t/s = 31.2 ms/token");
    println!("Our decode: 4.66 t/s = 214.6 ms/token");
    println!("Gap: {:.1}x", 214.6 / 31.2);

    println!(
        "\nIf GEMV is the bottleneck, our GEMV is {:.1}x slower",
        (total_gemv_time_ms / 31.2)
    );
}
