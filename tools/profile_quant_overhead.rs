// Profile quantization overhead in GEMV

use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::types::{BlockQ8K, QK_K};

fn zero_block() -> BlockQ8K {
    BlockQ8K { d: 0.0, qs: [0; 256], bsums: [0; 16] }
}

fn main() {
    println!("\n=== Quantization Overhead Profile ===\n");

    // Test different input sizes
    let test_sizes = [1024, 2048, 3072];

    for &k in &test_sizes {
        let nb = k / QK_K;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut y = vec![zero_block(); nb];

        // Benchmark quantization
        let n_iterations = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n_iterations {
            quantize_row_q8_k_neon(&x, &mut y);
        }
        let quant_time_us =
            start.elapsed().as_micros() as f64 / n_iterations as f64;

        // Benchmark cloning
        let start = std::time::Instant::now();
        for _ in 0..n_iterations {
            let _cloned = y.clone();
        }
        let clone_time_us =
            start.elapsed().as_micros() as f64 / n_iterations as f64;

        println!("K = {} (nb = {}):", k, nb);
        println!(
            "  Quantization: {:.2} µs ({:.2} µs/block)",
            quant_time_us,
            quant_time_us / nb as f64
        );
        println!(
            "  Clone: {:.2} µs ({:.2} MB)",
            clone_time_us,
            (nb * std::mem::size_of::<BlockQ8K>()) as f64 / 1_000_000.0
        );
        println!();
    }

    // Estimate total overhead per forward pass
    println!("=== Estimated Overhead per Forward Pass ===\n");

    // Count quantization operations per layer
    // Q GEMV (1024->2048): K=1024
    // K GEMV (1024->1024): K=1024
    // V GEMV (1024->1024): K=1024
    // Gate GEMV (1024->3072): K=1024
    // Up GEMV (1024->3072): K=1024
    // Down GEMV (3072->1024): K=3072
    // Wo GEMV (2048->1024): K=2048

    let layer_quant_ops = vec![
        ("Q GEMV", 1024),
        ("K GEMV", 1024),
        ("V GEMV", 1024),
        ("Gate GEMV", 1024),
        ("Up GEMV", 1024),
        ("Down GEMV", 3072),
        ("Wo GEMV", 2048),
    ];

    let mut total_quant_us = 0.0;
    let mut total_clone_us = 0.0;

    for (name, k) in &layer_quant_ops {
        let nb = k / QK_K;
        // Estimated from above benchmarks
        let quant_us = *k as f64 * 0.005; // ~5 ns per element
        let clone_us = (nb * std::mem::size_of::<BlockQ8K>()) as f64 * 0.001; // ~1 ns per byte
        println!(
            "{} (K={}): quant={:.2} µs, clone={:.2} µs",
            name, k, quant_us, clone_us
        );
        total_quant_us += quant_us;
        total_clone_us += clone_us;
    }

    println!(
        "\nPer layer total: {:.2} µs quant + {:.2} µs clone = {:.2} µs overhead",
        total_quant_us,
        total_clone_us,
        total_quant_us + total_clone_us
    );

    let n_layers = 28;
    let total_forward_overhead_ms =
        (total_quant_us + total_clone_us) * n_layers as f64 / 1000.0;
    println!(
        "Total forward pass overhead ({} layers): {:.2} ms",
        n_layers, total_forward_overhead_ms
    );
}
