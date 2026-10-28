// Measure actual GEMV time for Q5_K with real model dimensions

use std::time::Instant;

fn main() {
    // Qwen3.8-27B dimensions
    let hidden_size = 5120;
    let intermediate_size = 27648;
    let n_heads = 40;
    let n_kv_heads = 8;
    let head_dim = 128;
    let n_layers = 64;

    println!("=== GEMV Time Estimation for Qwen3.8-27B-Q5_K ===\n");
    println!("Model: 27B parameters, 64 layers");
    println!(
        "hidden_size: {}, intermediate_size: {}",
        hidden_size, intermediate_size
    );

    // GEMV calls per layer
    // 1. wq: hidden_size -> n_heads * head_dim = 5120 -> 5120
    // 2. wk: hidden_size -> n_kv_heads * head_dim = 5120 -> 1024
    // 3. wv: hidden_size -> n_kv_heads * head_dim = 5120 -> 1024
    // 4. wo: n_heads * head_dim -> hidden_size = 5120 -> 5120
    // 5. gate: hidden_size -> intermediate_size = 5120 -> 27648
    // 6. up: hidden_size -> intermediate_size = 5120 -> 27648
    // 7. down: intermediate_size -> hidden_size = 27648 -> 5120

    let gemv_sizes = [
        (hidden_size, n_heads * head_dim),    // wq
        (hidden_size, n_kv_heads * head_dim), // wk
        (hidden_size, n_kv_heads * head_dim), // wv
        (n_heads * head_dim, hidden_size),    // wo
        (hidden_size, intermediate_size),     // gate
        (hidden_size, intermediate_size),     // up
        (intermediate_size, hidden_size),     // down
    ];

    let total_bytes: usize =
        gemv_sizes.iter().map(|(k, m)| k * m * 4 / 2).sum(); // Q5_K is 4 bits, so 0.5 bytes per param
    println!(
        "\nWeight bytes per layer: {:.2} MB",
        total_bytes as f64 / 1024.0 / 1024.0
    );

    // Estimate memory bandwidth
    // Apple M3 Pro has ~100 GB/s memory bandwidth for unified memory
    // But for typical usage, let's assume 20-40 GB/s effective
    let bandwidth_gbps = 30.0;
    let time_per_layer_ms =
        (total_bytes as f64 / 1e9) / bandwidth_gbps * 1000.0;

    println!(
        "Estimated time per layer: {:.2} ms (at {} GB/s)",
        time_per_layer_ms, bandwidth_gbps
    );

    let total_time_ms = time_per_layer_ms * n_layers as f64;
    println!("Estimated total GEMV time: {:.2} ms", total_time_ms);

    println!("\n=== Expected Performance ===");
    let tokens_per_second = 1000.0 / total_time_ms;
    println!("Expected t/s (GEMV only): {:.2} t/s", tokens_per_second);

    println!("\n=== Actual Performance ===");
    println!("Observed: 0.18 t/s (Qwen3.8-27B)");
    println!("Gap: {:.0}x slower than expected", tokens_per_second / 0.18);

    println!("\n=== Analysis ===");
    println!(
        "If actual is {:.0}x slower, possible causes:",
        tokens_per_second / 0.18
    );
    println!("1. CPU cache misses (weights not cached)");
    println!("2. Non-vectorized GEMV");
    println!("3. Too many small allocations fragmenting memory");
    println!("4. Attention computation overhead (not accounted above)");
}
