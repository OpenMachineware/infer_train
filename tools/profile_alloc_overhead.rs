// Profile memory allocation overhead in CPU forward pass

fn main() {
    println!("=== Memory Allocation Estimation ===\n");

    // Estimate allocations per forward pass
    let hidden_size = 4096;
    let intermediate_size = 11008;
    let n_heads = 32;
    let n_kv_heads = 8;
    let head_dim = 128;
    let n_layers = 32;

    // Per layer allocations:
    // 1. RMS norm: Vec<f32> for normed (hidden_size)
    // 2. Q projection: Vec<f32> for q (n_heads * head_dim)
    // 3. K projection: Vec<f32> for k (n_kv_heads * head_dim)
    // 4. V projection: Vec<f32> for v (n_kv_heads * head_dim)
    // 5. RoPE Q: Vec<f32> for q_rope
    // 6. RoPE K: Vec<f32> for k_rope
    // 7. Attention output: Vec<f32> for attn_out
    // 8. Wo projection: Vec<f32> for attn_proj (hidden_size)
    // 9. Residual: Vec<f32> for residual (hidden_size)
    // 10. FFN norm: Vec<f32> for ffn_normed (hidden_size)
    // 11. Gate projection: Vec<f32> for gate (intermediate_size)
    // 12. Up projection: Vec<f32> for up (intermediate_size)
    // 13. SiLU * gate: Vec<f32> for fused (intermediate_size)
    // 14. Down projection: Vec<f32> for down (hidden_size)
    // 15. Final residual: Vec<f32> for output (hidden_size)

    let per_layer_bytes = hidden_size * 4 +  // normed
        n_heads * head_dim * 4 +  // q
        n_kv_heads * head_dim * 4 +  // k
        n_kv_heads * head_dim * 4 +  // v
        n_heads * head_dim * 4 +  // q_rope
        n_kv_heads * head_dim * 4 +  // k_rope
        n_heads * head_dim * 4 +  // attn_out
        hidden_size * 4 +  // attn_proj (wo)
        hidden_size * 4 +  // residual
        hidden_size * 4 +  // ffn_normed
        intermediate_size * 4 +  // gate
        intermediate_size * 4 +  // up
        intermediate_size * 4 +  // fused
        hidden_size * 4 +  // down
        hidden_size * 4; // final output

    let per_layer_allocs = 15;

    // Total for forward pass
    let total_layer_bytes = per_layer_bytes * n_layers;
    let total_layer_allocs = per_layer_allocs * n_layers;

    // Additional allocations:
    // - Embedding lookup: Vec<f32> for embedding (hidden_size)
    // - Final norm: Vec<f32> (hidden_size)
    // - Output projection: Vec<f32> for logits (vocab_size)

    let vocab_size = 32000; // Typical
    let additional_bytes = hidden_size * 4 + hidden_size * 4 + vocab_size * 4;
    let additional_allocs = 3;

    let total_bytes = total_layer_bytes + additional_bytes;
    let total_allocs = total_layer_allocs + additional_allocs;

    println!("Per layer:");
    println!("  Allocations: {}", per_layer_allocs);
    println!(
        "  Bytes: {} ({:.2} MB)",
        per_layer_bytes,
        per_layer_bytes as f64 / 1024.0 / 1024.0
    );

    println!("\nPer forward pass (decode, 1 token):");
    println!("  Allocations: {}", total_allocs);
    println!(
        "  Bytes: {} ({:.2} MB)",
        total_bytes,
        total_bytes as f64 / 1024.0 / 1024.0
    );

    println!("\nFor 100 tokens:");
    println!("  Allocations: {}", total_allocs * 100);
    println!(
        "  Bytes: {} ({:.2} GB)",
        total_bytes * 100,
        total_bytes as f64 * 100.0 / 1024.0 / 1024.0 / 1024.0
    );

    // Estimate time cost
    // Typical malloc time: ~100-500 ns for small allocations
    // For large allocations (>4KB): ~1-10 µs
    let avg_malloc_time_ns: f64 = 500.0; // Conservative estimate
    let total_malloc_time_us =
        (total_allocs as f64) * avg_malloc_time_ns / 1000.0;

    println!("\nEstimated malloc overhead per token:");
    println!(
        "  Time: {:.2} µs (assuming {} ns per malloc)",
        total_malloc_time_us, avg_malloc_time_ns as u64
    );

    // Compare with thread overhead
    println!("\n=== Comparison ===");
    println!("Thread creation overhead: ~35 ms per forward");
    println!(
        "Memory allocation overhead: {:.2} µs per forward",
        total_malloc_time_us
    );
    println!("Thread overhead dominates!");
}
