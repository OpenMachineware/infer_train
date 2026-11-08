#!/usr/bin/env cargo
//! Analyze attention FLOPs and bandwidth utilization

fn main() {
    // Qwen3-0.6B config
    let n_heads = 16;
    let n_kv_heads = 8; // GQA
    let head_dim = 128;
    let n_layers = 28;

    // Test scenarios
    for &seq_len in &[16, 32, 64] {
        println!("\n=== Attention Analysis: {} tokens ===", seq_len);

        // Per layer attention FLOPs
        // Q·K: n_heads × seq_len × seq_len × head_dim × 2 (mult+add)
        let qk_flops = n_heads * seq_len * seq_len * head_dim * 2;

        // Softmax: n_heads × seq_len × (exp + div) × seq_len
        let softmax_flops = n_heads * seq_len * seq_len * 4; // exp, sub, add, div

        // Weighted V: n_heads × seq_len × seq_len × head_dim × 2
        let wv_flops = n_heads * seq_len * seq_len * head_dim * 2;

        // Total per layer
        let total_per_layer = qk_flops + softmax_flops + wv_flops;
        let total_all_layers = total_per_layer * n_layers;

        println!("Per layer attention FLOPs:");
        println!("  Q·K dot:      {:>8.2} M", qk_flops as f64 / 1e6);
        println!("  Softmax:      {:>8.2} M", softmax_flops as f64 / 1e6);
        println!("  Weighted V:   {:>8.2} M", wv_flops as f64 / 1e6);
        println!("  Total:        {:>8.2} M", total_per_layer as f64 / 1e6);
        println!("  All layers:   {:>8.2} M", total_all_layers as f64 / 1e6);

        // Memory bandwidth
        // Q: seq_len × n_heads × head_dim × 4 bytes
        // K: seq_len × n_kv_heads × head_dim × 4 bytes
        // V: seq_len × n_kv_heads × head_dim × 4 bytes
        let q_bytes = seq_len * n_heads * head_dim * 4;
        let k_bytes = seq_len * n_kv_heads * head_dim * 4;
        let v_bytes = seq_len * n_kv_heads * head_dim * 4;

        // For each Q token, need to read all K and V
        let total_k_access = seq_len * seq_len * n_kv_heads * head_dim * 4;
        let total_v_access = seq_len * seq_len * n_kv_heads * head_dim * 4;

        let total_memory = (q_bytes + k_bytes + v_bytes + total_k_access + total_v_access) * n_layers;

        println!("Memory accesses:");
        println!("  Q size:       {:>8.2} KB", q_bytes as f64 / 1024.0);
        println!("  K size:       {:>8.2} KB", k_bytes as f64 / 1024.0);
        println!("  V size:       {:>8.2} KB", v_bytes as f64 / 1024.0);
        println!("  Total memory: {:>8.2} MB", total_memory as f64 / (1024.0 * 1024.0));

        // From benchmark: attention time for 64 tokens = 617ms
        let attention_ms = match seq_len {
            16 => 82.0,
            32 => 210.0,
            64 => 617.0,
            _ => 0.0,
        };

        println!("Achieved performance:");
        let gflops = total_all_layers as f64 / (attention_ms * 1e-3) / 1e9;
        let gbps = total_memory as f64 / (attention_ms * 1e-3) / 1e9;
        println!("  Compute:      {:>8.2} GFLOPS", gflops);
        println!("  Bandwidth:    {:>8.2} GB/s", gbps);

        // Theoretical peak (M4)
        let peak_flops = 2000.0; // GFLOPS
        let peak_bw = 100.0; // GB/s (estimated for L2 cache)

        println!("Efficiency:");
        println!("  Compute:      {:>8.1} %", gflops / peak_flops * 100.0);
        println!("  Bandwidth:    {:>8.1} %", gbps / peak_bw * 100.0);
    }
}
