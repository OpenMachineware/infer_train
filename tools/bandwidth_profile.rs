// Bandwidth profiling benchmark for GPU decode
// Measures memory bandwidth utilization per kernel

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::model::tensor::DataType;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::profiling::{
    BandwidthProfiler, calc_flash_attn_bandwidth, calc_gemv_bandwidth,
    calc_kv_update_bandwidth,
};
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{ComputePipelineState, MTLSize};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== Bandwidth Profiling: {} ===\n", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    let config = &ctx.config;
    println!("Model config:");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  intermediate_size: {:?}", config.intermediate_size);
    println!("  n_heads: {}", config.n_heads);
    println!("  n_kv_heads: {:?}", config.n_kv_heads);
    println!("  head_dim: {}", config.head_dim);
    println!("  n_layers: {}", config.n_layers);

    // Calculate per-kernel bandwidth requirements
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;

    // Q4_K weight size (144 bytes per 256 elements)
    let q4_k_bytes_per_block = 144;
    let qk_k = 256;

    println!("\n=== Per-Kernel Bandwidth Analysis ===\n");

    // Q projection GEMV
    let q_out_size = n_heads * head_dim;
    let q_weight_bytes = (hidden / qk_k) * q4_k_bytes_per_block * q_out_size;
    let (q_read, q_write) =
        calc_gemv_bandwidth(q_out_size, hidden, q_weight_bytes);
    println!("Q projection (Q4_K GEMV):");
    println!(
        "  M={}, K={}, weight_bytes={}",
        q_out_size, hidden, q_weight_bytes
    );
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        q_read as f64 / 1e6,
        q_write as f64 / 1e6
    );

    // K projection GEMV
    let k_out_size = n_kv_heads * head_dim;
    let k_weight_bytes = (hidden / qk_k) * q4_k_bytes_per_block * k_out_size;
    let (k_read, k_write) =
        calc_gemv_bandwidth(k_out_size, hidden, k_weight_bytes);
    println!("\nK projection (Q4_K GEMV):");
    println!("  M={}, K={}", k_out_size, hidden);
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        k_read as f64 / 1e6,
        k_write as f64 / 1e6
    );

    // V projection GEMV
    let v_weight_bytes = (hidden / qk_k) * q4_k_bytes_per_block * k_out_size;
    let (v_read, v_write) =
        calc_gemv_bandwidth(k_out_size, hidden, v_weight_bytes);
    println!("\nV projection (Q4_K GEMV):");
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        v_read as f64 / 1e6,
        v_write as f64 / 1e6
    );

    // Gate projection GEMV (FFN)
    let gate_weight_bytes =
        (hidden / qk_k) * q4_k_bytes_per_block * intermediate;
    let (gate_read, gate_write) =
        calc_gemv_bandwidth(intermediate, hidden, gate_weight_bytes);
    println!("\nGate projection (Q4_K GEMV):");
    println!("  M={}, K={}", intermediate, hidden);
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        gate_read as f64 / 1e6,
        gate_write as f64 / 1e6
    );

    // Up projection GEMV (FFN)
    let (up_read, up_write) =
        calc_gemv_bandwidth(intermediate, hidden, gate_weight_bytes);
    println!("\nUp projection (Q4_K GEMV):");
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        up_read as f64 / 1e6,
        up_write as f64 / 1e6
    );

    // Down projection GEMV (FFN)
    let down_weight_bytes =
        (intermediate / qk_k) * q4_k_bytes_per_block * hidden;
    let (down_read, down_write) =
        calc_gemv_bandwidth(hidden, intermediate, down_weight_bytes);
    println!("\nDown projection (Q4_K GEMV):");
    println!("  M={}, K={}", hidden, intermediate);
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        down_read as f64 / 1e6,
        down_write as f64 / 1e6
    );

    // Flash Attention (decode, seq_len=10 for example)
    let seq_len = 10;
    let (attn_read, attn_write) =
        calc_flash_attn_bandwidth(seq_len, n_heads, n_kv_heads, head_dim);
    println!("\nFlash Attention (seq_len={}):", seq_len);
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        attn_read as f64 / 1e6,
        attn_write as f64 / 1e6
    );

    // KV cache update
    let (kv_read, kv_write) = calc_kv_update_bandwidth(head_dim, n_kv_heads);
    println!("\nKV cache update (K or V):");
    println!(
        "  Read: {:.2} MB, Write: {:.2} MB",
        kv_read as f64 / 1e6,
        kv_write as f64 / 1e6
    );

    // Total per token
    println!("\n=== Total Memory Access Per Token ===");
    let total_gates = (gate_read + gate_write) as f64;
    let total_attn = (q_read
        + q_write
        + k_read
        + k_write
        + v_read
        + v_write
        + attn_read
        + attn_write) as f64;
    let total_kv = (kv_read + kv_write) as f64 * 2.0; // K and V

    // Per layer
    let per_layer = total_gates * 2.0 + total_attn; // gate+up+down + Q+K+V+attn
    println!("Per layer: {:.2} MB", per_layer / 1e6);
    println!(
        "Total ({} layers): {:.2} MB",
        config.n_layers,
        per_layer * config.n_layers as f64 / 1e6
    );

    // If we achieve 120 t/s, what's our effective bandwidth?
    let tokens_per_sec = 120.0;
    let total_bytes_per_token = per_layer * config.n_layers as f64;
    let effective_bw = total_bytes_per_token * tokens_per_sec / 1e9;
    println!("\nAt {} t/s:", tokens_per_sec);
    println!("  Effective bandwidth: {:.2} GB/s", effective_bw);
    println!("  Utilization: {:.1}%", effective_bw / 100.0 * 100.0);

    // If llama.cpp achieves 184 t/s
    let llama_tokens_per_sec = 184.0;
    let llama_effective_bw = total_bytes_per_token * llama_tokens_per_sec / 1e9;
    println!("\nllama.cpp at {} t/s:", llama_tokens_per_sec);
    println!("  Effective bandwidth: {:.2} GB/s", llama_effective_bw);
    println!("  Utilization: {:.1}%", llama_effective_bw / 100.0 * 100.0);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!(
        "This benchmark requires Metal support (macOS with metal feature)"
    );
}
