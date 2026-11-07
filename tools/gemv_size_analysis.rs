// Analyze GEMV sizes for Qwen3-0.6B and DeepSeek-R1-Distill-Qwen-1.5B
fn main() {
    // Qwen3-0.6B
    let hidden_06 = 1024;
    let intermediate_06 = 8960;
    let vocab_06 = 151936;
    let n_heads_06 = 16;
    let n_kv_heads_06 = 8;
    let head_dim_06 = 64;

    println!("=== Qwen3-0.6B GEMV sizes ===");
    println!(
        "QKV: {}x{} (q), {}x{} (k), {}x{} (v)",
        hidden_06,
        n_heads_06 * head_dim_06,
        hidden_06,
        n_kv_heads_06 * head_dim_06,
        hidden_06,
        n_kv_heads_06 * head_dim_06
    );
    println!("wo: {}x{}", n_kv_heads_06 * head_dim_06, hidden_06);
    println!("FFN gate+up: {}x{}", hidden_06, intermediate_06);
    println!("w2: {}x{}", intermediate_06, hidden_06);
    println!("output: {}x{}", hidden_06, vocab_06);

    // FLOPs estimation
    println!("\n=== FLOPs per GEMV (ops = 2*m*k) ===");
    let flops_qkv_06 = 3
        * 2
        * hidden_06
        * (n_heads_06 * head_dim_06 + 2 * n_kv_heads_06 * head_dim_06);
    let flops_wo_06 = 2 * (n_kv_heads_06 * head_dim_06) * hidden_06;
    let flops_ffn_06 = 2 * 2 * hidden_06 * intermediate_06;
    let flops_w2_06 = 2 * intermediate_06 * hidden_06;
    let flops_out_06 = 2 * hidden_06 * vocab_06;
    let total_06 =
        flops_qkv_06 + flops_wo_06 + flops_ffn_06 + flops_w2_06 + flops_out_06;

    println!("Qwen3-0.6B (28 layers):");
    println!("  QKV: {:.2} M ops", flops_qkv_06 as f64 / 1e6);
    println!("  wo: {:.2} M ops", flops_wo_06 as f64 / 1e6);
    println!("  FFN: {:.2} M ops", flops_ffn_06 as f64 / 1e6);
    println!("  w2: {:.2} M ops", flops_w2_06 as f64 / 1e6);
    println!("  output: {:.2} M ops", flops_out_06 as f64 / 1e6);
    println!("  Total per layer: {:.2} M ops", total_06 as f64 / 1e6);
    println!(
        "  Percentage: QKV {:.1}%, wo {:.1}%, FFN {:.1}%, w2 {:.1}%, out {:.1}%",
        100.0 * flops_qkv_06 as f64 / total_06 as f64,
        100.0 * flops_wo_06 as f64 / total_06 as f64,
        100.0 * flops_ffn_06 as f64 / total_06 as f64,
        100.0 * flops_w2_06 as f64 / total_06 as f64,
        100.0 * flops_out_06 as f64 / total_06 as f64
    );
}
