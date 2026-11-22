// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Operation-level profiling for prefill

use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "op_profile")]
struct Cli {
    /// Model path
    #[arg(short, long)]
    model: String,

    /// Number of tokens for prefill
    #[arg(long, default_value = "10")]
    tokens: usize,

    /// Number of iterations
    #[arg(long, default_value = "3")]
    iterations: usize,
}

fn main() {
    let cli = Cli::parse();

    use infer_train::inference::ForwardExecutor;
    use infer_train::inference::SimpleModel;
    use infer_train::model::GgufFile;
    use infer_train::tokenizer::create_tokenizer;

    let gguf = GgufFile::load(&cli.model).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(&cli.model, tokenizer).expect("Failed to load model");
    let config = model.config().clone();

    println!("=== Operation Profile ===");
    println!(
        "Model: {} layers, hidden={}",
        config.n_layers, config.hidden_size
    );
    println!("Tokens: {}", cli.tokens);
    println!();

    // Create token sequence
    let tokens: Vec<u32> = (0..cli.tokens as u32).collect();

    let mut executor = ForwardExecutor::new(model);

    // Warmup
    let _ = executor.forward_batch(&tokens);

    // Profile each iteration
    let mut total_times = Vec::new();

    for _iter in 0..cli.iterations {
        let gguf = GgufFile::load(&cli.model).expect("Failed to load model");
        let tokenizer = create_tokenizer(
            gguf.tokenizer_metadata().expect("Failed to get metadata"),
        )
        .expect("Failed to create tokenizer");
        let model = SimpleModel::load(&cli.model, tokenizer)
            .expect("Failed to load model");
        let mut exec = ForwardExecutor::new(model);

        let start = Instant::now();
        let _ = exec.forward_batch(&tokens);
        let elapsed = start.elapsed();
        total_times.push(elapsed.as_millis() as f64);
    }

    let avg_total: f64 =
        total_times.iter().sum::<f64>() / cli.iterations as f64;
    println!(
        "Total prefill: {:.2} ms ({:.1} t/s)",
        avg_total,
        cli.tokens as f64 / (avg_total / 1000.0)
    );
    println!();

    // Profile individual operations by instrumenting forward
    println!("=== Detailed Timing ===");

    // Run with timing inside forward.rs requires code changes
    // For now, break down by operation counts

    let n_layers = config.n_layers;
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let n_heads = config.n_heads;
    let head_dim = config.head_dim();

    println!("\nModel structure:");
    println!("  Layers: {}", n_layers);
    println!("  Hidden: {}", hidden);
    println!("  Intermediate: {}", intermediate);
    println!("  Heads: {}, head_dim: {}", n_heads, head_dim);

    println!("\nPer-layer operations:");
    println!("  GEMV: q,k,v,o (4x{}x{})", hidden, hidden);
    println!("  GEMV: gate,up,down (3x{}x{})", hidden, intermediate);
    println!(
        "  Attention: {} heads, {} queries, max {} KV",
        n_heads, cli.tokens, cli.tokens
    );

    // Estimate FLOPs
    let gemv_flops =
        (hidden * hidden * 4 + hidden * intermediate * 3) * 2 * n_layers;
    let attn_flops =
        cli.tokens * n_heads * cli.tokens * head_dim * 2 * n_layers;
    let total_flops = gemv_flops + attn_flops;

    println!("\nFLOP estimate:");
    println!("  GEMV: {:.2e}", gemv_flops as f64);
    println!("  Attention: {:.2e}", attn_flops as f64);
    println!("  Total: {:.2e}", total_flops as f64);
    println!(
        "  Compute: {:.1} GFLOPS",
        total_flops as f64 / (avg_total / 1000.0) / 1e9
    );

    // Estimate memory bandwidth
    let weight_bytes = 400_000_000.0; // ~400MB for 0.6B Q4
    let act_bytes =
        cli.tokens * (hidden * 10 + intermediate * 2) * 4 * n_layers;
    let total_bytes = weight_bytes + act_bytes as f64;

    println!("\nMemory estimate:");
    println!("  Weights: {:.0} MB", weight_bytes / 1e6);
    println!("  Activations: {:.0} MB", act_bytes as f64 / 1e6);
    println!(
        "  Bandwidth: {:.1} GB/s",
        total_bytes / (avg_total / 1000.0) / 1e9
    );
}
