// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed operation-level profiling tool
//!
//! Usage:
//!   infer-train-op-timing -m <model.gguf> -n 10
//!
//! Measures time for each operation type during inference

use clap::Parser;
use infer_train::inference::{ForwardExecutor, SimpleModel, TimingStats};
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "op-timing")]
#[command(about = "Operation-level timing analysis")]
struct Cli {
    /// Model path
    #[arg(short, long)]
    model: String,

    /// Number of decode tokens
    #[arg(short, long, default_value = "20")]
    decode_tokens: usize,

    /// Number of warmup iterations
    #[arg(long, default_value = "2")]
    warmup: usize,

    /// Number of measurement iterations
    #[arg(long, default_value = "3")]
    iterations: usize,

    /// Thread count for decode (0 = auto)
    #[arg(long, default_value = "0")]
    threads: usize,
}

fn main() {
    let cli = Cli::parse();

    println!("=== Operation Timing Analysis ===");
    println!("Model: {}", cli.model);
    println!("Decode tokens: {}", cli.decode_tokens);
    println!("Iterations: {}", cli.iterations);
    println!();

    // Load model
    let gguf = GgufFile::load(&cli.model).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(&cli.model, tokenizer).expect("Failed to load model");
    let config = model.config().clone();

    println!(
        "Model: {} layers, hidden={}",
        config.n_layers, config.hidden_size
    );

    // Create executor with timing enabled
    let mut executor = ForwardExecutor::new_with_threads(model, cli.threads, 0);
    executor.enable_timing();

    // Warmup
    println!("\n=== Warmup ===");
    for i in 0..cli.warmup {
        println!("Warmup {}/{}...", i + 1, cli.warmup);
        executor.reset();
        // Single token warmup
        let _ = executor.forward(0);
    }

    // Profile decode
    println!("\n=== Decode Timing ===");
    let mut all_decode_times = Vec::new();
    let mut total_timing: Option<TimingStats> = None;

    for iter in 0..cli.iterations {
        executor.reset();
        executor.enable_timing(); // Reset timing for this iteration

        let start = Instant::now();
        for _ in 0..cli.decode_tokens {
            let _ = executor.forward(0);
        }
        let elapsed = start.elapsed();
        all_decode_times.push(elapsed.as_secs_f64() * 1000.0);

        // Aggregate timing
        if let Some(timing) = executor.timing() {
            if let Some(ref mut total) = total_timing {
                for (name, op) in &timing.ops {
                    let entry =
                        total.ops.entry(name.clone()).or_insert_with(|| {
                            infer_train::inference::OpTiming {
                                name: name.clone(),
                                ..Default::default()
                            }
                        });
                    entry.count += op.count;
                    entry.total_us += op.total_us;
                    entry.min_us = if entry.min_us == 0 {
                        op.min_us
                    } else {
                        entry.min_us.min(op.min_us)
                    };
                    entry.max_us = entry.max_us.max(op.max_us);
                }
            } else {
                total_timing = Some(timing.clone());
            }
        }

        println!(
            "Iteration {}: {:.1} ms ({} tokens, {:.1} t/s)",
            iter + 1,
            elapsed.as_secs_f64() * 1000.0,
            cli.decode_tokens,
            cli.decode_tokens as f64 / elapsed.as_secs_f64()
        );
    }

    // Print timing breakdown
    if let Some(ref timing) = total_timing {
        timing.print_summary();

        // Calculate bandwidth
        let hidden = config.hidden_size as f64;
        let intermediate = config.intermediate_size as f64;
        let n_layers = config.n_layers as f64;

        // Weight bytes per token (Q4_K ~0.5 bytes per element)
        let weight_bytes = (hidden * hidden * 4.0 * 0.5 // Q,K,V,O
            + hidden * intermediate * 3.0 * 0.5) // gate,up,down
            * n_layers;

        let avg_decode_ms = all_decode_times.iter().sum::<f64>()
            / (all_decode_times.len() * cli.decode_tokens) as f64;
        let bandwidth_gbps = weight_bytes / (avg_decode_ms / 1000.0) / 1e9;

        println!("\n=== Performance Summary ===");
        println!(
            "Decode: {:.1} t/s ({:.2} ms/token)",
            1000.0 / avg_decode_ms,
            avg_decode_ms
        );
        println!("Memory bandwidth: {:.1} GB/s", bandwidth_gbps);

        // GEMV analysis
        let gemv_ops = [
            "gemv_q",
            "gemv_k",
            "gemv_v",
            "gemv_o",
            "gemv_gate",
            "gemv_up",
            "gemv_down",
        ];
        let mut gemv_total_us: u64 = 0;
        let mut gemv_count: usize = 0;
        for op_name in &gemv_ops {
            if let Some(op) = timing.ops.get(*op_name) {
                gemv_total_us += op.total_us;
                gemv_count += op.count;
            }
        }
        let gemv_total_ms = gemv_total_us as f64 / 1000.0;
        // total_ms is the sum of all decode iteration times (already includes all iterations)
        let total_ms = all_decode_times.iter().sum::<f64>();
        let gemv_pct = gemv_total_ms / total_ms * 100.0;

        println!(
            "\nGEMV total: {:.1} ms ({:.1}% of total time)",
            gemv_total_ms, gemv_pct
        );
        println!(
            "GEMV calls: {} (per layer: {})",
            gemv_count,
            gemv_count / config.n_layers
        );

        // Non-GEMV overhead
        let non_gemv_ms = total_ms - gemv_total_ms;
        let non_gemv_pct = 100.0 - gemv_pct;
        println!(
            "Non-GEMV: {:.1} ms ({:.1}% of total time)",
            non_gemv_ms, non_gemv_pct
        );

        // Breakdown by category
        println!("\n=== Category Breakdown ===");
        let categories = [
            (
                "GEMV (attention)",
                &["gemv_q", "gemv_k", "gemv_v", "gemv_o"] as &[&str],
            ),
            ("GEMV (FFN)", &["gemv_gate", "gemv_up", "gemv_down"]),
            ("Quantization", &["quantize_attn", "quantize_ffn"]),
            ("Norm", &["rms_norm_attn", "rms_norm_ffn", "qk_norm"]),
            ("Attention", &["flash_attn", "rope", "kv_update"]),
            ("Activation", &["swiglu"]),
        ];

        for (cat_name, ops) in categories {
            let mut cat_us: u64 = 0;
            for op_name in ops {
                if let Some(op) = timing.ops.get(*op_name) {
                    cat_us += op.total_us;
                }
            }
            let cat_ms = cat_us as f64 / 1000.0;
            let cat_pct = cat_ms / total_ms * 100.0;
            println!("{}: {:.1} ms ({:.1}%)", cat_name, cat_ms, cat_pct);
        }
    }
}
