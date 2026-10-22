// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Kernel-level timing analysis tool
//!
//! Usage:
//!   INFER_TIMING=1 cargo run --release --features metal --bin kernel-timing -- <model.gguf>
//!
//! This will output timing breakdown by operation type

use infer_train::{
    Device, InferenceContext,
    model::{LlamaModel, Tensor},
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);
    println!();

    // Load model
    let model = LlamaModel::load(model_path, Device::Gpu)
        .expect("Failed to load model");

    // Get model info
    println!("=== Model Info ===");
    println!("Hidden size: {}", model.hidden_size());
    println!("Layers: {}", model.n_layers());
    println!("Heads: {}", model.n_heads());
    println!("KV heads: {}", model.n_kv_heads());
    println!("Head dim: {}", model.head_dim());
    println!();

    // Create inference context
    let mut ctx = InferenceContext::new(model.clone());

    // Run a few tokens to warm up
    println!("=== Warmup ===");
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    for &token in &tokens {
        let start = Instant::now();
        let _ = ctx.forward(token);
        println!("Token {}: {:?}", token, start.elapsed());
    }
    println!();

    // Detailed timing
    println!("=== Kernel Timing Analysis ===");
    println!();

    // Count operations per layer
    println!("Operations per layer:");
    println!("  - GEMV: 6 (Q, K, V, Gate, Up, Down)");
    println!("  - RoPE: 2 (Q, K)");
    println!("  - KVCacheUpdate: 2 (K, V)");
    println!("  - RmsNorm: 2 (attn_norm, ffn_norm)");
    println!("  - FlashAttention: 1");
    println!("  - Add: 2 (residual connections)");
    println!("  - FusedSwiGLU: 1");
    println!("  Total: 16 operations/layer");
    println!();

    // Estimate timing based on our measurements
    let total_ops = model.n_layers() * 16 + 3; // +3 for embedding, output norm, output proj
    println!("Total dispatches per token: {}", total_ops);

    // Measured timing
    let measured_time_ms = 11.6; // From benchmark
    let time_per_dispatch_us = (measured_time_ms * 1000.0) / total_ops as f64;

    println!("Measured time per token: {:.2} ms", measured_time_ms);
    println!("Time per dispatch: {:.2} µs", time_per_dispatch_us);
    println!();

    // Breakdown by operation type (estimated from profiling)
    println!("=== Estimated Time Breakdown ===");
    println!();

    let gemv_ops = model.n_layers() * 6; // Q, K, V, Gate, Up, Down
    let rope_ops = model.n_layers() * 2;
    let kv_ops = model.n_layers() * 2;
    let norm_ops = model.n_layers() * 2;
    let flash_ops = model.n_layers();
    let add_ops = model.n_layers() * 2;
    let silu_ops = model.n_layers();

    // Estimated GPU time per operation (µs)
    // These are rough estimates - need actual measurement
    let gemv_time = 30.0; // GEMV is the heaviest
    let rope_time = 5.0;
    let kv_time = 5.0;
    let norm_time = 5.0;
    let flash_time = 25.0;
    let add_time = 5.0;
    let silu_time = 5.0;

    let total_gemv = gemv_ops as f64 * gemv_time;
    let total_rope = rope_ops as f64 * rope_time;
    let total_kv = kv_ops as f64 * kv_time;
    let total_norm = norm_ops as f64 * norm_time;
    let total_flash = flash_ops as f64 * flash_time;
    let total_add = add_ops as f64 * add_time;
    let total_silu = silu_ops as f64 * silu_time;

    let grand_total = total_gemv
        + total_rope
        + total_kv
        + total_norm
        + total_flash
        + total_add
        + total_silu;

    println!(
        "{:<20} {:>6} ops × {:>6.1} µs = {:>8.1} µs ({:>5.1}%)",
        "GEMV",
        gemv_ops,
        gemv_time,
        total_gemv,
        total_gemv / grand_total * 100.0
    );
    println!(
        "{:<20} {:>6} ops × {:>6.1} µs = {:>8.1} µs ({:>5.1}%)",
        "FlashAttention",
        flash_ops,
        flash_time,
        total_flash,
        total_flash / grand_total * 100.0
    );
    println!(
        "{:<20} {:>6} ops × {:>6.1} µs = {:>8.1} µs ({:>5.1}%)",
        "RoPE",
        rope_ops,
        rope_time,
        total_rope,
        total_rope / grand_total * 100.0
    );
    println!(
        "{:<20} {:>6} ops × {:>6.1} µs = {:>8.1} µs ({:>5.1}%)",
        "KVCacheUpdate",
        kv_ops,
        kv_time,
        total_kv,
        total_kv / grand_total * 100.0
    );
    println!(
        "{:<20} {:>6} ops × {:>6.1} µs = {:>8.1} µs ({:>5.1}%)",
        "RmsNorm",
        norm_ops,
        norm_time,
        total_norm,
        total_norm / grand_total * 100.0
    );
    println!(
        "{:<20} {:>6} ops × {:>6.1} µs = {:>8.1} µs ({:>5.1}%)",
        "Add",
        add_ops,
        add_time,
        total_add,
        total_add / grand_total * 100.0
    );
    println!(
        "{:<20} {:>6} ops × {:>6.1} µs = {:>8.1} µs ({:>5.1}%)",
        "FusedSwiGLU",
        silu_ops,
        silu_time,
        total_silu,
        total_silu / grand_total * 100.0
    );
    println!();
    println!(
        "Total estimated: {:.1} µs ({:.2} ms)",
        grand_total,
        grand_total / 1000.0
    );
    println!("Measured: {:.2} ms", measured_time_ms);
    println!(
        "Gap: {:.2} ms (dispatch overhead + other)",
        measured_time_ms - grand_total / 1000.0
    );
    println!();

    // llama.cpp comparison
    println!("=== llama.cpp Comparison ===");
    println!();
    let llama_time_ms = 4.17; // 239.59 t/s
    let our_time_ms = 11.6;
    let gap = our_time_ms / llama_time_ms;

    println!("llama.cpp: {:.2} ms/token (239.59 t/s)", llama_time_ms);
    println!("Ours: {:.2} ms/token (86 t/s)", our_time_ms);
    println!("Gap: {:.2}x slower", gap);
    println!();

    // Possible causes
    println!("=== Possible Causes of Gap ===");
    println!();
    println!("1. GEMV kernel efficiency");
    println!("   - Check if kernel matches llama.cpp implementation");
    println!("   - Verify memory access patterns");
    println!("   - Check dispatch configuration (NSG, threadgroup size)");
    println!();
    println!("2. Dispatch overhead");
    println!(
        "   - {:.2} ms gap could be dispatch overhead",
        our_time_ms - grand_total / 1000.0
    );
    println!("   - llama.cpp uses Residency Sets to reduce overhead");
    println!("   - Consider batch encoding");
    println!();
    println!("3. Memory management");
    println!("   - llama.cpp uses shared buffers (zero-copy)");
    println!("   - We may have extra memory copies");
    println!();
    println!("4. Metal pipeline compilation");
    println!("   - First run may have JIT compilation overhead");
    println!("   - llama.cpp pre-compiles all pipelines");
}
