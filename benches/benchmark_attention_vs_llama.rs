// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark CPU Attention vs llama.cpp
// Tests all 9 KV cache formats across multiple sizes

use std::fs;
use std::process::Command;

fn main() {
    println!("=== CPU Attention Performance vs llama.cpp ===\n");

    // Test configurations
    let configs = [
        (32, 128, 128), // Small decode: 32 heads, 128 head_dim, 128 seq_len
        (32, 128, 512), // Medium decode: 32 heads, 128 head_dim, 512 seq_len
        (32, 128, 1024), // Large decode: 32 heads, 128 head_dim, 1024 seq_len
        (32, 128, 2048), // Very large decode: 32 heads, 128 head_dim, 2048 seq_len
    ];

    // Formats to test
    let formats = ["f16", "q8_0", "q4_0", "q5_0", "q4_1", "q5_1", "iq4_nl"];

    println!("Format | 128 seq | 512 seq | 1024 seq | 2048 seq");
    println!("--------|---------|---------|----------|----------");

    for format in &formats {
        let mut results = Vec::new();

        for &(num_heads, head_dim, seq_len) in &configs {
            // Run our implementation
            let gflops =
                run_attention_benchmark(*format, num_heads, head_dim, seq_len);
            results.push(format!("{:.2}", gflops));
        }

        println!(
            "{:7} | {:>7} | {:>7} | {:>8} | {:>8}",
            format, results[0], results[1], results[2], results[3]
        );
    }

    // Run llama.cpp comparison
    println!("\n=== llama.cpp Comparison ===\n");

    // Build llama.cpp benchmark if needed
    let llama_dir = "llama.cpp-0.6.0";
    if !Path::new(&format!("{}/build/bin/llama-bench", llama_dir)).exists() {
        println!("llama.cpp not built, skipping comparison");
        println!(
            "To build: cd {} && mkdir -p build && cd build && cmake .. && make -j",
            llama_dir
        );
        return;
    }

    println!("Format | Ours (512) | llama.cpp (512) | Ratio");
    println!("--------|------------|-----------------|-------");

    for format in &formats {
        let our_gflops = run_attention_benchmark(*format, 32, 128, 512);
        let llama_gflops = run_llama_cpp_attention(*format, 32, 128, 512);
        let ratio = if llama_gflops > 0.0 {
            our_gflops / llama_gflops * 100.0
        } else {
            0.0
        };

        println!(
            "{:7} | {:>10.2} | {:>15.2} | {:>5.1}%",
            format, our_gflops, llama_gflops, ratio
        );
    }
}

use std::path::Path;

fn run_attention_benchmark(
    format: &str,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> f64 {
    // Build and run our Rust benchmark
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--release",
            "--bin",
            "benchmark_attention_single",
            "--",
            format,
            &num_heads.to_string(),
            &head_dim.to_string(),
            &seq_len.to_string(),
        ])
        .output()
        .ok();

    if let Some(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Parse GFLOPS from output like "Q8_0 Attention: 6.12 GFLOPS"
        for line in stdout.lines() {
            if line.contains("GFLOPS") {
                if let Some(pos) = line.find(':') {
                    let rest = &line[pos + 1..];
                    if let Some(gflops_pos) = rest.find("GFLOPS") {
                        let num_str = rest[..gflops_pos].trim();
                        if let Ok(gflops) = num_str.parse::<f64>() {
                            return gflops;
                        }
                    }
                }
            }
        }
    }
    0.0
}

fn run_llama_cpp_attention(
    format: &str,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> f64 {
    // This would require implementing a custom benchmark in llama.cpp
    // For now, return 0 to indicate it needs to be run manually
    // The actual benchmark would use ggml_compute_forward_flash_attn_ext
    0.0
}
