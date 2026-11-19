// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Fine-grained operator profiler with comparison to llama.cpp
//!
//! Strategy:
//! 1. Micro-benchmark individual operations (vec_dot, gemv, quantize)
//! 2. Compare with llama.cpp's implementation
//! 3. Identify exact bottleneck operations

use clap::Parser;
use std::process::Command;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "op_profile")]
#[command(about = "Fine-grained operator profiling with llama.cpp comparison")]
struct Cli {
    /// Model path
    #[arg(short, long)]
    model: String,

    /// Number of iterations for micro-benchmarks
    #[arg(long, default_value = "100")]
    bench_iters: usize,

    /// Run vec_dot benchmark
    #[arg(long)]
    vec_dot: bool,

    /// Run GEMV benchmark
    #[arg(long)]
    gemv: bool,

    /// Run quantization benchmark
    #[arg(long)]
    quant: bool,

    /// Run all benchmarks
    #[arg(long)]
    all: bool,

    /// Compare with llama.cpp
    #[arg(long)]
    compare: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.all || !(cli.vec_dot || cli.gemv || cli.quant) {
        // Run all benchmarks by default
        println!("=== Operator Micro-benchmarks ===\n");

        benchmark_quantization(cli.bench_iters);
        benchmark_vec_dot(cli.bench_iters);
        benchmark_gemv(cli.bench_iters);

        if cli.compare {
            println!("\n=== llama.cpp Reference ===\n");
            run_llama_cpp_benchmark(&cli.model);
        }
    } else {
        if cli.quant {
            benchmark_quantization(cli.bench_iters);
        }
        if cli.vec_dot {
            benchmark_vec_dot(cli.bench_iters);
        }
        if cli.gemv {
            benchmark_gemv(cli.bench_iters);
        }
    }
}

fn fp16_to_bits(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
}

fn benchmark_quantization(iters: usize) {
    use infer_train::backend::arm::quant::*;
    use infer_train::types::*;

    println!("--- Quantization ---\n");

    // Test different sizes
    for &n in &[512, 1024, 2048, 4096] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();

        // Q8_0
        let nb = (n + QK8_0 - 1) / QK8_0;
        let mut q8_0: Vec<BlockQ8_0> =
            vec![BlockQ8_0 { d: 0, qs: [0; 32] }; nb];

        // Warmup
        for _ in 0..10 {
            quantize_q8_0(&input, &mut q8_0);
        }

        let start = Instant::now();
        for _ in 0..iters {
            quantize_q8_0(&input, &mut q8_0);
        }
        let time = start.elapsed();
        let avg_us = time.as_micros() as f64 / iters as f64;
        let throughput_gbs = (n * 4) as f64 / avg_us / 1000.0; // GB/s

        println!(
            "Q8_0  {:5} elements: {:7.2} us ({:.2} GB/s)",
            n, avg_us, throughput_gbs
        );

        // Q8_K
        let nb = (n + QK_K - 1) / QK_K;
        let mut q8_k: Vec<BlockQ8_K> =
            vec![BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; nb];

        let start = Instant::now();
        for _ in 0..iters {
            quantize_q8_k(&input, &mut q8_k);
        }
        let time = start.elapsed();
        let avg_us = time.as_micros() as f64 / iters as f64;
        let throughput_gbs = (n * 4) as f64 / avg_us / 1000.0;

        println!(
            "Q8_K  {:5} elements: {:7.2} us ({:.2} GB/s)",
            n, avg_us, throughput_gbs
        );
        println!();
    }
}

fn benchmark_vec_dot(iters: usize) {
    use infer_train::backend::arm::vec_dot::*;
    use infer_train::types::*;

    println!("--- vec_dot (single block) ---\n");

    // Q4_0 vec_dot
    let x_q4_0 = BlockQ4_0 { d: fp16_to_bits(0.1), qs: [0x88; 16] };
    let y_q8_0 = BlockQ8_0 { d: fp16_to_bits(1.0), qs: [1; 32] };

    // Warmup
    for _ in 0..100 {
        let _ = vec_dot_q4_0_q8_0(&[x_q4_0], &[y_q8_0]);
    }

    let start = Instant::now();
    for _ in 0..iters * 10 {
        let _ = vec_dot_q4_0_q8_0(&[x_q4_0], &[y_q8_0]);
    }
    let time = start.elapsed();
    let avg_ns = time.as_nanos() as f64 / (iters * 10) as f64;

    println!("Q4_0 vec_dot (32 elements): {:.2} ns", avg_ns);

    // Q4_K vec_dot
    let x_q4_k = BlockQ4_K {
        d: fp16_to_bits(0.1),
        dmin: fp16_to_bits(0.0),
        scales: [0; 12],
        qs: [0x88; 128],
    };
    let y_q8_k = BlockQ8_K { d: 1.0f32, qs: [1; 256], bsums: [0; 16] };

    let start = Instant::now();
    for _ in 0..iters * 10 {
        let _ = vec_dot_q4_k_q8_k(&[x_q4_k], &[y_q8_k]);
    }
    let time = start.elapsed();
    let avg_ns = time.as_nanos() as f64 / (iters * 10) as f64;

    println!("Q4_K vec_dot (256 elements): {:.2} ns", avg_ns);

    // Q5_K vec_dot
    let x_q5_k = BlockQ5_K {
        d: fp16_to_bits(0.1),
        dmin: fp16_to_bits(0.0),
        scales: [0; 12],
        qh: [0; 32],
        qs: [0x88; 128],
    };

    let start = Instant::now();
    for _ in 0..iters * 10 {
        let _ = vec_dot_q5_k_q8_k(&[x_q5_k], &[y_q8_k]);
    }
    let time = start.elapsed();
    let avg_ns = time.as_nanos() as f64 / (iters * 10) as f64;

    println!("Q5_K vec_dot (256 elements): {:.2} ns", avg_ns);

    // Q6_K vec_dot
    let x_q6_k = BlockQ6_K {
        d: fp16_to_bits(0.1),
        ql: [0; 128],
        qh: [0; 64],
        scales: [0; 16],
    };

    let start = Instant::now();
    for _ in 0..iters * 10 {
        let _ = vec_dot_q6_k_q8_k(&[x_q6_k], &[y_q8_k]);
    }
    let time = start.elapsed();
    let avg_ns = time.as_nanos() as f64 / (iters * 10) as f64;

    println!("Q6_K vec_dot (256 elements): {:.2} ns", avg_ns);

    // Calculate throughput
    println!("\n--- vec_dot Throughput ---");
    let elements = 256;
    // Q4_K: 256 elements in avg_ns
    let start = Instant::now();
    for _ in 0..iters * 10 {
        let _ = vec_dot_q4_k_q8_k(&[x_q4_k], &[y_q8_k]);
    }
    let time = start.elapsed();
    let avg_ns = time.as_nanos() as f64 / (iters * 10) as f64;
    let gops = elements as f64 / avg_ns * 1000.0; // Giga ops/sec

    println!("Q4_K: {:.2} GOPS ({:.2} ns for 256 mul-adds)", gops, avg_ns);
    println!();
}

fn benchmark_gemv(iters: usize) {
    use infer_train::backend::arm::gemv::*;
    use infer_train::types::*;

    println!("--- GEMV (Matrix-Vector Multiplication) ---\n");

    // Typical sizes for Qwen3-0.6B
    let test_cases = [
        (1024, 2048, "Q projection"), // hidden_size -> n_heads * head_dim
        (1024, 1024, "K/V projection"), // hidden_size -> n_kv_heads * head_dim
        (3072, 1024, "FFN Gate/Up"),  // intermediate_size -> hidden_size
        (1024, 3072, "FFN Down"),     // hidden_size -> intermediate_size
    ];

    for (in_features, out_features, name) in test_cases {
        // Q4_K GEMV
        let nb = (in_features + QK_K - 1) / QK_K;
        let weights: Vec<BlockQ4_K> = (0..out_features * nb)
            .map(|i| BlockQ4_K {
                d: fp16_to_bits(0.1 + (i % 10) as f32 * 0.01),
                dmin: fp16_to_bits(0.0),
                scales: [((i % 16) + 1) as u8; 12],
                qs: [((i % 256) + 1) as u8; 128],
            })
            .collect();
        let input: Vec<BlockQ8_K> = (0..nb)
            .map(|i| BlockQ8_K {
                d: 1.0 + (i % 10) as f32 * 0.1,
                qs: [((i % 128) + 1) as i8; 256],
                bsums: [0; 16],
            })
            .collect();
        let mut output = vec![0.0f32; out_features];

        // Warmup
        for _ in 0..5 {
            gemv_q4_k(
                &weights,
                &input,
                &mut output,
                out_features,
                in_features,
                0,
            );
        }

        let start = Instant::now();
        for _ in 0..iters {
            gemv_q4_k(
                &weights,
                &input,
                &mut output,
                out_features,
                in_features,
                0,
            );
        }
        let time = start.elapsed();
        let avg_us = time.as_micros() as f64 / iters as f64;

        // Calculate FLOPs: 2 * M * N (multiply-add)
        let flops = 2 * out_features * in_features;
        let gflops = flops as f64 / avg_us / 1000.0;

        println!(
            "Q4_K GEMV {} ({}x{}): {:7.2} us ({:.2} GFLOPS)",
            name, out_features, in_features, avg_us, gflops
        );

        // Memory bandwidth: weights + input + output
        let weight_bytes = weights.len() * std::mem::size_of::<BlockQ4_K>();
        let input_bytes = input.len() * std::mem::size_of::<BlockQ8_K>();
        let output_bytes = output.len() * 4;
        let total_bytes = weight_bytes + input_bytes + output_bytes;
        let bandwidth_gbs = total_bytes as f64 / avg_us / 1000.0;

        println!(
            "    Memory: {:.2} MB, Bandwidth: {:.2} GB/s",
            total_bytes as f64 / 1e6,
            bandwidth_gbs
        );
    }

    println!();

    // Multi-threaded scaling test
    println!("--- GEMV Threading Scaling ---\n");
    let in_features = 2048;
    let out_features = 1024;
    let nb = (in_features + QK_K - 1) / QK_K;

    let weights: Vec<BlockQ4_K> = vec![
        BlockQ4_K {
            d: fp16_to_bits(0.1),
            dmin: fp16_to_bits(0.0),
            scales: [0; 12],
            qs: [0x88; 128],
        };
        out_features * nb
    ];
    let input: Vec<BlockQ8_K> =
        vec![BlockQ8_K { d: 1.0, qs: [1; 256], bsums: [0; 16] }; nb];
    let mut output = vec![0.0f32; out_features];

    for &n_threads in &[1, 2, 4, 8] {
        let start = Instant::now();
        for _ in 0..iters {
            gemv_q4_k(
                &weights,
                &input,
                &mut output,
                out_features,
                in_features,
                n_threads,
            );
        }
        let time = start.elapsed();
        let avg_us = time.as_micros() as f64 / iters as f64;
        println!("{} threads: {:7.2} us", n_threads, avg_us);
    }
}

fn run_llama_cpp_benchmark(model_path: &str) {
    // Run llama-cli with verbose output to extract per-layer timing
    let output = Command::new("llama-cli")
        .arg("-m")
        .arg(model_path)
        .arg("-p")
        .arg("Hello")
        .arg("-n")
        .arg("1")
        .arg("--temp")
        .arg("0")
        .arg("--verbose")
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            println!("llama.cpp output:\n{}\n{}", stdout, stderr);
        }
        Err(e) => {
            println!("Failed to run llama-cli: {}", e);
        }
    }
}
