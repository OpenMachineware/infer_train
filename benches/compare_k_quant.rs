// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Comprehensive K-quant vec_dot benchmark
// Tests Q2_K, Q3_K, Q4_K, Q5_K, Q6_K with K=4096

use half::f16;
use infer_train::quant::types::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K,
};
use infer_train::quant::vec_dot::arm::{
    vec_dot_q2_k_q8_k_neon, vec_dot_q3_k_q8_k_neon, vec_dot_q4_k_q8_k_neon,
    vec_dot_q5_k_q8_k_neon, vec_dot_q6_k_q8_k_neon,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// CPU warmup to stabilize M1 frequency
/// More aggressive warmup: 100M iterations + 1 second sleep
fn warmup_cpu() {
    let mut sum = 0.0f64;
    for i in 0..100_000_000 {
        sum = sum.sqrt() + i as f64;
    }
    std::thread::sleep(Duration::from_millis(1000));
    let _ = black_box(sum);
}

/// Generate test blocks for Q2_K
fn generate_q2k_blocks(nb: usize) -> (Vec<BlockQ2K>, Vec<BlockQ8K>) {
    let x: Vec<BlockQ2K> = (0..nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 16];
            for j in 0..64 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            for j in 0..16 {
                scales[j] = ((j % 4) + 1) as u8;
            }
            BlockQ2K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect();

    let y: Vec<BlockQ8K> = (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) as i8) - 32;
            }
            for j in 0..16 {
                bsums[j] = ((j + i) % 100) as i16;
            }
            BlockQ8K { d: 1.0 + (i % 10) as f32 * 0.1, qs, bsums }
        })
        .collect();

    (x, y)
}

/// Generate test blocks for Q3_K
fn generate_q3k_blocks(nb: usize) -> (Vec<BlockQ3K>, Vec<BlockQ8K>) {
    let x: Vec<BlockQ3K> = (0..nb)
        .map(|i| {
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 12];
            for j in 0..32 {
                hmask[j] = ((i * j) & 0xFF) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            for j in 0..12 {
                scales[j] = ((j % 4) + 1) as u8;
            }
            BlockQ3K { d: f16::from_f32(1.0).to_bits(), hmask, qs, scales }
        })
        .collect();

    let y: Vec<BlockQ8K> = (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) as i8) - 32;
            }
            for j in 0..16 {
                bsums[j] = ((j + i) % 100) as i16;
            }
            BlockQ8K { d: 1.0 + (i % 10) as f32 * 0.1, qs, bsums }
        })
        .collect();

    (x, y)
}

/// Generate test blocks for Q4_K
fn generate_q4k_blocks(nb: usize) -> (Vec<BlockQ4K>, Vec<BlockQ8K>) {
    let x: Vec<BlockQ4K> = (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((j % 4) + 1) as u8;
            }
            for j in 0..128 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect();

    let y: Vec<BlockQ8K> = (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) as i8) - 32;
            }
            for j in 0..16 {
                bsums[j] = ((j + i) % 100) as i16;
            }
            BlockQ8K { d: 1.0 + (i % 10) as f32 * 0.1, qs, bsums }
        })
        .collect();

    (x, y)
}

/// Generate test blocks for Q5_K
fn generate_q5k_blocks(nb: usize) -> (Vec<BlockQ5K>, Vec<BlockQ8K>) {
    let x: Vec<BlockQ5K> = (0..nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            for j in 0..32 {
                qh[j] = ((i * j) & 0xFF) as u8;
            }
            for j in 0..12 {
                scales[j] = ((j % 4) + 1) as u8;
            }
            BlockQ5K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                qs,
                qh,
                scales,
            }
        })
        .collect();

    let y: Vec<BlockQ8K> = (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) as i8) - 32;
            }
            for j in 0..16 {
                bsums[j] = ((j + i) % 100) as i16;
            }
            BlockQ8K { d: 1.0 + (i % 10) as f32 * 0.1, qs, bsums }
        })
        .collect();

    (x, y)
}

/// Generate test blocks for Q6_K
fn generate_q6k_blocks(nb: usize) -> (Vec<BlockQ6K>, Vec<BlockQ8K>) {
    let x: Vec<BlockQ6K> = (0..nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = ((i + j) & 0xFF) as u8;
            }
            for j in 0..64 {
                qh[j] = ((i * j) & 0xFF) as u8;
            }
            for j in 0..16 {
                scales[j] = ((j % 4) + 1) as i8;
            }
            BlockQ6K { d: f16::from_f32(1.0).to_bits(), ql, qh, scales }
        })
        .collect();

    let y: Vec<BlockQ8K> = (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) as i8) - 32;
            }
            for j in 0..16 {
                bsums[j] = ((j + i) % 100) as i16;
            }
            BlockQ8K { d: 1.0 + (i % 10) as f32 * 0.1, qs, bsums }
        })
        .collect();

    (x, y)
}

fn main() {
    unsafe {
        println!("K-quant vec_dot Performance Benchmark");
        println!("=====================================");
        println!("Testing all K-quant formats with K=4096\n");

        warmup_cpu();

        let nb = 16; // 16 blocks = 4096 elements
        let n = nb * QK_K;
        let iterations = 50000; // Increased from 10000 for more stable measurements

        // Reference llama.cpp performance (GFLOPS, best runs from project memory)
        let ref_perf = [
            ("Q2_K", 58.4),
            ("Q3_K", 41.8),
            ("Q4_K", 79.2),
            ("Q5_K", 51.5),
            ("Q6_K", 47.1),
        ];

        println!("Format | Rust GFLOPS | llama.cpp GFLOPS | Ratio | Status");
        println!("-------|-------------|------------------|-------|--------");

        // Q2_K
        {
            let (x, y) = generate_q2k_blocks(nb);
            let mut result = 0.0f32;

            let start = Instant::now();
            for _ in 0..iterations {
                result = black_box(vec_dot_q2_k_q8_k_neon(n, &x, &y));
            }
            let elapsed = start.elapsed();

            let time_ns = elapsed.as_nanos() as f64;
            let flops = 2.0 * (n as f64) * (iterations as f64);
            let rust_gflops = flops / (time_ns / 1e9) / 1e9;
            let ratio = (rust_gflops / ref_perf[0].1) * 100.0;
            let status = if ratio >= 100.0 { "✓" } else { "✗" };

            println!(
                "Q2_K   | {:11.1} | {:16.1} | {:5.0}% | {}",
                rust_gflops, ref_perf[0].1, ratio, status
            );
        }

        // Q3_K
        {
            let (x, y) = generate_q3k_blocks(nb);
            let mut result = 0.0f32;

            let start = Instant::now();
            for _ in 0..iterations {
                result = black_box(vec_dot_q3_k_q8_k_neon(n, &x, &y));
            }
            let elapsed = start.elapsed();

            let time_ns = elapsed.as_nanos() as f64;
            let flops = 2.0 * (n as f64) * (iterations as f64);
            let rust_gflops = flops / (time_ns / 1e9) / 1e9;
            let ratio = (rust_gflops / ref_perf[1].1) * 100.0;
            let status = if ratio >= 100.0 { "✓" } else { "✗" };

            println!(
                "Q3_K   | {:11.1} | {:16.1} | {:5.0}% | {}",
                rust_gflops, ref_perf[1].1, ratio, status
            );
        }

        // Q4_K
        {
            let (x, y) = generate_q4k_blocks(nb);
            let mut result = 0.0f32;

            let start = Instant::now();
            for _ in 0..iterations {
                result = black_box(vec_dot_q4_k_q8_k_neon(n, &x, &y));
            }
            let elapsed = start.elapsed();

            let time_ns = elapsed.as_nanos() as f64;
            let flops = 2.0 * (n as f64) * (iterations as f64);
            let rust_gflops = flops / (time_ns / 1e9) / 1e9;
            let ratio = (rust_gflops / ref_perf[2].1) * 100.0;
            let status = if ratio >= 100.0 { "✓" } else { "✗" };

            println!(
                "Q4_K   | {:11.1} | {:16.1} | {:5.0}% | {}",
                rust_gflops, ref_perf[2].1, ratio, status
            );
        }

        // Q5_K
        {
            let (x, y) = generate_q5k_blocks(nb);
            let mut result = 0.0f32;

            let start = Instant::now();
            for _ in 0..iterations {
                result = black_box(vec_dot_q5_k_q8_k_neon(n, &x, &y));
            }
            let elapsed = start.elapsed();

            let time_ns = elapsed.as_nanos() as f64;
            let flops = 2.0 * (n as f64) * (iterations as f64);
            let rust_gflops = flops / (time_ns / 1e9) / 1e9;
            let ratio = (rust_gflops / ref_perf[3].1) * 100.0;
            let status = if ratio >= 100.0 { "✓" } else { "✗" };

            println!(
                "Q5_K   | {:11.1} | {:16.1} | {:5.0}% | {}",
                rust_gflops, ref_perf[3].1, ratio, status
            );
        }

        // Q6_K
        {
            let (x, y) = generate_q6k_blocks(nb);
            let mut result = 0.0f32;

            let start = Instant::now();
            for _ in 0..iterations {
                result = black_box(vec_dot_q6_k_q8_k_neon(n, &x, &y));
            }
            let elapsed = start.elapsed();

            let time_ns = elapsed.as_nanos() as f64;
            let flops = 2.0 * (n as f64) * (iterations as f64);
            let rust_gflops = flops / (time_ns / 1e9) / 1e9;
            let ratio = (rust_gflops / ref_perf[4].1) * 100.0;
            let status = if ratio >= 100.0 { "✓" } else { "✗" };

            println!(
                "Q6_K   | {:11.1} | {:16.1} | {:5.0}% | {}",
                rust_gflops, ref_perf[4].1, ratio, status
            );
        }

        println!("\nNote: llama.cpp numbers from project memory (best runs)");
        println!("      Variance 20-40% due to M1 frequency scaling");
    }
}
