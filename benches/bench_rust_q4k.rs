// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Rust Q4_K vec_dot benchmark
// 直接调用Rust函数，无FFI开销
// 使用与llama.cpp benchmark完全相同的测试数据

use std::hint::black_box;
use std::time::{Duration, Instant};

use half::f16;
use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon;

const WARMUP_ROUNDS: usize = 3;
const WARMUP_ITERATIONS: usize = 100_000_000;
const WARMUP_SLEEP_MS: u64 = 2000;

fn warmup_cpu() {
    // Multi-round warmup to stabilize CPU frequency
    for round in 0..WARMUP_ROUNDS {
        let mut sum = 0.0f64;
        for i in 0..WARMUP_ITERATIONS {
            sum = sum * 0.9999 + 1.0;
        }
        let _ = black_box(sum);
        if round < WARMUP_ROUNDS - 1 {
            std::thread::sleep(Duration::from_millis(WARMUP_SLEEP_MS));
        }
    }
}

// 使用与C benchmark完全相同的数据生成逻辑
fn generate_q4k_blocks(nb: usize, seed: usize) -> Vec<BlockQ4K> {
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((j + 1) % 4 + 1) as u8;
            }
            for j in 0..128 {
                qs[j] = ((i + j + seed) & 0xFF) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q8k_blocks(nb: usize, seed: usize) -> Vec<BlockQ8K> {
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) - 32) as i8;
            }
            for j in 0..16 {
                bsums[j] = ((i + j + seed) % 256) as i16;
            }
            BlockQ8K { d: 1.0 + (i % 10) as f32 * 0.1, qs, bsums }
        })
        .collect()
}

fn calc_gflops(k: usize, iterations: usize, elapsed: Duration) -> f64 {
    let flops = 2.0 * (k as f64) * (iterations as f64);
    flops / elapsed.as_secs_f64() / 1e9
}

fn main() {
    println!("Rust Q4_K vec_dot Benchmark");
    println!("============================\n");

    println!("Warming up CPU...");
    warmup_cpu();
    println!("Warmup complete\n");

    let seed = 42; // 与llama.cpp benchmark相同的种子

    let sizes: [(usize, usize); 5] = [
        (256, 50000),
        (512, 50000),
        (1024, 50000),
        (2048, 10000),
        (4096, 10000),
    ];

    println!("Format | K     | GFLOPS  | Time (ms)");
    println!("-------|-------|---------|----------");

    unsafe {
        for (k, iterations) in sizes.iter() {
            let nb = k / QK_K;

            let x = generate_q4k_blocks(nb, seed);
            let y = generate_q8k_blocks(nb, seed);

            let mut result = 0.0f32;
            let start = Instant::now();
            for _ in 0..*iterations {
                result = black_box(vec_dot_q4_k_q8_k_neon(*k, &x, &y));
            }
            let elapsed = start.elapsed();

            let gflops = calc_gflops(*k, *iterations, elapsed);
            let time_ms = elapsed.as_secs_f64() * 1000.0 / *iterations as f64;

            println!("Q4_K   | {:5} | {:7.1} | {:.4}", k, gflops, time_ms);
        }
    }

    println!(
        "\nNote: Test data generated with seed={} (same as llama.cpp benchmark)",
        seed
    );
}
