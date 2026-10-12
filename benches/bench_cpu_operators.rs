// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Unified CPU operator benchmark: llama.cpp vs Rust
// Direct comparison in the same process with identical test data
// Method: 10 iterations, no warmup, same environment

use std::hint::black_box;
use std::time::Instant;

// FFI bindings to llama.cpp
mod ffi {
    #![allow(non_camel_case_types)]

    use infer_train::quant::types::*;

    unsafe extern "C" {
        pub fn llama_vec_dot_q4_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ4K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q5_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ5K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q6_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ6K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q2_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ2K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q3_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ3K,
            vy: *const BlockQ8K,
        );
    }
}

use ffi::*;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;

const QK_K: usize = 256;
const ITERATIONS: usize = 10000; // More iterations for stable timing

// Generate test data (matching llama.cpp exactly)
fn generate_q4k_blocks(nb: usize) -> Vec<BlockQ4K> {
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = 1;
            }
            for j in 0..128 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            BlockQ4K {
                d: 15360, // FP16 1.0
                dmin: 0,
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q5k_blocks(nb: usize) -> Vec<BlockQ5K> {
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            for j in 0..12 {
                scales[j] = 1;
            }
            for j in 0..128 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            for j in 0..32 {
                qh[j] = ((i * j) & 0xFF) as u8;
            }
            BlockQ5K { d: 15360, dmin: 0, scales, qh, qs }
        })
        .collect()
}

fn generate_q6k_blocks(nb: usize) -> Vec<BlockQ6K> {
    (0..nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..16 {
                scales[j] = 1;
            }
            for j in 0..128 {
                ql[j] = ((i + j) & 0xFF) as u8;
            }
            for j in 0..64 {
                qh[j] = ((i * j) & 0xFF) as u8;
            }
            BlockQ6K { ql, qh, scales, d: 15360 }
        })
        .collect()
}

fn generate_q2k_blocks(nb: usize) -> Vec<BlockQ2K> {
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 16];
            let mut qs = [0u8; 64];
            for j in 0..16 {
                scales[j] = 1;
            }
            for j in 0..64 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            BlockQ2K { d: 15360, dmin: 0, scales, qs }
        })
        .collect()
}

fn generate_q3k_blocks(nb: usize) -> Vec<BlockQ3K> {
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut hmask = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..12 {
                scales[j] = 1;
            }
            for j in 0..64 {
                qs[j] = ((i + j) & 0xFF) as u8;
            }
            for j in 0..32 {
                hmask[j] = ((i * j) & 0xFF) as u8;
            }
            BlockQ3K { d: 15360, hmask, qs, scales }
        })
        .collect()
}

fn generate_q8k_blocks(nb: usize) -> Vec<BlockQ8K> {
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) as i32 - 32) as i8;
            }
            for j in 0..16 {
                bsums[j] = ((i + j) % 256) as i16;
            }
            BlockQ8K { d: 1.0, qs, bsums }
        })
        .collect()
}

fn benchmark_k_quant<X, F>(
    name: &str,
    n: usize,
    llama_fn: unsafe extern "C" fn(i32, *mut f32, *const X, *const BlockQ8K),
    rust_fn: F,
    weights: &[X],
    input: &[BlockQ8K],
) where
    F: Fn(usize, &[X], &[BlockQ8K]) -> f32,
{
    let mut result = 0.0f32;

    // llama.cpp: 10 iterations, no warmup
    let llama_start = Instant::now();
    unsafe {
        for _ in 0..ITERATIONS {
            llama_fn(n as i32, &mut result, weights.as_ptr(), input.as_ptr());
        }
    }
    let llama_ns = llama_start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

    // Rust: 10 iterations, no warmup
    let rust_start = Instant::now();
    for _ in 0..ITERATIONS {
        result = black_box(rust_fn(n, weights, input));
    }
    let rust_ns = rust_start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

    let ratio = llama_ns / rust_ns;
    let status = if ratio >= 1.0 { "✓" } else { "✗" };

    let flops = 2.0 * n as f64;
    let llama_gflops = flops / (llama_ns / 1e9) / 1e9;
    let rust_gflops = flops / (rust_ns / 1e9) / 1e9;

    println!(
        "{:<8} | {:7.1} | {:7.1} | {:5.0}% | {}",
        name,
        llama_gflops,
        rust_gflops,
        ratio * 100.0,
        status
    );
}

fn main() {
    println!("=== CPU Operator Direct Comparison ===");
    println!("Method: 10 iterations, no warmup, same process\n");

    let nb = 16; // 16 blocks = 4096 elements
    let n = nb * QK_K;

    println!("Format   | llama.cpp | Rust    | Ratio | Status");
    println!("---------|-----------|---------|-------|-------");

    unsafe {
        // Q4_K
        {
            let x = generate_q4k_blocks(nb);
            let y = generate_q8k_blocks(nb);
            benchmark_k_quant(
                "Q4_K",
                n,
                llama_vec_dot_q4_K_q8_K,
                |n, x, y| vec_dot_q4_k_q8_k_neon(n, x, y),
                &x,
                &y,
            );
        }

        // Q2_K
        {
            let x = generate_q2k_blocks(nb);
            let y = generate_q8k_blocks(nb);
            benchmark_k_quant(
                "Q2_K",
                n,
                llama_vec_dot_q2_K_q8_K,
                |n, x, y| vec_dot_q2_k_q8_k_neon(n, x, y),
                &x,
                &y,
            );
        }

        // Q3_K
        {
            let x = generate_q3k_blocks(nb);
            let y = generate_q8k_blocks(nb);
            benchmark_k_quant(
                "Q3_K",
                n,
                llama_vec_dot_q3_K_q8_K,
                |n, x, y| vec_dot_q3_k_q8_k_neon(n, x, y),
                &x,
                &y,
            );
        }

        // Q5_K
        {
            let x = generate_q5k_blocks(nb);
            let y = generate_q8k_blocks(nb);
            benchmark_k_quant(
                "Q5_K",
                n,
                llama_vec_dot_q5_K_q8_K,
                |n, x, y| vec_dot_q5_k_q8_k_neon(n, x, y),
                &x,
                &y,
            );
        }

        // Q6_K
        {
            let x = generate_q6k_blocks(nb);
            let y = generate_q8k_blocks(nb);
            benchmark_k_quant(
                "Q6_K",
                n,
                llama_vec_dot_q6_K_q8_K,
                |n, x, y| vec_dot_q6_k_q8_k_neon(n, x, y),
                &x,
                &y,
            );
        }
    }

    println!("\n✓ = Rust faster than llama.cpp");
    println!("✗ = Rust slower than llama.cpp");
}
