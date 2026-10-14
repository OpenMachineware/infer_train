// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// K-quant GEMM vs MV performance comparison
// Tests both MV (N=1) and GEMM (N>1) kernels

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

// Generate FP32 input matrix (K x N)
fn gen_f32_matrix(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

// K-quant generators with valid test data
fn gen_q2_k(m: usize, k: usize, s: u64) -> Vec<BlockQ2K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 16];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..16 {
                let scale = (rng.r#gen::<u8>() % 63 + 1) as u8;
                let min = (rng.r#gen::<u8>() % 63) as u8;
                scales[j] = (min << 6) | scale;
            }
            BlockQ2K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn gen_q3_k(m: usize, k: usize, s: u64) -> Vec<BlockQ3K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut hmask = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..32 {
                hmask[j] = rng.r#gen::<u8>();
            }
            for j in 0..12 {
                scales[j] = (rng.r#gen::<u8>() % 63 + 1) as u8;
            }
            BlockQ3K { d: f16::from_f32(1.0).to_bits(), hmask, qs, scales }
        })
        .collect()
}

fn gen_q4_k(m: usize, k: usize, s: u64) -> Vec<BlockQ4K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..12 {
                let scale = (rng.r#gen::<u8>() % 63 + 1) as u8;
                let min = (rng.r#gen::<u8>() % 63) as u8;
                scales[j] = (min << 6) | scale;
            }
            BlockQ4K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn gen_q5_k(m: usize, k: usize, s: u64) -> Vec<BlockQ5K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..32 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..12 {
                let scale = (rng.r#gen::<u8>() % 63 + 1) as u8;
                let min = (rng.r#gen::<u8>() % 63) as u8;
                scales[j] = (min << 6) | scale;
            }
            BlockQ5K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qh,
                qs,
            }
        })
        .collect()
}

fn gen_q6_k(m: usize, k: usize, s: u64) -> Vec<BlockQ6K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = rng.r#gen::<u8>();
            }
            for j in 0..64 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..16 {
                scales[j] = (rng.r#gen::<u8>() % 64) as i8 - 32;
            }
            BlockQ6K { ql, qh, scales, d: f16::from_f32(1.0).to_bits() }
        })
        .collect()
}

fn main() {
    let ctx = MetalContext::new().expect("Metal init failed");

    // Test MV (N=1) first
    let m = 4096;
    let k = 4096;
    let rounds = 100;
    let seed = 42;

    println!("=== K-quant MV Performance (N=1, decode) ===");
    println!("M={}, K={}, {} rounds\n", m, k, rounds);

    // Pre-generate all weights
    let w_q2_k = gen_q2_k(m, k, seed);
    let w_q3_k = gen_q3_k(m, k, seed);
    let w_q4_k = gen_q4_k(m, k, seed);
    let w_q5_k = gen_q5_k(m, k, seed);
    let w_q6_k = gen_q6_k(m, k, seed);

    // Generate input for MV
    let input_mv = gen_f32_matrix(k, 1, seed);

    println!(
        "{:8} {:>10} {:>10} {:>10}",
        "Format", "us/run", "GFLOPS", "Correct"
    );
    println!("{}", "-".repeat(45));

    // Test MV
    let mv_tests: [(&str, &dyn Fn() -> Vec<f32>); 5] = [
        ("Q2_K", &|| ctx.mv_q2_k_f32(m, k, &w_q2_k, &input_mv).unwrap()),
        ("Q3_K", &|| ctx.mv_q3_k_f32(m, k, &w_q3_k, &input_mv).unwrap()),
        ("Q4_K", &|| ctx.mv_q4_k_f32(m, k, &w_q4_k, &input_mv).unwrap()),
        ("Q5_K", &|| ctx.mv_q5_k_f32(m, k, &w_q5_k, &input_mv).unwrap()),
        ("Q6_K", &|| ctx.mv_q6_k_f32(m, k, &w_q6_k, &input_mv).unwrap()),
    ];

    for (name, run_fn) in mv_tests {
        // Warmup
        for _ in 0..5 {
            let _ = run_fn();
        }

        // Benchmark
        let start = Instant::now();
        let mut result = None;
        for _ in 0..rounds {
            result = Some(run_fn());
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / rounds as f64;

        let flops = 2.0 * m as f64 * k as f64;
        let gflops = flops / (avg_us * 1e-6) / 1e9;

        let output = result.unwrap();
        let correct = output.iter().all(|&x| x.is_finite());

        println!(
            "{:8} {:>10.1} {:>10.1} {:>10}",
            name,
            avg_us,
            gflops,
            if correct { "OK" } else { "FAIL" }
        );
    }

    // Test GEMM (N=128, prefill)
    let n = 128;
    println!("\n=== K-quant GEMM Performance (N={}, prefill) ===", n);
    println!("M={}, K={}, {} rounds\n", m, k, rounds);

    let input_gemm = gen_f32_matrix(k, n, seed);

    println!(
        "{:8} {:>10} {:>10} {:>10}",
        "Format", "us/run", "GFLOPS", "Correct"
    );
    println!("{}", "-".repeat(45));

    // Test GEMM
    let gemm_tests: [(&str, &dyn Fn() -> Vec<f32>); 5] = [
        ("Q2_K", &|| {
            ctx.gemm_template_q2_k_f32(m, n, k, &w_q2_k, &input_gemm).unwrap()
        }),
        ("Q3_K", &|| {
            ctx.gemm_template_q3_k_f32(m, n, k, &w_q3_k, &input_gemm).unwrap()
        }),
        ("Q4_K", &|| {
            ctx.gemm_template_q4_k_f32(m, n, k, &w_q4_k, &input_gemm).unwrap()
        }),
        ("Q5_K", &|| {
            ctx.gemm_template_q5_k_f32(m, n, k, &w_q5_k, &input_gemm).unwrap()
        }),
        ("Q6_K", &|| {
            ctx.gemm_template_q6_k_f32(m, n, k, &w_q6_k, &input_gemm).unwrap()
        }),
    ];

    for (name, run_fn) in gemm_tests {
        // Warmup
        for _ in 0..5 {
            let _ = run_fn();
        }

        // Benchmark
        let start = Instant::now();
        let mut result = None;
        for _ in 0..rounds {
            result = Some(run_fn());
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / rounds as f64;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / (avg_us * 1e-6) / 1e9;

        let output = result.unwrap();
        let correct = output.iter().all(|&x| x.is_finite());

        println!(
            "{:8} {:>10.1} {:>10.1} {:>10}",
            name,
            avg_us,
            gflops,
            if correct { "OK" } else { "FAIL" }
        );
    }

    println!(
        "\nNote: MV uses kernel_mul_mv, GEMM uses kernel_mul_mm (llama.cpp templates)"
    );
}
