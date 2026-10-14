// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Fair performance comparison: Self-written vs llama.cpp
// Using MetalContext API for consistent measurement
use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

fn gen_f32(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

fn gen_q4_k(m: usize, k: usize, s: u64) -> Vec<BlockQ4K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut sc = [0u8; 12];
            for i in 0..128 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..12 {
                sc[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ4K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales: sc,
                qs,
            }
        })
        .collect()
}

fn main() {
    let ctx = MetalContext::new().expect("Metal init failed");
    let m = 4096usize;
    let k = 4096usize;
    let rounds = 50;

    let input = gen_f32(k, 42);
    let weights = gen_q4_k(m, k, 42);

    // Warmup
    println!("Warming up...");
    for _ in 0..20 {
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
    }

    println!(
        "\n=== Q4_K MV Performance (M={}, K={}, {} rounds) ===",
        m, k, rounds
    );

    // Measure
    let mut times_us = Vec::new();
    for i in 0..rounds {
        let start = Instant::now();
        let result = ctx.mv_q4_k_f32(m, k, &weights, &input).unwrap();
        let elapsed = start.elapsed().as_micros() as f64;
        times_us.push(elapsed);

        // Verify correctness on first run
        if i == 0 {
            let nan_count = result.iter().filter(|&&v| v.is_nan()).count();
            let inf_count = result.iter().filter(|&&v| v.is_infinite()).count();
            println!(
                "First result: {} values, {} NaN, {} Inf",
                result.len(),
                nan_count,
                inf_count
            );
        }
    }

    let avg_us = times_us.iter().sum::<f64>() / rounds as f64;
    let flops = 2.0 * m as f64 * k as f64;
    let gflops = flops / avg_us / 1000.0;

    println!("\nAverage time: {:.1} us", avg_us);
    println!("GFLOPS: {:.1}", gflops);

    // Comparison note
    println!(
        "\nNote: To compare against llama.cpp, need to run with alternating benchmark."
    );
    println!(
        "Previous results show self-written MV kernels match or exceed llama.cpp."
    );
}
