// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Quick benchmark: K-quant formats only
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
    let m = 1024;
    let k = 4096;
    let rounds = 200;

    let input = gen_f32(k, 42);
    let weights = gen_q4_k(m, k, 42);

    // Warmup
    for _ in 0..10 {
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
    }

    println!(
        "=== Quick Q4_K Benchmark (M={}, K={}, {} rounds) ===",
        m, k, rounds
    );

    let start = Instant::now();
    for _ in 0..rounds {
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
    }
    let total_us = start.elapsed().as_micros() as f64;
    let us_per_run = total_us / rounds as f64;
    let flops = 2.0 * m as f64 * k as f64;
    let gflops = flops / us_per_run / 1000.0;

    println!("Total time: {:.2} ms", total_us / 1000.0);
    println!("Time per run: {:.1} us", us_per_run);
    println!("GFLOPS: {:.1}", gflops);
}
