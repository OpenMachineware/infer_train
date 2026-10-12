// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple Q4_K MV test - no warmup tricks, just measure
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    let m = 4096usize;
    let k = 4096usize;
    let nb = m * (k / 256);

    // Generate test data
    let weights: Vec<BlockQ4K> = (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((j + 1) % 4) as u8 + 1;
            }
            for j in 0..128 {
                qs[j] = ((i + j + 42) & 0xFF) as u8;
            }
            BlockQ4K { d: 15360, dmin: 0, scales, qs }
        })
        .collect();

    let input = vec![1.0f32; k];

    // Run 10 times
    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).unwrap();
        times.push(start.elapsed().as_nanos() as f64);
    }

    let avg_ns = times.iter().sum::<f64>() / 10.0;
    let gflops = 2.0 * (m as f64) * (k as f64) / (avg_ns / 1e9) / 1e9;

    println!(
        "Rust Metal Q4_K MV M=4096 K=4096: {:.2} GFLOPS (avg {:.2} ms)",
        gflops,
        avg_ns / 1e6
    );
}
