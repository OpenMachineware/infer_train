// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Q6_K GEMM test
use infer_train::quant::types::BlockQ6K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().expect("Failed to create MetalContext");
    let m = 4096usize;
    let k = 4096usize;
    let n = 64usize;
    let nb = m * (k / 256);

    let weights: Vec<BlockQ6K> = (0..nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = ((i + j + 42) & 0xFF) as u8;
            }
            for j in 0..64 {
                qh[j] = ((i + j) % 256) as u8;
            }
            for j in 0..16 {
                scales[j] = (((j + 1) % 8) as i8) - 4;
            }
            BlockQ6K { d: half::f16::from_f32(1.0).to_bits(), ql, qh, scales }
        })
        .collect();

    let input = vec![1.0f32; n * k];

    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = ctx.matmul_q6_k_f32(m, n, k, &weights, &input).unwrap();
        times.push(start.elapsed().as_nanos() as f64);
    }

    let avg_ns = times.iter().sum::<f64>() / 10.0;
    let gflops =
        2.0 * (m as f64) * (k as f64) * (n as f64) / (avg_ns / 1e9) / 1e9;
    println!("Q6_K GEMM: {:.2} GFLOPS", gflops);
}
