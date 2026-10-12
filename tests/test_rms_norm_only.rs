// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple RMS Norm test - no warmup tricks, just measure
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    let size = 4096usize;
    let data = vec![1.0f32; size];
    let weight = vec![1.0f32; size];

    // Run 10 times
    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = ctx.rms_norm_f32(1, size, &data, &weight, 1e-5).unwrap();
        times.push(start.elapsed().as_nanos() as f64);
    }

    let avg_us = times.iter().sum::<f64>() / 10.0 / 1000.0;
    println!("Rust Metal RMS Norm size=4096: {:.2} µs", avg_us);
}
