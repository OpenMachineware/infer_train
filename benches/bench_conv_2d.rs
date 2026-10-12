// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark CONV_2D CPU performance

use infer_train::ops::conv::conv_2d_f32_simple;
use std::time::Instant;

fn main() {
    println!("=== CONV_2D CPU Benchmark ===\n");

    // Test configurations
    let configs = [
        // (c_in, c_out, h, w, kh, kw)
        (64, 64, 56, 56, 3, 3), // ResNet-like
        (128, 128, 28, 28, 3, 3),
        (256, 256, 14, 14, 3, 3),
        (512, 512, 7, 7, 3, 3),
    ];

    for &(c_in, c_out, h, w, kh, kw) in &configs {
        let h_out = h - kh + 1;
        let w_out = w - kw + 1;

        let input: Vec<f32> =
            (0..c_in * h * w).map(|i| i as f32 * 0.001).collect();
        let weights: Vec<f32> =
            (0..c_out * c_in * kh * kw).map(|i| i as f32 * 0.001).collect();
        let mut output = vec![0.0f32; c_out * h_out * w_out];

        // Warmup
        for _ in 0..3 {
            conv_2d_f32_simple(
                &input,
                &weights,
                &mut output,
                c_in,
                c_out,
                h,
                w,
                kh,
                kw,
            );
        }

        // Benchmark
        let rounds = 10;
        let start = Instant::now();
        for _ in 0..rounds {
            conv_2d_f32_simple(
                &input,
                &weights,
                &mut output,
                c_in,
                c_out,
                h,
                w,
                kh,
                kw,
            );
        }
        let elapsed = start.elapsed();

        let flops = 2.0
            * c_out as f64
            * c_in as f64
            * h_out as f64
            * w_out as f64
            * kh as f64
            * kw as f64;
        let gflops = flops * rounds as f64 / elapsed.as_secs_f64() / 1e9;

        println!(
            "C_in={}, C_out={}, {}x{} -> {}x{}, kernel {}x{}: {:.2} GFLOPS",
            c_in, c_out, h, w, h_out, w_out, kh, kw, gflops
        );
    }

    println!("\nNote: llama.cpp does not have optimized CPU CONV_2D");
    println!("This is a simple unoptimized implementation for CNN workloads");
}
