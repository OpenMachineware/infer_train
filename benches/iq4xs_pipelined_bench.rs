// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

// IQ4_XS pipelined GEMM CPU benchmark
// Compares our pipelined implementation with sequential baseline

use half::f16;
use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::matmul::iq4xs_pipelined::matmul_iq4xs_f32_pipelined;
use infer_train::quant::matmul::k_quant::matmul_iq4_xs_q8_k;
use infer_train::quant::types::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

const QK_K: usize = 256;

fn gen_f32_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

fn gen_iq4_xs_weights(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4XS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut scales_l = [0u8; 4];
            for j in 0..128 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..4 {
                scales_l[j] = rng.r#gen::<u8>();
            }
            BlockIQ4XS {
                d: f16::from_f32(1.0).to_bits(),
                scales_h: 0,
                scales_l,
                qs,
            }
        })
        .collect()
}

fn bench_sequential(
    weights: &[BlockIQ4XS],
    inputs: &[&[f32]],
    outputs: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
    q8k_buffer: &mut [BlockQ8K],
    iterations: usize,
) -> f64 {
    let nb = k / QK_K;
    let mut x_padded = vec![0.0f32; k];
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();

        // Sequential quantization
        for (i, input) in inputs.iter().enumerate() {
            let row_blocks = &mut q8k_buffer[i * nb..(i + 1) * nb];
            if input.len() != k {
                for j in input.len()..k {
                    x_padded[j] = 0.0;
                }
                x_padded[..input.len()].copy_from_slice(input);
                quantize_row_q8_k_neon(&x_padded, row_blocks);
            } else {
                quantize_row_q8_k_neon(input, row_blocks);
            }
        }

        // Sequential GEMV loop
        for col in 0..n {
            let x_vec = &q8k_buffer[col * nb..(col + 1) * nb];
            let out_col = &mut outputs[col * m..(col + 1) * m];
            unsafe {
                matmul_iq4_xs_q8_k(weights, x_vec, out_col, k, m);
            }
        }

        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iterations / 2]
}

fn bench_pipelined(
    weights: &[BlockIQ4XS],
    inputs: &[&[f32]],
    outputs: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
    q8k_buffer: &mut [BlockQ8K],
    iterations: usize,
) -> f64 {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();

        unsafe {
            matmul_iq4xs_f32_pipelined(
                weights, inputs, outputs, k, m, n, q8k_buffer,
            );
        }

        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iterations / 2]
}

fn main() {
    let m = 4096; // Output dimension (like FFN gate)
    let k = 4096; // Input dimension
    let n = 128; // Batch size (prefill tokens)
    let seed = 42u64;
    let iterations = 30;

    println!("=== IQ4_XS Pipelined GEMM CPU Benchmark ===");
    println!("M={}, K={}, N={}, {} iterations\n", m, k, n, iterations);

    // Generate test data
    let weights = gen_iq4_xs_weights(m, k, seed);
    let input_data = gen_f32_matrix(k, n, seed);
    let inputs: Vec<&[f32]> =
        (0..n).map(|i| &input_data[i * k..(i + 1) * k]).collect();

    let nb = k / QK_K;
    let mut q8k_buffer =
        vec![BlockQ8K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; n * nb];
    let mut output_seq = vec![0.0f32; m * n];
    let mut output_pipe = vec![0.0f32; m * n];

    // Warmup
    for _ in 0..5 {
        unsafe {
            matmul_iq4xs_f32_pipelined(
                &weights,
                &inputs,
                &mut output_pipe,
                k,
                m,
                n,
                &mut q8k_buffer,
            );
        }
    }

    // Benchmark sequential
    println!("Benchmarking sequential (quant + GEMV loop)...");
    let seq_time = bench_sequential(
        &weights,
        &inputs,
        &mut output_seq,
        k,
        m,
        n,
        &mut q8k_buffer,
        iterations,
    );

    // Benchmark pipelined
    println!(
        "Benchmarking pipelined (parallel quant + barrier + parallel GEMM)..."
    );
    let pipe_time = bench_pipelined(
        &weights,
        &inputs,
        &mut output_pipe,
        k,
        m,
        n,
        &mut q8k_buffer,
        iterations,
    );

    // Verify correctness
    let max_diff = output_seq
        .iter()
        .zip(output_pipe.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let seq_ms = seq_time / 1e6;
    let pipe_ms = pipe_time / 1e6;
    let speedup = seq_time / pipe_time;

    println!("\n=== Results ===");
    println!("Sequential:  {:8.2} ms", seq_ms);
    println!("Pipelined:   {:8.2} ms", pipe_ms);
    println!("Speedup:     {:8.2}x", speedup);
    println!(
        "Correctness: max_diff = {:.6} ({})",
        max_diff,
        if max_diff < 1e-4 { "OK" } else { "FAIL" }
    );

    // Calculate GFLOPS
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let seq_gflops = flops / (seq_time / 1e9) / 1e9;
    let pipe_gflops = flops / (pipe_time / 1e9) / 1e9;

    println!("\n=== Performance ===");
    println!("Sequential:  {:8.2} GFLOPS", seq_gflops);
    println!("Pipelined:   {:8.2} GFLOPS", pipe_gflops);
}
