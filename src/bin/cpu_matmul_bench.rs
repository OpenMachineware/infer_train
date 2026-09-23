// CPU matmul benchmark - compare with llama.cpp
use infer_train::quant::matmul::*;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;
use half::f16;
use std::time::Instant;

fn generate_q8_k_input(k: usize) -> Vec<BlockQ8K> {
    let nb = k / 256;
    (0..nb).map(|i| {
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        for j in 0..256 { qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8; }
        for j in 0..16 { bsums[j] = ((i * 23 + j * 7) % 256) as i16; }
        BlockQ8K { d: 0.01 + (i % 10) as f32 * 0.001, qs, bsums }
    }).collect()
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 128];
        let mut scales = [0u8; 12];
        for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ4K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
            scales,
            qs,
        }
    }).collect()
}

fn generate_q6_k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for j in 0..128 { ql[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..64 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..16 { scales[j] = ((i * 23 + j * 7) % 128) as i8 - 64; }
        BlockQ6K {
            ql,
            qh,
            scales,
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
        }
    }).collect()
}

fn benchmark_matmul_q4_k(m: usize, k: usize, iterations: usize) -> f64 {
    let weights = generate_q4_k_weights(m, k);
    let x = generate_q8_k_input(k);
    let mut dst = vec![0.0f32; m];  // M outputs

    // Warm up
    unsafe {
        matmul_q4_k_q8_k(&weights, &x, &mut dst, k, m);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            matmul_q4_k_q8_k(&weights, &x, &mut dst, k, m);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Print raw timing for debug
    if m == 1 && iterations == 10000 {
        println!("DEBUG: M=1, {} iters, elapsed={:.6}s, per iter={:.3}us",
                 iterations, elapsed, elapsed * 1e6 / iterations as f64);
    }

    // matmul: M * K multiply-add operations = 2 * M * K FLOPS
    let flops = 2.0 * (m as f64) * (k as f64) * (iterations as f64);
    flops / elapsed / 1e9
}

fn benchmark_matmul_q6_k(m: usize, k: usize, iterations: usize) -> f64 {
    let weights = generate_q6_k_weights(m, k);
    let x = generate_q8_k_input(k);
    let mut dst = vec![0.0f32; m];

    // Warm up
    unsafe {
        matmul_q6_k_q8_k(&weights, &x, &mut dst, k, m);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            matmul_q6_k_q8_k(&weights, &x, &mut dst, k, m);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    let flops = 2.0 * (m as f64) * (k as f64);
    flops / elapsed / 1e9
}

fn benchmark_vec_dot_q4_k(k: usize) -> f64 {
    let nb = k / 256;
    let x = generate_q8_k_input(k);
    let weights = &generate_q4_k_weights(1, k)[..nb];

    // Warm up
    unsafe {
        let _ = vec_dot_q4_k_q8_k_neon(k, weights, &x);
    }

    // Run many iterations to get accurate timing
    let iterations = 1000000;
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let _ = vec_dot_q4_k_q8_k_neon(k, weights, &x);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    // vec_dot: 2 * K FLOPS
    let flops = 2.0 * (k as f64) * (iterations as f64);
    flops / elapsed / 1e9
}

fn main() {
    println!("=== CPU Matmul Benchmark (Matrix-Vector) ===\n");

    // Test different sizes - typical LLM dimensions
    let sizes = [
        (1, 4096),      // Decode scenario: single row
        (16, 4096),     // Small batch
        (256, 4096),    // Medium batch
        (4096, 4096),   // Full matrix
    ];

    let iterations = 100;

    println!("Format: M x K | GFLOPS | Time (us)");
    println!("---");

    for &(m, k) in &sizes {
        let iters = if m == 1 { iterations * 100 } else { iterations };
        let gflops = benchmark_matmul_q4_k(m, k, iters);
        let time_us = (2.0 * m as f64 * k as f64) / gflops / 1e3;
        println!("Q4_K {:5} x {:5} | {:6.1} GFLOPS | {:6.1} us", m, k, gflops, time_us);
    }

    println!();

    for &(m, k) in &sizes {
        let gflops = benchmark_matmul_q6_k(m, k, iterations);
        let time_us = (2.0 * m as f64 * k as f64) / gflops / 1e3;
        println!("Q6_K {:5} x {:5} | {:6.1} GFLOPS | {:6.1} us", m, k, gflops, time_us);
    }

    println!("\n=== Vec_dot Single Row ===\n");

    for &k in &[256, 1024, 4096, 16384] {
        let gflops = benchmark_vec_dot_q4_k(k);
        println!("Q4_K vec_dot K={:5} | {:6.1} GFLOPS", k, gflops);
    }
}
