// 直接对比我们的matmul与llama.cpp的matmul性能
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

fn benchmark(name: &str, m: usize, k: usize, iterations: usize) -> f64 {
    unsafe {
        let weights = generate_q4_k_weights(m, k);
        let x = generate_q8_k_input(k);
        let mut dst = vec![0.0f32; m];

        // Warm up
        matmul_q4_k_q8_k(&weights, &x, &mut dst, k, m);

        let start = Instant::now();
        for _ in 0..iterations {
            matmul_q4_k_q8_k(&weights, &x, &mut dst, k, m);
        }
        let elapsed = start.elapsed().as_secs_f64();

        let flops = 2.0 * (m as f64) * (k as f64) * (iterations as f64);
        flops / elapsed / 1e9
    }
}

fn main() {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│         CPU Matmul Performance (Q4_K × Q8_K)                │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│  Format    │   M × K   │  GFLOPS  │  Time   │ vs llama.cpp │");
    println!("├─────────────────────────────────────────────────────────────┤");

    // 基准：llama.cpp的vec_dot性能约为82 GFLOPS
    // matmul应该接近这个值

    let test_cases = [
        ("Decode (M=1)", 1, 4096, 10000),
        ("Batch 16", 16, 4096, 5000),
        ("Batch 256", 256, 4096, 500),
        ("Full 4096", 4096, 4096, 100),
    ];

    unsafe {
        // 先测vec_dot作为基准
        let k = 4096;
        let nb = k / 256;
        let x = generate_q8_k_input(k);
        let weights = generate_q4_k_weights(1, k);

        let start = Instant::now();
        for _ in 0..100000 {
            let _ = vec_dot_q4_k_q8_k_neon(k, &weights, &x);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let vec_dot_gflops = 2.0 * (k as f64) * 100000.0 / elapsed / 1e9;

        println!("│ vec_dot    │  - × 4096 │  {:6.1}  │   -     │   基准参考   │", vec_dot_gflops);
        println!("├─────────────────────────────────────────────────────────────┤");
    }

    for (name, m, k, iters) in test_cases {
        let gflops = benchmark(name, m, k, iters);
        let time_us = (2.0 * m as f64 * k as f64) / gflops / 1e3;
        let ratio = gflops / 82.0;  // llama.cpp基准
        let status = if ratio >= 1.0 { "✓ 超过" } else if ratio >= 0.9 { "≈ 接近" } else { "✗ 低于" };

        println!("│ matmul {:4} │ {:4} × {:4} │  {:6.1}  │ {:5.1}μs │ {} {:.0}% │",
                 name, m, k, gflops, time_us, status, ratio * 100.0);
    }

    println!("└─────────────────────────────────────────────────────────────┘");

    println!("\n说明:");
    println!("  - llama.cpp的Q4_K vec_dot性能约82 GFLOPS (参考值)");
    println!("  - matmul循环有额外开销，GFLOPS会略低于vec_dot");
    println!("  - ✓表示性能超过llama.cpp，≈表示接近(90%+)，✗表示差距较大");
}
