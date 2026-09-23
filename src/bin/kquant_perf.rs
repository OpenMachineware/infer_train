// 测试所有K-quant格式的matmul性能对比
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

fn generate_q2_k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut scales = [0u8; 16];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..16 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ2K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales,
            qs,
        }
    }).collect()
}

fn generate_q3_k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut hmask = [0u8; 32];
        let mut scales = [0u8; 12];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..32 { hmask[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ3K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            hmask,
            qs,
            scales,
        }
    }).collect()
}

fn generate_q5_k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        let mut scales = [0u8; 12];
        for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..32 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ5K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales,
            qh,
            qs,
        }
    }).collect()
}

fn benchmark_matmul_q2_k(m: usize, k: usize) -> f64 {
    unsafe {
        let weights = generate_q2_k_weights(m, k);
        let x = generate_q8_k_input(k);
        let mut dst = vec![0.0f32; m];

        let iterations = if m <= 16 { 1000 } else { 100 };
        let start = Instant::now();
        for _ in 0..iterations {
            matmul_q2_k_q8_k(&weights, &x, &mut dst, k, m);
        }
        let elapsed = start.elapsed().as_secs_f64();
        2.0 * (m as f64) * (k as f64) * (iterations as f64) / elapsed / 1e9
    }
}

fn benchmark_matmul_q3_k(m: usize, k: usize) -> f64 {
    unsafe {
        let weights = generate_q3_k_weights(m, k);
        let x = generate_q8_k_input(k);
        let mut dst = vec![0.0f32; m];

        let iterations = if m <= 16 { 1000 } else { 100 };
        let start = Instant::now();
        for _ in 0..iterations {
            matmul_q3_k_q8_k(&weights, &x, &mut dst, k, m);
        }
        let elapsed = start.elapsed().as_secs_f64();
        2.0 * (m as f64) * (k as f64) * (iterations as f64) / elapsed / 1e9
    }
}

fn benchmark_matmul_q5_k(m: usize, k: usize) -> f64 {
    unsafe {
        let weights = generate_q5_k_weights(m, k);
        let x = generate_q8_k_input(k);
        let mut dst = vec![0.0f32; m];

        let iterations = if m <= 16 { 1000 } else { 100 };
        let start = Instant::now();
        for _ in 0..iterations {
            matmul_q5_k_q8_k(&weights, &x, &mut dst, k, m);
        }
        let elapsed = start.elapsed().as_secs_f64();
        2.0 * (m as f64) * (k as f64) * (iterations as f64) / elapsed / 1e9
    }
}

fn benchmark_vec_dot_q2_k(k: usize) -> f64 {
    unsafe {
        let weights = generate_q2_k_weights(1, k);
        let x = generate_q8_k_input(k);

        let start = Instant::now();
        for _ in 0..10000 {
            let _ = vec_dot_q2_k_q8_k_neon(k, &weights, &x);
        }
        let elapsed = start.elapsed().as_secs_f64();
        2.0 * (k as f64) * 10000.0 / elapsed / 1e9
    }
}

fn benchmark_vec_dot_q3_k(k: usize) -> f64 {
    unsafe {
        let weights = generate_q3_k_weights(1, k);
        let x = generate_q8_k_input(k);

        let start = Instant::now();
        for _ in 0..10000 {
            let _ = vec_dot_q3_k_q8_k_neon(k, &weights, &x);
        }
        let elapsed = start.elapsed().as_secs_f64();
        2.0 * (k as f64) * 10000.0 / elapsed / 1e9
    }
}

fn benchmark_vec_dot_q5_k(k: usize) -> f64 {
    unsafe {
        let weights = generate_q5_k_weights(1, k);
        let x = generate_q8_k_input(k);

        let start = Instant::now();
        for _ in 0..10000 {
            let _ = vec_dot_q5_k_q8_k_neon(k, &weights, &x);
        }
        let elapsed = start.elapsed().as_secs_f64();
        2.0 * (k as f64) * 10000.0 / elapsed / 1e9
    }
}

fn main() {
    let k = 4096;

    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│      K-quant Matmul Performance (K=4096)               │");
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│  格式   │ vec_dot │ matmul M=1 │ matmul M=4096 │ 状态  │");
    println!("├─────────────────────────────────────────────────────────┤");

    // Q4_K (已优化)
    let q4_vec = 88.0;  // 从之前的测试
    let q4_m1 = 51.3;
    let q4_m4096 = 67.1;
    println!("│  Q4_K   │ {:6.1}  │   {:6.1}   │    {:6.1}      │ 优化  │", q4_vec, q4_m1, q4_m4096);

    // Q2_K (未优化)
    let q2_vec = benchmark_vec_dot_q2_k(k);
    let q2_m1 = benchmark_matmul_q2_k(1, k);
    let q2_m4096 = benchmark_matmul_q2_k(4096, k);
    let q2_status = if q2_m1 < q2_vec * 0.8 { "需优化" } else { "正常" };
    println!("│  Q2_K   │ {:6.1}  │   {:6.1}   │    {:6.1}      │ {} │", q2_vec, q2_m1, q2_m4096, q2_status);

    // Q3_K (未优化)
    let q3_vec = benchmark_vec_dot_q3_k(k);
    let q3_m1 = benchmark_matmul_q3_k(1, k);
    let q3_m4096 = benchmark_matmul_q3_k(4096, k);
    let q3_status = if q3_m1 < q3_vec * 0.8 { "需优化" } else { "正常" };
    println!("│  Q3_K   │ {:6.1}  │   {:6.1}   │    {:6.1}      │ {} │", q3_vec, q3_m1, q3_m4096, q3_status);

    // Q5_K (未优化)
    let q5_vec = benchmark_vec_dot_q5_k(k);
    let q5_m1 = benchmark_matmul_q5_k(1, k);
    let q5_m4096 = benchmark_matmul_q5_k(4096, k);
    let q5_status = if q5_m1 < q5_vec * 0.8 { "需优化" } else { "正常" };
    println!("│  Q5_K   │ {:6.1}  │   {:6.1}   │    {:6.1}      │ {} │", q5_vec, q5_m1, q5_m4096, q5_status);

    // Q6_K (已优化，但有问题)
    println!("│  Q6_K   │   -     │    BUG     │      -        │ 待修复│");

    println!("└─────────────────────────────────────────────────────────┘");
}
