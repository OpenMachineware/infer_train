// GPU GEMM Performance Benchmark
// Compares GPU GEMM performance with CPU for all quantization formats

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::quant::types::*;
use half::f16;
use std::time::Instant;

fn generate_f32_input(n: usize, k: usize) -> Vec<f32> {
    (0..n * k)
        .map(|i| ((i % 100) as f32 * 0.01 - 0.5))
        .collect()
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
            for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q2_k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 16];
            for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
            for j in 0..16 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
            BlockQ2K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q3_k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut hmask = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
            for j in 0..32 { hmask[j] = ((i * 19 + j * 11) % 256) as u8; }
            for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
            BlockQ3K {
                d: f16::from_f32(0.5).to_bits(),
                hmask,
                qs,
                scales,
            }
        })
        .collect()
}

fn generate_q5_k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
            for j in 0..32 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
            for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
            BlockQ5K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q6_k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 { ql[j] = ((i * 17 + j * 13) % 256) as u8; }
            for j in 0..64 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
            for j in 0..16 { scales[j] = ((i * 23 + j * 7) % 128) as i8 - 64; }
            BlockQ6K { ql, qh, scales, d: f16::from_f32(0.5).to_bits() }
        })
        .collect()
}

fn main() {
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│           GPU GEMM Performance Benchmark (Apple Metal)                   │");
    println!("├──────────────────────────────────────────────────────────────────────────┤");
    println!("│ Format │ M × N × K   │ Time (µs) │ GFLOPS  │ Throughput (GB/s)           │");
    println!("├──────────────────────────────────────────────────────────────────────────┤");

    let metal_ctx = MetalContext::new().expect("Failed to create Metal context");

    let k = 4096;
    let batch_sizes = [1, 4, 16, 64, 256];

    // Test Q4_K GEMM
    for &n in &batch_sizes {
        let m = 4096;
        let weights = generate_q4_k_weights(m, k);
        let input = generate_f32_input(n, k);

        let start = Instant::now();
        let iterations = if n <= 4 { 100 } else { 10 };
        for _ in 0..iterations {
            let _ = metal_ctx.gemm_q4_k_f32(m, n, k, &weights, &input);
        }
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / iterations as f64;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / avg_time / 1000.0;

        // Memory bandwidth: weights (m * nb * sizeof(BlockQ4K)) + input (n * k * 4) + output (m * n * 4)
        let nb = k / 256;
        let bytes_read = m as f64 * nb as f64 * std::mem::size_of::<BlockQ4K>() as f64
                       + n as f64 * k as f64 * 4.0;
        let bytes_write = m as f64 * n as f64 * 4.0;
        let bandwidth = (bytes_read + bytes_write) / avg_time / 1000.0;

        println!("│ Q4_K  │ {}×{}×{} │ {:9.1} │ {:7.1} │ {:7.1}                 │",
                 m, n, k, avg_time, gflops, bandwidth);
    }

    println!("├──────────────────────────────────────────────────────────────────────────┤");

    // Test Q2_K GEMM
    let m = 4096;
    let n = 16;
    let weights = generate_q2_k_weights(m, k);
    let input = generate_f32_input(n, k);

    let start = Instant::now();
    for _ in 0..10 {
        let _ = metal_ctx.gemm_q2_k_f32(m, n, k, &weights, &input);
    }
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_micros() as f64 / 10.0;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / avg_time / 1000.0;

    println!("│ Q2_K  │ {}×{}×{} │ {:9.1} │ {:7.1} │                         │",
             m, n, k, avg_time, gflops);

    // Test Q3_K GEMM
    let weights = generate_q3_k_weights(m, k);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = metal_ctx.gemm_q3_k_f32(m, n, k, &weights, &input);
    }
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_micros() as f64 / 10.0;
    let gflops = flops / avg_time / 1000.0;

    println!("│ Q3_K  │ {}×{}×{} │ {:9.1} │ {:7.1} │                         │",
             m, n, k, avg_time, gflops);

    // Test Q5_K GEMM
    let weights = generate_q5_k_weights(m, k);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = metal_ctx.gemm_q5_k_f32(m, n, k, &weights, &input);
    }
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_micros() as f64 / 10.0;
    let gflops = flops / avg_time / 1000.0;

    println!("│ Q5_K  │ {}×{}×{} │ {:9.1} │ {:7.1} │                         │",
             m, n, k, avg_time, gflops);

    // Test Q6_K GEMM
    let weights = generate_q6_k_weights(m, k);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = metal_ctx.gemm_q6_k_f32(m, n, k, &weights, &input);
    }
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_micros() as f64 / 10.0;
    let gflops = flops / avg_time / 1000.0;

    println!("│ Q6_K  │ {}×{}×{} │ {:9.1} │ {:7.1} │                         │",
             m, n, k, avg_time, gflops);

    println!("└──────────────────────────────────────────────────────────────────────────┘");
    println!("\n说明:");
    println!("  - M × N × K: 权重矩阵M行, N个输入向量, 每个向量K维");
    println!("  - GFLOPS: 十亿次浮点运算每秒");
    println!("  - GPU GEMM用于batch推理(prefill), MV用于单向量推理(decode)");
}
