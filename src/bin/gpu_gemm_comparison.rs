// GPU GEMM Performance Comparison: Rust vs llama.cpp
// Test all K sizes from 256 to 4096

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::quant::types::*;
use half::f16;
use std::time::Instant;

fn generate_f32_input(n: usize, k: usize) -> Vec<f32> {
    (0..n * k).map(|i| ((i % 100) as f32 * 0.01 - 0.5)).collect()
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
    let metal_ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test sizes: K from 256 to 4096
    let k_sizes: Vec<usize> = vec![256, 512, 1024, 2048, 4096];
    let m = 1024;  // Fixed M for comparison
    let n = 16;    // Batch size

    println!("┌────────────────────────────────────────────────────────────────────────────┐");
    println!("│        GPU GEMM Performance: Rust Metal vs llama.cpp Baseline            │");
    println!("├────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Format │ K    │ Rust GFLOPS │ llama.cpp* │ Ratio │ Status                │");
    println!("├────────────────────────────────────────────────────────────────────────────┤");

    // Baseline from llama.cpp MV kernels (already tested)
    // These are MV GFLOPS for K=4096, multiply by batch efficiency for GEMM estimate
    let llama_cpp_baseline: std::collections::HashMap<&str, f64> = [
        ("Q4_K", 150.0),  // llama.cpp MV ~150 GFLOPS for large K
        ("Q2_K", 170.0),
        ("Q6_K", 150.0),
    ].iter().cloned().collect();

    for k in &k_sizes {
        println!("│        │──────│─────────────│────────────│───────│                       │");

        // Test Q4_K
        let weights = generate_q4_k_weights(m, *k);
        let input = generate_f32_input(n, *k);

        let start = Instant::now();
        let iterations = if *k <= 512 { 100 } else { 20 };
        for _ in 0..iterations {
            let _ = metal_ctx.gemm_q4_k_f32(m, n, *k, &weights, &input);
        }
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / iterations as f64;
        let flops = 2.0 * m as f64 * n as f64 * *k as f64;
        let rust_gflops = flops / avg_time / 1000.0;

        let baseline = llama_cpp_baseline.get("Q4_K").unwrap_or(&150.0);
        let ratio = rust_gflops / baseline * 100.0;
        let status = if ratio >= 100.0 { "✓ EXCEEDS" } else if ratio >= 90.0 { "≈ CLOSE" } else { "✗ BELOW" };

        println!("│ Q4_K   │ {:4} │ {:11.1} │ {:10.1} │ {:5.0}% │ {}              │",
                 k, rust_gflops, baseline, ratio, status);

        // Test Q2_K
        let weights = generate_q2_k_weights(m, *k);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = metal_ctx.gemm_q2_k_f32(m, n, *k, &weights, &input);
        }
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / iterations as f64;
        let rust_gflops = flops / avg_time / 1000.0;

        let baseline = llama_cpp_baseline.get("Q2_K").unwrap_or(&170.0);
        let ratio = rust_gflops / baseline * 100.0;
        let status = if ratio >= 100.0 { "✓ EXCEEDS" } else if ratio >= 90.0 { "≈ CLOSE" } else { "✗ BELOW" };

        println!("│ Q2_K   │ {:4} │ {:11.1} │ {:10.1} │ {:5.0}% │ {}              │",
                 k, rust_gflops, baseline, ratio, status);

        // Test Q6_K
        let weights = generate_q6_k_weights(m, *k);
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = metal_ctx.gemm_q6_k_f32(m, n, *k, &weights, &input);
        }
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / iterations as f64;
        let rust_gflops = flops / avg_time / 1000.0;

        let baseline = llama_cpp_baseline.get("Q6_K").unwrap_or(&150.0);
        let ratio = rust_gflops / baseline * 100.0;
        let status = if ratio >= 100.0 { "✓ EXCEEDS" } else if ratio >= 90.0 { "≈ CLOSE" } else { "✗ BELOW" };

        println!("│ Q6_K   │ {:4} │ {:11.1} │ {:10.1} │ {:5.0}% │ {}              │",
                 k, rust_gflops, baseline, ratio, status);
    }

    println!("└────────────────────────────────────────────────────────────────────────────┘");
    println!("\n说明:");
    println!("  - K: 输入向量维度");
    println!("  - llama.cpp*: 基于已测试的MV性能估算（K=4096数据）");
    println!("  - 需要实际对比llama.cpp的GEMM实现");
    println!("  - 如果Ratio低于100%，需要优化kernel");
}