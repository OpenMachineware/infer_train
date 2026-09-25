// Benchmark template kernel performance
// Run: cargo run --release --bin benchmark_template_perf

use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;
use std::time::Instant;

fn main() {
    println!("=== Template Kernel Performance Benchmark ===\n");

    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    let m = 4096;
    let k = 4096;
    let n = 128;

    println!("M={}, K={}, N={} (typical batch size)\n", m, k, n);

    // Create weights
    let weights: Vec<BlockQ4K> = (0..m * k / 256).map(|i| {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for j in 0..12 { scales[j] = (40 + j as u8 % 8) as u8; }
        for j in 0..128 { qs[j] = ((i * 17 + j * 3) % 256) as u8; }
        BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        }
    }).collect();

    // Create input (pre-convert to FP16 to exclude conversion time from benchmark)
    let input_f32: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let input_f16: Vec<u16> = input_f32.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();

    // Warmup
    let _ = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input);
    let _ = ctx.gemm_q4_k_f32(m, n, k, &weights, &input);

    // Benchmark template kernel
    let n_iter = 10;
    let start = Instant::now();
    for _ in 0..n_iter {
        let _ = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input);
    }
    let template_time = start.elapsed().as_micros() as f64 / n_iter as f64;

    // Benchmark hand-written kernel
    let start = Instant::now();
    for _ in 0..n_iter {
        let _ = ctx.gemm_q4_k_f32(m, n, k, &weights, &input);
    }
    let hand_time = start.elapsed().as_micros() as f64 / n_iter as f64;

    // Calculate GFLOPS
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let template_gflops = flops / (template_time * 1e-6) / 1e9;
    let hand_gflops = flops / (hand_time * 1e-6) / 1e9;

    println!("┌────────────────────┬────────────┬──────────┐");
    println!("│ Kernel             │ Time (µs)  │ GFLOPS   │");
    println!("├────────────────────┼────────────┼──────────┤");
    println!("│ Template (llama.cpp) │ {:10.2} │ {:8.2} │", template_time, template_gflops);
    println!("│ Hand-written       │ {:10.2} │ {:8.2} │", hand_time, hand_gflops);
    println!("└────────────────────┴────────────┴──────────┘");
    println!();
    println!("Speedup: {:.2}x", hand_time / template_time);
}