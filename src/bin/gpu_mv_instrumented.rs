// For Instruments Time Profiler analysis
// Run with: cargo run --bin gpu_mv_instrumented --release
// Profile with: Instruments -t "Time Profiler" target/release/gpu_mv_instrumented

use half::f16;
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i + j) % 64) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn main() {
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 10000;

    println!("GPU MV Instrumentation Test");
    println!("M={}, K={}, iterations={}", m, k, iterations);
    println!(
        "Run with: Instruments -t 'Time Profiler' target/release/gpu_mv_instrumented"
    );
    println!("\nStarting in 3 seconds...");
    std::thread::sleep(std::time::Duration::from_secs(3));
    println!("Starting...\n");

    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let weights = generate_q4_k_weights(m, k);
    let input: Vec<f32> = (0..k).map(|i| (i as f32).sin()).collect();

    // Warmup
    let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let gflops = (m * k * 2) as f64 / (avg_us * 1e-6) / 1e9;

    println!("Completed {} iterations", iterations);
    println!("Total time: {:.2} ms", elapsed.as_millis());
    println!("Avg time: {:.2} us", avg_us);
    println!("GFLOPS: {:.2}", gflops);
}
