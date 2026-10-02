// Rust Q4_K MV test - calls real MetalContext API
// Run: cargo run --bin rust_q4_k_test --release

use half::f16;
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn generate_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
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

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m = if args.len() > 1 { args[1].parse().unwrap() } else { 4096 };
    let k = if args.len() > 2 { args[2].parse().unwrap() } else { 4096 };
    let iter = if args.len() > 3 { args[3].parse().unwrap() } else { 10000 };

    println!("Rust Q4_K MV: M={}, K={}, iter={}", m, k, iter);

    let weights = generate_weights(m, k);
    let input = generate_input(k);

    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    // Warmup
    let result = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");

    // Print sample outputs
    println!("Sample outputs:");
    for i in 0..5.min(m) {
        println!("  output[{}] = {:.6}", i, result[i]);
    }

    // Benchmark
    let t0 = Instant::now();
    for _ in 0..iter {
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");
    }
    let us = t0.elapsed().as_micros() as f64 / iter as f64;
    println!(
        "Avg: {:.2} us, GFLOPS: {:.2}",
        us,
        (2.0 * m as f64 * k as f64) / (us * 1e-6) / 1e9
    );
}
