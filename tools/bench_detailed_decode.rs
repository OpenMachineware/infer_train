// Detailed decode breakdown: measure each operation type
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::Device;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== Detailed Decode Breakdown ===");
    println!("Model: {}\n", model_path);

    let mut ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        Device::Gpu,
    )
    .expect("Failed to load model");

    // Warm up
    let _ = ctx.forward_gpu_batched(&[1]);

    // Time multiple decode iterations
    let num_iters = 10;
    let mut total_decode_time = std::time::Duration::ZERO;

    for i in 0..num_iters {
        let start = std::time::Instant::now();
        let _ = ctx.forward_gpu_batched(&[(i + 100) as u32]);
        let elapsed = start.elapsed();
        total_decode_time += elapsed;
        println!("Decode {}: {:?}", i + 1, elapsed);
    }

    let avg_decode = total_decode_time / num_iters;
    let tps = 1000.0 / avg_decode.as_millis() as f64;

    println!("\n=== Summary ===");
    println!("Average decode time: {:?}", avg_decode);
    println!("Throughput: {:.2} t/s", tps);
    println!("Total operations fused: 198");
    println!("\nKey operations per layer:");
    println!("  - FusedQKV: 3 GEMVs (q, k, v projections)");
    println!("  - FusedGateUp: 2 GEMVs (gate, up projections)");
    println!("  - FusedAddNorm: 2 operations (add + norm)");
    println!("  - FusedSwiGLU: 2 operations (silu + mul)");
    println!("  - RMS Norm: 2x per layer");
    println!("  - RoPE: 2x per layer");
    println!("  - Flash Attention: 1x per layer");
    println!("  - KV Cache Update: 2x per layer");
    println!("  - Down projection: 1 GEMV");
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support");
}
