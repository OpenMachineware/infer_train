// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Measure single kernel execution time using Metal GPU counters
//!
//! Usage: cargo run --release --bin measure-single-kernel

#[cfg(feature = "metal")]
fn main() {
    use std::time::Instant;

    // Measure Q4_K GEMV kernel
    // Problem size: 1024 rows × 2048 cols, batch=1
    // Expected time: ~200-300 µs

    println!("Single Kernel Timing Measurement");
    println!("=================================");
    println!();
    println!("Method 1: CPU wall clock (includes dispatch overhead)");
    println!("  - Use Instant::now() before and after encode");
    println!("  - Measures total time including command buffer overhead");
    println!();
    println!("Method 2: Metal GPU time (kernel only)");
    println!("  - Use MTLCommandBuffer.gpuStartTime/gpuEndTime");
    println!("  - Measures actual kernel execution time");
    println!();
    println!("To measure with llama.cpp:");
    println!("  cd llama.cpp-0.7.0");
    println!("  ./build/bin/llama-bench -m ../Qwen3-0.6B-UD-Q4_K_XL.gguf -p 1 -n 1 -ngl 99");
    println!();
    println!("To measure with our implementation:");
    println!("  cargo run --release --bin bench-decode -- --model Qwen3-0.6B-UD-Q4_K_XL.gguf");
}

#[cfg(not(feature = "metal"))]
fn main() {
    println!("This tool requires Metal support");
}