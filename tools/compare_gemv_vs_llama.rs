// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare Q5_K GEMV with llama.cpp's implementation

fn main() {
    println!("=== Q5_K GEMV Analysis ===\n");

    // From our measurements:
    // - Single GEMV (1024x2048): 260µs
    // - 7 GEMVs batched: 866µs (124µs per GEMV)
    // - Speedup from batching: 2.1x

    println!("Single GEMV (1024x2048):");
    println!("  Our time: 260µs");
    println!("  Performance: 16.1 GOPS");

    println!("\n7 GEMVs batched (one layer):");
    println!("  Our time: 866µs");
    println!("  Per GEMV: 124µs");
    println!("  Performance: 33.8 GOPS per GEMV");

    println!("\n28 layers (196 GEMVs):");
    println!("  Our time: 24.3ms (if all batched)");

    println!("\n=== llama.cpp Comparison ===");
    println!("  Total forward time: 5.35ms");
    println!("  Assuming similar kernel performance,");
    println!("  gap is likely from other operations.");

    println!("\n=== What to optimize ===");
    println!("1. Check RoPE (230µs per layer = 6.4ms total)");
    println!("2. Check Flash Attention");
    println!("3. Check RMS Norm, SiLU, Mul, Add");
    println!("4. Consider dispatch overhead (kernel launch latency)");
}
