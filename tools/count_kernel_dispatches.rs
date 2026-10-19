// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Count kernel dispatches per layer

fn main() {
    println!("=== Kernel Dispatch Count Per Layer ===\n");

    // Per layer operations:
    println!("1. RMS norm (attention) - 1 dispatch");
    println!("2. QKV projections (GEMV) - 3 dispatches");
    println!("3. RoPE (Q and K) - 2 dispatches");
    println!("4. Flash Attention - 1 dispatch");
    println!("5. Output projection (GEMV) - 1 dispatch");
    println!("6. Residual add - 1 dispatch");
    println!("7. RMS norm (FFN) - 1 dispatch");
    println!("8. Gate projection (GEMV) - 1 dispatch");
    println!("9. Up projection (GEMV) - 1 dispatch");
    println!("10. SiLU activation - 1 dispatch");
    println!("11. Mul (gate * up) - 1 dispatch");
    println!("12. Down projection (GEMV) - 1 dispatch");
    println!("13. Residual add - 1 dispatch");

    let total_per_layer = 1 + 3 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1;
    println!("\nTotal per layer: {} dispatches", total_per_layer);

    println!("\nFor 28 layers: {} dispatches", total_per_layer * 28);

    println!("\nPlus final operations:");
    println!("14. Final norm - 1 dispatch");
    println!("15. Output projection - 1 dispatch");

    let total = total_per_layer * 28 + 2;
    println!("\nTotal forward pass: {} kernel dispatches", total);

    // Estimate dispatch overhead
    println!("\n=== Dispatch Overhead Estimate ===");
    let overhead_per_dispatch_us = 5.0;  // Conservative estimate
    let total_overhead_ms = (total as f64 * overhead_per_dispatch_us) / 1000.0;
    println!("If each dispatch has {}µs overhead:", overhead_per_dispatch_us);
    println!("  Total overhead: {:.1}ms", total_overhead_ms);

    println!("\nIf overhead is 10µs per dispatch:");
    let overhead_10us = (total as f64 * 10.0) / 1000.0;
    println!("  Total overhead: {:.1}ms", overhead_10us);

    println!("\nOur measured GPU time: 13-14ms");
    println!("llama.cpp total time: 5.35ms");
}
