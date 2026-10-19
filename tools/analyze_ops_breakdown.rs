// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Analyze operations breakdown per layer

fn main() {
    println!("=== Per-Layer Operation Breakdown ===");
    println!("Based on timing measurements:\n");

    // GEMV operations (Q5_K)
    let gemv_per_layer = 7;
    let gemv_time_per_layer_us = 697.0;  // measured
    let gemv_time_per_op_us = gemv_time_per_layer_us / gemv_per_layer as f64;

    // RoPE
    let rope_time_us = 230.9;  // measured per layer
    let rope_total_us = rope_time_us * 2.0;  // Q and K

    // Other ops (estimated from total - GEMV - RoPE)
    let total_layer_time_us = 697.0;  // measured
    let other_time_us = total_layer_time_us - (gemv_per_layer as f64 * gemv_time_per_op_us) - rope_total_us;

    println!("GEMV (Q5_K):");
    println!("  7 operations per layer");
    println!("  {:.1}µs per operation", gemv_time_per_op_us);
    println!("  {:.1}µs total per layer", gemv_per_layer as f64 * gemv_time_per_op_us);

    println!("\nRoPE:");
    println!("  {:.1}µs per application", rope_time_us);
    println!("  {:.1}µs total (Q and K)", rope_total_us);

    println!("\nOther ops (estimated):");
    println!("  {:.1}µs", other_time_us);
    println!("  Includes: RMS norm (2x), Flash attention, SiLU, Mul, Add (2x)");

    println!("\n=== Total Per Layer ===");
    println!("  {:.1}µs", total_layer_time_us);

    println!("\n=== For 28 Layers ===");
    println!("  GEMV: {:.1}ms", gemv_per_layer as f64 * gemv_time_per_op_us * 28.0 / 1000.0);
    println!("  RoPE: {:.1}ms", rope_total_us * 28.0 / 1000.0);
    println!("  Other: {:.1}ms", other_time_us * 28.0 / 1000.0);
    println!("  Total: {:.1}ms", total_layer_time_us * 28.0 / 1000.0);

    println!("\n=== Bottleneck Analysis ===");
    let gemv_pct = (gemv_per_layer as f64 * gemv_time_per_op_us / total_layer_time_us) * 100.0;
    let rope_pct = (rope_total_us / total_layer_time_us) * 100.0;
    let other_pct = (other_time_us / total_layer_time_us) * 100.0;

    println!("  GEMV: {:.1}%", gemv_pct);
    println!("  RoPE: {:.1}%", rope_pct);
    println!("  Other: {:.1}%", other_pct);

    println!("\n=== Comparison with llama.cpp ===");
    let our_total_ms = total_layer_time_us * 28.0 / 1000.0;
    let llama_total_ms = 5.35;  // 187 tok/s
    println!("  Our time: {:.1}ms", our_total_ms);
    println!("  llama.cpp: {:.1}ms", llama_total_ms);
    println!("  Gap: {:.1}x slower", our_total_ms / llama_total_ms);
}
