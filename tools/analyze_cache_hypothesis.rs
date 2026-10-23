// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test if weights are being read from cache vs main memory

fn main() {
    println!("=== Cache Hypothesis Analysis ===\n");

    // M1 GPU specs
    println!("M1 GPU Memory Hierarchy:");
    println!("  - System Memory: 8-16 GB (unified)");
    println!("  - GPU L2 Cache: ~2 MB (estimated)");
    println!("  - Memory Bandwidth: 100 GB/s peak\n");

    // Our weights
    let weight_bytes = 311_033_856u64; // 311 MB
    println!("Weight size: 311 MB\n");

    // Problem: 311 MB >> 2 MB L2 cache
    // So weights CANNOT be fully cached in L2
    println!("PROBLEM: 311 MB weights >> 2 MB L2 cache");
    println!("→ Weights must be read from main memory each time\n");

    // But why does llama.cpp read less?
    println!("Mystery: llama.cpp achieves faster decode with 'less' bandwidth\n");

    // Possible explanation: Metal's texture/buffer cache
    // Metal has a larger buffer cache (tens of MB) beyond L2
    println!("Possible explanation:");
    println!("  1. Metal has additional buffer cache beyond L2");
    println!("  2. Different access patterns affect cache efficiency");
    println!("  3. llama.cpp might have better sequential access\n");

    // Let's analyze access pattern
    println!("=== Access Pattern Analysis ===\n");

    // Per decode step, we access:
    // - 7 weights per layer × 28 layers = 196 weight reads
    // - Order: norm_weight, wq, wk, wv, wo, gate, up, down
    // - Each layer is processed sequentially

    println!("Our access pattern:");
    println!("  For each layer:");
    println!("    1. Read attention_norm weight (4 KB)");
    println!("    2. Read wq weight (1.4 MB)");
    println!("    3. Read wk weight (0.7 MB)");
    println!("    4. Read wv weight (0.7 MB)");
    println!("    5. Read wo weight (1.4 MB)");
    println!("    6. Read gate weight (1.8 MB)");
    println!("    7. Read up weight (1.8 MB)");
    println!("    8. Read down weight (1.8 MB)");
    println!("  Total per layer: 9.6 MB\n");

    // Check if weights are accessed sequentially
    println!("Weight access is SEQUENTIAL within each layer.");
    println!("This is GOOD for cache - sequential reads can be prefetched.\n");

    // Compare with llama.cpp
    println!("Key question: What's different in llama.cpp?\n");

    // Actually, let me reconsider the bandwidth calculation
    println!("=== Re-examining Bandwidth Calculation ===\n");

    let our_time_ms = 8.35;
    let llama_time_ms = 5.43;
    let our_bandwidth_gbs = 37.2;
    let llama_bandwidth_gbs = 31.0;

    let our_effective_mb = our_bandwidth_gbs * our_time_ms;
    let llama_effective_mb = llama_bandwidth_gbs * llama_time_ms;

    println!("Our effective memory: {:.1} MB", our_effective_mb);
    println!("llama.cpp effective memory: {:.1} MB", llama_effective_mb);
    println!("Difference: {:.1} MB\n", our_effective_mb - llama_effective_mb);

    // If both read the same 311 MB weights...
    println!("If both read 311 MB weights:");
    let our_bandwidth_check = 311.0 / our_time_ms;
    let llama_bandwidth_check = 311.0 / llama_time_ms;
    println!("  Our bandwidth would be: {:.1} GB/s", our_bandwidth_check);
    println!("  llama.cpp bandwidth would be: {:.1} GB/s", llama_bandwidth_check);
    println!("\nThese don't match the measured values!\n");

    // This means our bandwidth measurement is off
    println!("CONCLUSION: The bandwidth measurements are inconsistent.");
    println!("\nPossible causes:");
    println!("  1. Bandwidth calculation includes intermediate buffers");
    println!("  2. Bandwidth measurement is wrong");
    println!("  3. Metal reports different bandwidth metrics\n");

    // Let's use a simpler metric: time per token
    println!("=== Simpler Analysis: Time per Token ===\n");
    println!("We: {:.2} ms/token", our_time_ms);
    println!("llama.cpp: {:.2} ms/token", llama_time_ms);
    println!("Gap: {:.2}x slower\n", our_time_ms / llama_time_ms);

    // What can we optimize?
    println!("Potential optimizations:");
    println!("  1. Reduce kernel dispatch overhead");
    println!("  2. Improve kernel scheduling");
    println!("  3. Fuse more operations");
    println!("  4. Better memory access patterns");
}
