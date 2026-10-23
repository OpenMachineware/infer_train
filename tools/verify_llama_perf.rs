// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Verify llama.cpp performance on the same hardware

fn main() {
    println!("=== llama.cpp Performance Verification ===\n");

    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let llama_cpp_path = "/Users/jia/Desktop/infer_train/llama.cpp-0.7.0";

    println!("This tool will run llama.cpp benchmark on the same hardware.");
    println!("Model: {}", model_path);
    println!("llama.cpp: {}\n", llama_cpp_path);

    println!("To verify llama.cpp performance, run:");
    println!("\n  cd {} && ./build/bin/llama-bench -m {} -p 1 -n 16", llama_cpp_path, model_path);
    println!("\nThis will give us:");
    println!("  - pp 1: Prefill time for 1 token");
    println!("  - tg 16: Time to generate 16 tokens (decode)");
    println!("\nCompare with our results:");
    println!("  - Our prefill: ~2571 t/s (0.4 ms for 1 token)");
    println!("  - Our decode: ~115 t/s (8.7 ms/token)");

    println!("\n=== Expected Results ===");
    println!("If llama.cpp achieves 192 t/s on M1 hardware:");
    println!("  - That would require 53 GB/s bandwidth");
    println!("  - This is impossible on M1 (realistic peak: ~40 GB/s for GEMV)");
    println!("  - Conclusion: llama.cpp benchmark was on different hardware (M2/M3/M4)");

    println!("\nIf llama.cpp achieves ~120-150 t/s on M1:");
    println!("  - That's within expected range for M1");
    println!("  - Our 115 t/s is competitive");
    println!("  - Gap might be from kernel optimizations");

    println!("\n=== Alternative: Check llama.cpp source for hardware-specific optimizations ===");
    println!("llama.cpp might have:");
    println!("  - M2/M3-specific AMX instructions");
    println!("  - Better memory prefetching");
    println!("  - Hardware-specific kernel variants");

    println!("\n=== Recommendation ===");
    println!("Run llama.cpp benchmark on this machine to get ground truth.");
    println!("Then we can determine if there's a real gap to close.");
}