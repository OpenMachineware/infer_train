// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Minimal Q5_K GEMM test without qh contribution

use std::env;

fn main() {
    println!("=== Minimal Q5_K GEMM test ===\n");
    println!("Current issue: GEMM produces overflow/NaN when qh bits are used");
    println!("\nNext debugging steps:");
    println!("1. Create test with synthetic Q5_K block (known values)");
    println!("2. Test basic formula without qh contribution");
    println!("3. Verify scale extraction matches GEMV");
    println!("4. Add qh contribution with correct bit mapping");
    println!("\nRecommended approach:");
    println!("- Use llama.cpp's mul_mm_q5_K as reference implementation");
    println!("- Compare byte-by-byte with our dequantize_q5_K");
    println!("- Focus on: il value mapping, scale extraction, qh bit selection");
}
