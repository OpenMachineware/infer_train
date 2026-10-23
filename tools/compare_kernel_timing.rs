// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compare kernel execution timing between our implementation and llama.cpp
//!
//! This tool measures:
//! 1. Individual kernel execution time
//! 2. Dispatch overhead
//! 3. Total decode time

use std::time::Instant;

fn main() {
    println!("Kernel Timing Comparison Tool");
    println!("==============================");
    println!();
    println!("TODO:");
    println!("1. Add Metal GPU timestamp support to both implementations");
    println!("2. Measure individual kernel times:");
    println!("   - Q4_K GEMV (1024 rows, batch=1)");
    println!("   - Q5_K GEMV (3072 rows, batch=1)");
    println!("   - Q6_K GEMV (1024 rows, batch=1)");
    println!("   - Flash Attention");
    println!("   - RMS Norm");
    println!("3. Measure dispatch overhead:");
    println!("   - Time between kernel submission and start");
    println!("   - Command buffer encode time");
    println!("4. Compare with llama.cpp timing");
}
