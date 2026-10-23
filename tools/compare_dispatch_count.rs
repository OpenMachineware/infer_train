// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compare dispatch count between our implementation and llama.cpp

use std::process::Command;

fn main() {
    // Count our dispatches
    println!("=== Our Implementation ===");
    let our_output = Command::new("cargo")
        .args(&["run", "--release", "--bin", "count-dispatches", "--features", "metal"])
        .current_dir("/Users/jia/Desktop/infer_train")
        .output()
        .expect("Failed to run our dispatch counter");

    let our_stdout = String::from_utf8_lossy(&our_output.stdout);
    for line in our_stdout.lines() {
        if line.contains("Total dispatches") || line.contains("dispatches per decode") {
            println!("{}", line);
        }
    }

    // Count llama.cpp dispatches
    println!("\n=== llama.cpp Implementation ===");
    println!("To count llama.cpp dispatches, you need to:");
    println!("1. Set environment variable: GGML_METAL_DEBUG=1");
    println!("2. Run llama-cli with a short prompt");
    println!("3. Look for kernel dispatch count in the output");

    println!("\nOr use Metal frame capture to count dispatches.");
}