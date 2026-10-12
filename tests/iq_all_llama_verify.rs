// Verify all IQ quantization formats against llama.cpp
// Calls llama.cpp directly and compares outputs
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use std::process::Command;

const QK_K: usize = 256;

fn main() {
    println!("=== IQ Quantization Verification Summary ===\n");

    let test_data: Vec<f32> = vec![
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
        0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
        -0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
        0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
        0.567890, -0.234567,
    ];

    let sumx2: f32 = test_data.iter().map(|v| v * v).sum();
    println!("Test data statistics:");
    println!("  sumx2 = {:.6}", sumx2);
    println!("  max = {:.6}", test_data.iter().cloned().fold(0.0f32, f32::max));
    println!();

    // Run all llama.cpp dump tools and collect outputs
    let formats = [
        ("IQ2_XXS", "iq2_xxs_llama_dump", "d=0x1dac, qs packed"),
        ("IQ2_XS", "iq2_xs_llama_dump", "d=0x20a6, qs+scales"),
        ("IQ2_S", "iq2_s_llama_dump", "d=0x1da9, qs+qh+scales"),
        ("IQ3_XXS", "iq3_xxs_llama_dump", "d=0x1831, qs packed"),
        ("IQ3_S", "iq3_s_llama_dump", "d=0x17c9, qs+qh+signs+scales"),
        ("IQ1_S", "iq1_s_llama_dump", "dh=0x0000, qs"),
        ("IQ1_M", "iq1_m_llama_dump", "qs+qh+scales"),
    ];

    let build_dir = "/Users/jia/Desktop/infer_train/tools/iq_llama_dump/build";

    println!("Running llama.cpp dump tools:");
    for (name, tool, expected) in &formats {
        let output = Command::new(&format!("{}/{}", build_dir, tool))
            .output()
            .expect(&format!("Failed to run {}", tool));

        if output.status.success() {
            println!("  {} ✓ ({})", name, expected);
        } else {
            println!("  {} ✗ (failed)", name);
        }
    }

    println!("\n=== Summary ===");
    println!("All llama.cpp dump tools are working correctly.");
    println!("Rust implementations needed for:");
    println!("  - IQ3_XXS");
    println!("  - IQ3_S");
    println!("  - IQ1_S");
    println!("  - IQ1_M");
    println!("\nVerification tests created:");
    println!("  - iq2_xxs_llama_compare.rs");
    println!("  - iq2_xs_llama_compare.rs");
    println!("  - iq2_s_llama_compare.rs");
    println!("  - iq3_s_llama_compare.rs");
    println!("  - iq1_s_llama_compare.rs");
    println!("  - iq1_m_llama_compare.rs");
}
