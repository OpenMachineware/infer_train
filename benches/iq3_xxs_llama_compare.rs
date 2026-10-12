// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare IQ3_XXS quantization output with llama.cpp
// Uses same test data as iq3_xxs_llama_dump.c
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;

const QK_K: usize = 256;

fn main() {
    println!("=== IQ3_XXS vs llama.cpp comparison ===\n");

    // Test data from iq3_xxs_llama_dump.c (only 32 values, rest are 0)
    let mut test_data: Vec<f32> = vec![
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
        0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
        -0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
        0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
        0.567890, -0.234567,
    ];
    // Pad to 256 with zeros
    test_data.extend(vec![0.0f32; QK_K - 32]);

    // Expected llama.cpp output (from iq3_xxs_llama_dump)
    println!("Expected llama.cpp output:");
    println!("  d = 0x1831");
    println!("  qs[0:12] = 0xca 0xea 0xef 0xa9 0xa9 0xb8 0x28 0x82 0x00 0x00 0x00 0x00");

    println!("\nNote: IQ3_XXS quantization not yet implemented in Rust");
    println!("This test verifies the llama.cpp output for future comparison");

    // Show the test data statistics
    let sumx2: f32 = test_data.iter().map(|v| v * v).sum();
    let sigma2 = sumx2 / QK_K as f32;
    println!("\nTest data statistics:");
    println!("  sumx2 = {:.6}", sumx2);
    println!("  sigma2 = {:.6}", sigma2);
    println!("  max = {:.6}", test_data.iter().cloned().fold(0.0f32, f32::max));

    // Show what d represents
    let d = f16::from_bits(0x1831).to_f32();
    println!("\nGlobal scale d = {:.6} (from FP16 0x1831)", d);

    // IQ3_XXS uses 256-element grid with 3-bit encoding
    println!("\nIQ3_XXS block structure:");
    println!("  d = FP16 global scale (2 bytes)");
    println!("  qs[96] = Packed grid indices + signs + scales (3 bytes per 8 values)");
    println!("  Total size = 98 bytes for 256 values (3.0625 bpw)");
}
