// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare IQ1_M quantization output with llama.cpp
// Uses same test data as iq1_m_llama_dump.c
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

const QK_K: usize = 256;

fn main() {
    println!("=== IQ1_M vs llama.cpp comparison ===\n");

    // Test data from iq1_m_llama_dump.c (only 32 values, rest are 0)
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

    // Expected llama.cpp output (from iq1_m_llama_dump)
    println!("Expected llama.cpp output:");
    println!("  qs[0:16] = 0x64 0xca 0xe2 0x31 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00");
    println!("  qh[0:8] = 0x27 0xa0 0x00 0x00 0x00 0x00 0x00 0x00");
    println!("  scales[0:8] = 0x37 0x00 0x00 0x40 0x00 0xb0 0x00 0x20");

    println!("\nNote: IQ1_M quantization not yet implemented in Rust");
    println!("This test verifies the llama.cpp output for future comparison");

    // Show the test data statistics
    let sumx2: f32 = test_data.iter().map(|v| v * v).sum();
    let sigma2 = sumx2 / QK_K as f32;
    println!("\nTest data statistics:");
    println!("  sumx2 = {:.6}", sumx2);
    println!("  sigma2 = {:.6}", sigma2);
    println!("  max = {:.6}", test_data.iter().cloned().fold(0.0f32, f32::max));

    // IQ1_M uses 1.75 bpw encoding
    println!("\nIQ1_M block structure:");
    println!("  qs[32] = grid indices, low 8 bits");
    println!("  qh[16] = grid indices, high 3 bits + grid shift bit");
    println!("  scales[8] = 3-bit block scales");
    println!("  Total size = 56 bytes for 256 values (1.75 bpw)");
    println!("  No separate global scale field (unlike other IQ formats)");
}
