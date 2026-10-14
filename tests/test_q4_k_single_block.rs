// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test single Q4_K block to find the bug
// Usage: cargo run --bin test_q4_k_single_block --release

use half::f16;

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

fn main() {
    // Simple test: one block, 256 input values
    let block = BlockQ4K {
        d: f16::from_f32(1.0).to_bits(),
        dmin: f16::from_f32(0.0).to_bits(),
        scales: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        qs: [0x12; 128], // All values = 0x12 (low nibble 2, high nibble 1)
    };

    let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();

    println!("Block:");
    let d = block.d;
    let dmin = block.dmin;
    println!("  d = {} (f16 bits = {:#04x})", f16::from_bits(d), d);
    println!("  dmin = {} (f16 bits = {:#04x})", f16::from_bits(dmin), dmin);
    println!("  scales = {:?}", block.scales);
    println!("  qs[0..8] = {:02x?}", &block.qs[0..8]);

    // CPU reference computation (simplified)
    // For each 32-element sub-block, we need to:
    // 1. Extract scales
    // 2. Compute dot product

    // This is just a debug print - we need to verify with llama.cpp output
    println!("\nInput values [0..8]: {:?}", &input[0..8]);
    println!("\nTo verify correctness, compare with llama.cpp output");
}
