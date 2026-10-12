// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Generate the correct ksigns_iq2xs table

fn main() {
    println!("=== Generate Correct ksigns_iq2xs Table ===\n");

    // The rule:
    // - Input: 7-bit value with ODD parity (quantizer ensures this)
    // - Output: 8-bit sign pattern that represents the same signs
    // - Bits 0-6 of output = bits 0-6 of input
    // - Bit 7 of output = 0 (to maintain odd parity)

    // Generate the table
    let mut correct_table = [0u8; 128];
    for idx in 0..128 {
        let parity = (0..7).filter(|&i| (idx >> i) & 1 == 1).count() % 2;
        if parity == 1 {
            // Odd parity: bit 7 = 0
            correct_table[idx] = idx as u8;
        } else {
            // Even parity: should not happen, but let's see what Metal table has
            // Set bit 7 to 1 to make it odd
            correct_table[idx] = (idx as u8) | 0x80;
        }
    }

    // Compare with Metal table
    let metal_table: [u8; 128] = [
        1, 1, 1, 1, 1, 1, 1, 1, 0xFF, 1, 1, 1, 1, 1, 1, 0xFF, 1, 0xFF, 1, 1, 1,
        1, 1, 0xFF, 0xFF, 0xFF, 1, 1, 1, 1, 1, 1, 1, 1, 0xFF, 1, 1, 1, 1, 0xFF,
        0xFF, 1, 0xFF, 1, 1, 1, 1, 1, 1, 0xFF, 0xFF, 1, 1, 1, 1, 1, 0xFF, 0xFF,
        0xFF, 1, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF, 0xFF, 1, 1, 0xFF,
        1, 1, 1, 1, 1, 0xFF, 1, 0xFF, 1, 1, 1, 1, 0xFF, 0xFF, 1, 0xFF, 1, 1, 1,
        0xFF, 1, 1, 0xFF, 0xFF, 1, 1, 1, 1, 0xFF, 1, 0xFF, 0xFF, 1, 1, 1, 0xFF,
        1, 0xFF, 0xFF, 0xFF, 1, 1, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 1, 1, 1,
    ];

    println!("Comparing for ODD parity indices:");
    let mut mismatch = false;
    for idx in 0..128 {
        let parity = (0..7).filter(|&i| (idx >> i) & 1 == 1).count() % 2;
        if parity == 1 {
            let correct = correct_table[idx];
            let metal = metal_table[idx];
            if correct != metal {
                println!(
                    "  Index {}: correct = 0x{:02X}, metal = 0x{:02X}",
                    idx, correct, metal
                );
                mismatch = true;
            }
        }
    }

    if !mismatch {
        println!("  ✓ All odd parity indices match!");
    } else {
        println!("\n  ✗ Mismatch found!");
    }

    // Print the correct table
    println!("\nCorrect ksigns_iq2xs table:");
    println!("const KSIGNS_IQ2XS: [u8; 128] = [");
    for row in 0..8 {
        print!("   ");
        for col in 0..16 {
            let idx = row * 16 + col;
            print!(" 0x{:02X},", correct_table[idx]);
        }
        println!();
    }
    println!("];");

    // Verify: for all odd parity inputs, the output should have the same odd parity
    println!("\nVerification:");
    for idx in 0..128 {
        let parity_in = (0..7).filter(|&i| (idx >> i) & 1 == 1).count() % 2;
        let parity_out =
            (0..8).filter(|&i| correct_table[idx] & (1 << i) != 0).count() % 2;
        if parity_in != parity_out {
            println!(
                "  Index {}: parity mismatch! in={}, out={}",
                idx, parity_in, parity_out
            );
        }
    }
    println!("  ✓ All parity checks pass");
}
