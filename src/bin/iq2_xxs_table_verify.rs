// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Verify the ksigns_iq2xs table matches llama.cpp

fn main() {
    println!("=== IQ2_XXS ksigns_iq2xs Table Verification ===\n");

    // Table from Metal shader (int8_t cast to uint8_t)
    let table_metal: [u8; 128] = [
        1, 1, 1, 1, 1, 1, 1, 1, 0xFF, 1, 1, 1, 1, 1, 1, 0xFF, 1, 0xFF, 1, 1, 1,
        1, 1, 0xFF, 0xFF, 0xFF, 1, 1, 1, 1, 1, 1, 1, 1, 0xFF, 1, 1, 1, 1, 0xFF,
        0xFF, 1, 0xFF, 1, 1, 1, 1, 1, 1, 0xFF, 0xFF, 1, 1, 1, 1, 1, 0xFF, 0xFF,
        0xFF, 1, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF, 0xFF, 1, 1, 0xFF,
        1, 1, 1, 1, 1, 0xFF, 1, 0xFF, 1, 1, 1, 1, 0xFF, 0xFF, 1, 0xFF, 1, 1, 1,
        0xFF, 1, 1, 0xFF, 0xFF, 1, 1, 1, 1, 0xFF, 1, 0xFF, 0xFF, 1, 1, 1, 0xFF,
        1, 0xFF, 0xFF, 0xFF, 1, 1, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 1, 1, 1,
    ];

    // Check: for each 7-bit input, the output should have odd parity
    // Because the quantizer enforces odd number of negatives

    // Count entries with even vs odd parity
    let mut odd_count = 0;
    let mut even_count = 0;

    for idx in 0..128 {
        let sign_pattern = table_metal[idx];
        let parity =
            (0..8).filter(|&i| sign_pattern & (1 << i) != 0).count() % 2;
        if parity == 1 {
            odd_count += 1;
        } else {
            even_count += 1;
        }
    }

    println!("Parity check:");
    println!("  Odd parity entries: {}", odd_count);
    println!("  Even parity entries: {}", even_count);

    if even_count == 0 {
        println!("  ✓ All entries have odd parity");
    } else {
        println!("  ✗ Some entries have even parity!");
    }

    // Actually, maybe the constraint is not about the table entries
    // but about the 7-bit inputs that are stored in the block
    // The quantizer ensures the 7-bit value has odd parity
    // Then the table maps 7-bit to 8-bit

    // Let me check: for 7-bit input with odd parity, what's the 8-bit output?
    println!("\n7-bit input -> 8-bit output for odd parity inputs:");
    for idx in 0..128 {
        let parity_7bit = (0..7).filter(|&i| (idx >> i) & 1 == 1).count() % 2;
        if parity_7bit == 1 {
            let sign_pattern = table_metal[idx];
            let parity_8bit =
                (0..8).filter(|&i| sign_pattern & (1 << i) != 0).count() % 2;
            if parity_8bit != 1 {
                println!("  Index {} (odd 7-bit) -> even 8-bit!", idx);
            }
        }
    }

    // The key insight: the 7-bit input has odd parity
    // The table should map this to an 8-bit pattern that represents the actual signs
    // The 8-bit pattern should also have odd parity to represent the same number of negatives

    // Let me trace through a simple example:
    // Input has 1 negative (value 0) -> 7-bit = 0x01 -> odd parity
    // Table[0x01] = 1 = 0x01 = bit 0 set = only value 0 negative ✓

    // Input has 3 negatives (values 0, 1, 2) -> 7-bit = 0x07 -> odd parity
    // Table[0x07] = 0xFF = all bits set = all 8 values negative ✗
    // Wait, this doesn't match!

    println!("\nSpecific examples:");

    // Example 1: 1 negative (value 0)
    let idx1 = 0x01;
    let pattern1 = table_metal[idx1];
    println!(
        "7-bit = 0x01 (1 negative at pos 0): table[1] = 0x{:02X}",
        pattern1
    );
    println!(
        "  Expected: 0x01 (bit 0 set), Got: 0x{:02X} {}",
        pattern1,
        if pattern1 == 0x01 { "✓" } else { "✗" }
    );

    // Example 2: 7 negatives (all but one)
    let idx2 = 0x7F;
    let pattern2 = table_metal[idx2];
    println!(
        "\n7-bit = 0x7F (7 negatives at pos 0-6): table[127] = 0x{:02X}",
        pattern2
    );
    println!(
        "  Expected: 0xFF (all bits set), Got: 0x{:02X} {}",
        pattern2,
        if pattern2 == 0xFF { "✓" } else { "✗" }
    );

    // The issue: table[127] = 1, not 0xFF!
    // This means for 7 negatives, the output shows only 1 negative!
}
