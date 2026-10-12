// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug IQ4_NL nibble extraction and y-value mapping
use half::f16;

fn main() {
    // Example: qs = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, ...]
    let qs: [u8; 16] = [
        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44,
        0x55, 0x66, 0x77, 0x88,
    ];

    // CPU interpretation: each byte has two nibbles
    // qs[0] = 0x12: lo_nibble=2 for y[0], hi_nibble=1 for y[1]
    println!("CPU interpretation:");
    println!(
        "  Each qs byte provides kvalue indices for two consecutive y values"
    );
    for i in 0..4 {
        println!(
            "  qs[{}] = 0x{:02x}: y[{}] uses kvalue[{}], y[{}] uses kvalue[{}]",
            i,
            qs[i],
            2 * i,
            qs[i] & 0xF,
            2 * i + 1,
            qs[i] >> 4
        );
    }

    // GPU interpretation: q4 is uint16_t array
    let q4: [u16; 8] = [
        (qs[0] as u16) | ((qs[1] as u16) << 8),
        (qs[2] as u16) | ((qs[3] as u16) << 8),
        (qs[4] as u16) | ((qs[5] as u16) << 8),
        (qs[6] as u16) | ((qs[7] as u16) << 8),
        (qs[8] as u16) | ((qs[9] as u16) << 8),
        (qs[10] as u16) | ((qs[11] as u16) << 8),
        (qs[12] as u16) | ((qs[13] as u16) << 8),
        (qs[14] as u16) | ((qs[15] as u16) << 8),
    ];

    println!("\nGPU interpretation:");
    println!("  Each iteration i processes q4[2*i] and q4[2*i+1]");

    for i in 0..2 {
        let combined = (q4[2 * i] as u32) | ((q4[2 * i + 1] as u32) << 16);

        // il=0
        let aux32_il0 = combined & 0x0f0f0f0f;
        let q8_il0: [u8; 4] = [
            (aux32_il0 & 0xFF) as u8,
            ((aux32_il0 >> 8) & 0xFF) as u8,
            ((aux32_il0 >> 16) & 0xFF) as u8,
            ((aux32_il0 >> 24) & 0xFF) as u8,
        ];

        // il=1
        let aux32_il1 = (combined >> 4) & 0x0f0f0f0f;
        let q8_il1: [u8; 4] = [
            (aux32_il1 & 0xFF) as u8,
            ((aux32_il1 >> 8) & 0xFF) as u8,
            ((aux32_il1 >> 16) & 0xFF) as u8,
            ((aux32_il1 >> 24) & 0xFF) as u8,
        ];

        println!(
            "\n  i={}: q4[{},{}] = 0x{:04x}, 0x{:04x}",
            i,
            2 * i,
            2 * i + 1,
            q4[2 * i],
            q4[2 * i + 1]
        );
        println!("    combined = 0x{:08x}", combined);
        println!(
            "    il=0: q8=[{:2}, {:2}, {:2}, {:2}] -> kvalue indices for y[{:2},{:2},{:2},{:2}]",
            q8_il0[0],
            q8_il0[1],
            q8_il0[2],
            q8_il0[3],
            4 * i * 2 + 0,
            4 * i * 2 + 2,
            4 * i * 2 + 4,
            4 * i * 2 + 6
        );
        println!(
            "    il=1: q8=[{:2}, {:2}, {:2}, {:2}] -> kvalue indices for y[{:2},{:2},{:2},{:2}]",
            q8_il1[0],
            q8_il1[1],
            q8_il1[2],
            q8_il1[3],
            4 * i * 2 + 1,
            4 * i * 2 + 3,
            4 * i * 2 + 5,
            4 * i * 2 + 7
        );
    }

    println!(
        "\n  Total: 2 iterations * 2 il values * 4 q8 values = 16 nibbles"
    );
    println!("  With i=0,1,2,3: 4 * 2 * 4 = 32 nibbles total (complete)");

    println!("\nKey insight: nibble extraction is NOT interleaved!");
    println!("  il=0 extracts EVEN nibbles -> y at positions 0,2,4,6,8,...");
    println!("  il=1 extracts ODD nibbles  -> y at positions 1,3,5,7,9,...");
}
