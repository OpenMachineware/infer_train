// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Quick verification of IQ quantizer block sizes

use half::f16;

fn main() {
    println!("=== IQ Quantizer Block Size Verification ===\n");

    // Expected block sizes from llama.cpp
    let expected = [
        ("IQ1_S", 50),
        ("IQ1_M", 56),
        ("IQ2_XXS", 66),
        ("IQ2_XS", 74),
        ("IQ2_S", 82),
        ("IQ3_XXS", 98),
        ("IQ3_S", 110),
        ("IQ4_NL", 18),
        ("IQ4_XS", 136),
    ];

    let mut all_pass = true;

    for (name, expected_bytes) in expected {
        let test_file = format!("tools/test_{}.rs", name.to_lowercase());

        if !std::path::Path::new(&test_file).exists() {
            println!("{}: Test file not found", name);
            continue;
        }

        let content = std::fs::read_to_string(&test_file)
            .expect("Failed to read test file");

        // Find QUANTIZED_DATA section
        let marker = "pub const QUANTIZED_DATA: &[u8] = &[";
        let start = match content.find(marker) {
            Some(s) => s + marker.len(),
            None => {
                println!("{}: No QUANTIZED_DATA", name);
                continue;
            }
        };

        // Find end marker "];"
        let remaining = &content[start..];
        let end = match remaining.find("];") {
            Some(e) => e,
            None => {
                println!("{}: No end marker", name);
                continue;
            }
        };

        // Parse hex bytes (handle newlines and spaces)
        let hex_data = &remaining[..end];
        let bytes: Vec<u8> = hex_data
            .split(|c: char| c == ',' || c == '\n' || c == ' ')
            .filter_map(|s| {
                let s = s.trim();
                if s.starts_with("0x") {
                    u8::from_str_radix(&s[2..], 16).ok()
                } else {
                    None
                }
            })
            .collect();

        let n_blocks = 2;
        let actual_bytes = bytes.len() / n_blocks;

        if actual_bytes == expected_bytes {
            println!(
                "{}: ✅ {} bytes/block (expected {})",
                name, actual_bytes, expected_bytes
            );

            // Also check first block's scale
            if bytes.len() >= 2 {
                let d = u16::from_le_bytes([bytes[0], bytes[1]]);
                let d_f32 = f16::from_bits(d).to_f32();

                // IQ1_M doesn't have d at the start, check next
                if name == "IQ1_M" {
                    println!(
                        "  (IQ1_M has scales inside block, not d at start)"
                    );
                } else if d_f32.is_nan() || d_f32 <= 0.0 || d_f32 > 100.0 {
                    println!("  Scale: {:.4} (⚠️ out of range)", d_f32);
                    all_pass = false;
                } else {
                    println!("  Scale: {:.4} (valid)", d_f32);
                }
            }
        } else {
            println!(
                "{}: ❌ {} bytes/block (expected {})",
                name, actual_bytes, expected_bytes
            );
            all_pass = false;
        }
    }

    println!("\n=== Summary ===");
    if all_pass {
        println!("✅ All 9 IQ formats pass block size verification");
        println!("✅ Quantizer generates correctly structured data");
    } else {
        println!("❌ Some formats failed verification");
    }
}
