// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Debug sign encoding in IQ2_XXS

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;

fn main() {
    println!("=== IQ2_XXS Sign Encoding Debug ===\n");

    // Use simple input: mix of positive and negative
    let data: Vec<f32> =
        (0..256).map(|i| if i % 16 < 8 { 1.0 } else { -1.0 }).collect();

    // Print first 16 values
    println!("Input data[0..16]:");
    for i in 0..16 {
        print!("{:.1} ", data[i]);
    }
    println!("\n");

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    let block = &blocks[0];

    // Read aux32 for first sub-block
    let aux32: [u32; 2] = {
        let qs_bytes: [u8; 8] = {
            let mut bytes = [0u8; 8];
            for i in 0..4 {
                let u16_bytes = block.qs[i].to_le_bytes();
                bytes[2 * i] = u16_bytes[0];
                bytes[2 * i + 1] = u16_bytes[1];
            }
            bytes
        };
        [
            u32::from_le_bytes([
                qs_bytes[0],
                qs_bytes[1],
                qs_bytes[2],
                qs_bytes[3],
            ]),
            u32::from_le_bytes([
                qs_bytes[4],
                qs_bytes[5],
                qs_bytes[6],
                qs_bytes[7],
            ]),
        ]
    };

    let aux8: [u8; 8] = unsafe { std::mem::transmute(aux32) };

    // Extract signs for each group
    const KSIGNS_IQ2XS: [u8; 128] = [
        1, 1, 1, 1, 1, 1, 1, 1, 0xFF, 1, 1, 1, 1, 1, 1, 0xFF, 1, 0xFF, 1, 1, 1,
        1, 1, 0xFF, 0xFF, 0xFF, 1, 1, 1, 1, 1, 1, 1, 1, 0xFF, 1, 1, 1, 1, 0xFF,
        0xFF, 1, 0xFF, 1, 1, 1, 1, 1, 1, 0xFF, 0xFF, 1, 1, 1, 1, 1, 0xFF, 0xFF,
        0xFF, 1, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF, 0xFF, 1, 1, 0xFF,
        1, 1, 1, 1, 1, 0xFF, 1, 0xFF, 1, 1, 1, 1, 0xFF, 0xFF, 1, 0xFF, 1, 1, 1,
        0xFF, 1, 1, 0xFF, 0xFF, 1, 1, 1, 1, 0xFF, 1, 0xFF, 0xFF, 1, 1, 1, 0xFF,
        1, 0xFF, 0xFF, 0xFF, 1, 1, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 1, 1, 1,
    ];
    const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    println!("Sign extraction from aux32[1] = 0x{:08X}:", aux32[1]);
    println!("(Bits 0-27 encode signs, 7 bits per group)");

    for l in 0..4 {
        let signs_byte = KSIGNS_IQ2XS[(aux32[1] >> (7 * l)) as usize & 127];
        let signs_bits = (aux32[1] >> (7 * l)) & 0x7F;
        println!(
            "  Group {}: signs_bits = 0x{:02X}, signs_byte = {}",
            l, signs_bits, signs_byte
        );

        for j in 0..8 {
            let has_sign = signs_byte & KMASK_IQ2XS[j] != 0;
            println!(
                "    [{}]: bit {} = {}, sign = {}",
                j,
                j,
                (signs_bits >> j) & 1,
                if has_sign { "-" } else { "+" }
            );
        }
    }

    // Dequantize and compare
    let d = half::f16::from_bits(block.d).to_f32();
    let db = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;

    println!("\nDequantized values (db = {:.6}):", db);
    for l in 0..4 {
        let grid_idx = aux8[l];
        let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
        let signs = KSIGNS_IQ2XS[(aux32[1] >> (7 * l)) as usize & 127];

        print!("  Group {}: ", l);
        for j in 0..8 {
            let q = grid[j] as f32;
            let sign =
                if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            let val = db * q * sign;
            let expected = data[l * 8 + j];
            let match_str =
                if (val >= 0.0) == (expected >= 0.0) { "✓" } else { "✗" };
            print!("{:+.2}{} ", val, match_str);
        }
        println!();
    }

    // Check sign bits in block_signs
    // The quantizer should extract signs and ensure odd parity per group
    println!("\nExpected signs per group:");
    for k in 0..4 {
        let mut nflip = 0;
        for i in 0..8 {
            if data[8 * k + i] < 0.0 {
                nflip += 1;
            }
        }
        println!(
            "  Group {}: {} negatives, parity = {}",
            k,
            nflip,
            if nflip % 2 == 0 { "even" } else { "odd" }
        );
    }

    // IQ2_XXS requires odd parity per group of 8
    // If parity is even, one sign bit must be flipped
}
