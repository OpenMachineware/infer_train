// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Debug sign encoding for specific problematic indices

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== IQ2_XXS Sign Encoding Debug ===\n");

    // Use seed 42
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut data = vec![0.0f32; 256];

    for i in 0..256 {
        let base = rng.gen_range(-1.0..1.0) as f32;
        let outlier = if rng.gen_range(0.0..1.0) < 0.1 {
            rng.gen_range(-2.0..2.0) as f32
        } else {
            0.0
        };
        data[i] = base + outlier;
    }

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    let block = &blocks[0];

    // Focus on sub-block 3 (contains index 95)
    let ib32 = 3;

    println!("Input values for sub-block 3 (indices 96-127):");
    for i in 96..128 {
        print!("{:+.3} ", data[i]);
        if (i - 96) % 8 == 7 {
            println!();
        }
    }

    // Read encoded data
    let aux32: [u32; 2] = {
        let qs_bytes: [u8; 8] = {
            let mut bytes = [0u8; 8];
            for i in 0..4 {
                let u16_bytes = block.qs[4 * ib32 + i].to_le_bytes();
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

    println!("\nEncoded data for sub-block 3:");
    println!("  aux32[0] = 0x{:08X}", aux32[0]);
    println!("  aux32[1] = 0x{:08X}", aux32[1]);
    println!("  Grid indices (aux8[0..4]): {:?}", &aux8[0..4]);

    // Extract signs
    println!("\nSigns encoding:");
    for l in 0..4 {
        let signs_7bit = (aux32[1] >> (7 * l)) & 0x7F;
        println!(
            "  Group {}: 7-bit value = 0x{:02X} = {:07b}",
            l, signs_7bit, signs_7bit
        );

        let parity = (0..7).filter(|&i| (signs_7bit >> i) & 1 == 1).count();
        println!(
            "           Number of set bits = {}, parity = {}",
            parity,
            if parity % 2 == 0 { "even" } else { "odd" }
        );

        // The 7-bit value should represent signs of values 0-6
        // The 8th value's sign is implicit
        for i in 0..7 {
            let is_negative = (signs_7bit >> i) & 1 == 1;
            println!(
                "           Value {}: sign = {}",
                i,
                if is_negative { "-" } else { "+" }
            );
        }
    }

    // Check what the quantizer should have encoded
    println!("\n\nExpected signs (from input data):");
    for l in 0..4 {
        print!("  Group {}: ", l);
        for i in 0..8 {
            let idx = 96 + l * 8 + i;
            let sign = if data[idx] >= 0.0 { "+" } else { "-" };
            print!("{} ", sign);
        }
        println!();
    }

    // The issue: the quantizer's sign parity constraint
    println!("\n\nParity constraint analysis:");
    for l in 0..4 {
        let mut n_neg = 0;
        for i in 0..8 {
            let idx = 96 + l * 8 + i;
            if data[idx] < 0.0 {
                n_neg += 1;
            }
        }
        println!(
            "  Group {}: {} negatives, parity = {}",
            l,
            n_neg,
            if n_neg % 2 == 0 { "even" } else { "odd" }
        );

        if n_neg % 2 == 0 {
            println!("           ⚠️ Even parity! Quantizer must flip one sign");
        }
    }
}
