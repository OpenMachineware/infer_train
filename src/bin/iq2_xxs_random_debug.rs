// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Debug IQ2_XXS precision test with random data

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== IQ2_XXS Random Data Debug ===\n");

    // Use seed 42
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut data = vec![0.0f32; 256];

    // Same random data as the test
    for i in 0..256 {
        let base = rng.gen_range(-1.0..1.0) as f32;
        let outlier = if rng.gen_range(0.0..1.0) < 0.1 {
            rng.gen_range(-2.0..2.0) as f32
        } else {
            0.0
        };
        data[i] = base + outlier;
    }

    // Print first 16 values
    println!("Input data[0..16]:");
    for i in 0..16 {
        print!("{:.3} ", data[i]);
    }
    println!("\n");

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    let block = &blocks[0];

    let d = half::f16::from_bits(block.d).to_f32();
    println!("Block scale d = {:.6}", d);

    // Dequantize first sub-block
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

    let db = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;
    println!("Sub-block scale db = {:.6}", db);

    let aux8: [u8; 8] = unsafe { std::mem::transmute(aux32) };

    const KSIGNS_IQ2XS: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
    ];
    const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    println!("\nFirst 32 values comparison:");
    println!("Index | Input    | Dequant  | Error");
    println!("------|----------|----------|--------");

    for l in 0..4 {
        let grid_idx = aux8[l];
        let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
        let signs = KSIGNS_IQ2XS[(aux32[1] >> (7 * l)) as usize & 127];

        for j in 0..8 {
            let idx = l * 8 + j;
            let q = grid[j] as f32;
            let sign =
                if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            let dequant = db * q * sign;
            let input = data[idx];
            let err = (input - dequant).abs();
            println!(
                "  {:3}  | {:8.3} | {:8.3} | {:6.3}",
                idx, input, dequant, err
            );
        }
    }

    // Check if values are within expected range
    let max_abs_input = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("\nMax absolute input: {:.3}", max_abs_input);

    // IQ2_XXS quantization is 2-bit, so it has limited precision
    // Typical quantization error is 10-20% for reasonable data

    // But the test shows MSE = 6.82, which is huge!
    // Let me check if there's an issue with the dequantization
}
