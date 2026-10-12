// SPDX-FileCopyrightText: © 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Detailed trace of IQ2_XXS error for random data

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== IQ2_XXS Error Analysis for Random Data ===\n");

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

    // Full dequantization
    let d = half::f16::from_bits(block.d).to_f32();
    println!("Block scale d = {:.6}", d);

    const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    let mut all_errors = Vec::new();

    for ib32 in 0..8 {
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

        let db = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;
        let aux8: [u8; 8] = unsafe { std::mem::transmute(aux32) };

        for l in 0..4 {
            let grid_idx = aux8[l];
            let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
            let signs_7bit = (aux32[1] >> (7 * l)) & 0x7F;
            let signs = signs_7bit as u8
                | if (0..7).filter(|&i| (signs_7bit >> i) & 1 == 1).count() % 2
                    == 0
                {
                    0x80
                } else {
                    0
                };

            for j in 0..8 {
                let idx = ib32 * 32 + l * 8 + j;
                let q = grid[j] as f32;
                let sign =
                    if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                let dequant = db * q * sign;
                let input = data[idx];
                let err = (input - dequant).abs();
                all_errors.push((idx, input, dequant, err));
            }
        }
    }

    // Sort by error magnitude
    all_errors.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("\nTop 20 largest errors:");
    println!("Index | Input    | Dequant  | Error    | Rel Err");
    println!("------|----------|----------|----------|--------");
    for (idx, input, dequant, err) in all_errors.iter().take(20) {
        let rel_err =
            if input.abs() > 0.01 { err / input.abs() } else { f32::NAN };
        println!(
            "  {:3}  | {:8.3} | {:8.3} | {:8.3} | {:6.1}%",
            idx,
            input,
            dequant,
            err,
            rel_err * 100.0
        );
    }

    // Statistics
    let mse: f32 =
        all_errors.iter().map(|(_, _, _, e)| e * e).sum::<f32>() / 256.0;
    let max_err =
        all_errors.iter().map(|(_, _, _, e)| *e).fold(0.0f32, f32::max);
    let avg_err: f32 =
        all_errors.iter().map(|(_, _, _, e)| *e).sum::<f32>() / 256.0;

    println!("\nError statistics:");
    println!("  MSE: {:.6}", mse);
    println!("  Max: {:.6}", max_err);
    println!("  Avg: {:.6}", avg_err);

    // Compare with expected
    println!("\nExpected for IQ2_XXS:");
    println!("  2-bit quantization with 256-entry grid");
    println!("  Typical error: 5-15% for well-distributed data");
    println!("  Can be higher for data that doesn't match grid well");

    // Check if there's a systematic issue
    let pos_count = all_errors
        .iter()
        .filter(|(_, input, dequant, _)| *dequant > *input)
        .count();
    let neg_count = all_errors
        .iter()
        .filter(|(_, input, dequant, _)| *dequant < *input)
        .count();
    println!("\nError distribution:");
    println!("  Dequant > Input: {} times", pos_count);
    println!("  Dequant < Input: {} times", neg_count);
}
