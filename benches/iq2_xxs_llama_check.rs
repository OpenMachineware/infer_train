// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compare Rust quantization with llama.cpp reference

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;

fn main() {
    println!("=== IQ2_XXS llama.cpp Reference Check ===\n");

    // Use simple input: all 1.0
    let data = vec![1.0f32; 256];

    // Quantize with Rust
    let blocks = quantize_iq2_xxs(&data, None);
    let block = &blocks[0];

    // Decode block
    let d = half::f16::from_bits(block.d).to_f32();
    println!("Block scale d = {:.6}", d);

    // Decode first sub-block (ib32=0) using exact llama.cpp dequant formula
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

    // llama.cpp dequant formula: db = d * (0.5 + (aux32[1] >> 28)) * 0.25
    let db = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;
    println!("Sub-block scale db = {:.6}", db);

    // aux8 view of aux32
    let aux8: [u8; 8] = unsafe { std::mem::transmute(aux32) };

    // Dequantize
    const KSIGNS_IQ2XS: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
    ];
    const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    println!("\nFirst 32 dequantized values:");
    for l in 0..4 {
        let grid_idx = aux8[l];
        let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
        let signs = KSIGNS_IQ2XS[(aux32[1] >> (7 * l)) as usize & 127];

        for j in 0..8 {
            let q = grid[j] as f32;
            let sign =
                if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            let val = db * q * sign;
            print!("{:.3} ", val);
        }
    }
    println!();

    // Expected: all 1.0
    println!("\nExpected: all 1.0");

    // Compute error
    let mut sum_sq_err = 0.0f32;
    let mut max_err = 0.0f32;
    for l in 0..4 {
        let grid_idx = aux8[l];
        let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
        let signs = KSIGNS_IQ2XS[(aux32[1] >> (7 * l)) as usize & 127];

        for j in 0..8 {
            let q = grid[j] as f32;
            let sign =
                if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            let val = db * q * sign;
            let err = (val - 1.0).abs();
            sum_sq_err += err * err;
            max_err = max_err.max(err);
        }
    }

    let mse = sum_sq_err / 32.0;
    println!("\nMSE = {:.6}, Max Error = {:.4}", mse, max_err);

    // Also check what grid point was chosen
    println!("\nGrid analysis:");
    for l in 0..4 {
        let grid_idx = aux8[l];
        println!("  Group {}: grid_idx = {}", l, grid_idx);
    }

    // Check if there's a better grid point for uniform values
    println!("\nBest grid point for uniform L values:");
    for target_l in 0..=2 {
        // Build grid index for uniform L values
        // The grid stores dequant values, not L values
        // We need to find a grid entry with all same values
        let target_val = 8 + 17 * target_l; // 8, 25, or 43
        println!("  Target L = {}, dequant value = {}", target_l, target_val);

        // Check Grid[0] (all 8s)
        let grid0 = IQ2XXS_GRID[0].to_le_bytes();
        if grid0[0] == target_val && grid0.iter().all(|&v| v == grid0[0]) {
            println!("    Grid[0] matches!");
        }
    }
}
