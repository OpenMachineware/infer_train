// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! IQ2_XXS quantization precision test
//!
//! Method:
//! 1. Generate random FP32 data (256 values = 1 block)
//! 2. Quantize with Rust quantize_iq2_xxs
//! 3. Dequantize back to FP32
//! 4. Compute MSE and max error
//! 5. Compare with expected error range from llama.cpp

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== IQ2_XXS Quantization Precision Test ===\n");

    // Test with multiple seeds to ensure robustness
    let seeds = [42u64, 123, 456, 789, 1001];
    let mut all_mse = Vec::new();
    let mut all_max_err = Vec::new();
    let mut all_relative_err = Vec::new();

    for seed in seeds {
        let (mse, max_err, rel_err) = test_single_seed(seed);
        all_mse.push(mse);
        all_max_err.push(max_err);
        all_relative_err.push(rel_err);
        println!(
            "Seed {}: MSE = {:.6}, Max Error = {:.4}, Relative = {:.2}%",
            seed,
            mse,
            max_err,
            rel_err * 100.0
        );
    }

    let avg_mse: f32 = all_mse.iter().sum::<f32>() / all_mse.len() as f32;
    let avg_max_err: f32 =
        all_max_err.iter().sum::<f32>() / all_max_err.len() as f32;
    let avg_rel_err: f32 =
        all_relative_err.iter().sum::<f32>() / all_relative_err.len() as f32;
    let max_mse = all_mse.iter().cloned().fold(0.0f32, f32::max);
    let max_of_max_err = all_max_err.iter().cloned().fold(0.0f32, f32::max);

    println!("\n=== Summary ===");
    println!("Average MSE: {:.6}", avg_mse);
    println!("Average Max Error: {:.4}", avg_max_err);
    println!("Average Relative Error: {:.2}%", avg_rel_err * 100.0);
    println!("Worst MSE: {:.6}", max_mse);
    println!("Worst Max Error: {:.4}", max_of_max_err);

    // IQ2_XXS uses 2-bit quantization with 256-point grid
    // Expected: relative error should be < 15% for reasonable data
    println!("\n=== Expected Range ===");
    println!("Relative Error: < 15% for random data");
    println!("Max Error: depends on data range");

    if avg_rel_err < 0.15 {
        println!("\n✅ PASS - Errors within acceptable range");
    } else {
        println!("\n❌ FAIL - Errors too large, quantization may be incorrect");
    }
}

fn test_single_seed(seed: u64) -> (f32, f32, f32) {
    // Generate random data with normal distribution
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; 256];

    // Use a mix of small and large values to test scale handling
    for i in 0..256 {
        // Random value in range [-1, 1] with some outliers
        let base = rng.gen_range(-1.0..1.0) as f32;
        let outlier = if rng.gen_range(0.0..1.0) < 0.1 {
            rng.gen_range(-2.0..2.0) as f32
        } else {
            0.0
        };
        data[i] = base + outlier;
    }

    // Compute data statistics
    let max_val = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    assert_eq!(blocks.len(), 1, "Should produce 1 block for 256 values");

    // Dequantize
    let dequant = dequantize_iq2_xxs(&blocks);

    // Compute errors
    let mut sum_sq_err = 0.0f32;
    let mut max_err = 0.0f32;
    let mut sum_abs_rel = 0.0f32;
    let mut count_rel = 0;

    for i in 0..256 {
        let err = (data[i] - dequant[i]).abs();
        sum_sq_err += err * err;
        max_err = max_err.max(err);

        // Relative error (avoid division by zero)
        if data[i].abs() > 0.01 {
            sum_abs_rel += err / data[i].abs();
            count_rel += 1;
        }
    }

    let mse = sum_sq_err / 256.0;
    let avg_rel_err =
        if count_rel > 0 { sum_abs_rel / count_rel as f32 } else { 0.0 };

    (mse, max_err, avg_rel_err)
}

fn dequantize_iq2_xxs(blocks: &[BlockIQ2XXS]) -> Vec<f32> {
    // From llama.cpp dequantize_row_iq2_xxs
    // Block format: d (FP16) + qs[32] (16 u16 values)
    // Each 32-element sub-block uses 4 u16 values (8 bytes) from qs

    let mut result = vec![0.0f32; blocks.len() * 256];

    // ksigns_iq2xs: 7-bit index -> 8-bit sign pattern
    // Rule: bits 0-6 = input, bit 7 = parity fix (0 if odd, 1 if even)
    const KSIGNS_IQ2XS: [u8; 128] = [
        0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, 0x11, 0x13, 0x15, 0x17,
        0x19, 0x1B, 0x1D, 0x1F, 0x21, 0x23, 0x25, 0x27, 0x29, 0x2B, 0x2D, 0x2F,
        0x31, 0x33, 0x35, 0x37, 0x39, 0x3B, 0x3D, 0x3F, 0x41, 0x43, 0x45, 0x47,
        0x49, 0x4B, 0x4D, 0x4F, 0x51, 0x53, 0x55, 0x57, 0x59, 0x5B, 0x5D, 0x5F,
        0x61, 0x63, 0x65, 0x67, 0x69, 0x6B, 0x6D, 0x6F, 0x71, 0x73, 0x75, 0x77,
        0x79, 0x7B, 0x7D, 0x7F, 0x81, 0x83, 0x85, 0x87, 0x89, 0x8B, 0x8D, 0x8F,
        0x91, 0x93, 0x95, 0x97, 0x99, 0x9B, 0x9D, 0x9F, 0xA1, 0xA3, 0xA5, 0xA7,
        0xA9, 0xAB, 0xAD, 0xAF, 0xB1, 0xB3, 0xB5, 0xB7, 0xB9, 0xBB, 0xBD, 0xBF,
        0xC1, 0xC3, 0xC5, 0xC7, 0xC9, 0xCB, 0xCD, 0xCF, 0xD1, 0xD3, 0xD5, 0xD7,
        0xD9, 0xDB, 0xDD, 0xDF, 0xE1, 0xE3, 0xE5, 0xE7, 0xE9, 0xEB, 0xED, 0xEF,
        0xF1, 0xF3, 0xF5, 0xF7, 0xF9, 0xFB, 0xFD, 0xFF,
    ];
    const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    for (ib, block) in blocks.iter().enumerate() {
        let d = half::f16::from_bits(block.d).to_f32();

        // 8 sub-blocks of 32 values each
        for ib32 in 0..8 {
            // Copy 4 u16 values (8 bytes) to aux32
            // From llama.cpp: memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            // This means we read 8 bytes starting at byte offset 4*ib32*2 = 8*ib32
            let byte_offset = 8 * ib32;
            let qs_bytes = unsafe {
                std::slice::from_raw_parts(
                    block.qs.as_ptr() as *const u8,
                    32 * 2,
                )
            };

            let aux32: [u32; 2] = [
                u32::from_le_bytes([
                    qs_bytes[byte_offset],
                    qs_bytes[byte_offset + 1],
                    qs_bytes[byte_offset + 2],
                    qs_bytes[byte_offset + 3],
                ]),
                u32::from_le_bytes([
                    qs_bytes[byte_offset + 4],
                    qs_bytes[byte_offset + 5],
                    qs_bytes[byte_offset + 6],
                    qs_bytes[byte_offset + 7],
                ]),
            ];

            // Decode scale from bits 28-31 of aux32[1]
            // Metal kernel: const float d = db * (0.5f + (aux32 >> 28));
            // Metal kernel multiplies final sum by 0.25 (line 1969)
            let d = d * (0.5 + (aux32[1] >> 28) as f32) * 0.25;

            // aux8 is a view of aux32 as 8 bytes
            let aux8: [u8; 8] = unsafe { std::mem::transmute(aux32) };

            // Decode 4 groups of 8 values
            for l in 0..4 {
                let grid_idx = aux8[l];
                let grid = IQ2XXS_GRID[grid_idx as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[(aux32[1] >> (7 * l)) as usize & 127];

                for j in 0..8 {
                    // grid values are uint8 (8, 25, 43) - use directly, no division
                    let q = grid[j] as f32;
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0f32
                    } else {
                        1.0f32
                    };

                    let idx = ib * 256 + ib32 * 32 + l * 8 + j;
                    result[idx] = sign * d * q;
                }
            }
        }
    }

    result
}
