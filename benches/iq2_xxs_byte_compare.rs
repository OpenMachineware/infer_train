// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Trace exact byte layout for IQ2_XXS

use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::IQ2XXS_GRID;

fn main() {
    println!("=== IQ2_XXS Byte Layout Comparison ===\n");

    // Use simple input: all 1.0
    let data = vec![1.0f32; 256];

    // Quantize
    let blocks = quantize_iq2_xxs(&data, None);
    let block = &blocks[0];

    println!("qs array (u16 values):");
    for i in 0..8 {
        println!("  qs[{}] = 0x{:04X}", i, block.qs[i]);
    }

    println!("\n=== Metal kernel view ===");
    // Metal: q2 = qs + 4*ib = qs + 0 (for ib=0)
    // q2 is a uint16_t pointer, so q2[0] = qs[0], q2[1] = qs[1], etc.
    // aux32 = q2[2] | (q2[3] << 16)
    let metal_aux32: u32 = (block.qs[2] as u32) | ((block.qs[3] as u32) << 16);
    println!("aux32 = qs[2] | (qs[3] << 16) = 0x{:08X}", metal_aux32);

    // aux8 = (uint8_t*)q2
    // This is the byte view of qs[0..4]
    let metal_aux8: [u8; 8] = {
        let mut bytes = [0u8; 8];
        for i in 0..4 {
            let u16_bytes = block.qs[i].to_le_bytes();
            bytes[2 * i] = u16_bytes[0];
            bytes[2 * i + 1] = u16_bytes[1];
        }
        bytes
    };
    println!("aux8 (bytes of qs[0..4]): {:?}", metal_aux8);
    println!("aux8[0..4] (grid indices): {:?}", &metal_aux8[0..4]);

    // Scale bits from aux32
    let scale_bits = metal_aux32 >> 28;
    println!("scale_bits (aux32 >> 28): {}", scale_bits);

    println!("\n=== My test view ===");
    // My test: byte_offset = 8 * ib32 = 0
    // Read 8 bytes starting at byte 0
    let test_qs_bytes: [u8; 8] = {
        let mut bytes = [0u8; 8];
        for i in 0..4 {
            let u16_bytes = block.qs[i].to_le_bytes();
            bytes[2 * i] = u16_bytes[0];
            bytes[2 * i + 1] = u16_bytes[1];
        }
        bytes
    };
    println!("qs_bytes[0..8]: {:?}", test_qs_bytes);

    let test_aux32: [u32; 2] = [
        u32::from_le_bytes([
            test_qs_bytes[0],
            test_qs_bytes[1],
            test_qs_bytes[2],
            test_qs_bytes[3],
        ]),
        u32::from_le_bytes([
            test_qs_bytes[4],
            test_qs_bytes[5],
            test_qs_bytes[6],
            test_qs_bytes[7],
        ]),
    ];
    println!("aux32[0] = 0x{:08X}", test_aux32[0]);
    println!("aux32[1] = 0x{:08X}", test_aux32[1]);

    // Grid indices from aux8
    let test_aux8: [u8; 8] = unsafe { std::mem::transmute(test_aux32) };
    println!("aux8: {:?}", test_aux8);
    println!("aux8[0..4] (grid indices): {:?}", &test_aux8[0..4]);

    println!("\n=== Comparison ===");
    println!("Metal aux32:  0x{:08X}", metal_aux32);
    println!("Test aux32[1]: 0x{:08X}", test_aux32[1]);

    if metal_aux32 == test_aux32[1] {
        println!("✓ Both read the same aux32!");
    } else {
        println!("✗ Different aux32 values!");
    }

    // Check grid indices
    println!("\nGrid indices:");
    println!("  Metal: {:?}", &metal_aux8[0..4]);
    println!("  Test:  {:?}", &test_aux8[0..4]);

    if metal_aux8[0..4] == test_aux8[0..4] {
        println!("✓ Same grid indices!");
    } else {
        println!("✗ Different grid indices!");
    }

    // Dequant with both formulas
    let d = half::f16::from_bits(block.d).to_f32();

    println!("\n=== Dequantization ===");

    // Metal formula
    let metal_db = d * (0.5 + scale_bits as f32);
    println!("Metal: d = {:.6}, db = {:.6}", d, metal_db);
    println!("  Formula: d_block * (0.5 + scale_bits)");
    println!("  Final result: db * grid * 0.25");

    // My test formula
    let test_db = d * (0.5 + scale_bits as f32) * 0.25;
    println!("\nTest: d = {:.6}, db = {:.6}", d, test_db);
    println!("  Formula: d_block * (0.5 + scale_bits) * 0.25");
    println!("  Result: db * grid");

    // Compare dequantized values
    println!("\nDequantized values for grid=25:");
    println!("  Metal: db * 25 * 0.25 = {:.6}", metal_db * 25.0 * 0.25);
    println!("  Test:  db * 25        = {:.6}", test_db * 25.0);

    if (metal_db * 25.0 * 0.25 - test_db * 25.0).abs() < 1e-6 {
        println!("✓ Same dequantized values!");
    } else {
        println!("✗ Different dequantized values!");
    }
}
