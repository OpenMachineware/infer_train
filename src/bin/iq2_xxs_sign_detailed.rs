// Trace sign encoding for IQ2_XXS with actual quantized data
// Focus on why some signs are wrong

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::{
    IQ2GridData, IQ2XXS_GRID, KGRID_2BIT_256,
};

fn main() {
    println!("=== Detailed Sign Encoding Trace ===\n");

    // Create test data matching previous test
    let test_data: Vec<f32> = (0..256)
        .map(|i| if i % 7 == 0 { -1.0 } else { (i as f32) / 100.0 })
        .collect();

    // Quantize
    let quant_weights: Vec<f32> = vec![1.0; 256];
    let blocks = quantize_iq2_xxs(&test_data, Some(&quant_weights));

    // Dequant first sub-block
    let d = f16::from_bits(blocks[0].d).to_f32();
    let q2_0 = (blocks[0].qs[0] as u32) | ((blocks[0].qs[1] as u32) << 16);
    let q2_1 = (blocks[0].qs[2] as u32) | ((blocks[0].qs[3] as u32) << 16);

    let aux8: [u8; 4] = [
        (q2_0 & 0xFF) as u8,
        ((q2_0 >> 8) & 0xFF) as u8,
        ((q2_0 >> 16) & 0xFF) as u8,
        ((q2_0 >> 24) & 0xFF) as u8,
    ];

    let scale_bits = q2_1 >> 28;
    let db = d * (0.5 + scale_bits as f32) * 0.25;

    println!("Block 0 first sub-block:");
    println!("  d = {:.6}, scale_bits = {}, db = {:.6}", d, scale_bits, db);
    println!("  Grid indices: {:?}", aux8);

    // Signs packed in q2_1
    println!("\n  Sign indices (7-bit each):");
    for l in 0..4 {
        let signs_7bit = (q2_1 >> (7 * l)) & 0x7F;
        println!(
            "    Group {}: 7-bit index = {} ({:07b})",
            l, signs_7bit, signs_7bit
        );
    }

    // ksigns table
    let ksigns_iq2xs: [u8; 128] = [
        0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144,
        17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160,
        33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48,
        177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
        192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
        80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
        96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237,
        238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
        252, 125, 126, 255,
    ];
    let kmask_iq2xs: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    println!("\n  Detailed decoding:");
    for l in 0..4 {
        let grid_idx = aux8[l] as usize;
        let grid = IQ2XXS_GRID[grid_idx].to_le_bytes();
        let signs_7bit = (q2_1 >> (7 * l)) & 0x7F;
        let signs = ksigns_iq2xs[signs_7bit as usize];

        println!("\n  Group {}:", l);
        println!("    Grid[{}] = {:?}", grid_idx, grid);
        println!("    7-bit index = {} ({:07b})", signs_7bit, signs_7bit);
        println!("    ksigns[{}] = {} ({:08b})", signs_7bit, signs, signs);

        println!("    Values:");
        for j in 0..8 {
            let q = grid[j] as f32;
            let sign_bit = signs & kmask_iq2xs[j];
            let sign = if sign_bit != 0 { -1.0f32 } else { 1.0f32 };
            let deq_val = db * q * sign;
            let orig_idx = l * 8 + j;
            let orig_val = test_data[orig_idx];
            let correct = (orig_val >= 0.0) == (sign > 0.0);
            let status = if correct { "✓" } else { "✗" };
            println!(
                "      j={}: grid={:.0}, sign_bit={}, sign={:.0}, deq={:.3}, orig={:.3} {}",
                j, q, sign_bit, sign, deq_val, orig_val, status
            );
        }
    }

    // Now manually trace what the quantizer should produce
    println!("\n\n=== Manual Sign Encoding Trace ===");

    // For Group 0 (indices 0-7): [-1.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -1.0]
    let group0_input: [f32; 8] =
        [-1.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -1.0];
    let weight: [f32; 8] = [1.0; 8];

    let mut nflip = 0;
    let mut s: u8 = 0;
    let mut xval = [0.0f32; 8];

    println!("\nGroup 0 input: {:?}", group0_input);
    for i in 0..8 {
        if group0_input[i] >= 0.0 {
            xval[i] = group0_input[i];
        } else {
            xval[i] = -group0_input[i];
            nflip += 1;
            s |= 1 << i;
        }
    }

    println!("  nflip = {}, s = {:07b}", nflip, s);
    println!("  xval = {:?}", xval);

    // Parity constraint
    if nflip % 2 != 0 {
        let mut min_idx = 0;
        let mut min_val = weight[0] * group0_input[0] * group0_input[0];
        for i in 1..8 {
            let ax = weight[i] * group0_input[i] * group0_input[i];
            if ax < min_val {
                min_val = ax;
                min_idx = i;
            }
        }
        xval[min_idx] = -xval[min_idx];
        s ^= 1 << min_idx;
        println!(
            "  nflip is ODD, flipping bit {} (minimum at {:.6})",
            min_idx, min_val
        );
    }

    let stored = s & 127;
    println!("  Stored 7-bit index = {} ({:07b})", stored, stored);

    let signs_lookup = ksigns_iq2xs[stored as usize];
    println!("  ksigns[{}] = {} ({:08b})", stored, signs_lookup, signs_lookup);

    println!("\n  Expected signs:");
    for j in 0..8 {
        let sign_bit = signs_lookup & kmask_iq2xs[j];
        let sign = if sign_bit != 0 { -1.0f32 } else { 1.0f32 };
        let orig_sign = if group0_input[j] >= 0.0 { 1.0f32 } else { -1.0f32 };
        let match_status = if sign == orig_sign { "✓" } else { "✗ WRONG" };
        println!(
            "    j={}: sign_bit={}, decoded_sign={:.0}, original_sign={:.0} {}",
            j, sign_bit, sign, orig_sign, match_status
        );
    }
}
