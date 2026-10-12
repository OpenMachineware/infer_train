// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test llama.cpp IQ2_XXS quantization-dequantization consistency

// Constants from llama.cpp
const QK_K: usize = 256;
const KMAX_Q: i32 = 3;

// Grid data (first 32 entries)
const IQ2XXS_GRID: [u64; 32] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
];

const KGRID_2BIT_256: [u16; 32] = [
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97, 100, 128, 130,
    138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642,
];

fn nearest_int(x: f32) -> i32 {
    (x + 0.5).floor() as i32
}

fn main() {
    println!("=== Testing Grid Value Relationship ===\n");

    // Check the relationship between quantization grid and dequantization grid
    println!("Quantization grid uses: q = 2*l + 1 = {{1, 3, 5}}");
    println!("Dequantization grid uses: {{8, 25, 43}}");
    println!();

    println!("Checking if grid values match:");
    for l in 0..3 {
        let q = 2 * l + 1;
        let expected_dequant = 8 * q;
        let actual_dequant = match l {
            0 => 8,
            1 => 25,
            2 => 43,
            _ => 0,
        };
        println!(
            "  l={}: q={}, expected_dequant={}, actual_dequant={}, diff={}",
            l,
            q,
            expected_dequant,
            actual_dequant,
            actual_dequant as f32 / 8.0 - q as f32
        );
    }
    println!();

    // Test: If we quantize with q=3, and dequantize with 25*0.25=6.25,
    // the dequant value is 6.25/scale, but we quantized as 3/scale.
    // That's a factor of 2.0833x difference!

    println!("=== Scale Relationship Test ===\n");
    println!("If quantization: xval ≈ scale * q (where q = 2*l + 1)");
    println!("And dequantization: value = db * grid_value");
    println!("  where db = d * (0.5 + lval) * 0.25");
    println!();
    println!("From scale encoding: scale = d * (2*lval + 1)");
    println!("Therefore: db = scale * 0.125");
    println!();
    println!("So: value = scale * 0.125 * grid_value");
    println!("But quantization says: xval = scale * q");
    println!();
    println!("For these to match:");
    println!("  scale * 0.125 * grid_value = scale * q");
    println!("  grid_value = q / 0.125 = 8 * q");
    println!();
    println!("Check: q=1 → grid=8 ✓, q=3 → grid=24 ≠ 25, q=5 → grid=40 ≠ 43");
    println!();

    println!("=== The Problem ===\n");
    println!(
        "The dequantization grid has values 25 and 43 instead of 24 and 40!"
    );
    println!(
        "This causes a ~4% error in dequantization for L=1 and ~7.5% for L=2."
    );
    println!();

    // Calculate the actual error
    println!("Actual errors:");
    for l in 1..3 {
        let q = 2 * l + 1;
        let actual_grid = match l {
            1 => 25,
            2 => 43,
            _ => 0,
        };
        let expected_grid = 8 * q;
        let dequant_val = actual_grid as f32 * 0.125;
        let expected_val = q as f32;
        let error = (dequant_val - expected_val) / expected_val * 100.0;
        println!(
            "  l={}: dequant gives {:.3} instead of {}, error = {:.1}%",
            l, dequant_val, expected_val, error
        );
    }
    println!();

    println!("=== Solution ===\n");
    println!(
        "The quantization should use the ACTUAL dequantization grid values (8, 25, 43),"
    );
    println!("not the theoretical values (1, 3, 5).");
    println!();
    println!("In quantization: xval ≈ scale * grid_value * 0.125");
    println!("Instead of: xval ≈ scale * q");
    println!();
    println!("This means the L values should be extracted from the grid as:");
    println!("  l = round((grid_value * 0.125 - 1) / 2)");
    println!("Not: l = (q - 1) / 2 = (grid_value / 8 - 1) / 2");
}
