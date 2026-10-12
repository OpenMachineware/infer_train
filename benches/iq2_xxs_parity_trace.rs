// Trace sign encoding for each group to find the parity bug
// The issue: Groups 1-3 have s with even parity but nflip=1

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use infer_train::quant::weights::iq_grid::{
    IQ2GridData, IQ2XXS_GRID, KGRID_2BIT_256,
};

fn main() {
    println!("=== Parity Bug Trace ===\n");

    // Test data
    let test_data: Vec<f32> = (0..256)
        .map(|i| if i % 7 == 0 { -1.0 } else { (i as f32) / 100.0 })
        .collect();

    // First sub-block (indices 0-31)
    let xb = &test_data[0..32];

    // Check each group
    for k in 0..4 {
        println!("Group {} (indices {}-{}):", k, k * 8, k * 8 + 7);

        // Input values
        print!("  Input: ");
        for i in 0..8 {
            print!("{:.2} ", xb[8 * k + i]);
        }
        println!();

        // Count negatives and build s
        let mut nflip = 0;
        let mut s: u8 = 0;
        for i in 0..8 {
            if xb[8 * k + i] < 0.0 {
                nflip += 1;
                s |= 1 << i;
            }
        }

        println!("  nflip = {}", nflip);
        println!("  s (before parity) = {} ({:08b})", s, s);

        // Check parity
        let parity = (0..8).filter(|&i| (s >> i) & 1 == 1).count() % 2;
        println!(
            "  s parity = {} ({})",
            parity,
            if parity == 0 { "even" } else { "odd" }
        );
        println!("  nflip parity = {}", nflip % 2);

        // This is the key: the parity check should be based on nflip, not s!
        // nflip counts the number of negative values
        // s records which bits are set
        // They should have the same parity!

        if nflip % 2 != 0 {
            // Need to flip one bit
            println!("  ODD nflip: need to flip one bit");
            // Find minimum weight element to flip
            let min_idx = find_min_idx(&xb[8 * k..8 * k + 8]);
            s ^= 1 << min_idx;
            println!(
                "  s (after flipping bit {}) = {} ({:08b})",
                min_idx, s, s
            );
        }

        let stored = s & 127;
        println!("  Stored 7-bit index = {} ({:07b})", stored, stored);
        println!();
    }
}

fn find_min_idx(x: &[f32]) -> usize {
    let mut min_idx = 0;
    let mut min_val = x[0].abs();
    for i in 1..8 {
        let ax = x[i].abs();
        if ax < min_val {
            min_val = ax;
            min_idx = i;
        }
    }
    min_idx
}
