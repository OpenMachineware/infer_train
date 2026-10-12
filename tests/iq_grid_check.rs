// Check if KGRID_2BIT_256 covers all possible l-value patterns

use infer_train::quant::weights::iq_grid::KGRID_2BIT_256;

fn main() {
    println!("=== Checking KGRID_2BIT_256 coverage ===\n");

    // Build a set of all kgrid values
    let mut kset = vec![false; 65536];
    let mut max_val = 0u16;
    for &v in &KGRID_2BIT_256 {
        kset[v as usize] = true;
        max_val = max_val.max(v);
    }

    println!("Grid size: {}", KGRID_2BIT_256.len());
    println!("Max kgrid value: {}", max_val);

    // Check if the grid covers all possible l-value patterns
    // For 8 2-bit values, there are 3^8 = 6561 possible patterns
    // But KGRID_2BIT_256 has only 256 entries

    // Count off-grid patterns
    let mut on_grid = 0;
    let mut off_grid = 0;
    for u in 0..65536 {
        // Check if u encodes a valid l-value pattern (all l values in 0..3)
        let mut valid = true;
        for i in 0..8 {
            let l = (u >> (2 * i)) & 0x3;
            if l > 2 {
                valid = false;
                break;
            }
        }
        if valid {
            if kset[u as usize] {
                on_grid += 1;
            } else {
                off_grid += 1;
            }
        }
    }

    println!("\nValid l-value patterns (l in 0..2):");
    println!("  On-grid: {}", on_grid);
    println!("  Off-grid: {}", off_grid);

    // Show some example off-grid patterns
    println!("\nExample off-grid patterns:");
    let mut count = 0;
    for u in 0..=65535u16 {
        let mut valid = true;
        for i in 0..8 {
            let l = (u >> (2 * i)) & 0x3;
            if l > 2 {
                valid = false;
                break;
            }
        }
        if valid && !kset[u as usize] && count < 10 {
            println!("  u={}", u);
            // Show l values
            let mut lvals = String::new();
            for i in 0..8 {
                let l = (u >> (2 * i)) & 0x3;
                lvals.push_str(&format!("{} ", l));
            }
            println!("    l values: {}", lvals);
            count += 1;
        }
    }

    // Check specific values from the error
    println!("\nChecking problematic values:");
    for u in [52479u16, 62963u16, 63475u16, 53107u16] {
        let mut valid = true;
        for i in 0..8 {
            let l = (u >> (2 * i)) & 0x3;
            if l > 2 {
                valid = false;
                break;
            }
        }
        println!("  u={}: valid={}, in_grid={}", u, valid, kset[u as usize]);
    }
}
