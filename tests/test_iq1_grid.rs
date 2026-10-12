// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test IQ1_S grid construction

use infer_train::quant::weights::iq_grid::*;

fn main() {
    println!("NGRID_IQ1S = {}", NGRID_IQ1S);
    println!("KGRID_1BIT_2048 length = {}", KGRID_1BIT_2048.len());

    // Test creating grid data
    let grid_data = IQ2GridData::new(NGRID_IQ1S, &KGRID_1BIT_2048, 3);
    println!("Grid size: {}", grid_data.grid.len());
    println!("Map size: {}", grid_data.map.len());
    println!("Neighbors size: {}", grid_data.neighbors.len());

    // Check some map values
    let mut on_grid = 0;
    let mut off_grid = 0;
    for &m in &grid_data.map {
        if m >= 0 {
            on_grid += 1;
        } else {
            off_grid += 1;
        }
    }
    println!("On-grid entries: {}", on_grid);
    println!("Off-grid entries: {}", off_grid);

    // Sample first few neighbor entries
    let mut idx = 0;
    for &m in &grid_data.map {
        if m < 0 {
            let neighbor_idx = (-m - 1) as usize;
            let num_neighbors = grid_data.neighbors[neighbor_idx] as usize;
            println!(
                "Index {}: {} neighbors at offset {}",
                idx, num_neighbors, neighbor_idx
            );
            if num_neighbors > 0 {
                print!("  Neighbors: ");
                for j in 0..num_neighbors.min(5) {
                    let n = grid_data.neighbors[neighbor_idx + 1 + j];
                    print!("{} ", n);
                }
                println!();
            }
            break;
        }
        idx += 1;
    }
}
