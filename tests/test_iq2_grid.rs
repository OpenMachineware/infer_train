// Test IQ2_XXS grid construction

use infer_train::quant::weights::iq_grid::*;

fn main() {
    println!("=== Testing IQ2_XXS Grid ===");
    let grid_data = IQ2GridData::new(256, &KGRID_2BIT_256, 2);
    println!("Grid size: {}", grid_data.grid.len());
    println!("Neighbors size: {}", grid_data.neighbors.len());

    // Count on-grid vs off-grid
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

    // Sample neighbor entries
    let mut sample_count = 0;
    for (idx, &m) in grid_data.map.iter().enumerate() {
        if m < 0 && sample_count < 3 {
            let neighbor_idx = (-m - 1) as usize;
            if neighbor_idx < grid_data.neighbors.len() {
                let num_neighbors = grid_data.neighbors[neighbor_idx] as usize;
                println!(
                    "\nMap[{}] = {} -> neighbor_idx = {}",
                    idx, m, neighbor_idx
                );
                println!(
                    "  Count at neighbors[{}] = {}",
                    neighbor_idx, num_neighbors
                );

                // Check how many indices are available
                let available = grid_data.neighbors.len() - neighbor_idx - 1;
                println!("  Available indices after count: {}", available);

                if num_neighbors > 0 && num_neighbors <= available.min(20) {
                    print!("  First few indices: ");
                    for j in 0..num_neighbors.min(5) {
                        print!(
                            "{} ",
                            grid_data.neighbors[neighbor_idx + 1 + j]
                        );
                    }
                    println!();
                }
            }
            sample_count += 1;
        }
    }

    // Test a specific u16 value
    let test_u: u16 = 1; // Small value, likely off-grid
    let map_val = grid_data.map[test_u as usize];
    println!("\n=== Testing map[{}] = {} ===", test_u, map_val);
    if map_val < 0 {
        let neighbor_idx = (-map_val - 1) as usize;
        println!("neighbor_idx = {}", neighbor_idx);
        if neighbor_idx < grid_data.neighbors.len() {
            let num_neighbors = grid_data.neighbors[neighbor_idx] as usize;
            println!("num_neighbors = {}", num_neighbors);
            let available = grid_data.neighbors.len() - neighbor_idx;
            println!("Slice length from neighbor_idx = {}", available);
        }
    }
}
