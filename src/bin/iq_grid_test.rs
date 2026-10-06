use infer_train::quant::weights::iq_grid::{
    nearest_int, IQ2GridData, KGRID_2BIT_256,
};

fn main() {
    println!("=== Testing IQ2 Grid Data ===\n");

    let grid_data = IQ2GridData::new(256, &KGRID_2BIT_256, 2);

    // Test 1: Verify grid values
    println!("Test 1: Grid values for first few entries");
    for i in 0..5 {
        let bytes = grid_data.grid[i].to_le_bytes();
        print!("  grid[{}] = {:016x} -> bytes: [", i, grid_data.grid[i]);
        for (j, b) in bytes.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{}", b);
        }
        println!("]");
    }

    // Expected: grid[0] should have all bytes = 1 (for l=0: 2*0+1=1)
    // grid[1] should have first byte = 5 (for l=2: 2*2+1=5), rest = 1

    // Test 2: Verify map
    println!("\nTest 2: Map for known patterns");

    // Pattern u=0: all l=0 -> should map to grid[0]
    println!("  map[0] = {} (expected 0)", grid_data.map[0]);

    // Pattern u=2: l[0]=2, l[1..7]=0 -> should map to grid[1] (kgrid[1]=2)
    println!("  map[2] = {} (expected 1)", grid_data.map[2]);

    // Pattern u=5: l[0]=1, l[1]=1, rest=0 -> should map to grid[2] (kgrid[2]=5)
    println!("  map[5] = {} (expected 2)", grid_data.map[5]);

    // Test 3: Off-grid pattern
    println!("\nTest 3: Off-grid pattern");
    // u=1: l[0]=1, rest=0 -> should be off-grid (not in kgrid)
    println!(
        "  map[1] = {} (should be negative, indicating off-grid)",
        grid_data.map[1]
    );

    // Test 4: Find best neighbor for off-grid pattern
    println!("\nTest 4: Find best neighbor");
    let xval = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let weight = [1.0; 8];
    let mut l = [0i8; 8];
    let result = grid_data.find_best_neighbor(1, &xval, &weight, 1.0, &mut l);
    println!("  find_best_neighbor(1, ...) = {}", result);
    println!("  l values: {:?}", l);

    println!("\nDone.");
}
