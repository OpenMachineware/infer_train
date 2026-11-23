// Verify BlockQ5_K memory layout matches llama.cpp
use infer_train::types::BlockQ5_K;

fn main() {
    println!("BlockQ5_K size: {} bytes", std::mem::size_of::<BlockQ5_K>());
    println!(
        "BlockQ5_K alignment: {} bytes",
        std::mem::align_of::<BlockQ5_K>()
    );

    // Check field offsets
    let test_block: BlockQ5_K = unsafe { std::mem::zeroed() };
    let base_ptr = &test_block as *const BlockQ5_K as usize;

    let d_offset = &test_block.d as *const _ as usize - base_ptr;
    let dmin_offset = &test_block.dmin as *const _ as usize - base_ptr;
    let scales_offset = &test_block.scales as *const _ as usize - base_ptr;
    let qh_offset = &test_block.qh as *const _ as usize - base_ptr;
    let qs_offset = &test_block.qs as *const _ as usize - base_ptr;

    println!("\nField offsets:");
    println!(
        "  d: offset {} (size {})",
        d_offset,
        std::mem::size_of_val(&test_block.d)
    );
    println!(
        "  dmin: offset {} (size {})",
        dmin_offset,
        std::mem::size_of_val(&test_block.dmin)
    );
    println!(
        "  scales: offset {} (size {})",
        scales_offset,
        std::mem::size_of_val(&test_block.scales)
    );
    println!(
        "  qh: offset {} (size {})",
        qh_offset,
        std::mem::size_of_val(&test_block.qh)
    );
    println!(
        "  qs: offset {} (size {})",
        qs_offset,
        std::mem::size_of_val(&test_block.qs)
    );

    // According to llama.cpp (ggml-common.h), the layout should be:
    // offset 0: d (ggml_half, 2 bytes)
    // offset 2: dmin (ggml_half, 2 bytes)
    // offset 4: scales (12 bytes)
    // offset 16: qh (32 bytes)
    // offset 48: qs (128 bytes)
    // Total: 2 + 2 + 12 + 32 + 128 = 176 bytes

    println!("\nExpected layout (llama.cpp):");
    println!("  d: offset 0 (2 bytes)");
    println!("  dmin: offset 2 (2 bytes)");
    println!("  scales: offset 4 (12 bytes)");
    println!("  qh: offset 16 (32 bytes)");
    println!("  qs: offset 48 (128 bytes)");

    println!("\nCurrent layout (our code):");
    println!("  d: offset {}", d_offset);
    println!("  dmin: offset {}", dmin_offset);
    println!("  scales: offset {}", scales_offset);
    println!("  qh: offset {}", qh_offset);
    println!("  qs: offset {}", qs_offset);

    if d_offset == 0 && dmin_offset == 2 && scales_offset == 4 {
        println!("\n✓ Layout matches llama.cpp");
    } else {
        println!("\n✗ Layout DOES NOT MATCH llama.cpp!");
    }
}
