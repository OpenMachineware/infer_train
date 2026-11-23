// Verify BlockQ6_K memory layout matches llama.cpp
use infer_train::types::BlockQ6_K;

fn main() {
    println!("BlockQ6_K size: {} bytes", std::mem::size_of::<BlockQ6_K>());
    println!(
        "BlockQ6_K alignment: {} bytes",
        std::mem::align_of::<BlockQ6_K>()
    );

    // Check field offsets
    let test_block: BlockQ6_K = unsafe { std::mem::zeroed() };
    let base_ptr = &test_block as *const BlockQ6_K as usize;

    let ql_offset = &test_block.ql as *const _ as usize - base_ptr;
    let qh_offset = &test_block.qh as *const _ as usize - base_ptr;
    let scales_offset = &test_block.scales as *const _ as usize - base_ptr;
    let d_offset = &test_block.d as *const _ as usize - base_ptr;

    println!("\nField offsets:");
    println!(
        "  ql: offset {} (size {})",
        ql_offset,
        std::mem::size_of_val(&test_block.ql)
    );
    println!(
        "  qh: offset {} (size {})",
        qh_offset,
        std::mem::size_of_val(&test_block.qh)
    );
    println!(
        "  scales: offset {} (size {})",
        scales_offset,
        std::mem::size_of_val(&test_block.scales)
    );
    println!(
        "  d: offset {} (size {})",
        d_offset,
        std::mem::size_of_val(&test_block.d)
    );

    // According to llama.cpp (ggml-common.h), the layout is:
    // offset 0: ql (128 x u8, 128 bytes)
    // offset 128: qh (64 x u8, 64 bytes)
    // offset 192: scales (16 x i8, 16 bytes)
    // offset 208: d (ggml_half, 2 bytes)
    // Total: 128 + 64 + 16 + 2 = 210 bytes

    println!("\nExpected layout (llama.cpp):");
    println!("  ql: offset 0 (128 bytes)");
    println!("  qh: offset 128 (64 bytes)");
    println!("  scales: offset 192 (16 bytes)");
    println!("  d: offset 208 (2 bytes)");

    let correct = ql_offset == 0
        && qh_offset == 128
        && scales_offset == 192
        && d_offset == 208;

    if correct {
        println!("\n✓ Layout matches llama.cpp");
    } else {
        println!("\n✗ Layout DOES NOT MATCH llama.cpp!");
    }
}
