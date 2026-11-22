// Test to understand and verify interleaved layout

use infer_train::backend::arm::repack::repack_q5_k_to_x8;
use infer_train::types::{BlockQ5_K, BlockQ5_Kx8};

#[test]
fn test_interleaved_layout() {
    println!("=== Interleaved Layout Analysis ===\n");

    // Create 8 blocks with distinct patterns
    let mut blocks: Vec<BlockQ5_K> = Vec::with_capacity(8);
    for row in 0..8 {
        let mut block = BlockQ5_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [32u8; 12],
            qh: [0u8; 32],
            qs: [0u8; 128],
        };

        // Fill qs with a pattern that makes it easy to identify which row it belongs to
        // For row i, store value (i * 10 + j) in byte j
        for j in 0..128 {
            block.qs[j] = ((row * 10 + j) % 256) as u8;
        }

        // Fill qh similarly
        for j in 0..32 {
            block.qh[j] = ((row * 10 + j) % 256) as u8;
        }

        blocks.push(block);
    }

    // Repack
    let src_stride = 1 * std::mem::size_of::<BlockQ5_K>();
    let mut packed: Vec<u8> = vec![0u8; 1 * std::mem::size_of::<BlockQ5_Kx8>()];
    let src_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            8 * std::mem::size_of::<BlockQ5_K>(),
        )
    };
    repack_q5_k_to_x8(src_bytes, src_stride, &mut packed, 8, 1);

    let packed_block: &BlockQ5_Kx8 =
        unsafe { &*(packed.as_ptr() as *const BlockQ5_Kx8) };

    // Analyze qs layout
    println!("QS Layout Analysis (first 64 bytes):");
    for i in 0..16 {
        let offset = i * 4;
        let row_id = i % 8;
        let src_offset = (i / 8) * 4;
        print!("  qs[{:3}..{:3}] = ", offset, offset + 4);
        for j in 0..4 {
            print!("{:3} ", packed_block.qs[offset + j]);
        }
        println!("  <- row {} qs[{}..{}]", row_id, src_offset, src_offset + 4);
    }

    // Verify pattern
    println!("\nVerification:");
    for i in 0..16 {
        let offset = i * 4;
        let row_id = i % 8;
        let src_offset = (i / 8) * 4;

        let expected: [u8; 4] = std::array::from_fn(|j| {
            ((row_id * 10 + src_offset + j) % 256) as u8
        });
        let actual: [u8; 4] =
            std::array::from_fn(|j| packed_block.qs[offset + j]);

        if expected == actual {
            println!(
                "  qs[{}] matches row {} qs[{}..{}]",
                offset,
                row_id,
                src_offset,
                src_offset + 4
            );
        } else {
            println!(
                "  MISMATCH at qs[{}]: expected {:?}, got {:?}",
                offset, expected, actual
            );
        }
    }

    // For subblock 0, we need values 0..63 (64 values)
    // Each value uses low 4 bits, so we need 32 bytes per row
    // But in interleaved format, these 32 bytes are spread across 8 groups of 4 bytes
    println!("\n\nSubblock 0 (values 0..63) layout for each row:");
    for row in 0..8 {
        print!("  Row {}: qs bytes at offsets ", row);
        for group in 0..8 {
            let byte_offset = group * 32 + row * 4; // Each group is 32 bytes (8 rows * 4)
            print!("{}..{} ", byte_offset, byte_offset + 4);
        }
        println!();
    }

    // Now let's verify what we need for subblock 0, column group 0 (values 0..15)
    println!("\n\nSubblock 0, values 0..15 (first 16 values) for each row:");
    for row in 0..8 {
        // These are stored in the first group of interleaved data
        // Position in interleaved array: row * 4
        let byte_offset = row * 4;
        println!(
            "  Row {}: qs[{}..{}] = {:?}",
            row,
            byte_offset,
            byte_offset + 4,
            &packed_block.qs[byte_offset..byte_offset + 4]
        );
    }

    // Similarly for qh
    println!("\n\nQH Layout Analysis (first 32 bytes):");
    for i in 0..8 {
        let offset = i * 4;
        let row_id = i % 8;
        let src_offset = (i / 8) * 4;
        print!("  qh[{:3}..{:3}] = ", offset, offset + 4);
        for j in 0..4 {
            print!("{:3} ", packed_block.qh[offset + j]);
        }
        println!("  <- row {} qh[{}..{}]", row_id, src_offset, src_offset + 4);
    }
}
