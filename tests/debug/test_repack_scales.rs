// Test Q5_K scales encoding/decoding

use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};
use infer_train::backend::arm::repack::repack_q5_k_to_x8;
use infer_train::backend::arm::vec_dot::vec_dot_q5_k_q8_k;
use infer_train::backend::arm::gemv::gemv_q5_k_x8_neon;

fn main() {
    // Create 8 Q5_K blocks with known scales
    let mut src_blocks: [BlockQ5_K; 8] = unsafe { std::mem::zeroed() };

    // Set known scale values
    for i in 0..8 {
        src_blocks[i].d = half::f16::to_bits(half::f16::from_f32(1.0));
        src_blocks[i].dmin = half::f16::to_bits(half::f16::from_f32(0.1));

        // Set scales - each block has 12 bytes encoding 8 scales + 8 mins
        // Use different values to test all bits
        for j in 0..12 {
            src_blocks[i].scales[j] = (j as u8 + i as u8 * 12) % 64;
        }

        // Set qs to 1 (so q5 value is 1 after decode)
        src_blocks[i].qs.fill(0x11); // Low 4 bits = 1, high 4 bits = 1

        // Set qh to 0 (so high bit is 0, q5 value = 1)
        src_blocks[i].qh.fill(0);
    }

    println!("=== Original scales ===");
    for i in 0..8 {
        println!("Block {} scales: {:?}", i, &src_blocks[i].scales);
    }

    // Repack to x8 format
    let src_bytes: Vec<u8> = src_blocks.iter()
        .flat_map(|b| unsafe {
            std::slice::from_raw_parts(b as *const _ as *const u8, std::mem::size_of::<BlockQ5_K>()).iter().copied()
        })
        .collect();

    let mut dst_bytes = vec![0u8; std::mem::size_of::<BlockQ5_Kx8>()];

    repack_q5_k_to_x8(&src_bytes, std::mem::size_of::<BlockQ5_K>(), &mut dst_bytes, 8, 1);

    let dst_block: &BlockQ5_Kx8 = unsafe {
        &*(dst_bytes.as_ptr() as *const BlockQ5_Kx8)
    };

    println!("\n=== Repacked scales (96 bytes) ===");
    println!("First 48 bytes: {:?}", &dst_block.scales[..48]);
    println!("Last 48 bytes: {:?}", &dst_block.scales[48..]);

    // Decode scales using the GEMV decode function
    println!("\n=== Decoded scales ===");
    for sb in 0..8 {
        let scales_in = &dst_block.scales[sb * 12..];

        // Decode from 12 bytes to 8 scales + 8 mins
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut sm: [u32; 3] = [0; 3];
        unsafe {
            std::ptr::copy_nonoverlapping(
                scales_in.as_ptr(),
                sm.as_mut_ptr() as *mut u8,
                12,
            );
        }

        let scales_0_3 = sm[0] & KMASK1;
        let scales_4_7 = (sm[2] & KMASK2) | (((sm[0] >> 6) & KMASK3) << 4);
        let mins_0_3 = sm[1] & KMASK1;
        let mins_4_7 = ((sm[2] >> 4) & KMASK2) | (((sm[1] >> 6) & KMASK3) << 4);

        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];

        for (i, b) in scales_0_3.to_le_bytes()[..4].iter().enumerate() {
            scales[i] = *b;
        }
        for (i, b) in scales_4_7.to_le_bytes()[..4].iter().enumerate() {
            scales[4 + i] = *b;
        }
        for (i, b) in mins_0_3.to_le_bytes()[..4].iter().enumerate() {
            mins[i] = *b;
        }
        for (i, b) in mins_4_7.to_le_bytes()[..4].iter().enumerate() {
            mins[4 + i] = *b;
        }

        println!("Subblock {}: scales={:?}, mins={:?}", sb, scales, mins);
    }

    // Test vec_dot equivalence
    println!("\n=== Vec_dot comparison ===");

    // Vec_dot: 8 blocks of Q5_K dot 8 blocks of Q8_K = single value
    // GEMV: 8 rows (each row = 1 block of 256 elements) dot 1 block of Q8_K = 8 values

    // For fair comparison, we need:
    // - vec_dot: 1 block Q5_K dot 1 block Q8_K
    // - GEMV: 8 rows (interleaved as 1 BlockQ5_Kx8) dot 1 block Q8_K

    // Create 8 separate Q5_K blocks (each row has 1 block = 256 elements)
    let mut row_blocks: Vec<BlockQ5_K> = vec![unsafe { std::mem::zeroed() }; 8];
    for (i, blk) in row_blocks.iter_mut().enumerate() {
        blk.d = half::f16::to_bits(half::f16::from_f32(1.0 + i as f32 * 0.1));
        blk.dmin = half::f16::to_bits(half::f16::from_f32(0.1));

        // Set scales to simple values
        for j in 0..12 {
            blk.scales[j] = 1; // All scales = 1
        }

        // Set qs to 1
        blk.qs.fill(0x11);
        blk.qh.fill(0);
    }

    // Create single Q8_K block
    let mut q8_block: BlockQ8_K = unsafe { std::mem::zeroed() };
    q8_block.d = 1.0;
    q8_block.qs.fill(1);

    // Compute vec_dot for each row
    println!("=== Vec_dot per row ===");
    let mut vec_dot_results = [0.0f32; 8];
    for i in 0..8 {
        vec_dot_results[i] = vec_dot_q5_k_q8_k(&[row_blocks[i]], &[q8_block]);
        println!("Row {}: {}", i, vec_dot_results[i]);
    }

    // Repack 8 rows to interleaved format
    let src_bytes: Vec<u8> = row_blocks.iter()
        .flat_map(|b| unsafe {
            std::slice::from_raw_parts(b as *const _ as *const u8, std::mem::size_of::<BlockQ5_K>()).iter().copied()
        })
        .collect();

    let mut dst_bytes = vec![0u8; std::mem::size_of::<BlockQ5_Kx8>()];
    repack_q5_k_to_x8(&src_bytes, std::mem::size_of::<BlockQ5_K>(), &mut dst_bytes, 8, 1);

    // Compute interleaved GEMV
    println!("\n=== Interleaved GEMV ===");
    let mut gemv_results = [0.0f32; 8];
    unsafe {
        // Create slice of BlockQ5_Kx8 (1 row group × 1 block)
        let weights_x8: &[BlockQ5_Kx8] = std::slice::from_raw_parts(
            dst_bytes.as_ptr() as *const BlockQ5_Kx8,
            1
        );
        gemv_q5_k_x8_neon(weights_x8, &[q8_block], &mut gemv_results, 8, 256);
    }
    println!("GEMV results: {:?}", gemv_results);

    // Compare
    println!("\n=== Comparison ===");
    for i in 0..8 {
        let diff = (vec_dot_results[i] - gemv_results[i]).abs();
        println!("Row {}: vec_dot={:.2}, gemv={:.2}, diff={:.2}",
                 i, vec_dot_results[i], gemv_results[i], diff);
    }
}
