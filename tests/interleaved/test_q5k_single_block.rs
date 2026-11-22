//! Debug test for Q5_K repack and GEMV
use infer_train::backend::arm::{gemv, repack, vec_dot};
use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

#[test]
fn test_single_block_debug() {
    // Create 8 identical BlockQ5_K for simplicity
    let create_block = |row: usize| -> BlockQ5_K {
        let mut block: BlockQ5_K = unsafe { std::mem::zeroed() };
        block.d =
            half::f16::to_bits(half::f16::from_f32(1.0 + row as f32 * 0.1));
        block.dmin =
            half::f16::to_bits(half::f16::from_f32(0.1 + row as f32 * 0.01));

        // Set scales: all 1s
        for i in 0..12 {
            block.scales[i] = 1;
        }

        // Set qs: simple pattern - each byte = row number
        for i in 0..128 {
            block.qs[i] = row as u8;
        }

        // Set qh: all zeros
        for i in 0..32 {
            block.qh[i] = 0;
        }

        block
    };

    let blocks: Vec<BlockQ5_K> = (0..8).map(create_block).collect();

    // Create Q8_K input: all ones
    let mut q8: BlockQ8_K = unsafe { std::mem::zeroed() };
    q8.d = 1.0;
    for i in 0..256 {
        q8.qs[i] = 1;
    }
    for s in 0..16 {
        q8.bsums[s] = 16;
    }

    // Repack to BlockQ5_Kx8
    let stride = std::mem::size_of::<BlockQ5_K>();
    let weight_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, 8 * stride)
            .to_vec()
    };

    let size_x8 = std::mem::size_of::<BlockQ5_Kx8>();
    let mut repacked: Vec<u8> = vec![0u8; size_x8];
    unsafe {
        repack::repack_q5_k_to_x8(&weight_bytes, stride, &mut repacked, 8, 1);
    }

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(repacked.as_ptr() as *const BlockQ5_Kx8, 1)
    };

    // Test GEMV
    let mut output_gemv = vec![0.0f32; 8];
    unsafe {
        gemv::gemv_q5_k_x8_neon(weights_x8, &[q8], &mut output_gemv, 8, QK_K);
    }

    println!("vec_dot results:");
    for row in 0..8 {
        let result =
            unsafe { vec_dot::vec_dot_q5_k_q8_k(&[blocks[row]], &[q8]) };
        println!("  row {}: {:.2}", row, result);
    }

    println!("\nGEMV results:");
    for row in 0..8 {
        println!("  row {}: {:.2}", row, output_gemv[row]);
    }

    // Manual calculation:
    // For row j with qs = j, qh = 0:
    // Each qs byte = j, low nibble = j, high nibble = 0 (since j < 8)
    // Each subblock (64 elements): 32 qs bytes -> 32 low nibbles * j + 32 high nibbles * 0
    // Sum of low nibbles per subblock = 32 * j
    // Sum of high nibbles per subblock = 0
    // Total sum per subblock = (32 * j) * scale_0 + 0 * scale_1
    // With scale_0 = scale_1 = 1: sumi = 32 * j
    // With d = 1 + j*0.1, q8.d = 1: sumf = sumi * (1 + j*0.1) * 1
    // For 4 subblocks: sumf = 4 * 32 * j * (1 + j*0.1) = 128 * j * (1 + j*0.1)
    // For row 0: 128 * 0 * 1 = 0
    // For row 1: 128 * 1 * 1.1 = 140.8
    // For row 7: 128 * 7 * 1.7 = 1523.2

    // But vec_dot shows different values, so either my understanding is wrong
    // or vec_dot has a different calculation

    println!("\nExpected (manual):");
    for j in 0..8 {
        let expected = 128.0 * j as f32 * (1.0 + j as f32 * 0.1);
        println!("  row {}: {:.2}", j, expected);
    }
}
