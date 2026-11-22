// Test to verify repack correctness

use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

fn create_test_block_q5_k() -> BlockQ5_K {
    let mut block: BlockQ5_K = unsafe { std::mem::zeroed() };
    // Set some test values
    block.d = 15360; // scale in f16 (approx 1.0)
    block.dmin = 14336; // min scale in f16 (approx 0.5)
    // Fill qs with test pattern
    for i in 0..128 {
        block.qs[i] = (i % 128) as u8;
    }
    // Fill qh with test pattern
    for i in 0..32 {
        block.qh[i] = (i % 32) as u8;
    }
    // Fill scales with test pattern
    for i in 0..12 {
        block.scales[i] = i as u8;
    }
    block
}

fn test_qh_interleaved_layout() {
    // Create 8 test blocks
    let mut blocks: [BlockQ5_K; 8] = unsafe { std::mem::zeroed() };
    for i in 0..8 {
        blocks[i] = create_test_block_q5_k();
        // Give each block a unique pattern (qh is 32 bytes per block)
        for j in 0..32 {
            blocks[i].qh[j] = ((i * 32 + j) % 256) as u8;
        }
    }

    // Reconstruct row i's qh bytes from interleaved layout
    // According to our analysis:
    // Row i's qh bytes 0-3 are at offset 4*i
    // Row i's qh bytes 4-7 are at offset 32 + 4*i
    // etc.

    // Expected for row 0: qh bytes should be [0,1,2,3], [4,5,6,7], ...
    // After repack with 4-byte interleaving:
    // dst[0..4] = row0 qh[0..4]
    // dst[4..8] = row1 qh[0..4]
    // ...
    // dst[32..36] = row0 qh[4..8]

    // So for row 0:
    // bytes 0-3 at offset 0
    // bytes 4-7 at offset 32
    // bytes 8-11 at offset 64
    // bytes 12-15 at offset 96

    // Let's simulate the repack and verify
    let mut qh_interleaved: [u8; 256] = [0; 256];
    for i in 0..64 {
        let src_id = i % 8; // which row
        let src_offset = (i / 8) * 4; // which 4-byte chunk
        let dst_offset = i * 4;
        qh_interleaved[dst_offset..dst_offset + 4]
            .copy_from_slice(&blocks[src_id].qh[src_offset..src_offset + 4]);
    }

    // Now verify row 0's qh reconstruction
    let mut row0_qh: [u8; 32] = [0; 32];
    for j in 0..8 {
        let src_offset = j * 32; // each 4-byte chunk is 32 bytes apart
        row0_qh[j * 4..j * 4 + 4]
            .copy_from_slice(&qh_interleaved[src_offset..src_offset + 4]);
    }

    // Expected: row 0's qh should be [0,1,2,3,4,5,6,7,...,31]
    for i in 0..32 {
        assert_eq!(
            row0_qh[i], i as u8,
            "Row 0 qh byte {} mismatch: expected {}, got {}",
            i, i, row0_qh[i]
        );
    }

    println!("Row 0 qh reconstruction: {:?}", &row0_qh[..]);
    println!("Test passed: qh interleaved layout is correct");
}

fn test_gemv_qh_loading() {
    // Test the qh loading logic in gemv_q5_k_x8_neon
    let mut blocks: [BlockQ5_K; 8] = unsafe { std::mem::zeroed() };
    for i in 0..8 {
        blocks[i] = create_test_block_q5_k();
        for j in 0..32 {
            blocks[i].qh[j] = (i * 32 + j) as u8;
        }
    }

    // Create interleaved buffer
    let mut qh_interleaved: [u8; 256] = [0; 256];
    for i in 0..64 {
        let src_id = i % 8;
        let src_offset = (i / 8) * 4;
        let dst_offset = i * 4;
        qh_interleaved[dst_offset..dst_offset + 4]
            .copy_from_slice(&blocks[src_id].qh[src_offset..src_offset + 4]);
    }

    // Now test the loading logic
    for i in 0..8 {
        let mut row_qh_0: [u8; 16] = [0; 16];
        let mut row_qh_1: [u8; 16] = [0; 16];

        for j in 0..4 {
            // c=0: bytes 0-3, 4-7, 8-11, 12-15 for row i
            row_qh_0[j * 4 + 0] = qh_interleaved[j * 32 + i * 4 + 0];
            row_qh_0[j * 4 + 1] = qh_interleaved[j * 32 + i * 4 + 1];
            row_qh_0[j * 4 + 2] = qh_interleaved[j * 32 + i * 4 + 2];
            row_qh_0[j * 4 + 3] = qh_interleaved[j * 32 + i * 4 + 3];

            // c=1: bytes 16-19, 20-23, 24-27, 28-31 for row i
            row_qh_1[j * 4 + 0] = qh_interleaved[(j + 4) * 32 + i * 4 + 0];
            row_qh_1[j * 4 + 1] = qh_interleaved[(j + 4) * 32 + i * 4 + 1];
            row_qh_1[j * 4 + 2] = qh_interleaved[(j + 4) * 32 + i * 4 + 2];
            row_qh_1[j * 4 + 3] = qh_interleaved[(j + 4) * 32 + i * 4 + 3];
        }

        // Expected: row i should have qh bytes [i*32, i*32+1, ..., i*32+31]
        // row_qh_0 should be [i*32, ..., i*32+15]
        // row_qh_1 should be [i*32+16, ..., i*32+31]
        for j in 0..16 {
            let expected = (i * 32 + j) as u8;
            assert_eq!(
                row_qh_0[j], expected,
                "Row {} qh_0 byte {} mismatch",
                i, j
            );
        }
        for j in 0..16 {
            let expected = (i * 32 + 16 + j) as u8;
            assert_eq!(
                row_qh_1[j], expected,
                "Row {} qh_1 byte {} mismatch",
                i, j
            );
        }

        println!(
            "Row {}: qh_0 = {:?}, qh_1 = {:?}",
            i,
            &row_qh_0[..],
            &row_qh_1[..]
        );
    }

    println!("Test passed: GEMV qh loading is correct");
}

fn main() {
    test_qh_interleaved_layout();
    test_gemv_qh_loading();
    println!("\nAll tests passed!");
}
