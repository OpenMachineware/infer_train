// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Weight repacking for interleaved GEMV optimization
// Transforms row-interleaved weights into block-interleaved format

use crate::types::{
    BlockQ4_0, BlockQ4_0x4, BlockQ4_0x8, BlockQ4_K, BlockQ4_Kx8, BlockQ5_K,
    BlockQ5_Kx8, BlockQ6_K, BlockQ6_Kx8, QK_K, QK4_0,
};

/// Repack Q4_0 weights to 4-row interleaved format
/// Input: row-major weights (stride between rows)
/// Output: block_q4_0x4 format (4 rows packed together)
///
/// Layout follows llama.cpp's make_block_q4_0x4 with blck_size_interleave=4
/// XOR mask 0x88888888 converts nibbles from bias offset form to pure sign form
pub fn repack_q4_0_to_x4(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    n_rows: usize,
    n_blocks_per_row: usize,
) {
    assert!(n_rows % 4 == 0, "n_rows must be multiple of 4");

    let size_block_q4_0 = std::mem::size_of::<BlockQ4_0>();
    let size_block_q4_0x4 = std::mem::size_of::<BlockQ4_0x4>();
    const BLCK_SIZE: usize = 4; // blck_size_interleave

    // Process 4 rows at a time
    for row_group in 0..(n_rows / 4) {
        let base_row = row_group * 4;

        // Process each block column
        for blk in 0..n_blocks_per_row {
            let dst_blk = row_group * n_blocks_per_row + blk;
            let dst_offset = dst_blk * size_block_q4_0x4;
            let dst_block = unsafe {
                &mut *(dst.as_mut_ptr().add(dst_offset) as *mut BlockQ4_0x4)
            };

            // Load 4 source blocks
            let mut src_blocks: [BlockQ4_0; 4] = unsafe { std::mem::zeroed() };
            for i in 0..4 {
                let src_row = base_row + i;
                let src_offset = src_row * src_stride + blk * size_block_q4_0;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset),
                        &mut src_blocks[i] as *mut BlockQ4_0 as *mut u8,
                        size_block_q4_0,
                    );
                }
            }

            // Pack scales
            for i in 0..4 {
                dst_block.d[i] = src_blocks[i].d;
            }

            // Interleave qs: 4 bytes at a time from each block
            // end = QK4_0 * 2 / BLCK_SIZE = 32 * 2 / 4 = 16 iterations
            // XOR with 0x88888888 to convert from bias offset to pure sign form
            const XOR_MASK: u32 = 0x88888888;
            for i in 0..16 {
                let src_id = i % 4;
                let src_offset = (i / 4) * BLCK_SIZE;
                let dst_offset = i * BLCK_SIZE;

                let mut chunk: u32 = 0;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_blocks[src_id].qs.as_ptr().add(src_offset),
                        &mut chunk as *mut u32 as *mut u8,
                        BLCK_SIZE,
                    );
                }
                chunk ^= XOR_MASK;
                dst_block.qs[dst_offset..dst_offset + BLCK_SIZE]
                    .copy_from_slice(&chunk.to_le_bytes());
            }
        }
    }
}

/// Repack Q4_0 weights to 8-row interleaved format
/// Input: row-major weights (stride between rows)
/// Output: block_q4_0x8 format (8 rows packed together)
///
/// Layout follows llama.cpp's make_block_q4_0x8 with blck_size_interleave=8
/// XOR mask is applied to convert nibbles from bias offset form to signed form
pub fn repack_q4_0_to_x8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    n_rows: usize,
    n_blocks_per_row: usize,
) {
    assert!(n_rows % 8 == 0, "n_rows must be multiple of 8");

    let size_block_q4_0 = std::mem::size_of::<BlockQ4_0>();
    let size_block_q4_0x8 = std::mem::size_of::<BlockQ4_0x8>();
    const BLCK_SIZE: usize = 8; // blck_size_interleave

    // Process 8 rows at a time
    for row_group in 0..(n_rows / 8) {
        let base_row = row_group * 8;

        for blk in 0..n_blocks_per_row {
            let dst_blk = row_group * n_blocks_per_row + blk;
            let dst_offset = dst_blk * size_block_q4_0x8;
            let dst_block = unsafe {
                &mut *(dst.as_mut_ptr().add(dst_offset) as *mut BlockQ4_0x8)
            };

            // Load 8 source blocks
            let mut src_blocks: [BlockQ4_0; 8] = unsafe { std::mem::zeroed() };
            for i in 0..8 {
                let src_row = base_row + i;
                let src_offset = src_row * src_stride + blk * size_block_q4_0;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset),
                        &mut src_blocks[i] as *mut BlockQ4_0 as *mut u8,
                        size_block_q4_0,
                    );
                }
            }

            // Pack scales
            for i in 0..8 {
                dst_block.d[i] = src_blocks[i].d;
            }

            // Interleave qs: 8 bytes at a time from each block
            // XOR mask 0x8888888888888888 is applied to the whole byte
            // This converts nibbles from bias offset to signed form
            // After XOR:
            // - Low nibble: value becomes (A ^ 8) which after <<4 as i8 gives A-8
            // - High nibble: value becomes (B ^ 8) which when extracted as i8 gives B-8
            const XOR_MASK: u64 = 0x8888888888888888;
            for i in 0..16 {
                let src_id = i % 8;
                let src_offset = (i / 8) * BLCK_SIZE;
                let dst_offset = i * BLCK_SIZE;

                let mut chunk: u64 = 0;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_blocks[src_id].qs.as_ptr().add(src_offset),
                        &mut chunk as *mut u64 as *mut u8,
                        BLCK_SIZE,
                    );
                }
                chunk ^= XOR_MASK;
                dst_block.qs[dst_offset..dst_offset + BLCK_SIZE]
                    .copy_from_slice(&chunk.to_le_bytes());
            }
        }
    }
}

/// Repack Q5_K weights from interleaved to block-interleaved format
/// Input: row-interleaved weights (stride between rows)
/// Output: block_q5_Kx8 format (8 rows packed together)
///
/// Layout follows llama.cpp's make_block_q5_Kx8 with blck_size_interleave=4
pub fn repack_q5_k_to_x8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    n_rows: usize,
    n_blocks_per_row: usize,
) {
    assert!(n_rows % 8 == 0, "n_rows must be multiple of 8");

    let size_block_q5_k = std::mem::size_of::<BlockQ5_K>();
    let size_block_q5_kx8 = std::mem::size_of::<BlockQ5_Kx8>();
    const BLCK_SIZE: usize = 4; // blck_size_interleave for 8x4 interleaved GEMV

    // Detect input layout from stride
    // - GGUF factor-4 interleaved: stride = 4 * size_block_q5_k (704 for Q5_K)
    // - Standard row-major: stride = n_blocks_per_row * size_block_q5_k
    // - Factor-8 interleaved: stride = 8 * size_block_q5_k (already target format)
    let interleave_factor = src_stride / size_block_q5_k;
    let is_gguf_interleaved = interleave_factor == 4 && n_blocks_per_row > 4;

    // Process 8 rows at a time
    for row_group in 0..(n_rows / 8) {
        let base_row = row_group * 8;

        // Process each block column
        for blk in 0..n_blocks_per_row {
            let dst_blk = row_group * n_blocks_per_row + blk;
            let dst_offset = dst_blk * size_block_q5_kx8;
            let dst_block = unsafe {
                &mut *(dst.as_mut_ptr().add(dst_offset) as *mut BlockQ5_Kx8)
            };

            // Load 8 source blocks
            let mut src_blocks: [BlockQ5_K; 8] = unsafe { std::mem::zeroed() };
            for i in 0..8 {
                let src_row = base_row + i;

                let src_offset = if is_gguf_interleaved {
                    // GGUF factor-4 interleaved layout:
                    // - Data stored in 4-row chunks
                    // - Each chunk has 2 halves:
                    //   - First half (4*4 blocks): blocks 0-3 for 4 rows
                    //   - Second half (4*4 blocks): blocks 4-7 for 4 rows
                    // - Chunk size = 4 rows * 8 blocks * block_size / 2 = 5632 bytes
                    //   (stored as 2 consecutive 2816-byte halves)
                    let chunk_idx = src_row / 4;
                    let row_in_chunk = src_row % 4;
                    let block_half = blk / 4;
                    let block_in_half = blk % 4;

                    let chunk_size = 4 * 4 * size_block_q5_k * 2; // 5632 bytes
                    let half_size = 4 * 4 * size_block_q5_k; // 2816 bytes

                    chunk_idx * chunk_size
                        + block_half * half_size
                        + row_in_chunk * 4 * size_block_q5_k
                        + block_in_half * size_block_q5_k
                } else {
                    // Standard row-major format
                    src_row * src_stride + blk * size_block_q5_k
                };

                // Bounds check
                if src_offset + size_block_q5_k > src.len() {
                    continue;
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset),
                        &mut src_blocks[i] as *mut BlockQ5_K as *mut u8,
                        size_block_q5_k,
                    );
                }
            }

            // Pack d and dmin
            for i in 0..8 {
                dst_block.d[i] = src_blocks[i].d;
                dst_block.dmin[i] = src_blocks[i].dmin;
            }

            // Interleave qs: 4 bytes at a time from each block
            // end = QK_K * 4 / BLCK_SIZE = 256 * 4 / 4 = 256
            for i in 0..256 {
                let src_id = i % 8;
                let src_qs_offset = (i / 8) * BLCK_SIZE;
                let dst_qs_offset = i * BLCK_SIZE;
                dst_block.qs[dst_qs_offset..dst_qs_offset + BLCK_SIZE]
                    .copy_from_slice(
                        &src_blocks[src_id].qs
                            [src_qs_offset..src_qs_offset + BLCK_SIZE],
                    );
            }

            // Interleave qh: same pattern but qh is 32 bytes
            // end = 256 / 4 = 64 iterations
            // This produces 4-byte interleaved layout matching llama.cpp's make_block_q5_Kx8
            for i in 0..64 {
                let src_id = i % 8;
                let src_qh_offset = (i / 8) * BLCK_SIZE;
                let dst_qh_offset = i * BLCK_SIZE;
                dst_block.qh[dst_qh_offset..dst_qh_offset + BLCK_SIZE]
                    .copy_from_slice(
                        &src_blocks[src_id].qh
                            [src_qh_offset..src_qh_offset + BLCK_SIZE],
                    );
            }

            // Pack scales: same logic as llama.cpp's make_block_q5_Kx8
            // Each 12-byte chunk contains scales and mins for one subblock across 8 blocks
            for sb in 0..4 {
                let mut s = [0u8; 8];
                let mut m = [0u8; 8];

                // Extract scales and mins from each block
                for j in 0..8 {
                    s[j] = src_blocks[j].scales[sb] & 63;
                    m[j] = src_blocks[j].scales[sb + 4] & 63;
                }

                let off = sb * 12;
                dst_block.scales[off] = (s[0] & 63) + ((s[4] & 48) << 2);
                dst_block.scales[off + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                dst_block.scales[off + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                dst_block.scales[off + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                dst_block.scales[off + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                dst_block.scales[off + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                dst_block.scales[off + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                dst_block.scales[off + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                dst_block.scales[off + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                dst_block.scales[off + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                dst_block.scales[off + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                dst_block.scales[off + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }

            // Second half of scales
            for sb in 0..4 {
                let mut s = [0u8; 8];
                let mut m = [0u8; 8];

                for j in 0..8 {
                    s[j] = ((src_blocks[j].scales[sb] & 192) >> 2)
                        | (src_blocks[j].scales[sb + 8] & 15);
                    m[j] = ((src_blocks[j].scales[sb + 4] & 192) >> 2)
                        | ((src_blocks[j].scales[sb + 8] >> 4) & 15);
                }

                let off = sb * 12 + 48;
                dst_block.scales[off] = (s[0] & 63) + ((s[4] & 48) << 2);
                dst_block.scales[off + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                dst_block.scales[off + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                dst_block.scales[off + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                dst_block.scales[off + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                dst_block.scales[off + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                dst_block.scales[off + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                dst_block.scales[off + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                dst_block.scales[off + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                dst_block.scales[off + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                dst_block.scales[off + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                dst_block.scales[off + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }
        }
    }
}

/// Repack Q4_K weights to 8-row interleaved format
/// Matches llama.cpp's make_block_q4_Kx8
pub fn repack_q4_k_to_x8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    n_rows: usize,
    n_blocks_per_row: usize,
) {
    assert!(n_rows % 8 == 0, "n_rows must be multiple of 8");

    const BLCK_SIZE: usize = 4; // blck_size_interleave for 8x4 interleaved GEMV
    let size_block_q4_k = std::mem::size_of::<BlockQ4_K>();
    let size_block_q4_kx8 = std::mem::size_of::<BlockQ4_Kx8>();

    // Detect input layout from stride
    let interleave_factor = src_stride / size_block_q4_k;
    let is_gguf_interleaved = interleave_factor == 4 && n_blocks_per_row > 4;

    // Process 8 rows at a time
    for row_group in 0..(n_rows / 8) {
        let base_row = row_group * 8;

        // Process each block column
        for blk in 0..n_blocks_per_row {
            let dst_blk = row_group * n_blocks_per_row + blk;
            let dst_offset = dst_blk * size_block_q4_kx8;
            let dst_block = unsafe {
                &mut *(dst.as_mut_ptr().add(dst_offset) as *mut BlockQ4_Kx8)
            };

            // Load 8 source blocks
            let mut src_blocks: [BlockQ4_K; 8] = unsafe { std::mem::zeroed() };
            for i in 0..8 {
                let src_row = base_row + i;

                let src_offset = if is_gguf_interleaved {
                    let chunk_idx = src_row / 4;
                    let row_in_chunk = src_row % 4;
                    let block_half = blk / 4;
                    let block_in_half = blk % 4;

                    let chunk_size = 4 * 4 * size_block_q4_k * 2;
                    let half_size = 4 * 4 * size_block_q4_k;

                    chunk_idx * chunk_size
                        + block_half * half_size
                        + row_in_chunk * 4 * size_block_q4_k
                        + block_in_half * size_block_q4_k
                } else {
                    src_row * src_stride + blk * size_block_q4_k
                };

                if src_offset + size_block_q4_k > src.len() {
                    continue;
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset),
                        &mut src_blocks[i] as *mut BlockQ4_K as *mut u8,
                        size_block_q4_k,
                    );
                }
            }

            // Pack d and dmin
            for i in 0..8 {
                dst_block.d[i] = src_blocks[i].d;
                dst_block.dmin[i] = src_blocks[i].dmin;
            }

            // Interleave qs: 4 bytes at a time from each block
            // end = QK_K * 4 / BLCK_SIZE = 256 * 4 / 4 = 256
            for i in 0..256 {
                let src_id = i % 8;
                let src_qs_offset = (i / 8) * BLCK_SIZE;
                let dst_qs_offset = i * BLCK_SIZE;
                dst_block.qs[dst_qs_offset..dst_qs_offset + BLCK_SIZE]
                    .copy_from_slice(
                        &src_blocks[src_id].qs
                            [src_qs_offset..src_qs_offset + BLCK_SIZE],
                    );
            }

            // Pack scales: same logic as llama.cpp's make_block_q4_Kx8
            // Each 12-byte chunk contains scales and mins for one subblock across 8 blocks
            for sb in 0..4 {
                let mut s = [0u8; 8];
                let mut m = [0u8; 8];

                // Extract scales and mins from each block
                for j in 0..8 {
                    s[j] = src_blocks[j].scales[sb] & 63;
                    m[j] = src_blocks[j].scales[sb + 4] & 63;
                }

                let off = sb * 12;
                dst_block.scales[off] = (s[0] & 63) + ((s[4] & 48) << 2);
                dst_block.scales[off + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                dst_block.scales[off + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                dst_block.scales[off + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                dst_block.scales[off + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                dst_block.scales[off + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                dst_block.scales[off + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                dst_block.scales[off + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                dst_block.scales[off + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                dst_block.scales[off + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                dst_block.scales[off + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                dst_block.scales[off + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }

            // Second half of scales
            for sb in 0..4 {
                let mut s = [0u8; 8];
                let mut m = [0u8; 8];

                for j in 0..8 {
                    s[j] = ((src_blocks[j].scales[sb] & 192) >> 2)
                        | (src_blocks[j].scales[sb + 8] & 15);
                    m[j] = ((src_blocks[j].scales[sb + 4] & 192) >> 2)
                        | ((src_blocks[j].scales[sb + 8] >> 4) & 15);
                }

                let off = sb * 12 + 48;
                dst_block.scales[off] = (s[0] & 63) + ((s[4] & 48) << 2);
                dst_block.scales[off + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                dst_block.scales[off + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                dst_block.scales[off + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                dst_block.scales[off + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                dst_block.scales[off + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                dst_block.scales[off + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                dst_block.scales[off + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                dst_block.scales[off + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                dst_block.scales[off + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                dst_block.scales[off + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                dst_block.scales[off + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }
        }
    }
}

/// Repack Q6_K weights to 8-row interleaved format
/// Follows llama.cpp's make_block_q6_Kx8 with blck_size_interleave=4
pub fn repack_q6_k_to_x8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    n_rows: usize,
    n_blocks_per_row: usize,
) {
    assert!(n_rows % 8 == 0, "n_rows must be multiple of 8");

    let size_block_q6_k = std::mem::size_of::<BlockQ6_K>();
    let size_block_q6_kx8 = std::mem::size_of::<BlockQ6_Kx8>();
    const BLCK_SIZE: usize = 4; // blck_size_interleave

    // Detect input layout from stride
    let interleave_factor = src_stride / size_block_q6_k;
    let is_gguf_interleaved = interleave_factor == 4 && n_blocks_per_row > 4;

    for row_group in 0..(n_rows / 8) {
        let base_row = row_group * 8;

        for blk in 0..n_blocks_per_row {
            let dst_blk = row_group * n_blocks_per_row + blk;
            let dst_offset = dst_blk * size_block_q6_kx8;
            let dst_block = unsafe {
                &mut *(dst.as_mut_ptr().add(dst_offset) as *mut BlockQ6_Kx8)
            };

            // Load 8 source blocks
            let mut src_blocks: [BlockQ6_K; 8] = unsafe { std::mem::zeroed() };
            for i in 0..8 {
                let src_row = base_row + i;

                let src_offset = if is_gguf_interleaved {
                    let chunk_idx = src_row / 4;
                    let row_in_chunk = src_row % 4;
                    let block_half = blk / 4;
                    let block_in_half = blk % 4;

                    let chunk_size = 4 * 4 * size_block_q6_k * 2;
                    let half_size = 4 * 4 * size_block_q6_k;

                    chunk_idx * chunk_size
                        + block_half * half_size
                        + row_in_chunk * 4 * size_block_q6_k
                        + block_in_half * size_block_q6_k
                } else {
                    src_row * src_stride + blk * size_block_q6_k
                };

                if src_offset + size_block_q6_k > src.len() {
                    continue;
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_offset),
                        &mut src_blocks[i] as *mut BlockQ6_K as *mut u8,
                        size_block_q6_k,
                    );
                }
            }

            // Pack d
            for i in 0..8 {
                dst_block.d[i] = src_blocks[i].d;
            }

            // Interleave ql: 4 bytes at a time
            // end_ls = QK_K * 4 / BLCK_SIZE = 256 * 4 / 4 = 256
            for i in 0..256 {
                let src_id = i % 8;
                let src_offset = (i / 8) * BLCK_SIZE;
                let dst_offset = i * BLCK_SIZE;
                dst_block.ql[dst_offset..dst_offset + BLCK_SIZE]
                    .copy_from_slice(
                        &src_blocks[src_id].ql
                            [src_offset..src_offset + BLCK_SIZE],
                    );
            }

            // Interleave qh: 4 bytes at a time
            // end_hs = end_ls / 2 = 128
            for i in 0..128 {
                let src_id = i % 8;
                let src_offset = (i / 8) * BLCK_SIZE;
                let dst_offset = i * BLCK_SIZE;
                dst_block.qh[dst_offset..dst_offset + BLCK_SIZE]
                    .copy_from_slice(
                        &src_blocks[src_id].qh
                            [src_offset..src_offset + BLCK_SIZE],
                    );
            }

            // Interleave scales: scales[j * 8 + i] = in[i].scales[j]
            for i in 0..8 {
                for j in 0..16 {
                    dst_block.scales[j * 8 + i] = src_blocks[i].scales[j];
                }
            }
        }
    }
}

/// Check if repacking is beneficial for the given dimensions
pub fn should_repack(
    n_rows: usize,
    quant_type: crate::types::QuantType,
) -> bool {
    // Only repack if:
    // 1. Row count is multiple of 8 (for 8x4 or 8x8 interleaved GEMV)
    // 2. Quantization type is K-quant (Q4_K, Q5_K, Q6_K)
    if n_rows % 8 != 0 {
        return false;
    }

    matches!(
        quant_type,
        crate::types::QuantType::Q4_K
            | crate::types::QuantType::Q5_K
            | crate::types::QuantType::Q6_K
    )
}
