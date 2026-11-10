// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// Weight repacking for cache optimization

use super::types::{BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K};

/// Interleaved block formats for better cache utilization during GEMV
/// Based on llama.cpp's repack implementation

/// Q4_K interleaved 8x4/8x8: 8 rows packed together
/// Matches llama.cpp's block_q4_Kx8
/// Size: 16 (d) + 16 (dmin) + 96 (scales) + 1024 (qs) = 1152 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ4Kx8 {
    pub d: [u16; 8],       // Super-block scales for 8 rows
    pub dmin: [u16; 8],    // Super-block min scales for 8 rows
    pub scales: [u8; 96],  // Scales and mins for 8 rows × 12 bytes each
    pub qs: [u8; 1024],    // 4-bit quants: 8 rows × 128 bytes (full QK_K elements)
}

/// Q5_K interleaved 8x4/8x8
/// Size: 16 (d) + 16 (dmin) + 96 (scales) + 256 (qh) + 1024 (qs) = 1408 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ5Kx8 {
    pub d: [u16; 8],
    pub dmin: [u16; 8],
    pub scales: [u8; 96],
    pub qh: [u8; 256],     // High bits: QK_K * 8 / 8 = 256
    pub qs: [u8; 1024],    // Low 4 bits: QK_K * 8 / 2 = 1024
}

/// Q6_K interleaved 8x4/8x8
/// Size: 16 (d) + 128 (scales) + 1024 (ql) + 512 (qh) = 1680 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ6Kx8 {
    pub d: [u16; 8],       // Super-block scales for 8 rows
    pub scales: [i8; 128], // Scales: QK_K / 16 * 8 = 128
    pub ql: [u8; 1024],    // Lower 4 bits: QK_K / 2 * 8 = 1024
    pub qh: [u8; 512],     // Upper 2 bits: QK_K / 4 * 8 = 512
}

/// Repack type based on hardware detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepackType {
    None,
    Interleave8x4,  // M1: NEON + dotprod, no i8mm
    Interleave8x8,  // M2/M3/M4: NEON + i8mm
}

/// Detect optimal repack type based on CPU features and tensor dimensions
///
/// Returns the interleave strategy that should be used for the given parameters
pub fn get_optimal_repack_type(dtype: &str, ne1: usize) -> RepackType {
    // Dimension check: must be multiple of interleave factor
    if ne1 % 8 != 0 {
        return RepackType::None;
    }

    // Only K-quant formats support repacking currently
    match dtype {
        "Q4_K" | "Q5_K" | "Q6_K" => {}
        _ => return RepackType::None,
    }

    #[cfg(target_arch = "aarch64")]
    {
        // M2/M3/M4 with i8mm (ARMv8.6-A)
        #[cfg(target_feature = "matmul_int8")]
        {
            return RepackType::Interleave8x8;
        }

        // M1 with dotprod (ARMv8.2-A)
        #[cfg(target_feature = "dotprod")]
        {
            return RepackType::Interleave8x4;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // x86 AVX2/AVX512 could use 8x8, but not implemented yet
        RepackType::None
    }

    #[cfg(all(target_arch = "aarch64", not(any(target_feature = "matmul_int8", target_feature = "dotprod"))))]
    {
        RepackType::None
    }
}

/// Repack Q4_K weights to interleaved 8x4 format
///
/// # Arguments
/// * `src` - Original weights in row-major format [M, N] as flat array
/// * `dst` - Destination for interleaved weights
/// * `m` - Number of rows (must be multiple of 8)
/// * `n` - Number of blocks per row
/// * `blck_size_interleave` - Interleave granularity (4 for 8x4, 8 for 8x8)
///
/// # Layout
/// Original: [row0_block0, row0_block1, ..., row1_block0, row1_block1, ...]
/// Interleaved: [rows0-7_block0, rows0-7_block1, ...]
///
/// qs field: interleaved at `blck_size_interleave` byte granularity
/// scales field: repacked to group corresponding sub-block scales together
pub unsafe fn repack_q4k_to_interleaved(
    src: &[BlockQ4K],
    dst: &mut [BlockQ4Kx8],
    m: usize,
    n: usize,
    blck_size_interleave: usize,
) {
    assert!(m % 8 == 0, "Rows must be multiple of 8");
    assert!(blck_size_interleave == 4 || blck_size_interleave == 8);

    let n_interleaved = m / 8;  // Number of interleaved groups

    for i_group in 0..n_interleaved {
        let row_start = i_group * 8;
        let dst_group = &mut dst[i_group * n..(i_group + 1) * n];

        for j in 0..n {
            let dst = &mut dst_group[j];

            // Collect 8 source blocks
            let src_blocks: [BlockQ4K; 8] = std::array::from_fn(|r| {
                src[(row_start + r) * n + j]
            });

            // Copy d and dmin
            for r in 0..8 {
                dst.d[r] = src_blocks[r].d;
                dst.dmin[r] = src_blocks[r].dmin;
            }

            // Interleave qs field
            // Take blck_size_interleave bytes at a time from each of 8 rows
            let end = (QK_K * 4) / blck_size_interleave;  // 128 for interleave=4, 64 for interleave=8
            for i in 0..end {
                let src_id = i % 8;
                let src_offset = (i / 8) * blck_size_interleave;
                let dst_offset = i * blck_size_interleave;

                std::ptr::copy_nonoverlapping(
                    src_blocks[src_id].qs.as_ptr().add(src_offset),
                    dst.qs.as_mut_ptr().add(dst_offset),
                    blck_size_interleave,
                );
            }

            // Repack scales - complex interleaving logic
            // Each original block has 12 bytes of scales (8 scales + 8 mins, 6 bits each)
            // Output has 96 bytes total, organized as 8 groups of 12 bytes
            for i in 0..4 {
                // Extract scales and mins from 8 blocks
                let mut s: [u8; 8] = [0; 8];
                let mut m: [u8; 8] = [0; 8];

                for j in 0..8 {
                    s[j] = src_blocks[j].scales[i] & 63;
                    m[j] = src_blocks[j].scales[i + 4] & 63;
                }

                // Pack into first 48 bytes
                dst.scales[i * 12] = (s[0] & 63) + ((s[4] & 48) << 2);
                dst.scales[i * 12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                dst.scales[i * 12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                dst.scales[i * 12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                dst.scales[i * 12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                dst.scales[i * 12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                dst.scales[i * 12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                dst.scales[i * 12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                dst.scales[i * 12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                dst.scales[i * 12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                dst.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                dst.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }

            // Second half of scales
            for i in 0..4 {
                let mut s: [u8; 8] = [0; 8];
                let mut m: [u8; 8] = [0; 8];

                for j in 0..8 {
                    s[j] = ((src_blocks[j].scales[i] & 192) >> 2) | (src_blocks[j].scales[i + 8] & 15);
                    m[j] = ((src_blocks[j].scales[i + 4] & 192) >> 2) | ((src_blocks[j].scales[i + 8] & 240) >> 4);
                }

                let base = i * 12 + 48;
                dst.scales[base] = (s[0] & 63) + ((s[4] & 48) << 2);
                dst.scales[base + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
                dst.scales[base + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
                dst.scales[base + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
                dst.scales[base + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
                dst.scales[base + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
                dst.scales[base + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
                dst.scales[base + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
                dst.scales[base + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
                dst.scales[base + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
                dst.scales[base + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
                dst.scales[base + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
            }
        }
    }
}

// Q5_K and Q6_K repacking will be added when needed
// The key insight is that Q5_K adds qh field, Q6_K has different layout
// For now, focus on Q4_K which is the most common

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repack_type_detection() {
        // M1 should use 8x4
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod", not(target_feature = "matmul_int8")))]
        {
            assert_eq!(get_optimal_repack_type("Q4_K", 1024), RepackType::Interleave8x4);
            assert_eq!(get_optimal_repack_type("Q4_K", 7), RepackType::None); // Not multiple of 8
        }

        // M2+ should use 8x8
        #[cfg(all(target_arch = "aarch64", target_feature = "matmul_int8"))]
        {
            assert_eq!(get_optimal_repack_type("Q4_K", 1024), RepackType::Interleave8x8);
        }

        // Non-K-quant formats should not repack
        assert_eq!(get_optimal_repack_type("F32", 1024), RepackType::None);
    }
}
