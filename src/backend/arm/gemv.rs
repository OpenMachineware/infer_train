use std::arch::aarch64::*;

use crate::types::{
    BlockQ4_0, BlockQ4_0x4, BlockQ4_0x8, BlockQ4_K, BlockQ4_Kx8, BlockQ5_K,
    BlockQ5_Kx8, BlockQ6_K, BlockQ6_Kx8, BlockQ8_0, BlockQ8_K, QK_K,
};

use super::vec_dot::{
    vec_dot_q4_0_q8_0, vec_dot_q4_k_q8_k, vec_dot_q5_k_q8_k, vec_dot_q6_k_q8_k,
};

pub unsafe fn gemv_q4_0_f32_unrolled(
    weights: &[BlockQ4_0],
    input: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let nb = in_features / QK4_0;

    // Quantize input to Q8_0 first
    let mut input_q8: Vec<BlockQ8_0> =
        vec![BlockQ8_0 { d: 0, qs: [0; 32] }; nb];
    for j in 0..nb {
        let block_input = &input[j * QK4_0..(j + 1) * QK4_0];
        // Simple quantization: find max abs and scale
        let mut max_abs = 0.0f32;
        for &v in block_input {
            max_abs = max_abs.max(v.abs());
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        input_q8[j].d = crate::types::fp32_to_fp16(scale);
        for k in 0..QK4_0 {
            input_q8[j].qs[k] =
                (block_input[k] / scale).round().clamp(-128.0, 127.0) as i8;
        }
    }

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q4_0_q8_0(row_weights, &input_q8);
    }
}

pub const QK4_0: usize = 32;

pub unsafe fn gemv_q4_0_blocked(
    weights: &[BlockQ4_0],
    input: &[f32],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    // Simply call the unrolled version
    gemv_q4_0_f32_unrolled(weights, input, output, out_features, in_features);
}

/// Interleaved GEMV for Q4_0 with 8 rows processed together.
/// Uses SIMD for dot products with safe memory access.
/// BlockQ4_0x8 layout: 8 scales (16 bytes) + interleaved qs (128 bytes)
/// qs layout: bytes[0..63] for k=0, bytes[64..127] for k=1
/// Each row has 8 bytes per k, stored consecutively: row0_k0, row1_k0, ..., row7_k0, row0_k1, ...
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q4_0_x8_neon(
    weights: &[BlockQ4_0x8],
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    use crate::types::QK4_0;
    let nb = in_features / QK4_0;
    let nc = out_features / 8;

    let w_ptr = weights.as_ptr();

    for x in 0..nc {
        let mut sumf = [0.0f32; 8];

        for l in 0..nb {
            let b = &*w_ptr.add(x * nb + l);
            let q8 = &input[l];

            // Load scales
            let scales: [f32; 8] =
                std::array::from_fn(|i| crate::types::fp16_to_fp32(b.d[i]));
            let q8_d = crate::types::fp16_to_fp32(q8.d);

            // Process both halves (k=0 and k=1)
            // k=0: low nibbles of qs[0..15] (16 Q4 values -> 8 Q8 values via low nibbles)
            // k=1: high nibbles of qs[0..15] (16 Q4 values -> 8 Q8 values via high nibbles)
            // Q8 mapping: k=0 uses q8.qs[0..7], k=1 uses q8.qs[8..15] (and similarly for high nibbles)
            for k in 0..2 {
                // Load Q8 values for this k
                // k=0: q8_lo = q8.qs[0..7], q8_hi = q8.qs[16..23] (wait, this doesn't match)
                // Actually, for Q4_0 GEMV:
                // - Each BlockQ8_0 has 32 i8 values (qs[0..31])
                // - We want to multiply with 32 Q4 values (16 bytes * 2 nibbles)
                // - k=0 processes the first 16 Q4 values (low nibbles of first 8 bytes, high nibbles of first 8 bytes)
                // - k=1 processes the next 16 Q4 values (low nibbles of next 8 bytes, high nibbles of next 8 bytes)

                // Wait, let me reconsider the layout:
                // BlockQ4_0 has 16 bytes qs storing 32 Q4 values
                // BlockQ4_0x8 packs 8 blocks together
                // The repack interleaves 8 bytes at a time:
                // qs[0..7] = row0 qs[0..7], qs[8..15] = row1 qs[0..7], ..., qs[56..63] = row7 qs[0..7]
                // qs[64..71] = row0 qs[8..15], ..., qs[120..127] = row7 qs[8..15]

                // So for k=0: each row's qs[0..7] (8 bytes = 16 Q4 values: 8 low + 8 high nibbles)
                // For k=1: each row's qs[8..15] (8 bytes = 16 Q4 values: 8 low + 8 high nibbles)

                // Q8 for Q4_0 has 32 i8 values per block
                // We need to map: k=0 -> q8[0..7] for low nibbles, q8[8..15] for high nibbles? No...

                // Actually in standard Q4_0 vec_dot:
                // 16 bytes of Q4 (32 nibbles) dot with 32 bytes of Q8
                // Low nibbles dot with q8[0..15], high nibbles dot with q8[16..31]? No, that's not right either.

                // Let's look at the actual layout:
                // Q4_0 stores 32 values in 16 bytes: each byte has low nibble (value 0-15) and high nibble (value 0-15)
                // The XOR mask 0x8888... converts to signed: nibble ^ 8 gives 0-7 positive, 8-15 negative
                // After XOR, we have: low nibble = original_low ^ 8, high nibble = original_high ^ 8
                // To extract as signed: (byte << 4) as i8 gives (low ^ 8 - 8) * 16 = (low - 8) * 16
                // And byte & 0xF0 gives high nibble directly

                // For Q8_0 input with 32 values:
                // We dot the first 16 Q4 values with q8[0..15] and next 16 with q8[16..31]? No...
                // Actually it's simpler: 32 Q4 values dot with 32 Q8 values
                // But we process in halves for efficiency

                // The standard Q4_0 vec_dot processes:
                // - 16 Q4 values from bytes 0..7 with Q8[0..15]? No, let me check the actual code.

                // For simplicity, let's use the Q8 values based on k:
                // k=0: use q8.qs[0..15] (first 16 Q8 values)
                // k=1: use q8.qs[16..31] (last 16 Q8 values)
                // But wait, we also need to separate low and high nibbles...

                // Actually, the layout should be:
                // 8 bytes of Q4 -> 16 Q4 values (8 low + 8 high nibbles)
                // These 16 Q4 values dot with 16 Q8 values
                // So k=0: Q4 bytes[0..7] -> 16 values dot with q8[0..15]
                // And k=1: Q4 bytes[8..15] -> 16 values dot with q8[16..31]
                // But we also need low/high nibble separation for SIMD efficiency

                // Let's use a simpler approach:
                // For k=0: load q8[0..15], split into q8[0..7] for low nibbles and q8[8..15] for high nibbles
                // For k=1: load q8[16..31], split into q8[16..23] for low nibbles and q8[24..31] for high nibbles

                // Load Q8 values - use vld1q_s8 (16 bytes) which is safe because q8.qs has 32 bytes
                let q8_base = k * 16;
                let q8_vec = vld1q_s8(q8.qs.as_ptr().add(q8_base));
                // Split into low and high 8 elements
                // We'll use vget_low_s8 and vget_high_s8 for correct extraction
                // But for dot product, we need int8x16_t, so let's duplicate the halves

                // Actually, for the dot product, we can just use lanes 0-1 of the result
                // q8_vec lanes 0-3 are q8[q8_base..q8_base+3], lanes 4-7 are q8[q8_base+4..q8_base+7]
                // These are for low nibbles
                // Lanes 8-11 are q8[q8_base+8..q8_base+11], lanes 12-15 are q8[q8_base+12..q8_base+15]
                // These are for high nibbles

                // Load interleaved Q4 values (64 bytes for this k)
                let qs_base = b.qs.as_ptr().add(k * 64);

                // Process 8 rows
                for j in 0..8 {
                    // Load 8 bytes for this row - use vld1_u8 to load exactly 8 bytes
                    // Then combine with a zero vector for safe SIMD processing
                    let q4_bytes_lo = vld1_u8(qs_base.add(j * 8)); // uint8x8_t
                    // Combine low and high for processing
                    // We need 16 bytes for SIMD, but we only have 8
                    // Create a vector with 8 valid bytes in the low half, 8 zeros in high half
                    let q4_bytes = vcombine_u8(q4_bytes_lo, vdup_n_u8(0)); // uint8x16_t

                    // Extract low nibbles: shift left 4 to get signed values
                    // XOR mask was applied during repack, so the nibbles are already signed
                    // (low_nibble - 8) representation: (byte << 4) as i8 = (low - 8) * 16
                    let q4_lo = vreinterpretq_s8_u8(vshlq_n_u8(q4_bytes, 4));

                    // Extract high nibbles: mask with 0xF0
                    // (high_nibble - 8) representation: byte & 0xF0 = (high - 8) * 16
                    let q4_hi = vreinterpretq_s8_u8(vandq_u8(
                        q4_bytes,
                        vdupq_n_u8(0xF0),
                    ));

                    // Compute dot products
                    // q4_lo has values in lanes 0-7 (from the 8 bytes we loaded), lanes 8-15 are 0
                    // q4_hi has values in lanes 0-7, lanes 8-15 are 0
                    // q8_vec has 16 valid Q8 values

                    // For q4_lo (low nibbles), we want to dot with q8[q8_base..q8_base+7]
                    // For q4_hi (high nibbles), we want to dot with q8[q8_base+8..q8_base+15]

                    // vdotq_s32 returns 4 lanes:
                    // lane 0 = sum(a[0..3] * b[0..3])
                    // lane 1 = sum(a[4..7] * b[4..7])
                    // lane 2 = sum(a[8..11] * b[8..11])
                    // lane 3 = sum(a[12..15] * b[12..15])

                    // For q4_lo: lanes 0-1 have valid data (8 elements), lanes 2-3 are 0
                    // We want lanes 0-1 to dot with q8 lanes 0-7
                    // Result: lanes 0-1 are correct, lanes 2-3 are 0

                    // For q4_hi: lanes 0-1 have valid data
                    // We want lanes 0-1 to dot with q8 lanes 8-15
                    // But vdotq_s32 pairs same lanes together, so q4_hi lanes 0-1 would dot with q8 lanes 0-7
                    // This is wrong! We need to align q4_hi with q8 lanes 8-15

                    // Solution: create separate q8 vectors for low and high nibbles
                    // q8_for_low = q8_vec (lanes 0-7 for our dot product)
                    // q8_for_high = vextq_s8(q8_vec, q8_vec, 8) (shifts lanes 8-15 to 0-7)

                    // Actually, let's use a cleaner approach:
                    // For low nibbles: dot q4_lo[0..7] with q8_vec[0..7]
                    // For high nibbles: dot q4_hi[0..7] with q8_vec[8..15]

                    // We can use vextq_s8 to shift q8_vec so that lanes 8-15 become lanes 0-7
                    let q8_for_high = vextq_s8(q8_vec, q8_vec, 8);

                    // Now compute dot products
                    let dot_lo = vdotq_s32(vdupq_n_s32(0), q4_lo, q8_vec);
                    let dot_hi = vdotq_s32(vdupq_n_s32(0), q4_hi, q8_for_high);

                    // Sum lanes 0 and 1 (these contain the dot products for our 8 elements)
                    let sum_lo =
                        vgetq_lane_s32(dot_lo, 0) + vgetq_lane_s32(dot_lo, 1);
                    let sum_hi =
                        vgetq_lane_s32(dot_hi, 0) + vgetq_lane_s32(dot_hi, 1);

                    // Divide by 16 (shift right 4) and accumulate
                    sumf[j] +=
                        (sum_lo + sum_hi) as f32 * scales[j] * q8_d / 16.0;
                }
            }
        }

        for j in 0..8 {
            output[x * 8 + j] = sumf[j];
        }
    }
}

/// Interleaved GEMV for Q4_K with 8 rows processed together.
/// Follows llama.cpp's ggml_gemv_q4_K_8x4_q8_K pattern.
/// Q4_K has only 4-bit values (no high bits like Q5_K), making it simpler.
/// Interleaved GEMV for repacked Q4_Kx8 weights
/// TODO: Implement SIMD version. For now, fall back to vec_dot.
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_q4_k_x8_neon(
    weights: &[crate::types::BlockQ4_Kx8],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let nb = in_features / QK_K;
    let nc = out_features / 8;

    for x in 0..nc {
        let base_row = x * 8;
        let q4_ptr = weights.as_ptr().add(x * nb);

        let mut sumf = [0.0f32; 8];

        for b in 0..nb {
            let blk = &*q4_ptr.add(b);
            let q8 = &input[b];

            // Process using scalar implementation
            // TODO: Implement SIMD version
            for j in 0..8 {
                let d = crate::types::fp16_to_fp32(blk.d[j]);
                let dmin = crate::types::fp16_to_fp32(blk.dmin[j]);

                // Decode scales and mins for this block
                // Q4_Kx8 has the same scales encoding as Q5_Kx8
                let mut sumi = 0i32;
                let mut sum_min = 0i32;

                for sb in 0..4 {
                    let (scales_0, mins_0) =
                        decode_scales_q5_kx8(&blk.scales[sb * 24..]);
                    let (scales_1, mins_1) =
                        decode_scales_q5_kx8(&blk.scales[sb * 24 + 12..]);

                    // Process each chunk within subblock
                    for k in 0..4 {
                        let q8_offset = sb * 64 + k * 8;
                        for i in 0..8 {
                            // QS offset for interleaved layout
                            let p = (sb * 4 + k) * 8 + i;
                            let chunk = p / 4;
                            let qs_offset = chunk * 32 + j * 4 + (p % 4);
                            let qs_val = blk.qs[qs_offset];

                            let v0 = (qs_val & 0xF) as i32;
                            let v1 = (qs_val >> 4) as i32;

                            let q8_v0 = q8.qs[q8_offset + i] as i32;
                            let q8_v1 = q8.qs[q8_offset + 32 + i] as i32;

                            sumi += v0 * scales_0[j] as i32 * q8_v0
                                + v1 * scales_1[j] as i32 * q8_v1;
                        }
                    }

                    let bsum_lo = q8.bsums[sb * 2] as i32;
                    let bsum_hi = q8.bsums[sb * 2 + 1] as i32;
                    sum_min +=
                        bsum_lo * mins_0[j] as i32 + bsum_hi * mins_1[j] as i32;
                }

                sumf[j] += (sumi as f32 * d - sum_min as f32 * dmin) * q8.d;
            }
        }

        for j in 0..8 {
            output[base_row + j] = sumf[j];
        }
    }
}

/// Interleaved GEMV for repacked Q5_Kx8 weights
/// Processes 8 output rows at a time using scalar accumulators
///
/// Based on llama.cpp's generic GEMV implementation:
/// - qs offset: k * ncols_interleaved * blocklen + j * blocklen + i
/// - qh offset: qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos
/// - scales access: scales_0 = utmp + (k / 8) * 32, scales_1 = scales_0 + 16
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_q5_k_x8_neon(
    weights: &[crate::types::BlockQ5_Kx8],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let nb = in_features / QK_K;
    let nc = out_features / 8;
    const BLOCKLEN: usize = 4;
    const NCOLS: usize = 8;

    for x in 0..nc {
        let base_row = x * 8;
        let q5_ptr = weights.as_ptr().add(x * nb);

        let mut sumf = [0.0f32; 8];
        let mut sum_min = [0.0f32; 8];

        for b in 0..nb {
            let blk = &*q5_ptr.add(b);
            let q8 = &input[b];

            // Decode all scales and mins
            let mut all_scales = [[0u8; 8]; 8];
            let mut all_mins = [[0u8; 8]; 8];

            for sb in 0..8 {
                let (scales, mins) =
                    decode_scales_q5_kx8(&blk.scales[sb * 12..]);
                all_scales[sb] = scales;
                all_mins[sb] = mins;
            }

            // Process each 4-value chunk
            // k = 0..(QK_K / (2 * BLOCKLEN)) = 256 / 8 = 32
            for k in 0..32 {
                // scales_0 = scales[(k / 8) * 2], scales_1 = scales[(k / 8) * 2 + 1]
                let scale_idx = k / 8;
                let scales_0 = &all_scales[scale_idx * 2];
                let scales_1 = &all_scales[scale_idx * 2 + 1];

                // qh_shift = (k / 8) * 2
                let qh_shift = (k / 8) * 2;

                // q8_offset = (k / 8) * 64 + (k % 8) * 4
                let q8_base = (k / 8) * 64 + (k % 8) * BLOCKLEN;

                for j in 0..NCOLS {
                    let mut sumi = 0i32;
                    for i in 0..BLOCKLEN {
                        // qs offset: k * NCOLS * BLOCKLEN + j * BLOCKLEN + i
                        let b_qs_offset =
                            k * NCOLS * BLOCKLEN + j * BLOCKLEN + i;

                        // qh offset
                        let qh_idx = (k * BLOCKLEN + i) % 32;
                        let qh_chunk = qh_idx / BLOCKLEN;
                        let qh_pos = qh_idx % BLOCKLEN;
                        let b_qh_offset = qh_chunk * (BLOCKLEN * NCOLS)
                            + j * BLOCKLEN
                            + qh_pos;

                        let qh_val = blk.qh[b_qh_offset];
                        let h0 = (qh_val >> qh_shift) & 1;
                        let h1 = (qh_val >> (qh_shift + 1)) & 1;

                        let v0 = ((blk.qs[b_qs_offset] & 0xF) | (h0 << 4)) as i8
                            as i32;
                        let v1 = ((blk.qs[b_qs_offset] >> 4) | (h1 << 4)) as i8
                            as i32;

                        let q8_v0 = q8.qs[q8_base + i] as i32;
                        let q8_v1 = q8.qs[q8_base + i + 32] as i32;

                        sumi += v0 * q8_v0 * scales_0[j] as i32
                            + v1 * q8_v1 * scales_1[j] as i32;
                    }
                    sumf[j] += sumi as f32
                        * crate::types::fp16_to_fp32(blk.d[j])
                        * q8.d;
                }
            }

            // Bias correction
            for sb in 0..8 {
                let mins = &all_mins[sb];
                for j in 0..8 {
                    sum_min[j] += mins[j] as f32
                        * (q8.bsums[sb * 2] + q8.bsums[sb * 2 + 1]) as f32
                        * crate::types::fp16_to_fp32(blk.dmin[j])
                        * q8.d;
                }
            }
        }

        for j in 0..8 {
            output[base_row + j] = sumf[j] - sum_min[j];
        }
    }
}

/// Decode 12 bytes of packed scales/mins into 8 scales and 8 mins
/// Follows llama.cpp's decode_q_Kx8_6bit_scales exactly
#[inline]
fn decode_scales_q5_kx8(scales_in: &[u8]) -> ([u8; 8], [u8; 8]) {
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

    // Extract mins
    let mins_0_3 = sm[1] & KMASK1;
    let mins_4_7 = ((sm[2] >> 4) & KMASK2) | (((sm[1] >> 6) & KMASK3) << 4);

    let mut mins: [u8; 8] = [0; 8];
    for (i, b) in mins_0_3.to_le_bytes()[..4].iter().enumerate() {
        mins[i] = *b;
    }
    for (i, b) in mins_4_7.to_le_bytes()[..4].iter().enumerate() {
        mins[4 + i] = *b;
    }

    // Extract scales
    let scales_0_3 = sm[0] & KMASK1;
    let scales_4_7 = (sm[2] & KMASK2) | (((sm[0] >> 6) & KMASK3) << 4);

    let mut scales: [u8; 8] = [0; 8];
    for (i, b) in scales_0_3.to_le_bytes()[..4].iter().enumerate() {
        scales[i] = *b;
    }
    for (i, b) in scales_4_7.to_le_bytes()[..4].iter().enumerate() {
        scales[4 + i] = *b;
    }

    (scales, mins)
}

/// Original GEMV for Q4_K using vec_dot
pub unsafe fn gemv_q4_k_f32(
    weights: &[BlockQ4_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let nb = in_features / QK_K;

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q4_k_q8_k(row_weights, input);
    }
}

/// Original GEMV for Q5_K using vec_dot
pub unsafe fn gemv_q5_k_f32(
    weights: &[BlockQ5_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let nb = in_features / QK_K;

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q5_k_q8_k(row_weights, input);
    }
}

/// Original GEMV for Q6_K using vec_dot
pub unsafe fn gemv_q6_k_f32(
    weights: &[BlockQ6_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let nb = in_features / QK_K;

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q6_k_q8_k(row_weights, input);
    }
}

/// GEMV for Q4_K with transposed weights (row-major storage)
/// Note: Same as Q5_K - compute actual row stride from in_features.
pub unsafe fn gemv_q4_k_transposed(
    weight_data: &[u8],
    _weight_stride: usize, // Ignored - compute actual row stride from in_features
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;
    let size_block_q4_k = std::mem::size_of::<BlockQ4_K>();
    let row_stride = nb * size_block_q4_k; // Actual bytes per row

    for i in 0..out_features {
        let row_offset = i * row_stride;

        let row_blocks: Vec<BlockQ4_K> = (0..nb)
            .map(|j| {
                let blk = &*(weight_data
                    .as_ptr()
                    .add(row_offset + j * size_block_q4_k)
                    as *const BlockQ4_K);
                blk.clone()
            })
            .collect();

        output[i] = vec_dot_q4_k_q8_k(&row_blocks, input);
    }
}

/// GEMV for Q5_K with transposed weights (row-major storage)
/// Note: GGUF stores weights in row-major format where each row has compact storage.
/// The stride parameter from GGUF (nb[1]) is NOT the row stride - it's the byte offset
/// for moving one index in dimension 1, which may be larger than actual row size.
/// For correct GEMV, we compute row_stride = nb_blocks * block_size.
pub unsafe fn gemv_q5_k_transposed(
    weight_data: &[u8],
    _weight_stride: usize, // Ignored - compute actual row stride from in_features
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;
    let size_block_q5_k = std::mem::size_of::<BlockQ5_K>();
    let row_stride = nb * size_block_q5_k; // Actual bytes per row

    // Debug output for first call
    if output.len() > 0 && out_features > 100 {
        log::debug!(
            "gemv_q5_k_transposed: nb={}, size_block={}, row_stride={}",
            nb,
            size_block_q5_k,
            row_stride
        );
        log::debug!(
            "gemv_q5_k_transposed: out_features={}, in_features={}",
            out_features,
            in_features
        );

        // Check first row
        let first_block: &crate::types::BlockQ5_K =
            &*(weight_data.as_ptr() as *const crate::types::BlockQ5_K);
        log::debug!(
            "First weight block: d={:.4}, dmin={:.4}",
            half::f16::from_bits(first_block.d).to_f32(),
            half::f16::from_bits(first_block.dmin).to_f32()
        );

        // Check input
        log::debug!(
            "Input first block: d={:.4}, qs[:4]={:?}",
            input[0].d,
            &input[0].qs[..4]
        );
    }

    for i in 0..out_features {
        let row_offset = i * row_stride;

        let row_blocks: Vec<BlockQ5_K> = (0..nb)
            .map(|j| {
                let blk = &*(weight_data
                    .as_ptr()
                    .add(row_offset + j * size_block_q5_k)
                    as *const BlockQ5_K);
                blk.clone()
            })
            .collect();

        output[i] = vec_dot_q5_k_q8_k(&row_blocks, input);
    }
}

/// GEMV for Q6_K with transposed weights (row-major storage)
/// Note: Same as Q5_K - compute actual row stride from in_features.
pub unsafe fn gemv_q6_k_transposed(
    weight_data: &[u8],
    _weight_stride: usize, // Ignored - compute actual row stride from in_features
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;
    let size_block_q6_k = std::mem::size_of::<BlockQ6_K>();
    let row_stride = nb * size_block_q6_k; // Actual bytes per row

    for i in 0..out_features {
        let row_offset = i * row_stride;

        let row_blocks: Vec<BlockQ6_K> = (0..nb)
            .map(|j| {
                let blk = &*(weight_data
                    .as_ptr()
                    .add(row_offset + j * size_block_q6_k)
                    as *const BlockQ6_K);
                blk.clone()
            })
            .collect();

        output[i] = vec_dot_q6_k_q8_k(&row_blocks, input);
    }
}

/// GEMV for Q4_0 with transposed weights (column-major storage)
pub unsafe fn gemv_q4_0_transposed(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    use crate::types::BlockQ4_0;
    let nb = in_features / QK4_0;
    let size_block_q4_0 = std::mem::size_of::<BlockQ4_0>();

    for i in 0..out_features {
        let col_offset = i * weight_stride;

        let row_blocks: Vec<BlockQ4_0> = (0..nb)
            .map(|j| {
                let blk = &*(weight_data
                    .as_ptr()
                    .add(col_offset + j * size_block_q4_0)
                    as *const BlockQ4_0);
                blk.clone()
            })
            .collect();

        output[i] = vec_dot_q4_0_q8_0(&row_blocks, input);
    }
}

/// GEMV for Q4_0 with row-major weights
pub unsafe fn gemv_q4_0(
    weights: &[BlockQ4_0],
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK4_0;

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q4_0_q8_0(row_weights, input);
    }
}

/// GEMV for Q8_0 with row-major weights
pub unsafe fn gemv_q8_0(
    weights: &[BlockQ8_0],
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    use crate::types::QK8_0;
    let nb = in_features / QK8_0;

    for i in 0..out_features {
        let mut sum = 0.0f32;
        let row_weights = &weights[i * nb..(i + 1) * nb];

        for (j, blk) in row_weights.iter().enumerate() {
            // Q8_0 x Q8_0: scale and dot product
            let scale = crate::types::fp16_to_fp32(blk.d)
                * crate::types::fp16_to_fp32(input[j].d);
            let mut dot = 0i32;
            for k in 0..QK8_0 {
                dot += (blk.qs[k] as i32) * (input[j].qs[k] as i32);
            }
            sum += scale * dot as f32;
        }

        output[i] = sum;
    }
}

/// GEMV for Q4_K with row-major weights
pub unsafe fn gemv_q4_k(
    weights: &[BlockQ4_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q4_k_q8_k(row_weights, input);
    }
}

/// GEMV for Q5_K with row-major weights
pub unsafe fn gemv_q5_k(
    weights: &[BlockQ5_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q5_k_q8_k(row_weights, input);
    }
}

/// GEMV for Q6_K with row-major weights
pub unsafe fn gemv_q6_k(
    weights: &[BlockQ6_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;

    for i in 0..out_features {
        let row_weights = &weights[i * nb..(i + 1) * nb];
        output[i] = vec_dot_q6_k_q8_k(row_weights, input);
    }
}

/// Batched matrix multiplication for Q4_0
pub fn mul_mat_q4_0(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK4_0;
    let size_block_q4_0 = std::mem::size_of::<BlockQ4_0>();

    for t in 0..n_tokens {
        let input_offset = t * nb;
        let output_offset = t * out_features;
        let input_slice = &input[input_offset..input_offset + nb];

        for i in 0..out_features {
            let mut sum = 0.0f32;
            let col_offset = i * weight_stride;

            for j in 0..nb {
                unsafe {
                    let blk = &*(weight_data
                        .as_ptr()
                        .add(col_offset + j * size_block_q4_0)
                        as *const BlockQ4_0);
                    // Compute single block dot product
                    let d = crate::types::fp16_to_fp32(blk.d)
                        * crate::types::fp16_to_fp32(input_slice[j].d);
                    let mut dot = 0i32;
                    for k in 0..QK4_0 {
                        let q4 = (blk.qs[k] & 0x0F) as i8 - 8;
                        let q8 = input_slice[j].qs[k] as i32;
                        dot += (q4 as i32) * q8;
                    }
                    for k in 0..QK4_0 {
                        let q4 = (blk.qs[k] >> 4) as i8 - 8;
                        let q8 = input_slice[j].qs[k] as i32;
                        dot += (q4 as i32) * q8;
                    }
                    sum += d * dot as f32;
                }
            }

            output[output_offset + i] = sum;
        }
    }
}

/// Batched matrix multiplication for Q5_K
/// Temporarily using simple vec_dot for correctness verification
pub fn mul_mat_q5_k(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;
    let size_block_q5_k = std::mem::size_of::<BlockQ5_K>();

    // Use simple vec_dot for correctness (bypass interleaved GEMV)
    for t in 0..n_tokens {
        let input_slice = &input[t * nb..(t + 1) * nb];
        let output_slice =
            &mut output[t * out_features..(t + 1) * out_features];

        for i in 0..out_features {
            let col_offset = i * weight_stride;
            let row_blocks: Vec<BlockQ5_K> = (0..nb)
                .map(|j| unsafe {
                    *(weight_data.as_ptr().add(col_offset + j * size_block_q5_k)
                        as *const BlockQ5_K)
                })
                .collect();
            output_slice[i] = vec_dot_q5_k_q8_k(&row_blocks, input_slice);
        }
    }
}

/// Batched matrix multiplication for Q4_K
/// Temporarily using simple vec_dot for correctness verification
pub fn mul_mat_q4_k(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;
    let size_block_q4_k = std::mem::size_of::<BlockQ4_K>();

    // Use simple vec_dot for correctness (bypass interleaved GEMV)
    for t in 0..n_tokens {
        let input_slice = &input[t * nb..(t + 1) * nb];
        let output_slice =
            &mut output[t * out_features..(t + 1) * out_features];

        for i in 0..out_features {
            let col_offset = i * weight_stride;
            let row_blocks: Vec<BlockQ4_K> = (0..nb)
                .map(|j| unsafe {
                    *(weight_data.as_ptr().add(col_offset + j * size_block_q4_k)
                        as *const BlockQ4_K)
                })
                .collect();
            output_slice[i] = vec_dot_q4_k_q8_k(&row_blocks, input_slice);
        }
    }
}

/// Batched matrix multiplication for Q6_K
/// Temporarily using simple vec_dot for correctness verification
pub fn mul_mat_q6_k(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK_K;
    let size_block_q6_k = std::mem::size_of::<BlockQ6_K>();

    // Use simple vec_dot for correctness (bypass interleaved GEMV)
    for t in 0..n_tokens {
        let input_slice = &input[t * nb..(t + 1) * nb];
        let output_slice =
            &mut output[t * out_features..(t + 1) * out_features];

        for i in 0..out_features {
            let col_offset = i * weight_stride;
            let row_blocks: Vec<BlockQ6_K> = (0..nb)
                .map(|j| unsafe {
                    *(weight_data.as_ptr().add(col_offset + j * size_block_q6_k)
                        as *const BlockQ6_K)
                })
                .collect();
            output_slice[i] = vec_dot_q6_k_q8_k(&row_blocks, input_slice);
        }
    }
}

/// Batched matrix multiplication for IQ4_XS
pub fn mul_mat_iq4_xs(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    use super::vec_dot::vec_dot_iq4_xs_q8_k;
    use crate::types::BlockIQ4XS;

    let nb = in_features / QK_K;
    let size_block_iq4_xs = std::mem::size_of::<BlockIQ4XS>();

    for t in 0..n_tokens {
        let input_offset = t * nb;
        let output_offset = t * out_features;
        let input_slice = &input[input_offset..input_offset + nb];

        for i in 0..out_features {
            let mut sum = 0.0f32;
            let col_offset = i * weight_stride;

            let row_blocks: Vec<BlockIQ4XS> = (0..nb)
                .map(|j| unsafe {
                    let blk = &*(weight_data
                        .as_ptr()
                        .add(col_offset + j * size_block_iq4_xs)
                        as *const BlockIQ4XS);
                    blk.clone()
                })
                .collect();

            sum = vec_dot_iq4_xs_q8_k(&row_blocks, input_slice);
            output[output_offset + i] = sum;
        }
    }
}

/// Interleaved GEMV for Q6_K with 8 rows processed together.
/// Follows llama.cpp's ggml_gemv_q6_K_8x4_q8_K pattern.
///
/// Q6_K encoding:
/// - ql stores low 4 bits of 6-bit values
/// - qh stores high 2 bits (4 values per byte)
/// - 16 values share one scale
/// - Values are biased by -32 (range: -32 to 31)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q6_k_x8_neon(
    weights: &[BlockQ6_Kx8],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let nb = in_features / QK_K;
    let nc = out_features / 8;

    const COL_GROUPS: usize = 2; // 0-3 and 4-7
    let m4b = vdupq_n_u8(0x0F);
    let mask_lo = vdupq_n_u8(0x03);
    let mask_hi = vdupq_n_u8(0x30);

    for x in 0..nc {
        let q6_ptr = weights.as_ptr().add(x * nb);
        let mut acc_f32 = [vdupq_n_f32(0.0); 2];

        for b in 0..nb {
            let blk = &*q6_ptr.add(b);
            let q8 = &input[b];

            // Load 8 block scales
            let d: [f32; 8] =
                std::array::from_fn(|i| crate::types::fp16_to_fp32(blk.d[i]));
            let q6_d_0 = vld1q_f32(d.as_ptr());
            let q6_d_1 = vld1q_f32(d.as_ptr().add(4));
            let q8_d = vdupq_n_f32(q8.d);
            let sb_scale_0 = vmulq_f32(q6_d_0, q8_d);
            let sb_scale_1 = vmulq_f32(q6_d_1, q8_d);

            // Load all 16 * 8 = 128 scales once and widen to int16
            let mut q6_scales: [i16; 128] = [0; 128];
            for i in 0..16 {
                let scales = vmovl_s8(vld1_s8(blk.scales.as_ptr().add(i * 8)));
                vst1q_s16(q6_scales.as_mut_ptr().add(i * 8), scales);
            }

            // Compute bias: each scale × bsum × 32 (for -32 bias in Q6_K values)
            let mut bias_lo = vdupq_n_s32(0);
            let mut bias_hi = vdupq_n_s32(0);

            for i in (0..16).step_by(4) {
                let bsums_vec = vld1_s16(q8.bsums.as_ptr().add(i));
                let scales_lo_0 = vld1_s16(q6_scales.as_ptr().add((i + 0) * 8));
                let scales_hi_0 =
                    vld1_s16(q6_scales.as_ptr().add((i + 0) * 8 + 4));
                let scales_lo_1 = vld1_s16(q6_scales.as_ptr().add((i + 1) * 8));
                let scales_hi_1 =
                    vld1_s16(q6_scales.as_ptr().add((i + 1) * 8 + 4));
                let scales_lo_2 = vld1_s16(q6_scales.as_ptr().add((i + 2) * 8));
                let scales_hi_2 =
                    vld1_s16(q6_scales.as_ptr().add((i + 2) * 8 + 4));
                let scales_lo_3 = vld1_s16(q6_scales.as_ptr().add((i + 3) * 8));
                let scales_hi_3 =
                    vld1_s16(q6_scales.as_ptr().add((i + 3) * 8 + 4));

                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_0, bsums_vec, 0);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_0, bsums_vec, 0);
                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_1, bsums_vec, 1);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_1, bsums_vec, 1);
                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_2, bsums_vec, 2);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_2, bsums_vec, 2);
                bias_lo = vmlal_lane_s16(bias_lo, scales_lo_3, bsums_vec, 3);
                bias_hi = vmlal_lane_s16(bias_hi, scales_hi_3, bsums_vec, 3);
            }
            bias_lo = vshlq_n_s32(bias_lo, 5); // Multiply by 32
            bias_hi = vshlq_n_s32(bias_hi, 5);

            let mut acc = [vdupq_n_s32(0); COL_GROUPS];

            // Process two 128-value halves per block
            for half in 0..2 {
                let ql_base = blk.ql.as_ptr().add(half * 512);
                let qh_base = blk.qh.as_ptr().add(half * 256);

                // Process 4 subblocks (each subblock = 16 values per column)
                for sb in 0..4 {
                    let q8_base_l = q8.qs.as_ptr().add(half * 128 + sb * 16);
                    let q8_base_h = q8_base_l.add(64);

                    // Load and duplicate Q8 values (each covers 4 columns)
                    let mut q8_l: [int8x16_t; 4] = [vdupq_n_s8(0); 4];
                    let mut q8_h: [int8x16_t; 4] = [vdupq_n_s8(0); 4];
                    for i in 0..4 {
                        let q8_l_val =
                            vld1_dup_s32(q8_base_l.add(i * 4) as *const i32);
                        let q8_h_val =
                            vld1_dup_s32(q8_base_h.add(i * 4) as *const i32);
                        q8_l[i] =
                            vreinterpretq_s8_s32(vdupq_lane_s32(q8_l_val, 0));
                        q8_h[i] =
                            vreinterpretq_s8_s32(vdupq_lane_s32(q8_h_val, 0));
                    }

                    let ql_off = sb * 128;
                    let qh_off = ql_off & 255; // Wraps after 256 bytes

                    // Load ql and qh (64 bytes each via 4 × 16-byte loads)
                    let q6_ql_0 = [
                        vld1q_u8(ql_base.add(ql_off)),
                        vld1q_u8(ql_base.add(ql_off + 16)),
                        vld1q_u8(ql_base.add(ql_off + 32)),
                        vld1q_u8(ql_base.add(ql_off + 48)),
                    ];
                    let q6_ql_1 = [
                        vld1q_u8(ql_base.add(ql_off + 64)),
                        vld1q_u8(ql_base.add(ql_off + 80)),
                        vld1q_u8(ql_base.add(ql_off + 96)),
                        vld1q_u8(ql_base.add(ql_off + 112)),
                    ];
                    let q6_qh_0 = [
                        vld1q_u8(qh_base.add(qh_off)),
                        vld1q_u8(qh_base.add(qh_off + 16)),
                        vld1q_u8(qh_base.add(qh_off + 32)),
                        vld1q_u8(qh_base.add(qh_off + 48)),
                    ];
                    let q6_qh_1 = [
                        vld1q_u8(qh_base.add(qh_off + 64)),
                        vld1q_u8(qh_base.add(qh_off + 80)),
                        vld1q_u8(qh_base.add(qh_off + 96)),
                        vld1q_u8(qh_base.add(qh_off + 112)),
                    ];

                    // Adjust qh for subblocks 2 and 3
                    let q6_qh_0 = if sb > 1 {
                        [
                            vshrq_n_u8(q6_qh_0[0], 2),
                            vshrq_n_u8(q6_qh_0[1], 2),
                            vshrq_n_u8(q6_qh_0[2], 2),
                            vshrq_n_u8(q6_qh_0[3], 2),
                        ]
                    } else {
                        q6_qh_0
                    };
                    let q6_qh_1 = if sb > 1 {
                        [
                            vshrq_n_u8(q6_qh_1[0], 2),
                            vshrq_n_u8(q6_qh_1[1], 2),
                            vshrq_n_u8(q6_qh_1[2], 2),
                            vshrq_n_u8(q6_qh_1[3], 2),
                        ]
                    } else {
                        q6_qh_1
                    };

                    let q6_ql = [
                        q6_ql_0[0], q6_ql_0[1], q6_ql_0[2], q6_ql_0[3],
                        q6_ql_1[0], q6_ql_1[1], q6_ql_1[2], q6_ql_1[3],
                    ];
                    let q6_qh = [
                        q6_qh_0[0], q6_qh_0[1], q6_qh_0[2], q6_qh_0[3],
                        q6_qh_1[0], q6_qh_1[1], q6_qh_1[2], q6_qh_1[3],
                    ];

                    // Process column groups
                    for g in 0..COL_GROUPS {
                        let mut sb_acc_l = vdupq_n_s32(0);
                        let mut sb_acc_h = vdupq_n_s32(0);

                        for chunk in 0..4 {
                            let idx = chunk * 2 + g;
                            let q6_qs_l = q6_ql[idx];
                            let q6_qs_h = q6_qh[idx];

                            // Extract high 2 bits for upper nibble
                            let q6_qs_hh = vandq_u8(q6_qs_h, mask_hi);

                            // Dequantize: (low4 | high2<<4) and (high4 | high2h)
                            let q6_l = vreinterpretq_s8_u8(vsliq_n_u8(
                                vandq_u8(q6_qs_l, m4b),
                                vandq_u8(q6_qs_h, mask_lo),
                                4,
                            ));
                            let q6_h = vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8(q6_qs_l, 4),
                                q6_qs_hh,
                            ));

                            sb_acc_l = vdotq_s32(sb_acc_l, q6_l, q8_l[chunk]);
                            sb_acc_h = vdotq_s32(sb_acc_h, q6_h, q8_h[chunk]);
                        }

                        let scale_idx_l = half * 8 + sb;
                        let scale_idx_h = half * 8 + sb + 4;

                        let scale_vec_l = vmovl_s16(vld1_s16(
                            q6_scales.as_ptr().add(scale_idx_l * 8 + g * 4),
                        ));
                        let scale_vec_h = vmovl_s16(vld1_s16(
                            q6_scales.as_ptr().add(scale_idx_h * 8 + g * 4),
                        ));

                        acc[g] = vmlaq_s32(acc[g], sb_acc_l, scale_vec_l);
                        acc[g] = vmlaq_s32(acc[g], sb_acc_h, scale_vec_h);
                    }
                }
            }

            // Apply bias correction
            acc[0] = vsubq_s32(acc[0], bias_lo);
            acc[1] = vsubq_s32(acc[1], bias_hi);

            // Convert to float and apply block scale
            acc_f32[0] =
                vfmaq_f32(acc_f32[0], vcvtq_f32_s32(acc[0]), sb_scale_0);
            acc_f32[1] =
                vfmaq_f32(acc_f32[1], vcvtq_f32_s32(acc[1]), sb_scale_1);
        }

        let base = x * 8;
        vst1q_f32(output.as_mut_ptr().add(base), acc_f32[0]);
        vst1q_f32(output.as_mut_ptr().add(base + 4), acc_f32[1]);
    }
}
