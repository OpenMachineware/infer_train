// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Self-written K-quant Matrix-Vector multiplication kernels
// NOT copied from llama.cpp - original implementation based on algorithm understanding

#include <metal_stdlib>
using namespace metal;

// Include block structure definitions
#define GGML_COMMON_DECL_METAL
#include "../include/blocks.h"

#define QK_K 256
#define N_SIMDWIDTH 32

// MV args structure for K-quant kernels
struct kquant_mv_args {
    uint32_t ne00;  // K dimension (input size)
    uint32_t ne01;  // M dimension (output rows)
    uint64_t nb01;  // Byte stride between rows of weight matrix
};

// Convert f16 bits (u16) to f32
// Rust stores f16 as u16 bits, not as Metal half
static inline float f16_bits_to_f32(uint16_t bits) {
    uint16_t sign = (bits >> 15) & 1;
    uint16_t exp = (bits >> 10) & 0x1F;
    uint16_t mant = bits & 0x3FF;

    if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal: (-1)^sign * 2^-14 * (mant/1024)
        float val = (sign ? -1.0f : 1.0f) * pow(2.0f, -14.0f) * (mant / 1024.0f);
        return val;
    } else if (exp == 31) {
        // Infinity or NaN
        return sign ? -INFINITY : INFINITY;
    }

    // Normalized: (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    float val = (sign ? -1.0f : 1.0f) * pow(2.0f, float(exp) - 15.0f) * (1.0f + mant / 1024.0f);
    return val;
}

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Helper function for Q4_K/Q5_K scale extraction
// j: sub-block index (0-7), k: which sub-block within the pair (0 or 1)
// Returns (scale, min) packed as uchar2
static inline uchar2 get_scale_min_k4(int j, int k, device const uchar * scales) {
    if (j < 4) {
        // Lower 6 bits of scales[j+k] and scales[j+4+k]
        return uchar2(uchar(scales[j+k] & 63), uchar(scales[j+4+k] & 63));
    } else {
        // For j >= 4, scales are packed differently
        // scale: lower 4 bits of scales[j+4+k] combined with upper 2 bits of scales[j-4+k]
        // min: upper 4 bits of scales[j+4+k] combined with upper 2 bits of scales[j+k]
        return uchar2(
            uchar((scales[j+4+k] & 0x0F) | ((scales[j-4+k] & 0xC0) >> 2)),
            uchar((scales[j+4+k] >> 4) | ((scales[j+k] & 0xC0) >> 2))
        );
    }
}

// ============================================================================
// Q4_K Matrix-Vector Multiplication
// Block: 256 elements, 144 bytes
// Layout: d[2] + dmin[2] + scales[12] + qs[128]
// 8 sub-blocks of 32 elements each, each has scale and min
// ============================================================================

kernel void kernel_mul_mv_q4_K_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x;  // Each threadgroup handles one row
    if (row >= (int)args.ne01) return;

    const int nb = args.ne00 / QK_K;  // Number of blocks per row

    float sumf = 0.0f;

    // Get pointer to the weights for this row
    device const char * row_ptr = src0 + row * args.nb01;
    device const block_q4_K * blocks = (device const block_q4_K *)row_ptr;

    // Each thread processes blocks at stride N_SIMDWIDTH
    for (int ib = tiisg; ib < nb; ib += N_SIMDWIDTH) {
        device const block_q4_K * block = blocks + ib;
        device const float * y = src1 + ib * QK_K;

        // Super-block scales - read as u16 bits, convert to f32
        uint16_t d_bits = *((device const uint16_t*)(&block->d));
        uint16_t dmin_bits = *((device const uint16_t*)(&block->dmin));
        float d = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        // Process 8 sub-blocks of 32 elements
        for (int is = 0; is < 8; is++) {
            // Extract scale and min for this sub-block
            uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
            float sc = sc_min.x;
            float min = sc_min.y;

            float dl = d * sc;
            float ml = dmin * min;

            // 32 elements stored in 16 bytes (4 bits each)
            device const uchar * qs = block->qs + is * 16;

            // Process 16 bytes (32 elements)
            for (int k = 0; k < 16; k++) {
                uchar byte = qs[k];
                float v0 = (byte & 0x0F);  // Lower nibble
                float v1 = ((byte >> 4) & 0x0F);  // Upper nibble

                sumf += (dl * v0 - ml) * y[is * 32 + k];
                sumf += (dl * v1 - ml) * y[is * 32 + 16 + k];
            }
        }
    }

    // SIMD reduction across threads
    float sum_all = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = sum_all;
    }
}

// ============================================================================
// Q2_K Matrix-Vector Multiplication
// Block: 256 elements, 84 bytes
// Layout: scales[16] + qs[64] + d[2] + dmin[2]
// 16 sub-blocks of 16 elements each
// ============================================================================

kernel void kernel_mul_mv_q2_K_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x;  // Each threadgroup handles one row
    if (row >= (int)args.ne01) return;

    const int nb = args.ne00 / QK_K;  // Number of blocks per row

    float sumf = 0.0f;

    // Get pointer to the weights for this row
    device const char * row_ptr = src0 + row * args.nb01;
    device const block_q2_K * blocks = (device const block_q2_K *)row_ptr;

    // Each thread processes blocks at stride N_SIMDWIDTH
    for (int ib = tiisg; ib < nb; ib += N_SIMDWIDTH) {
        device const block_q2_K * block = blocks + ib;
        device const float * y = src1 + ib * QK_K;

        uint16_t d_bits = *((device const uint16_t*)(&block->d));
        uint16_t dmin_bits = *((device const uint16_t*)(&block->dmin));
        float d = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        // Q2_K has 16 sub-blocks of 16 elements
        // Each sub-block has its own scale (4 bits) and min (4 bits) packed in scales[16]
        // qs[64] stores 2-bit quantized values: 256 elements / 4 = 64 bytes

        // Process 16 sub-blocks
        for (int is = 0; is < 16; is++) {
            // Scale and min are each 4 bits, packed in one byte
            uchar sc_min = block->scales[is];
            float sc = (sc_min & 0x0F);  // Lower 4 bits
            float min = (sc_min >> 4);    // Upper 4 bits

            float dl = d * sc;
            float ml = dmin * min;

            // 16 elements stored in 4 bytes (2 bits each)
            // Each byte packs four 2-bit values
            device const uchar * qs = block->qs + is * 4;

            for (int k = 0; k < 4; k++) {
                uchar byte = qs[k];
                // Extract four 2-bit values
                for (int b = 0; b < 4; b++) {
                    float q = (byte >> (b * 2)) & 0x03;
                    // Q2_K: 2-bit unsigned -> signed: 0,1,2,3 -> -2,-1,0,1
                    float val = q - 2.0f;
                    float v = dl * val - ml;
                    sumf += v * y[is * 16 + k * 4 + b];
                }
            }
        }
    }

    // SIMD reduction across threads
    float sum_all = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = sum_all;
    }
}

// ============================================================================
// Q3_K Matrix-Vector Multiplication
// Block: 256 elements, 110 bytes
// Layout: hmask[32] + qs[64] + scales[12] + d[2]
// 16 sub-blocks of 16 elements each, 3-bit quantization
// ============================================================================

kernel void kernel_mul_mv_q3_K_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x;  // Each threadgroup handles one row
    if (row >= (int)args.ne01) return;

    const int nb = args.ne00 / QK_K;  // Number of blocks per row

    float sumf = 0.0f;

    // Get pointer to the weights for this row
    device const char * row_ptr = src0 + row * args.nb01;
    device const block_q3_K * blocks = (device const block_q3_K *)row_ptr;

    // Each thread processes blocks at stride N_SIMDWIDTH
    for (int ib = tiisg; ib < nb; ib += N_SIMDWIDTH) {
        device const block_q3_K * block = blocks + ib;
        device const float * y = src1 + ib * QK_K;

        uint16_t d_bits = *((device const uint16_t*)(&block->d));
        float d = f16_bits_to_f32(d_bits);

        // Q3_K: 16 sub-blocks, 3-bit quantization
        // scales[12] contains 16 scales (6 bits each), packed
        // qs[64] has lower 2 bits, hmask[32] has the high bit
        // Each sub-block: 16 elements
        // Formula: v = dl * q - (hbit ? 0 : ml) where ml = 4 * dl

        // Process 16 sub-blocks
        for (int is = 0; is < 16; is++) {
            // Extract 6-bit scale from scales[12]
            // scales are packed: 3 bytes for every 4 scales
            int byte_idx = (is / 4) * 3;
            int bit_offset = (is % 4) * 6;
            int byte_off = bit_offset / 8;
            int bit_off = bit_offset % 8;

            uchar scale;
            if (bit_off <= 2) {
                scale = (block->scales[byte_idx + byte_off] >> bit_off) & 0x3F;
            } else {
                // Spans two bytes
                scale = ((block->scales[byte_idx + byte_off] >> bit_off) |
                        (block->scales[byte_idx + byte_off + 1] << (8 - bit_off))) & 0x3F;
            }

            // Q3_K scales are stored as 6-bit values offset by 32
            float sc = scale - 32.0f;
            float dl = d * sc;
            float ml = 4.0f * dl;  // Offset for negative quant values

            // 16 elements per sub-block
            // Each element: 2 bits from qs[64], 1 bit from hmask[32]
            device const uchar * qs = block->qs + is * 4;
            device const uchar * hm = block->hmask + is * 2;

            for (int k = 0; k < 4; k++) {
                uchar byte = qs[k];
                uchar hbyte = hm[k / 2];  // 4 qs bytes share 2 hmask bytes
                for (int b = 0; b < 4; b++) {
                    int elem_idx = k * 4 + b;
                    int hbit = (hbyte >> (elem_idx % 8)) & 1;
                    int q = (byte >> (b * 2)) & 0x03;

                    // If hbit is set, v = dl * q
                    // If hbit is not set, v = dl * q - ml = dl * (q - 4)
                    float v = dl * q - (hbit ? 0.0f : ml);

                    sumf += v * y[is * 16 + elem_idx];
                }
            }
        }
    }

    // SIMD reduction across threads
    float sum_all = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = sum_all;
    }
}

// Decode Q5_K scales from 12-byte format
// Returns 8 scales and 4 mins packed in a struct
// Matches the CPU dequant_q5_k scale extraction exactly
struct q5_k_scales {
    uint8_t scales[8];
    uint8_t mins[4];
};

static inline struct q5_k_scales decode_q5_k_scales(device const uchar * scales) {
    // Read 12 bytes as 4 uint32
    // Match CPU's unsafe copy into utmp[4]
    uint32_t utmp[4];
    utmp[0] = scales[0] | (scales[1] << 8) | (scales[2] << 16) | (scales[3] << 24);
    utmp[1] = scales[4] | (scales[5] << 8) | (scales[6] << 16) | (scales[7] << 24);
    utmp[2] = scales[8] | (scales[9] << 8) | (scales[10] << 16) | (scales[11] << 24);

    const uint32_t KMASK1 = 0x3f3f3f3f;
    const uint32_t KMASK2 = 0x0f0f0f0f;
    const uint32_t KMASK3 = 0x03030303;

    // Match CPU code exactly
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    uint32_t uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // Now: utmp[0] = scales[0..3], utmp[1] = scales[4..7], utmp[2] = mins[0..3]
    // transmute to [u8; 16]
    // scales_bytes[0..7] = scales, scales_bytes[8..11] = mins

    struct q5_k_scales result;
    result.scales[0] = utmp[0] & 0x3F;
    result.scales[1] = (utmp[0] >> 8) & 0x3F;
    result.scales[2] = (utmp[0] >> 16) & 0x3F;
    result.scales[3] = (utmp[0] >> 24) & 0x3F;
    result.scales[4] = utmp[1] & 0x3F;
    result.scales[5] = (utmp[1] >> 8) & 0x3F;
    result.scales[6] = (utmp[1] >> 16) & 0x3F;
    result.scales[7] = (utmp[1] >> 24) & 0x3F;
    result.mins[0] = utmp[2] & 0x3F;
    result.mins[1] = (utmp[2] >> 8) & 0x3F;
    result.mins[2] = (utmp[2] >> 16) & 0x3F;
    result.mins[3] = (utmp[2] >> 24) & 0x3F;

    return result;
}

// ============================================================================
// Q5_K Matrix-Vector Multiplication
// Block: 256 elements, 176 bytes
// Layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
// 8 sub-blocks of 32 elements each, 5-bit quantization
// ============================================================================

kernel void kernel_mul_mv_q5_K_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x;  // Each threadgroup handles one row
    if (row >= (int)args.ne01) return;

    const int nb = args.ne00 / QK_K;  // Number of blocks per row

    float sumf = 0.0f;

    // Get pointer to the weights for this row
    device const char * row_ptr = src0 + row * args.nb01;
    device const block_q5_K * blocks = (device const block_q5_K *)row_ptr;

    // Each thread processes blocks at stride N_SIMDWIDTH
    for (int ib = tiisg; ib < nb; ib += N_SIMDWIDTH) {
        device const block_q5_K * block = blocks + ib;
        device const float * y = src1 + ib * QK_K;

        uint16_t d_bits = *((device const uint16_t*)(&block->d));
        uint16_t dmin_bits = *((device const uint16_t*)(&block->dmin));
        float d = f16_bits_to_f32(d_bits);
        float dmin = f16_bits_to_f32(dmin_bits);

        // Decode Q5_K scales: 8 scales + 4 mins
        struct q5_k_scales decoded = decode_q5_k_scales(block->scales);

        // Q5_K: 256 elements per block
        // Use 4 iterations of 64 elements each (matches CPU NEON pattern)
        for (int iter = 0; iter < 4; iter++) {
            // Get scales for this iteration
            float scale0 = decoded.scales[iter * 2];
            float scale1 = decoded.scales[iter * 2 + 1];
            float min0 = decoded.mins[iter / 2];  // mins are shared: [0] for iter 0,1; [1] for iter 2,3

            float dl0 = d * scale0;
            float ml0 = dmin * min0;
            float dl1 = d * scale1;
            float ml1 = dmin * min0;  // Same min for both scales

            // Process 64 elements per iteration
            // Groups 0-1: low nibbles, bit 0 of qh
            // Groups 2-3: high nibbles, bit 1 of qh
            for (int k = 0; k < 16; k++) {
                // Group 0: qs[iter*32 + k] low nibble, qh[k] bit 0
                uchar qs0 = block->qs[iter * 32 + k];
                float low0 = (qs0 & 0x0F);
                float h0 = ((block->qh[k] >> 0) & 1) * 16.0f;
                float q0 = low0 + h0;
                sumf += (dl0 * q0 - ml0) * y[iter * 64 + k];

                // Group 1: qs[iter*32 + 16 + k] low nibble, qh[16+k] bit 0
                uchar qs1 = block->qs[iter * 32 + 16 + k];
                float low1 = (qs1 & 0x0F);
                float h1 = ((block->qh[16 + k] >> 0) & 1) * 16.0f;
                float q1 = low1 + h1;
                sumf += (dl0 * q1 - ml0) * y[iter * 64 + 16 + k];

                // Group 2: qs[iter*32 + k] high nibble, qh[k] bit 1
                float high0 = ((qs0 >> 4) & 0x0F);
                float h2 = ((block->qh[k] >> 1) & 1) * 16.0f;
                float q2 = high0 + h2;
                sumf += (dl1 * q2 - ml1) * y[iter * 64 + 32 + k];

                // Group 3: qs[iter*32 + 16 + k] high nibble, qh[16+k] bit 1
                float high1 = ((qs1 >> 4) & 0x0F);
                float h3 = ((block->qh[16 + k] >> 1) & 1) * 16.0f;
                float q3 = high1 + h3;
                sumf += (dl1 * q3 - ml1) * y[iter * 64 + 48 + k];
            }
        }
    }

    // SIMD reduction across threads
    float sum_all = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = sum_all;
    }
}

// ============================================================================
// Q6_K Matrix-Vector Multiplication
// Block: 256 elements, 210 bytes
// Layout: ql[128] + qh[64] + scales[16] + d[2]
// 16 sub-blocks of 16 elements each, 6-bit quantization
// Note: ql/qh have interleaved layout, not linear per sub-block
// ============================================================================

kernel void kernel_mul_mv_q6_K_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x;  // Each threadgroup handles one row
    if (row >= (int)args.ne01) return;

    const int nb = args.ne00 / QK_K;  // Number of blocks per row

    float sumf = 0.0f;

    // Get pointer to the weights for this row
    device const char * row_ptr = src0 + row * args.nb01;
    device const block_q6_K * blocks = (device const block_q6_K *)row_ptr;

    // Each thread processes blocks at stride N_SIMDWIDTH
    for (int ib = tiisg; ib < nb; ib += N_SIMDWIDTH) {
        device const block_q6_K * block = blocks + ib;
        device const float * y = src1 + ib * QK_K;

        uint16_t d_bits = *((device const uint16_t*)(&block->d));
        float d = f16_bits_to_f32(d_bits);

        // Q6_K uses interleaved ql/qh layout
        // ql and qh are read as uint16 for SIMD processing
        device const uint16_t * ql16 = (device const uint16_t *)block->ql;
        device const uint16_t * qh16 = (device const uint16_t *)block->qh;

        // Process 16 sub-blocks (il = 0..15)
        for (int il = 0; il < 16; il++) {
            float sc = block->scales[il];
            float dl = d * sc;
            float ml = 32.0f * dl;  // Offset for signed quant

            // Compute offsets for interleaved layout
            device const uint16_t * ql_sub = ql16 + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
            device const uint16_t * qh_sub = qh16 + 16*(il/8) + 8*(il&1);

            // Process 16 elements using 4 iterations of 4 elements each
            // Use masks based on (il/2)&3
            int il2 = (il/2) & 3;
            const uint32_t kmask1 = il2>1 ? (il2>2 ? 0xC0C0C0C0 : 0x30303030) : (il2>0 ? 0x0C0C0C0C : 0x03030303);
            const uint32_t kmask2 = il2>1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
            const uint8_t shr_h = il2>2 ? 2 : 0;
            const uint8_t shl_h = il2>1 ? 0 : (il2>0 ? 2 : 4);
            const uint8_t shr_l = il2>1 ? 4 : 0;

            for (int i = 0; i < 4; i++) {
                const uint32_t low = (ql_sub[2*i] | ((uint32_t)ql_sub[2*i+1] << 16)) & kmask2;
                const uint32_t high = (qh_sub[2*i] | ((uint32_t)qh_sub[2*i+1] << 16)) & kmask1;
                const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);

                // Extract 4 bytes (4 elements)
                for (int b = 0; b < 4; b++) {
                    uint8_t qv = (q >> (b * 8)) & 0xFF;
                    // Convert to signed: qv - 32
                    float v = dl * (qv - 32.0f);
                    sumf += v * y[il * 16 + i * 4 + b];
                }
            }
        }
    }

    // SIMD reduction across threads
    float sum_all = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = sum_all;
    }
}
