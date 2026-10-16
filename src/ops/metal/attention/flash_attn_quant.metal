// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Self-written Flash Attention kernels with quantized KV support
// Dequantizes K/V on-the-fly during attention computation
//
// SPDX-License-Identifier: AGPL-3.0
// SPDX-FileCopyrightText: Copyright (c) 2025 Jia Liu

#include <metal_stdlib>
using namespace metal;

#define QK 32

// Block structures
struct block_q4_0 {
    half d;
    uint8_t qs[QK / 2];
};

struct block_q4_1 {
    half d;
    half m;
    uint8_t qs[QK / 2];
};

struct block_q5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[QK / 2];
};

struct block_q5_1 {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[QK / 2];
};

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};

// F16 and BF16 pseudo-blocks (4 elements each for type4 interface)
struct block_f16 {
    half h[4];
};

#if __METAL_VERSION__ >= 310 || defined(GGML_METAL_HAS_BF16)
struct block_bf16 {
    bfloat bf[4];
};
#endif

// IQ4_NL: 4-bit with non-linear lookup table (32 elements per block)
struct block_iq4_nl {
    half d;
    uint8_t qs[QK / 2];  // 16 bytes, each holds 2 4-bit indices
};

// K-quant block structures (256 elements per block)
struct block_q2_K {
    half d;
    half dmin;
    uint8_t scales[16];
    uint8_t qs[64];
};

struct block_q3_K {
    half d;
    uint8_t hmask[32];
    uint8_t qs[64];
    int8_t scales[12];
};

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];  // 4-bit quants, packed 2 per byte, 256 elements / 2 = 128 bytes
};

struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];   // High bit for each of 256 elements (256/8 = 32)
    uint8_t qs[128];  // 4-bit quants, packed 2 per byte, 256 elements / 2 = 128 bytes
};

struct block_q6_K {
    half d;
    int8_t scales[16];
    uint8_t ql[128];
    uint8_t qh[64];
};

// Dequantize functions (inline from dequantize.metal)
// Q4_0 dequantize - 4 elements at a time
template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

// Q4_1 dequantize - 4 elements at a time
template <typename type4>
void dequantize_q4_1_t4(device const block_q4_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = (il/4) ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float m = xb->m;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

// Q5_0 dequantize - 4 elements at a time
template <typename type4>
void dequantize_q5_0_t4(device const block_q5_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.0h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;
    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ? 0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

// Q5_1 dequantize - 4 elements at a time
template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;
    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ? 0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

// Q8_0 dequantize - 4 elements at a time
template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}

constant float kvalues_iq4nl[16] = {
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0
};

// IQ4_NL dequantize - 4 elements at a time
// Layout: qs[j] = indices[j] (low) | indices[j+16] (high)
// il=0..3: low nibbles (elements 0..15), il=4..7: high nibbles (elements 16..31)
template <typename type4>
void dequantize_iq4_nl_t4(device const block_iq4_nl *xb, short il, thread type4 & reg) {
    const float d = xb->d;
    // Use explicit pointer offset to avoid potential struct padding issues
    device const uint8_t * qs = (device const uint8_t *)((device const uint16_t *)xb + 1);

    // il determines which nibbles and which group of 4 bytes
    // il=0: bytes 0..3 low nibbles (elements 0..3)
    // il=1: bytes 4..7 low nibbles (elements 4..7)
    // il=2: bytes 8..11 low nibbles (elements 8..11)
    // il=3: bytes 12..15 low nibbles (elements 12..15)
    // il=4: bytes 0..3 high nibbles (elements 16..19)
    // il=5: bytes 4..7 high nibbles (elements 20..23)
    // il=6: bytes 8..11 high nibbles (elements 24..27)
    // il=7: bytes 12..15 high nibbles (elements 28..31)

    short byte_group = il % 4;  // Which group of 4 bytes
    bool high_nibble = il >= 4;

    // Read 4 bytes using explicit offset
    uint8_t b0 = qs[byte_group * 4 + 0];
    uint8_t b1 = qs[byte_group * 4 + 1];
    uint8_t b2 = qs[byte_group * 4 + 2];
    uint8_t b3 = qs[byte_group * 4 + 3];

    if (high_nibble) {
        reg[0] = kvalues_iq4nl[(b0 >> 4) & 0x0F] * d;
        reg[1] = kvalues_iq4nl[(b1 >> 4) & 0x0F] * d;
        reg[2] = kvalues_iq4nl[(b2 >> 4) & 0x0F] * d;
        reg[3] = kvalues_iq4nl[(b3 >> 4) & 0x0F] * d;
    } else {
        reg[0] = kvalues_iq4nl[b0 & 0x0F] * d;
        reg[1] = kvalues_iq4nl[b1 & 0x0F] * d;
        reg[2] = kvalues_iq4nl[b2 & 0x0F] * d;
        reg[3] = kvalues_iq4nl[b3 & 0x0F] * d;
    }
}

// F16 dequantize - identity conversion (half to float)
template <typename type4>
void dequantize_f16_t4(device const block_f16 * xb, short il, thread type4 & reg) {
    // il is ignored for F16 - each call reads 4 consecutive elements
    reg = (type4)float4(xb->h[0], xb->h[1], xb->h[2], xb->h[3]);
}

#if __METAL_VERSION__ >= 310 || defined(GGML_METAL_HAS_BF16)
// BF16 dequantize - identity conversion (bfloat to float)
template <typename type4>
void dequantize_bf16_t4(device const block_bf16 * xb, short il, thread type4 & reg) {
    // il is ignored for BF16 - each call reads 4 consecutive elements
    reg = (type4)float4(xb->bf[0], xb->bf[1], xb->bf[2], xb->bf[3]);
}
#endif

// K-quant dequantize functions - type4x4 (16 elements at a time)
// These handle the complex layout of K-quant formats

template <typename type4x4>
void dequantize_q2_K_t4x4(device const block_q2_K * xb, short il, thread type4x4 & reg) {
    // Q2_K: 2-bit quantization, 256 elements per block
    // qs[64]: 4 2-bit values per byte
    // scales[16]: one scale per 16-element sub-block
    // il ranges 0..15, each produces 16 elements

    const float d = xb->d;
    const float min = xb->dmin;

    // Each byte has 4 2-bit values
    // il * 4 = start element index for this iteration
    short byte_idx = il;  // 1 byte per il (4 elements), but we need 16 elements
    device const uint8_t * q = xb->qs + il * 4;  // 4 bytes = 16 elements

    // Scale for this sub-block (16 elements each)
    short sub_block = il;
    uint8_t sc = xb->scales[sub_block];
    float dl = d * (sc & 0x0F);
    float ml = min * (sc >> 4);

    for (int i = 0; i < 4; ++i) {
        uchar byte = q[i];
        for (int j = 0; j < 4; ++j) {
            float val = (byte >> (2 * j)) & 0x03;
            reg[i][j] = dl * val - ml;
        }
    }
}

template <typename type4x4>
void dequantize_q3_K_t4x4(device const block_q3_K * xb, short il, thread type4x4 & reg) {
    // Q3_K: 3-bit quantization, 256 elements per block
    // qs[64]: low 2 bits, 4 per byte
    // hmask[32]: high bit for each element
    // scales[12]: scale per sub-block

    const float d = xb->d;
    device const uint8_t * q = xb->qs + il * 4;  // 4 bytes = 16 elements
    device const uint8_t * h = xb->hmask;

    // Scale for this sub-block
    short sub_block = il / 2;  // 8 sub-blocks of 32 elements
    const int8_t sc = xb->scales[sub_block];

    for (int i = 0; i < 4; ++i) {
        uchar byte = q[i];
        for (int j = 0; j < 4; ++j) {
            short idx = il * 16 + i * 4 + j;
            float low2 = (byte >> (2 * j)) & 0x03;
            float high1 = (h[idx / 8] >> (idx % 8)) & 1;

            // 3-bit value: [0,7], signed as [−4, 3]
            float val = low2 + high1 * 4.0 - 4.0;

            reg[i][j] = d * sc * val;
        }
    }
}

template <typename type4x4>
void dequantize_q4_K_t4x4(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    // Q4_K: 4-bit quantization, 256 elements per block
    // qs[128]: two 4-bit values per byte
    // Layout: qs[j] = element j (low nibble) | element j+128 (high nibble)
    // For head_dim < 256, we use only the low nibbles (elements 0..127)
    // il ranges 0..15 for head_dim=256, 0..7 for head_dim=128

    const float d = xb->d;
    const float min = xb->dmin;

    // For head_dim=128: each il produces 16 elements from 16 bytes
    short byte_idx = il * 16;  // 16 bytes per il
    device const uchar * q = xb->qs + byte_idx;

    for (int i = 0; i < 4; ++i) {
        uchar byte0 = q[i * 4];
        uchar byte1 = q[i * 4 + 1];
        uchar byte2 = q[i * 4 + 2];
        uchar byte3 = q[i * 4 + 3];

        // Use only low nibbles (elements 0..127)
        reg[i][0] = (byte0 & 0x0F) * d + min;
        reg[i][1] = (byte1 & 0x0F) * d + min;
        reg[i][2] = (byte2 & 0x0F) * d + min;
        reg[i][3] = (byte3 & 0x0F) * d + min;
    }
}

template <typename type4x4>
void dequantize_q5_K_t4x4(device const block_q5_K * xb, short il, thread type4x4 & reg) {
    // Q5_K: 5-bit quantization, 256 elements per block
    // qs[128]: low 4 bits, two per byte
    // qh[32]: high bit for each element
    // Layout: qs[j] = element j (low nibble) | element j+128 (high nibble)
    // For head_dim < 256, use elements 0..127 (low nibbles)

    const float d = xb->d;
    const float min = xb->dmin;

    short byte_idx = il * 16;
    device const uchar * q = xb->qs + byte_idx;
    device const uchar * qh = xb->qh;

    for (int i = 0; i < 4; ++i) {
        uchar byte0 = q[i * 4];
        uchar byte1 = q[i * 4 + 1];
        uchar byte2 = q[i * 4 + 2];
        uchar byte3 = q[i * 4 + 3];

        // Low nibbles (elements 0..127)
        float low0 = (byte0 & 0x0F);
        float low1 = (byte1 & 0x0F);
        float low2 = (byte2 & 0x0F);
        float low3 = (byte3 & 0x0F);

        // High bits from qh (elements 0..127)
        short idx0 = byte_idx + i * 4;
        short idx1 = byte_idx + i * 4 + 1;
        short idx2 = byte_idx + i * 4 + 2;
        short idx3 = byte_idx + i * 4 + 3;
        float h0 = ((qh[idx0 / 8] >> (idx0 % 8)) & 1) * 16.0;
        float h1 = ((qh[idx1 / 8] >> (idx1 % 8)) & 1) * 16.0;
        float h2 = ((qh[idx2 / 8] >> (idx2 % 8)) & 1) * 16.0;
        float h3 = ((qh[idx3 / 8] >> (idx3 % 8)) & 1) * 16.0;

        reg[i][0] = (low0 + h0) * d + min;
        reg[i][1] = (low1 + h1) * d + min;
        reg[i][2] = (low2 + h2) * d + min;
        reg[i][3] = (low3 + h3) * d + min;
    }
}

template <typename type4x4>
void dequantize_q6_K_t4x4(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    // Q6_K: 6-bit signed quantization, 256 elements per block
    // ql[128]: lower 4 bits, two per byte
    // qh[64]: upper 2 bits, packed
    // Layout: ql[j] = element j (low nibble) | element j+128 (high nibble)
    // For head_dim < 256, use elements 0..127 (low nibbles)

    const float d = xb->d;

    short byte_idx = il * 16;
    device const uchar * ql = xb->ql + byte_idx;
    device const uchar * qh = xb->qh;

    for (int i = 0; i < 4; ++i) {
        uchar byte0 = ql[i * 4];
        uchar byte1 = ql[i * 4 + 1];
        uchar byte2 = ql[i * 4 + 2];
        uchar byte3 = ql[i * 4 + 3];

        // Low nibbles (elements 0..127)
        float low0 = (byte0 & 0x0F);
        float low1 = (byte1 & 0x0F);
        float low2 = (byte2 & 0x0F);
        float low3 = (byte3 & 0x0F);

        // Upper 2 bits from qh
        short idx0 = byte_idx + i * 4;
        short idx1 = byte_idx + i * 4 + 1;
        short idx2 = byte_idx + i * 4 + 2;
        short idx3 = byte_idx + i * 4 + 3;
        short qh_idx0 = idx0 / 2;
        short qh_idx1 = idx1 / 2;
        short qh_idx2 = idx2 / 2;
        short qh_idx3 = idx3 / 2;
        uchar qh_byte0 = qh[qh_idx0];
        uchar qh_byte1 = qh[qh_idx1];
        uchar qh_byte2 = qh[qh_idx2];
        uchar qh_byte3 = qh[qh_idx3];
        float h0 = (qh_byte0 >> (idx0 % 2 * 4)) & 0x03;
        float h1 = (qh_byte1 >> (idx1 % 2 * 4)) & 0x03;
        float h2 = (qh_byte2 >> (idx2 % 2 * 4)) & 0x03;
        float h3 = (qh_byte3 >> (idx3 % 2 * 4)) & 0x03;

        // Reconstruct: signed 6-bit [-32, 31]
        reg[i][0] = (low0 + h0 * 16.0 - 32.0) * d;
        reg[i][1] = (low1 + h1 * 16.0 - 32.0) * d;
        reg[i][2] = (low2 + h2 * 16.0 - 32.0) * d;
        reg[i][3] = (low3 + h3 * 16.0 - 32.0) * d;
    }
}

struct FlashAttnArgs {
    uint32_t batch;
    uint32_t heads;
    uint32_t seq_q;
    uint32_t seq_k;
    uint32_t head_dim;
    float scale;
    uint32_t q_stride_b;
    uint32_t q_stride_h;
    uint32_t q_stride_s;
    uint32_t k_stride_b;  // In bytes, stride between K rows
    uint32_t k_stride_h;
    uint32_t k_stride_s;
    uint32_t v_stride_b;
    uint32_t v_stride_h;
    uint32_t v_stride_s;
    uint32_t o_stride_b;
    uint32_t o_stride_h;
    uint32_t o_stride_s;
};

// ============================================================================
// Decode kernel: one SIMD group (32 threads) per head
// Each thread processes multiple KV positions
// ============================================================================

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &)>

kernel void flash_decode_attention_quant(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    constexpr ushort BLK_SIZE = 32;
    constexpr ushort BLK4 = BLK_SIZE / 4;
    constexpr ushort SIMD_SIZE = 32;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    // Online softmax state per thread
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Early return if this thread has no positions to process
    if (tiisg >= seq_k) {
        return;
    }

    // Process KV positions assigned to this thread
    for (uint kv_idx = tiisg; kv_idx < seq_k; kv_idx += SIMD_SIZE) {
        // Compute Q*K
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        for (ushort blk = 0; blk < head_dim / BLK_SIZE; blk++) {
            device const k_block_t * kb = (device const k_block_t *)(k_row + blk * sizeof(k_block_t));
            for (ushort il = 0; il < BLK4; il++) {
                type4 reg;
                deq_k(kb, il, reg);
                ushort d4 = blk * BLK4 + il;
                // Vectorized dot product (4 elements at once)
                float4 q4 = *(device const float4 *)(q_ptr + d4 * 4);
                score += dot((float4)reg, q4);
            }
        }
        score *= scale;

        // Online softmax update
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        for (ushort blk = 0; blk < head_dim / BLK_SIZE; blk++) {
            device const v_block_t * vb = (device const v_block_t *)(v_row + blk * sizeof(v_block_t));
            for (ushort il = 0; il < BLK4; il++) {
                type4 reg;
                deq_v(vb, il, reg);
                ushort d4 = blk * BLK4 + il;
                // Vectorized accumulation
                *(thread float4 *)(local_out + d4 * 4) += weight * (float4)reg;
            }
        }
    }

    // SIMD reduction
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    // Vectorized scaling
    for (ushort d4 = 0; d4 < head_dim / 4; d4++) {
        *(thread float4 *)(local_out + d4 * 4) *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);

    // SIMD sum for each dimension - ALL threads must participate
    float sg_out[128];
    // Vectorized SIMD sum
    for (ushort d4 = 0; d4 < head_dim / 4; d4++) {
        float4 tmp = *(thread float4 *)(local_out + d4 * 4);
        float4 sum4;
        sum4.x = simd_sum(tmp.x);
        sum4.y = simd_sum(tmp.y);
        sum4.z = simd_sum(tmp.z);
        sum4.w = simd_sum(tmp.w);
        *(thread float4 *)(sg_out + d4 * 4) = sum4;
    }

    // Write output - only thread 0 writes
    if (tiisg == 0) {
        for (ushort d = 0; d < head_dim; d++) {
            output[b * args.o_stride_b + h * args.o_stride_h + d] = sg_out[d] / sg_sum;
        }
    }
}

// ============================================================================
// Prefill kernel: one threadgroup per query position
// Uses multiple SIMD groups for parallel KV processing
// ============================================================================

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &)>

kernel void flash_prefill_attention_quant(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const uint sq = tgpig.x;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    constexpr ushort BLK_SIZE = 32;
    constexpr ushort BLK4 = BLK_SIZE / 4;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h + sq * args.q_stride_s;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = ntg.x;

    // Online softmax state
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Process KV positions
    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += num_threads) {
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        for (ushort blk = 0; blk < head_dim / BLK_SIZE; blk++) {
            device const k_block_t * kb = (device const k_block_t *)(k_row + blk * sizeof(k_block_t));
            for (ushort il = 0; il < BLK4; il++) {
                type4 reg;
                deq_k(kb, il, reg);
                ushort d4 = blk * BLK4 + il;
                // Vectorized dot product (4 elements at once)
                float4 q4 = *(device const float4 *)(q_ptr + d4 * 4);
                score += dot((float4)reg, q4);
            }
        }
        score *= scale;

        // Online softmax
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        for (ushort blk = 0; blk < head_dim / BLK_SIZE; blk++) {
            device const v_block_t * vb = (device const v_block_t *)(v_row + blk * sizeof(v_block_t));
            for (ushort il = 0; il < BLK4; il++) {
                type4 reg;
                deq_v(vb, il, reg);
                ushort d4 = blk * BLK4 + il;
                // Vectorized accumulation
                *(thread float4 *)(local_out + d4 * 4) += weight * (float4)reg;
            }
        }
    }

    // SIMD reduction
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);
    float sg_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        sg_out[d] = simd_sum(local_out[d]);
    }

    // Cross-SIMD reduction via threadgroup memory
    threadgroup float tg_max[4];
    threadgroup float tg_sum[4];
    threadgroup float tg_out[128 * 4];

    if (tiisg == 0) {
        tg_max[sgitg] = sg_max;
        tg_sum[sgitg] = sg_sum;
        for (ushort d = 0; d < head_dim; d++) {
            tg_out[sgitg * 128 + d] = sg_out[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float global_max = tg_max[0];
        for (ushort i = 1; i < 4; i++) {
            if (i * 32 < num_threads) {
                global_max = max(global_max, tg_max[i]);
            }
        }

        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort i = 0; i < 4; i++) {
            if (i * 32 < num_threads) {
                float scale_g = exp(tg_max[i] - global_max);
                global_sum += tg_sum[i] * scale_g;
                for (ushort d = 0; d < head_dim; d++) {
                    global_out[d] += tg_out[i * 128 + d] * scale_g;
                }
            }
        }

        device float * o_ptr = output + b * args.o_stride_b + h * args.o_stride_h + sq * args.o_stride_s;
        for (ushort d = 0; d < head_dim; d++) {
            o_ptr[d] = global_out[d] / global_sum;
        }
    }
}

// ============================================================================
// F16/BF16 kernels: linear format (no block structure)
// Each pseudo-block is 4 elements (8 bytes)
// ============================================================================

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &)>

kernel void flash_decode_attention_f16(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    constexpr ushort SIMD_SIZE = 32;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    // Online softmax state per thread
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Early return if this thread has no positions to process
    if (tiisg >= seq_k) {
        return;
    }

    // Process KV positions assigned to this thread
    for (uint kv_idx = tiisg; kv_idx < seq_k; kv_idx += SIMD_SIZE) {
        // Compute Q*K - linear format, each pseudo-block is 4 elements
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        for (ushort blk = 0; blk < head_dim / 4; blk++) {
            device const k_block_t * kb = (device const k_block_t *)(k_row + blk * sizeof(k_block_t));
            type4 reg;
            deq_k(kb, 0, reg);  // il is ignored for linear format
            for (ushort j = 0; j < 4; j++) {
                score += q_ptr[blk * 4 + j] * reg[j];
            }
        }
        score *= scale;

        // Online softmax update
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        for (ushort blk = 0; blk < head_dim / 4; blk++) {
            device const v_block_t * vb = (device const v_block_t *)(v_row + blk * sizeof(v_block_t));
            type4 reg;
            deq_v(vb, 0, reg);
            for (ushort j = 0; j < 4; j++) {
                local_out[blk * 4 + j] += weight * reg[j];
            }
        }
    }

    // SIMD reduction
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);
    float sg_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        sg_out[d] = simd_sum(local_out[d]);
    }

    if (tiisg == 0) {
        for (ushort d = 0; d < head_dim; d++) {
            output[b * args.o_stride_b + h * args.o_stride_h + d] = sg_out[d] / sg_sum;
        }
    }
}

// ============================================================================
// Adaptive Decode kernel: supports multiple SIMD groups for better GPU utilization
// Uses threadgroup memory for cross-SIMD reduction
// NSG (number of SIMD groups) determines parallelism based on seq_k
// ============================================================================

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &)>

kernel void flash_decode_attention_adaptive(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    threadgroup float * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    // Get NSG from function constant or default
    const ushort nsg = ntg.y;  // Number of SIMD groups
    const ushort n_threads = ntg.x * nsg;  // Total threads

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    // Threadgroup memory for cross-SIMD reduction
    // Layout: [nsg max_values][nsg sum_values][nsg * head_dim output]
    threadgroup float * sg_max_vals = shmem;
    threadgroup float * sg_sum_vals = shmem + nsg;
    threadgroup float * sg_outputs = shmem + 2 * nsg;

    // Online softmax state per thread
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Global thread index within the threadgroup
    const ushort tid = sgitg * 32 + tiisg;

    // Early return if this thread has no positions to process
    if (tid >= seq_k) {
        // Still need to write to threadgroup memory for reduction
        sg_max_vals[sgitg] = local_max;
        sg_sum_vals[sgitg] = 0.0f;
        for (ushort d = 0; d < head_dim; d++) {
            sg_outputs[sgitg * head_dim + d] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }

    // Process KV positions assigned to this thread
    // Each thread processes positions: tid, tid + n_threads, tid + 2*n_threads, ...
    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += n_threads) {
        // Compute Q*K
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        for (ushort blk = 0; blk < head_dim / 32; blk++) {
            device const k_block_t * kb = (device const k_block_t *)(k_row + blk * sizeof(k_block_t));
            for (ushort il = 0; il < 8; il++) {
                type4 reg;
                deq_k(kb, il, reg);
                ushort d4 = blk * 8 + il;
                // Vectorized dot product
                float4 q4 = *(device const float4 *)(q_ptr + d4 * 4);
                score += dot((float4)reg, q4);
            }
        }
        score *= scale;

        // Online softmax update
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        for (ushort blk = 0; blk < head_dim / 32; blk++) {
            device const v_block_t * vb = (device const v_block_t *)(v_row + blk * sizeof(v_block_t));
            for (ushort il = 0; il < 8; il++) {
                type4 reg;
                deq_v(vb, il, reg);
                ushort d4 = blk * 8 + il;
                // Vectorized accumulation
                *(thread float4 *)(local_out + d4 * 4) += weight * (float4)reg;
            }
        }
    }

    // SIMD-level reduction within each SIMD group
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    // Vectorized scaling
    for (ushort d4 = 0; d4 < head_dim / 4; d4++) {
        *(thread float4 *)(local_out + d4 * 4) *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);
    float sg_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        sg_out[d] = simd_sum(local_out[d]);
    }

    // Write SIMD group results to threadgroup memory
    if (tiisg == 0) {
        sg_max_vals[sgitg] = sg_max;
        sg_sum_vals[sgitg] = sg_sum;
        for (ushort d = 0; d < head_dim; d++) {
            sg_outputs[sgitg * head_dim + d] = sg_out[d];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups (only SIMD group 0)
    if (sgitg == 0) {
        // Find global max
        float global_max = sg_max_vals[0];
        for (ushort sg = 1; sg < nsg; sg++) {
            global_max = max(global_max, sg_max_vals[sg]);
        }

        // Scale sums and compute global sum
        float global_sum = 0.0f;
        for (ushort sg = 0; sg < nsg; sg++) {
            float scale_sg = exp(sg_max_vals[sg] - global_max);
            sg_sum_vals[sg] *= scale_sg;
            global_sum += sg_sum_vals[sg];

            // Scale outputs
            for (ushort d = 0; d < head_dim; d++) {
                sg_outputs[sg * head_dim + d] *= scale_sg;
            }
        }

        // Sum outputs and write result
        // Use all 32 threads in SIMD group 0 to write all head_dim outputs
        for (ushort d = tiisg; d < head_dim; d += 32) {
            float sum = 0.0f;
            for (ushort sg = 0; sg < nsg; sg++) {
                sum += sg_outputs[sg * head_dim + d];
            }
            output[b * args.o_stride_b + h * args.o_stride_h + d] = sum / global_sum;
        }
    }
}

// Template instantiations for adaptive decode (F16 format, 4 SIMD groups = 128 threads)
template [[host_name("flash_decode_attention_f16_adaptive")]]
kernel void flash_decode_attention_adaptive<block_f16, block_f16, float4, dequantize_f16_t4, dequantize_f16_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    threadgroup float *, uint3, ushort3, ushort, ushort);

template [[host_name("flash_decode_attention_q4_0_adaptive")]]
kernel void flash_decode_attention_adaptive<block_q4_0, block_q4_0, float4, dequantize_q4_0_t4, dequantize_q4_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    threadgroup float *, uint3, ushort3, ushort, ushort);

template [[host_name("flash_decode_attention_q8_0_adaptive")]]
kernel void flash_decode_attention_adaptive<block_q8_0, block_q8_0, float4, dequantize_q8_0_t4, dequantize_q8_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    threadgroup float *, uint3, ushort3, ushort, ushort);

// ============================================================================
// Uses NWG=32 workgroups, NSG=2 SIMD groups per workgroup
// Partial results are combined via reduction kernel
// ============================================================================

struct PartialResult {
    float m;        // max value
    float s;        // sum value
};

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &)>

kernel void flash_decode_attention_multi_wg(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * partial_out [[buffer(4)]],      // [nwg, head_dim] partial outputs
    device PartialResult * partial_ms [[buffer(5)]], // [nwg] partial max/sum
    threadgroup float * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort iwg = tgpig.x;  // workgroup index (0..nwg-1)
    const ushort nwg = ntg.x;    // number of workgroups

    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    // Each workgroup processes a contiguous chunk of KV positions
    const uint chunk_size = (seq_k + nwg - 1) / nwg;  // ceil division
    const uint kv_start = iwg * chunk_size;
    const uint kv_end = min(kv_start + chunk_size, seq_k);

    if (kv_start >= seq_k) {
        // No work for this workgroup - write zeros
        if (sgitg == 0 && tiisg == 0) {
            partial_ms[iwg] = (PartialResult){ -FLT_MAX/2, 0.0f };
            for (ushort d = 0; d < head_dim; d++) {
                partial_out[iwg * head_dim + d] = 0.0f;
            }
        }
        return;
    }

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    // Threadgroup memory for SIMD reduction
    threadgroup float * sg_max_vals = shmem;
    threadgroup float * sg_sum_vals = shmem + 2;
    threadgroup float * sg_outputs = shmem + 4;

    // Online softmax state per thread
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Process KV positions assigned to this thread
    // Each thread processes positions with stride nsg*32
    const ushort n_threads = 2 * 32;  // 2 SIMD groups * 32 threads
    const ushort tid = sgitg * 32 + tiisg;

    for (uint kv_idx = kv_start + tid; kv_idx < kv_end; kv_idx += n_threads) {
        // Compute Q*K
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        for (ushort blk = 0; blk < head_dim / 32; blk++) {
            device const k_block_t * kb = (device const k_block_t *)(k_row + blk * sizeof(k_block_t));
            for (ushort il = 0; il < 8; il++) {
                type4 reg;
                deq_k(kb, il, reg);
                ushort d4 = blk * 8 + il;
                // Vectorized dot product
                float4 q4 = *(device const float4 *)(q_ptr + d4 * 4);
                score += dot((float4)reg, q4);
            }
        }
        score *= scale;

        // Online softmax update
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        for (ushort blk = 0; blk < head_dim / 32; blk++) {
            device const v_block_t * vb = (device const v_block_t *)(v_row + blk * sizeof(v_block_t));
            for (ushort il = 0; il < 8; il++) {
                type4 reg;
                deq_v(vb, il, reg);
                ushort d4 = blk * 8 + il;
                // Vectorized accumulation
                *(thread float4 *)(local_out + d4 * 4) += weight * (float4)reg;
            }
        }
    }

    // SIMD-level reduction within each SIMD group
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    // Vectorized scaling
    for (ushort d4 = 0; d4 < head_dim / 4; d4++) {
        *(thread float4 *)(local_out + d4 * 4) *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);
    float sg_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        sg_out[d] = simd_sum(local_out[d]);
    }

    // Write SIMD group results to threadgroup memory
    if (tiisg == 0) {
        sg_max_vals[sgitg] = sg_max;
        sg_sum_vals[sgitg] = sg_sum;
        for (ushort d = 0; d < head_dim; d++) {
            sg_outputs[sgitg * head_dim + d] = sg_out[d];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups (only SIMD group 0)
    if (sgitg == 0) {
        float global_max = sg_max_vals[0];
        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort sg = 0; sg < 2; sg++) {
            float scale_g = exp(sg_max_vals[sg] - global_max);
            // Rescale previous
            if (sg > 0) {
                global_sum *= scale_g;
                for (ushort d = 0; d < head_dim; d++) {
                    global_out[d] *= scale_g;
                }
            }
            global_max = max(global_max, sg_max_vals[sg]);
            float new_scale = exp(sg_max_vals[sg] - global_max);
            global_sum += sg_sum_vals[sg] * new_scale;
            for (ushort d = 0; d < head_dim; d++) {
                global_out[d] += sg_outputs[sg * head_dim + d] * new_scale;
            }
        }

        // Store partial result for this workgroup
        partial_ms[iwg].m = global_max;
        partial_ms[iwg].s = global_sum;
        for (ushort d = tiisg; d < head_dim; d += 32) {
            partial_out[iwg * head_dim + d] = global_out[d];
        }
    }
}

// Reduction kernel: combines partial results from multiple workgroups
kernel void flash_decode_attention_reduce(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * partial_out [[buffer(1)]],      // [nwg, head_dim]
    device const PartialResult * partial_ms [[buffer(2)]], // [nwg]
    device float * output [[buffer(3)]],
    constant ushort & nwg [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort head_dim = args.head_dim;

    // Find global max
    float global_max = partial_ms[0].m;
    for (ushort wg = 1; wg < nwg; wg++) {
        global_max = max(global_max, partial_ms[wg].m);
    }

    // Combine partial results with online softmax
    float global_sum = 0.0f;
    float final_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        final_out[d] = 0.0f;
    }

    for (ushort wg = 0; wg < nwg; wg++) {
        float scale = exp(partial_ms[wg].m - global_max);
        global_sum += partial_ms[wg].s * scale;
        for (ushort d = 0; d < head_dim; d++) {
            final_out[d] += partial_out[wg * head_dim + d] * scale;
        }
    }

    // Write final output
    if (tiisg == 0) {
        for (ushort d = 0; d < head_dim; d++) {
            output[b * args.o_stride_b + h * args.o_stride_h + d] = final_out[d] / global_sum;
        }
    }
}

// Template instantiations for multi-wg decode
template [[host_name("flash_decode_attention_f16_multi_wg")]]
kernel void flash_decode_attention_multi_wg<block_f16, block_f16, float4, dequantize_f16_t4, dequantize_f16_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *, device PartialResult *,
    threadgroup float *, uint3, ushort3, ushort, ushort);

template [[host_name("flash_decode_attention_q4_0_multi_wg")]]
kernel void flash_decode_attention_multi_wg<block_q4_0, block_q4_0, float4, dequantize_q4_0_t4, dequantize_q4_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *, device PartialResult *,
    threadgroup float *, uint3, ushort3, ushort, ushort);

template [[host_name("flash_decode_attention_q8_0_multi_wg")]]
kernel void flash_decode_attention_multi_wg<block_q8_0, block_q8_0, float4, dequantize_q8_0_t4, dequantize_q8_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *, device PartialResult *,
    threadgroup float *, uint3, ushort3, ushort, ushort);

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &)>

kernel void flash_prefill_attention_f16(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const uint sq = tgpig.x;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h + sq * args.q_stride_s;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = ntg.x;

    // Online softmax state
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Process KV positions
    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += num_threads) {
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        for (ushort blk = 0; blk < head_dim / 4; blk++) {
            device const k_block_t * kb = (device const k_block_t *)(k_row + blk * sizeof(k_block_t));
            type4 reg;
            deq_k(kb, 0, reg);
            for (ushort j = 0; j < 4; j++) {
                score += q_ptr[blk * 4 + j] * reg[j];
            }
        }
        score *= scale;

        // Online softmax
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        for (ushort blk = 0; blk < head_dim / 4; blk++) {
            device const v_block_t * vb = (device const v_block_t *)(v_row + blk * sizeof(v_block_t));
            type4 reg;
            deq_v(vb, 0, reg);
            for (ushort j = 0; j < 4; j++) {
                local_out[blk * 4 + j] += weight * reg[j];
            }
        }
    }

    // SIMD reduction
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);
    float sg_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        sg_out[d] = simd_sum(local_out[d]);
    }

    // Cross-SIMD reduction via threadgroup memory
    threadgroup float tg_max[4];
    threadgroup float tg_sum[4];
    threadgroup float tg_out[128 * 4];

    if (tiisg == 0) {
        tg_max[sgitg] = sg_max;
        tg_sum[sgitg] = sg_sum;
        for (ushort d = 0; d < head_dim; d++) {
            tg_out[sgitg * 128 + d] = sg_out[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float global_max = tg_max[0];
        for (ushort i = 1; i < 4; i++) {
            if (i * 32 < num_threads) {
                global_max = max(global_max, tg_max[i]);
            }
        }

        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort i = 0; i < 4; i++) {
            if (i * 32 < num_threads) {
                float scale_g = exp(tg_max[i] - global_max);
                global_sum += tg_sum[i] * scale_g;
                for (ushort d = 0; d < head_dim; d++) {
                    global_out[d] += tg_out[i * 128 + d] * scale_g;
                }
            }
        }

        device float * o_ptr = output + b * args.o_stride_b + h * args.o_stride_h + sq * args.o_stride_s;
        for (ushort d = 0; d < head_dim; d++) {
            o_ptr[d] = global_out[d] / global_sum;
        }
    }
}

// ============================================================================
// K-quant kernels: type4x4 format (16 elements per dequantize call)
// K-quant blocks contain 256 elements each
// ============================================================================

template<
    typename k_block_t,
    typename v_block_t,
    typename type4x4,
    void (*deq_k)(device const k_block_t *, short, thread type4x4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4x4 &)>

kernel void flash_decode_attention_kquant(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    constexpr ushort SIMD_SIZE = 32;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    // Online softmax state per thread
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Early return if this thread has no positions to process
    if (tiisg >= seq_k) {
        return;
    }

    // Process KV positions assigned to this thread
    for (uint kv_idx = tiisg; kv_idx < seq_k; kv_idx += SIMD_SIZE) {
        // Compute Q*K - K-quant: 256 elements per block
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        // K-quant block covers 256 elements
        // For head_dim < 256, we use first head_dim elements from the block
        device const k_block_t * kb = (device const k_block_t *)k_row;
        for (ushort il = 0; il < head_dim / 16; il++) {
            type4x4 reg;
            deq_k(kb, il, reg);
            for (ushort i = 0; i < 16; i++) {
                score += q_ptr[il * 16 + i] * reg[i/4][i%4];
            }
        }
        score *= scale;

        // Online softmax update
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        device const v_block_t * vb = (device const v_block_t *)v_row;
        for (ushort il = 0; il < head_dim / 16; il++) {
            type4x4 reg;
            deq_v(vb, il, reg);
            for (ushort i = 0; i < 16; i++) {
                local_out[il * 16 + i] += weight * reg[i/4][i%4];
            }
        }
    }

    // SIMD reduction
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);
    float sg_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        sg_out[d] = simd_sum(local_out[d]);
    }

    if (tiisg == 0) {
        for (ushort d = 0; d < head_dim; d++) {
            output[b * args.o_stride_b + h * args.o_stride_h + d] = sg_out[d] / sg_sum;
        }
    }
}

template<
    typename k_block_t,
    typename v_block_t,
    typename type4x4,
    void (*deq_k)(device const k_block_t *, short, thread type4x4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4x4 &)>

kernel void flash_prefill_attention_kquant(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const uint sq = tgpig.x;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h + sq * args.q_stride_s;
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = ntg.x;

    // Online softmax state
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Process KV positions
    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += num_threads) {
        device const char * k_row = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;

        device const k_block_t * kb = (device const k_block_t *)k_row;
        for (ushort il = 0; il < head_dim / 16; il++) {
            type4x4 reg;
            deq_k(kb, il, reg);
            for (ushort i = 0; i < 16; i++) {
                score += q_ptr[il * 16 + i] * reg[i/4][i%4];
            }
        }
        score *= scale;

        // Online softmax
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Accumulate weighted V
        device const char * v_row = v_base + kv_idx * args.v_stride_s;
        device const v_block_t * vb = (device const v_block_t *)v_row;
        for (ushort il = 0; il < head_dim / 16; il++) {
            type4x4 reg;
            deq_v(vb, il, reg);
            for (ushort i = 0; i < 16; i++) {
                local_out[il * 16 + i] += weight * reg[i/4][i%4];
            }
        }
    }

    // SIMD reduction
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);
    float sg_out[128];
    for (ushort d = 0; d < head_dim; d++) {
        sg_out[d] = simd_sum(local_out[d]);
    }

    // Cross-SIMD reduction via threadgroup memory
    threadgroup float tg_max[4];
    threadgroup float tg_sum[4];
    threadgroup float tg_out[128 * 4];

    if (tiisg == 0) {
        tg_max[sgitg] = sg_max;
        tg_sum[sgitg] = sg_sum;
        for (ushort d = 0; d < head_dim; d++) {
            tg_out[sgitg * 128 + d] = sg_out[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float global_max = tg_max[0];
        for (ushort i = 1; i < 4; i++) {
            if (i * 32 < num_threads) {
                global_max = max(global_max, tg_max[i]);
            }
        }

        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort i = 0; i < 4; i++) {
            if (i * 32 < num_threads) {
                float scale_g = exp(tg_max[i] - global_max);
                global_sum += tg_sum[i] * scale_g;
                for (ushort d = 0; d < head_dim; d++) {
                    global_out[d] += tg_out[i * 128 + d] * scale_g;
                }
            }
        }

        device float * o_ptr = output + b * args.o_stride_b + h * args.o_stride_h + sq * args.o_stride_s;
        for (ushort d = 0; d < head_dim; d++) {
            o_ptr[d] = global_out[d] / global_sum;
        }
    }
}

// ============================================================================
// Template instantiations for each quantization format
// ============================================================================

// Q4_0
template [[host_name("flash_decode_attention_q4_0")]]
kernel void flash_decode_attention_quant<block_q4_0, block_q4_0, float4, dequantize_q4_0_t4, dequantize_q4_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q4_0")]]
kernel void flash_prefill_attention_quant<block_q4_0, block_q4_0, float4, dequantize_q4_0_t4, dequantize_q4_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q4_1
template [[host_name("flash_decode_attention_q4_1")]]
kernel void flash_decode_attention_quant<block_q4_1, block_q4_1, float4, dequantize_q4_1_t4, dequantize_q4_1_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q4_1")]]
kernel void flash_prefill_attention_quant<block_q4_1, block_q4_1, float4, dequantize_q4_1_t4, dequantize_q4_1_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q5_0
template [[host_name("flash_decode_attention_q5_0")]]
kernel void flash_decode_attention_quant<block_q5_0, block_q5_0, float4, dequantize_q5_0_t4, dequantize_q5_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q5_0")]]
kernel void flash_prefill_attention_quant<block_q5_0, block_q5_0, float4, dequantize_q5_0_t4, dequantize_q5_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q5_1
template [[host_name("flash_decode_attention_q5_1")]]
kernel void flash_decode_attention_quant<block_q5_1, block_q5_1, float4, dequantize_q5_1_t4, dequantize_q5_1_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q5_1")]]
kernel void flash_prefill_attention_quant<block_q5_1, block_q5_1, float4, dequantize_q5_1_t4, dequantize_q5_1_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q8_0
template [[host_name("flash_decode_attention_q8_0")]]
kernel void flash_decode_attention_quant<block_q8_0, block_q8_0, float4, dequantize_q8_0_t4, dequantize_q8_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q8_0")]]
kernel void flash_prefill_attention_quant<block_q8_0, block_q8_0, float4, dequantize_q8_0_t4, dequantize_q8_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// IQ4_NL
template [[host_name("flash_decode_attention_iq4_nl")]]
kernel void flash_decode_attention_quant<block_iq4_nl, block_iq4_nl, float4, dequantize_iq4_nl_t4<float4>, dequantize_iq4_nl_t4<float4>>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_iq4_nl")]]
kernel void flash_prefill_attention_quant<block_iq4_nl, block_iq4_nl, float4, dequantize_iq4_nl_t4<float4>, dequantize_iq4_nl_t4<float4>>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// F16 - linear format kernel
template [[host_name("flash_decode_attention_f16")]]
kernel void flash_decode_attention_f16<block_f16, block_f16, float4, dequantize_f16_t4, dequantize_f16_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_f16")]]
kernel void flash_prefill_attention_f16<block_f16, block_f16, float4, dequantize_f16_t4, dequantize_f16_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

#if __METAL_VERSION__ >= 310 || defined(GGML_METAL_HAS_BF16)
// BF16 - linear format kernel
template [[host_name("flash_decode_attention_bf16")]]
kernel void flash_decode_attention_f16<block_bf16, block_bf16, float4, dequantize_bf16_t4, dequantize_bf16_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_bf16")]]
kernel void flash_prefill_attention_f16<block_bf16, block_bf16, float4, dequantize_bf16_t4, dequantize_bf16_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);
#endif

// K-quant formats - type4x4 kernel
// Q2_K
template [[host_name("flash_decode_attention_q2_K")]]
kernel void flash_decode_attention_kquant<block_q2_K, block_q2_K, float4x4, dequantize_q2_K_t4x4, dequantize_q2_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q2_K")]]
kernel void flash_prefill_attention_kquant<block_q2_K, block_q2_K, float4x4, dequantize_q2_K_t4x4, dequantize_q2_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q3_K
template [[host_name("flash_decode_attention_q3_K")]]
kernel void flash_decode_attention_kquant<block_q3_K, block_q3_K, float4x4, dequantize_q3_K_t4x4, dequantize_q3_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q3_K")]]
kernel void flash_prefill_attention_kquant<block_q3_K, block_q3_K, float4x4, dequantize_q3_K_t4x4, dequantize_q3_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q4_K
template [[host_name("flash_decode_attention_q4_K")]]
kernel void flash_decode_attention_kquant<block_q4_K, block_q4_K, float4x4, dequantize_q4_K_t4x4, dequantize_q4_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q4_K")]]
kernel void flash_prefill_attention_kquant<block_q4_K, block_q4_K, float4x4, dequantize_q4_K_t4x4, dequantize_q4_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q5_K
template [[host_name("flash_decode_attention_q5_K")]]
kernel void flash_decode_attention_kquant<block_q5_K, block_q5_K, float4x4, dequantize_q5_K_t4x4, dequantize_q5_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q5_K")]]
kernel void flash_prefill_attention_kquant<block_q5_K, block_q5_K, float4x4, dequantize_q5_K_t4x4, dequantize_q5_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// Q6_K
template [[host_name("flash_decode_attention_q6_K")]]
kernel void flash_decode_attention_kquant<block_q6_K, block_q6_K, float4x4, dequantize_q6_K_t4x4, dequantize_q6_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort);

template [[host_name("flash_prefill_attention_q6_K")]]
kernel void flash_prefill_attention_kquant<block_q6_K, block_q6_K, float4x4, dequantize_q6_K_t4x4, dequantize_q6_K_t4x4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *, device float *,
    uint3, ushort3, ushort, ushort);

// ============================================================================
// High-performance decode: 8x4 grid with stride-8 access
// Each thread processes 4 float4s (16 elements) with stride 8 across head_dim
// Grid: 32 threads = 8 (tx) x 4 (ty), ty handles cache items, tx handles head_dim
// ============================================================================

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    short nl_k,  // Number of float4s per block (e.g., 1 for F16, 8 for Q4_0)
    short nl_v,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &)>

kernel void flash_decode_grid8x4(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    threadgroup half * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // 8x4 grid: tx (0-7) for head_dim, ty (0-3) for cache items within chunk
    const ushort tx = tiisg % 8;
    const ushort ty = tiisg / 8;

    const float scale = args.scale;
    const uint seq_k = args.seq_k;
    const uint b = tgpig.z;
    const uint h = tgpig.y;

    constexpr ushort DK4 = 32;   // 32 float4s = 128 elements
    constexpr ushort NL = 8;     // 8 tx threads
    constexpr ushort NE = 4;     // 4 cache items per chunk
    constexpr ushort ITER = 4;   // 4 float4s per thread

    // Threadgroup memory layout:
    // [0..256): query (32 half4)
    // [256..288): scores for current chunk (32 float = NE*NL)
    threadgroup char * tg_base = (threadgroup char *) shmem;
    threadgroup half4 * sq4 = (threadgroup half4 *) (tg_base + 0);
    threadgroup float * ss = (threadgroup float *) (tg_base + 256);

    // Load query
    device const float4 * q_dev = (device const float4 *)
        (q + b * args.q_stride_b + h * args.q_stride_h);
    if (tiisg < DK4) {
        sq4[tiisg] = half4(q_dev[tiisg]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread online softmax state (will be reduced across threads)
    float M = -FLT_MAX / 2;   // Local max
    float S = 0.0f;           // Local sum

    // Per-thread output accumulator (each thread handles 4 float4s)
    float4 O[ITER];
    for (ushort i = 0; i < ITER; i++) {
        O[i] = float4(0.0f);
    }

    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    // Process cache items in chunks of NE
    const uint num_chunks = (seq_k + NE - 1) / NE;

    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        const uint kv_idx = chunk * NE + ty;
        const bool valid = kv_idx < seq_k;

        // Step 1: Compute Q*K^T for my cache item
        float score = 0.0f;
        if (valid) {
            device const k_block_t * pk = (device const k_block_t *)
                (k_base + kv_idx * args.k_stride_s);
            for (ushort iter = 0; iter < ITER; iter++) {
                const ushort i = iter * NL + tx;
                type4 k_vec;
                deq_k(pk + i / nl_k, i % nl_k, k_vec);
                score += dot((float4) k_vec, (float4) sq4[i]);
            }
        } else {
            score = -FLT_MAX / 2;  // Invalid cache item
        }

        // Step 2: Reduce score across tx (within each ty group)
        score += simd_shuffle_xor(score, 4);
        score += simd_shuffle_xor(score, 2);
        score += simd_shuffle_xor(score, 1);
        // Now tx=0 has the reduced score for each ty

        // Step 3: Store score to threadgroup memory
        // After reduction, only tx=0 has the correct score for each ty
        // We need to broadcast to all tx threads
        if (tx == 0) {
            ss[ty] = score * scale;  // Store at position ty (0-3)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: Online softmax update
        // Process each cache item sequentially (online softmax algorithm)
        {
            for (ushort cc = 0; cc < NE; cc++) {
                const uint item_idx = chunk * NE + cc;
                if (item_idx >= seq_k) {
                    ss[cc] = 0.0f;  // Invalid item, zero weight
                    continue;
                }

                const float s = ss[cc];  // Score for cc-th cache item

                // Only tx==0 computes the new M and weight
                float ms = 1.0f;  // Scaling factor for old output
                float vs = 0.0f;  // Weight for new V

                if (tx == 0) {
                    const float m = M;
                    M = max(M, s);

                    ms = exp(m - M);  // Scaling factor
                    vs = exp(s - M);  // New weight

                    S = S * ms + vs;

                    ss[cc] = vs;  // Store weight for V accumulation
                    ss[NE] = M;   // Store updated M
                    ss[NE + 1] = ms;  // Store scaling factor
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // All threads read the scaling factor and weight
                if (tx != 0) {
                    M = ss[NE];
                    ms = ss[NE + 1];
                }

                // All threads scale their output
                for (ushort i = 0; i < ITER; i++) {
                    O[i] *= ms;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: V accumulation - ALL threads participate
        {
            for (ushort cc = 0; cc < NE; cc++) {
                const uint item_idx = chunk * NE + cc;
                if (item_idx >= seq_k) continue;

                const float weight = ss[cc];  // Weight for this cache item

                device const v_block_t * pv = (device const v_block_t *)
                    (v_base + item_idx * args.v_stride_s);

                for (ushort iter = 0; iter < ITER; iter++) {
                    const ushort i = iter * NL + tx;
                    type4 v_vec;
                    deq_v(pv + i / nl_v, i % nl_v, v_vec);
                    O[iter] += weight * (float4) v_vec;
                }
            }
        }
    }

    // After all chunks, broadcast S from tx==0 to all threads
    // (Only tx==0 updated S during softmax)
    if (tx == 0) {
        ss[0] = S;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float final_sum = ss[0];

    // Final output: normalize and write (only ty=0 to avoid duplicate writes)
    if (ty == 0) {
        const float inv_sum = 1.0f / final_sum;
        for (ushort iter = 0; iter < ITER; iter++) {
            const ushort i = iter * NL + tx;
            const float4 result = O[iter] * inv_sum;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 0] = result.x;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 1] = result.y;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 2] = result.z;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 3] = result.w;
        }
    }
}

// Instantiations
template [[host_name("flash_decode_grid8x4_f16")]]
kernel void flash_decode_grid8x4<block_f16, block_f16, float4, 1, 1, dequantize_f16_t4, dequantize_f16_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort);

template [[host_name("flash_decode_grid8x4_q4_0")]]
kernel void flash_decode_grid8x4<block_q4_0, block_q4_0, float4, 8, 8, dequantize_q4_0_t4, dequantize_q4_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort);

template [[host_name("flash_decode_grid8x4_q8_0")]]
kernel void flash_decode_grid8x4<block_q8_0, block_q8_0, float4, 8, 8, dequantize_q8_0_t4, dequantize_q8_0_t4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort);

// ============================================================================
// Multi-SIMD Group Flash Attention Decode
// Each threadgroup has multiple SIMD groups for parallel cache processing
// ============================================================================

template<
    typename k_block_t,
    typename v_block_t,
    typename type4,
    short nl_k,
    short nl_v,
    void (*deq_k)(device const k_block_t *, short, thread type4 &),
    void (*deq_v)(device const v_block_t *, short, thread type4 &),
    short NUM_GROUPS>  // Number of SIMD groups per threadgroup

kernel void flash_decode_multi_group(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    threadgroup half * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Thread organization
    const ushort lane_id = tiisg;          // 0-31, thread within SIMD group
    const ushort group_id = sgitg;         // 0..NUM_GROUPS-1, SIMD group within threadgroup

    const float scale = args.scale;
    const uint seq_k = args.seq_k;
    const uint b = tgpig.z;
    const uint h = tgpig.y;

    // Constants
    constexpr ushort DK4 = 32;     // 32 float4s = 128 elements
    constexpr ushort LANES = 32;   // Threads per SIMD group
    constexpr ushort CHUNK = 4;    // Cache items per chunk
    constexpr ushort ITER = 4;     // Float4s per thread for head_dim

    // Threadgroup memory layout
    // [0..128): query (32 float4s)
    // [128..128+NUM_GROUPS*16): per-group M and S (2 floats each, 16 aligned)
    // [128+NUM_GROUPS*16..]: per-group output accumulators
    threadgroup float4 * query_vec = (threadgroup float4 *) shmem;
    threadgroup float * group_ms = (threadgroup float *) (shmem + 128);
    threadgroup float4 * group_out = (threadgroup float4 *) (shmem + 128 + NUM_GROUPS * 16);

    // Load query (all threads participate)
    device const float4 * q_dev = (device const float4 *)
        (q + b * args.q_stride_b + h * args.q_stride_h);
    if (lane_id < DK4) {
        query_vec[lane_id] = float4(q_dev[lane_id]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-group state for online softmax
    float local_M = -FLT_MAX / 2;
    float local_S = 0.0f;
    float4 local_O[ITER];
    for (ushort i = 0; i < ITER; i++) {
        local_O[i] = float4(0.0f);
    }

    // Sub-thread indexing for head_dim parallelization
    const ushort tx = lane_id % 8;  // Head_dim dimension
    const ushort ty = lane_id / 8;  // Auxiliary

    // Pointers to K/V cache
    device const char * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const char * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    // Process cache items in chunks
    // Each SIMD group starts at a different offset
    const uint group_stride = NUM_GROUPS;  // Number of groups in this threadgroup
    const uint start_chunk = group_id;     // Starting chunk for this group

    for (uint chunk_idx = start_chunk; chunk_idx * CHUNK < seq_k; chunk_idx += group_stride) {
        const uint base_item = chunk_idx * CHUNK;

        // Process CHUNK cache items
        for (ushort c = 0; c < CHUNK; c++) {
            const uint item_idx = base_item + c;
            if (item_idx >= seq_k) continue;

            // Step 1: Compute Q*K^T score
            float score = 0.0f;
            device const k_block_t * pk = (device const k_block_t *)
                (k_base + item_idx * args.k_stride_s);

            for (ushort iter = 0; iter < ITER; iter++) {
                const ushort i = iter * 8 + tx;
                type4 k_vec;
                deq_k(pk + i / nl_k, i % nl_k, k_vec);
                score += dot((float4) k_vec, query_vec[i]);
            }

            // Reduce score across threads in SIMD group
            score += simd_shuffle_xor(score, 4);
            score += simd_shuffle_xor(score, 2);
            score += simd_shuffle_xor(score, 1);
            // Now thread 0 has the complete score

            // Threadgroup storage for this chunk's scores
            threadgroup float * chunk_scores = group_ms + group_id * 16;
            if (lane_id == 0) {
                chunk_scores[c] = score * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 2: Online softmax update
            if (lane_id == 0) {
                const float s = chunk_scores[c];
                const float old_M = local_M;
                local_M = max(local_M, s);

                const float scale_old = exp(old_M - local_M);
                const float weight = exp(s - local_M);

                local_S = local_S * scale_old + weight;

                chunk_scores[c] = weight;  // Store weight for V accumulation
                chunk_scores[CHUNK + 0] = scale_old;  // Store scale for all threads
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 3: All threads scale previous output
            const float scale_factor = chunk_scores[CHUNK + 0];
            for (ushort iter = 0; iter < ITER; iter++) {
                local_O[iter] *= scale_factor;
            }

            // Step 4: All threads accumulate weighted V
            const float weight = chunk_scores[c];
            device const v_block_t * pv = (device const v_block_t *)
                (v_base + item_idx * args.v_stride_s);
            for (ushort iter = 0; iter < ITER; iter++) {
                const ushort i = iter * 8 + tx;
                type4 v_vec;
                deq_v(pv + i / nl_v, i % nl_v, v_vec);
                local_O[iter] += weight * (float4) v_vec;
            }
        }
    }

    // Store per-group results to threadgroup memory
    // M and S
    if (lane_id == 0) {
        group_ms[group_id * 2 + 0] = local_S;
        group_ms[group_id * 2 + 1] = local_M;
    }

    // Output accumulator
    // Each thread has ITER float4s, indexed by (iter, tx)
    // Total per group: ITER * 8 float4s
    for (ushort iter = 0; iter < ITER; iter++) {
        const ushort idx = group_id * ITER * 8 + iter * 8 + tx;
        group_out[idx] = local_O[iter];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Tree reduction across SIMD groups
    for (ushort r = NUM_GROUPS / 2; r > 0; r >>= 1) {
        if (group_id < r) {
            // Combine group [group_id] and group [group_id + r]
            const float S0 = group_ms[(group_id) * 2 + 0];
            const float M0 = group_ms[(group_id) * 2 + 1];
            const float S1 = group_ms[(group_id + r) * 2 + 0];
            const float M1 = group_ms[(group_id + r) * 2 + 1];

            const float Mx = max(M0, M1);
            const float scale0 = exp(M0 - Mx);
            const float scale1 = exp(M1 - Mx);

            const float combined_S = S0 * scale0 + S1 * scale1;

            // Update group 0's state
            if (lane_id == 0) {
                group_ms[group_id * 2 + 0] = combined_S;
                group_ms[group_id * 2 + 1] = Mx;
            }

            // Combine outputs
            for (ushort iter = 0; iter < ITER; iter++) {
                const ushort idx0 = group_id * ITER * 8 + iter * 8 + tx;
                const ushort idx1 = (group_id + r) * ITER * 8 + iter * 8 + tx;
                const float4 O0 = group_out[idx0];
                const float4 O1 = group_out[idx1];
                group_out[idx0] = O0 * scale0 + O1 * scale1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 5: Final output (only group 0 writes)
    if (group_id == 0 && ty == 0) {
        const float final_S = group_ms[0];
        const float inv_S = 1.0f / final_S;

        for (ushort iter = 0; iter < ITER; iter++) {
            const ushort idx = iter * 8 + tx;
            const ushort i = iter * 8 + tx;  // Element index in head_dim
            const float4 result = group_out[idx] * inv_S;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 0] = result.x;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 1] = result.y;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 2] = result.z;
            output[b * args.o_stride_b + h * args.o_stride_h + i * 4 + 3] = result.w;
        }
    }
}

// Instantiations for F16 with different group counts
template [[host_name("flash_decode_multi_group_f16_g1")]]
kernel void flash_decode_multi_group<block_f16, block_f16, float4, 1, 1, dequantize_f16_t4, dequantize_f16_t4, 1>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_f16_g2")]]
kernel void flash_decode_multi_group<block_f16, block_f16, float4, 1, 1, dequantize_f16_t4, dequantize_f16_t4, 2>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_f16_g4")]]
kernel void flash_decode_multi_group<block_f16, block_f16, float4, 1, 1, dequantize_f16_t4, dequantize_f16_t4, 4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_f16_g8")]]
kernel void flash_decode_multi_group<block_f16, block_f16, float4, 1, 1, dequantize_f16_t4, dequantize_f16_t4, 8>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

// Q4_0
template [[host_name("flash_decode_multi_group_q4_0_g1")]]
kernel void flash_decode_multi_group<block_q4_0, block_q4_0, float4, 8, 8, dequantize_q4_0_t4, dequantize_q4_0_t4, 1>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q4_0_g2")]]
kernel void flash_decode_multi_group<block_q4_0, block_q4_0, float4, 8, 8, dequantize_q4_0_t4, dequantize_q4_0_t4, 2>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q4_0_g4")]]
kernel void flash_decode_multi_group<block_q4_0, block_q4_0, float4, 8, 8, dequantize_q4_0_t4, dequantize_q4_0_t4, 4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q4_0_g8")]]
kernel void flash_decode_multi_group<block_q4_0, block_q4_0, float4, 8, 8, dequantize_q4_0_t4, dequantize_q4_0_t4, 8>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

// Q8_0
template [[host_name("flash_decode_multi_group_q8_0_g1")]]
kernel void flash_decode_multi_group<block_q8_0, block_q8_0, float4, 8, 8, dequantize_q8_0_t4, dequantize_q8_0_t4, 1>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q8_0_g2")]]
kernel void flash_decode_multi_group<block_q8_0, block_q8_0, float4, 8, 8, dequantize_q8_0_t4, dequantize_q8_0_t4, 2>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q8_0_g4")]]
kernel void flash_decode_multi_group<block_q8_0, block_q8_0, float4, 8, 8, dequantize_q8_0_t4, dequantize_q8_0_t4, 4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q8_0_g8")]]
kernel void flash_decode_multi_group<block_q8_0, block_q8_0, float4, 8, 8, dequantize_q8_0_t4, dequantize_q8_0_t4, 8>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

// Q4_1
template [[host_name("flash_decode_multi_group_q4_1_g1")]]
kernel void flash_decode_multi_group<block_q4_1, block_q4_1, float4, 8, 8, dequantize_q4_1_t4, dequantize_q4_1_t4, 1>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q4_1_g2")]]
kernel void flash_decode_multi_group<block_q4_1, block_q4_1, float4, 8, 8, dequantize_q4_1_t4, dequantize_q4_1_t4, 2>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q4_1_g4")]]
kernel void flash_decode_multi_group<block_q4_1, block_q4_1, float4, 8, 8, dequantize_q4_1_t4, dequantize_q4_1_t4, 4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q4_1_g8")]]
kernel void flash_decode_multi_group<block_q4_1, block_q4_1, float4, 8, 8, dequantize_q4_1_t4, dequantize_q4_1_t4, 8>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

// Q5_0
template [[host_name("flash_decode_multi_group_q5_0_g1")]]
kernel void flash_decode_multi_group<block_q5_0, block_q5_0, float4, 8, 8, dequantize_q5_0_t4, dequantize_q5_0_t4, 1>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q5_0_g2")]]
kernel void flash_decode_multi_group<block_q5_0, block_q5_0, float4, 8, 8, dequantize_q5_0_t4, dequantize_q5_0_t4, 2>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q5_0_g4")]]
kernel void flash_decode_multi_group<block_q5_0, block_q5_0, float4, 8, 8, dequantize_q5_0_t4, dequantize_q5_0_t4, 4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q5_0_g8")]]
kernel void flash_decode_multi_group<block_q5_0, block_q5_0, float4, 8, 8, dequantize_q5_0_t4, dequantize_q5_0_t4, 8>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

// Q5_1
template [[host_name("flash_decode_multi_group_q5_1_g1")]]
kernel void flash_decode_multi_group<block_q5_1, block_q5_1, float4, 8, 8, dequantize_q5_1_t4, dequantize_q5_1_t4, 1>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q5_1_g2")]]
kernel void flash_decode_multi_group<block_q5_1, block_q5_1, float4, 8, 8, dequantize_q5_1_t4, dequantize_q5_1_t4, 2>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q5_1_g4")]]
kernel void flash_decode_multi_group<block_q5_1, block_q5_1, float4, 8, 8, dequantize_q5_1_t4, dequantize_q5_1_t4, 4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_q5_1_g8")]]
kernel void flash_decode_multi_group<block_q5_1, block_q5_1, float4, 8, 8, dequantize_q5_1_t4, dequantize_q5_1_t4, 8>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

// BF16
template [[host_name("flash_decode_multi_group_bf16_g1")]]
kernel void flash_decode_multi_group<block_bf16, block_bf16, float4, 1, 1, dequantize_bf16_t4, dequantize_bf16_t4, 1>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_bf16_g2")]]
kernel void flash_decode_multi_group<block_bf16, block_bf16, float4, 1, 1, dequantize_bf16_t4, dequantize_bf16_t4, 2>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_bf16_g4")]]
kernel void flash_decode_multi_group<block_bf16, block_bf16, float4, 1, 1, dequantize_bf16_t4, dequantize_bf16_t4, 4>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);

template [[host_name("flash_decode_multi_group_bf16_g8")]]
kernel void flash_decode_multi_group<block_bf16, block_bf16, float4, 1, 1, dequantize_bf16_t4, dequantize_bf16_t4, 8>(
    constant FlashAttnArgs &, device const float *, device const char *, device const char *,
    device float *, threadgroup half *, uint3, ushort, ushort);
