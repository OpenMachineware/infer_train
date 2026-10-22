// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GPU Embedding Lookup kernel
// Performs embedding lookup and dequantization on GPU
// Supports quantized formats: Q4_K, Q5_K, Q6_K

#include <metal_stdlib>
using namespace metal;

// Include block structure definitions
#define GGML_COMMON_DECL_METAL
#include "../include/blocks.h"

#define QK_K 256

// ============================================================================
// F16 Embedding Lookup
// ============================================================================

kernel void kernel_embedding_lookup_f16(
    device const half* embeddings [[buffer(0)]],  // [hidden_size, vocab_size] in F16
    device float* output [[buffer(1)]],            // [hidden_size] output
    constant uint& token [[buffer(2)]],            // Token ID to look up
    constant uint& hidden_size [[buffer(3)]],      // Hidden size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden_size) return;

    // GGUF stores embeddings transposed: shape is [hidden_size, vocab_size]
    // Token N's embedding starts at offset: N * hidden_size
    device const half* embed_ptr = embeddings + token * hidden_size;
    output[gid] = float(embed_ptr[gid]);
}

// ============================================================================
// Q4_K Embedding Lookup
// ============================================================================

// Helper function for Q4_K scale extraction
static inline uchar2 get_scale_min_k4(int j, int k, device const uchar * scales) {
    if (j < 4) {
        return uchar2(uchar(scales[j+0+k] & 63), uchar(scales[j+4+k] & 63));
    } else {
        return uchar2(
            uchar((scales[j+4+k] & 0x0F) | ((scales[j-4+k] & 0xC0) >> 2)),
            uchar((scales[j+4+k] >> 4) | ((scales[j-0+k] & 0xC0) >> 2))
        );
    }
}

kernel void kernel_embedding_lookup_q4_k(
    device const char* embeddings [[buffer(0)]],   // Q4_K quantized embeddings
    device float* output [[buffer(1)]],             // [hidden_size] output
    constant uint& token [[buffer(2)]],             // Token ID to look up
    constant uint& hidden_size [[buffer(3)]],       // Hidden size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden_size) return;

    const int nb = hidden_size / QK_K;  // Number of Q4_K blocks per embedding
    const int ib = gid / QK_K;          // Block index
    const int i_in_block = gid % QK_K;  // Index within block

    // Each embedding is nb blocks, token N's embedding starts at block N*nb
    device const block_q4_K* blocks = (device const block_q4_K*)embeddings;
    device const block_q4_K* block = blocks + token * nb + ib;

    // Dequantize this element
    device const half* dh = (device const half*)&block->d;
    float d = dh[0];
    float dmin = dh[1];

    const int is = i_in_block / 32;  // Sub-block index (8 sub-blocks per block)
    const int i_in_sub = i_in_block % 32;

    uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
    float sc = sc_min.x;
    float m = sc_min.y;
    float dl = d * sc;
    float ml = dmin * m;

    device const uchar* qs = block->qs + is * 16;
    const int byte_idx = i_in_sub / 2;
    const uchar byte = qs[byte_idx];

    float value;
    if (i_in_sub % 2 == 0) {
        value = float(byte & 0x0F);
    } else {
        value = float((byte >> 4) & 0x0F);
    }

    output[gid] = dl * value - ml;
}

// ============================================================================
// Q5_K Embedding Lookup
// ============================================================================

kernel void kernel_embedding_lookup_q5_k(
    device const char* embeddings [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden_size) return;

    const int nb = hidden_size / QK_K;
    const int ib = gid / QK_K;
    const int i_in_block = gid % QK_K;

    device const block_q5_K* blocks = (device const block_q5_K*)embeddings;
    device const block_q5_K* block = blocks + token * nb + ib;

    device const half* dh = (device const half*)&block->d;
    float d = dh[0];
    float dmin = dh[1];

    const int is = i_in_block / 32;
    const int i_in_sub = i_in_block % 32;

    uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
    float sc = sc_min.x;
    float m = sc_min.y;
    float dl = d * sc;
    float ml = dmin * m;

    device const uchar* qs = block->qs + is * 16;
    device const uchar* qh = block->qh + is * 4;
    const int byte_idx = i_in_sub / 2;
    const uchar byte = qs[byte_idx];
    const uchar high_bits = qh[byte_idx / 4];

    float value;
    if (i_in_sub % 2 == 0) {
        value = float(byte & 0x0F) + ((high_bits & 0x01) ? 16.0f : 0.0f);
    } else {
        value = float((byte >> 4) & 0x0F) + ((high_bits & 0x02) ? 16.0f : 0.0f);
    }

    output[gid] = dl * value - ml;
}

// ============================================================================
// Q6_K Embedding Lookup
// ============================================================================

kernel void kernel_embedding_lookup_q6_k(
    device const char* embeddings [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden_size) return;

    const int nb = hidden_size / QK_K;
    const int ib = gid / QK_K;
    const int i_in_block = gid % QK_K;

    device const block_q6_K* blocks = (device const block_q6_K*)embeddings;
    device const block_q6_K* block = blocks + token * nb + ib;

    device const half* dh = (device const half*)&block->d;
    float d = dh[0];

    const int is = i_in_block / 16;  // Q6_K has 16 sub-blocks of 16 elements
    const int i_in_sub = i_in_block % 16;

    float sc = block->scales[is];

    device const uchar* ql = block->ql + is * 8;
    device const uchar* qh = block->qh + is * 4;
    const int byte_idx = i_in_sub / 2;
    const uchar l = ql[byte_idx];
    const uchar h = qh[byte_idx];

    float value;
    if (i_in_sub % 2 == 0) {
        value = float(l & 0x0F) + float(h & 0x03) * 16.0f;
    } else {
        value = float((l >> 4) & 0x0F) + float((h >> 2) & 0x03) * 16.0f;
    }

    output[gid] = d * sc * value;
}