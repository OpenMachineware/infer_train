// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Transposed GEMV kernels for quantized formats
// Handles weights stored as [K, M] (transposed layout)

#include <metal_stdlib>
using namespace metal;

#define QK 256

// Q6_K block: 256 elements, 210 bytes
struct BlockQ6K {
    uint8_t ql[128];    // Lower 4 bits
    uint8_t qh[64];     // Upper 2 bits
    int8_t scales[16];  // Scales
    uint16_t d;         // Super-block scale
};

// Dequantize a single element from Q6_K block
// Returns value at position i (0-255) within the block
float dequantize_q6_k_elem(
    device const BlockQ6K& block,
    uint i
) {
    // Q6_K packs 256 values into:
    // - ql[128]: low 4 bits of 128 pairs (256 values)
    // - qh[64]: high 2 bits of 64 groups of 4 (256 values)
    // - scales[16]: one scale per 16 values (16 groups)
    // - d: super-block scale

    uint j = i / 16;  // Which group of 16
    uint l = i % 16;  // Position within group

    // Get the scale for this group of 16
    float scale = block.scales[j] / 32.0f;  // Scale normalization

    // Get low 4 bits
    uint ql_idx = i / 2;  // Which byte in ql
    uint ql_shift = (i % 2) * 4;  // Which nibble
    uint8_t ql_val = (block.ql[ql_idx] >> ql_shift) & 0x0F;

    // Get high 2 bits
    uint qh_idx = i / 4;  // Which byte in qh
    uint qh_shift = (i % 4) * 2;  // Which 2-bit value
    uint8_t qh_val = (block.qh[qh_idx] >> qh_shift) & 0x03;

    // Combine to get 6-bit value (0-63)
    uint q = (qh_val << 4) | ql_val;

    // Dequantize: q is in range 0-63, map to signed range
    // Q6_K uses a specific quantization scheme
    float val = (float(q) - 32.0f) * scale * float(block.d);

    return val;
}

// Transposed GEMV: y = W^T * x
// W is stored as [K, M] in Q6_K format
// Each row of W has (M/QK) blocks
// To compute y[j], we need to gather element j from each of K blocks
//
// For efficiency, we compute multiple y elements in parallel
// Each threadgroup processes a batch of outputs

kernel void mv_q6_k_f32_transposed(
    device const BlockQ6K* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],  // Output size (vocab_size)
    constant uint& k [[buffer(4)]],  // Input size (hidden_size)
    uint tid [[thread_position_in_grid]])
{
    // Each thread computes one output element
    if (tid >= m) return;

    float sum = 0.0f;

    // Number of blocks per row
    uint blocks_per_row = m / QK;

    // For output element tid, we need to sum:
    // sum = sum over i of W[i, tid] * x[i]
    // where W is stored as [K, M]
    //
    // W[i, tid] is at:
    // - Row i (in terms of blocks)
    // - Block tid/QK within that row
    // - Element tid % QK within that block

    uint tid_block = tid / QK;
    uint tid_offset = tid % QK;

    for (uint i = 0; i < k; i++) {
        // Block index for row i
        uint block_idx = i * blocks_per_row + tid_block;

        // Get the element from the block
        float w_val = dequantize_q6_k_elem(weights[block_idx], tid_offset);

        sum += w_val * input[i];
    }

    output[tid] = sum;
}

// F32 transposed GEMV: y = W^T * x where W is [K, M] stored row-major
kernel void mv_f32_f32_transposed(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],  // Output size
    constant uint& k [[buffer(4)]],  // Input size
    uint tid [[thread_position_in_grid]])
{
    // Each thread computes one output element
    if (tid >= m) return;

    float sum = 0.0f;

    // W is stored as [K, M], so W[i, j] is at position i * M + j
    for (uint i = 0; i < k; i++) {
        sum += weights[i * m + tid] * input[i];
    }

    output[tid] = sum;
}
