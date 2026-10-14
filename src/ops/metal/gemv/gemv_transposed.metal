// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Transposed GEMV kernels for quantized formats
// Handles weights stored as [K, M] (transposed layout)
// Used for output layer with tied embeddings

#include <metal_stdlib>
using namespace metal;

#define QK_K 256

// Q6_K block: 256 elements, 210 bytes
// Layout: ql[128] + qh[64] + scales[16] + d[2]
// Uses interleaved layout for ql/qh

// Dequantize a single element from Q6_K block
// elem_idx: 0-255 position within the block
float dequantize_q6_k_elem(
    device const uint8_t* ql,
    device const uint8_t* qh,
    device const int8_t* scales,
    float d,
    uint elem_idx
) {
    // Q6_K: 16 sub-blocks of 16 elements each
    uint il = elem_idx / 16;  // Which sub-block (0-15)
    uint k = elem_idx % 16;   // Position within sub-block (0-15)

    float sc = scales[il];
    float dl = d * sc;

    // Interleaved layout offsets (from self-written kernel)
    device const uint16_t* ql16 = (device const uint16_t*)ql;
    device const uint16_t* qh16 = (device const uint16_t*)qh;

    // Compute offsets
    uint ql_offset = 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    uint qh_offset = 16 * (il / 8) + 8 * (il & 1);

    // Masks based on (il/2)&3
    uint il2 = (il / 2) & 3;
    uint32_t kmask1, kmask2;
    uint8_t shr_h, shl_h, shr_l;

    if (il2 > 1) {
        kmask1 = (il2 > 2) ? 0xC0C0C0C0 : 0x30303030;
        kmask2 = 0xF0F0F0F0;
        shr_l = 4;
    } else {
        kmask1 = (il2 > 0) ? 0x0C0C0C0C : 0x03030303;
        kmask2 = 0x0F0F0F0F;
        shr_l = 0;
    }
    shr_h = (il2 > 2) ? 2 : 0;
    shl_h = (il2 > 1) ? 0 : ((il2 > 0) ? 2 : 4);

    // Determine which 4-element group and position within it
    uint i = k / 4;  // Group index (0-3)
    uint b = k % 4;  // Position within group (0-3)

    // Get the uint32 from ql and qh
    uint32_t low = (ql16[ql_offset + 2 * i] | ((uint32_t)ql16[ql_offset + 2 * i + 1] << 16)) & kmask2;
    uint32_t high = (qh16[qh_offset + 2 * i] | ((uint32_t)qh16[qh_offset + 2 * i + 1] << 16)) & kmask1;
    uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);

    // Extract the byte for this element
    uint8_t qv = (q >> (b * 8)) & 0xFF;

    // Convert to signed and scale
    return dl * (qv - 32.0f);
}

// Transposed GEMV for Q6_K: y = W^T * x
// W is stored as [K, M] in Q6_K format (transposed for tied embeddings)
// K = input_size (hidden_size), M = output_size (vocab_size)
// Each thread computes one output element

kernel void mv_q6_k_f32_transposed(
    device const char* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],  // Output size (vocab_size)
    constant uint& k [[buffer(4)]],  // Input size (hidden_size)
    uint tid [[thread_position_in_grid]])
{
    // Each thread computes one output element
    if (tid >= m) return;

    float sum = 0.0f;

    // Q6_K block size is 256
    uint blocks_per_row = m / QK_K;  // Number of blocks per row

    // For output element tid, compute:
    // sum = sum over i of W[i, tid] * x[i]
    // W[i, tid] is in block i * blocks_per_row + tid / 256, element tid % 256

    uint tid_block = tid / QK_K;
    uint tid_elem = tid % QK_K;

    // Block size in bytes: 210
    const uint block_bytes = 210;

    for (uint i = 0; i < k; i++) {
        // Block index for row i, column tid
        uint block_idx = i * blocks_per_row + tid_block;

        // Get block pointer
        device const char* block_ptr = weights + block_idx * block_bytes;

        // Q6_K structure offsets within block
        device const uint8_t* ql = (device const uint8_t*)(block_ptr);
        device const uint8_t* qh = (device const uint8_t*)(block_ptr + 128);
        device const int8_t* scales = (device const int8_t*)(block_ptr + 192);
        device const uint16_t* d_ptr = (device const uint16_t*)(block_ptr + 208);

        float d = float(*d_ptr);
        if (d != 0.0f) {
            // Convert f16 bits to float
            // f16: 1 sign, 5 exponent, 10 mantissa
            uint16_t d_bits = *d_ptr;
            uint16_t sign = (d_bits >> 15) & 1;
            uint16_t exp = (d_bits >> 10) & 0x1F;
            uint16_t mant = d_bits & 0x3FF;

            if (exp == 0) {
                d = 0.0f;  // Zero or denormal (treat as zero)
            } else {
                d = (sign ? -1.0f : 1.0f) * (1.0f + float(mant) / 1024.0f) * pow(2.0f, float(exp) - 15.0f);
            }
        }

        float w_val = dequantize_q6_k_elem(ql, qh, scales, d, tid_elem);
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
