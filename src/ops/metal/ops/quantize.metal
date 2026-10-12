// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Quantization kernels for GPU
// Convert FP32 to quantized formats

#include <metal_stdlib>
using namespace metal;

#define QK 32

// Block structures
struct block_q4_0 {
    half d;
    uint8_t qs[QK / 2];
};

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};

// Quantize F32 to F16
kernel void kernel_quantize_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = half(src[gid]);
}

// Quantize F32 to F16 (vectorized, 4 elements per thread)
kernel void kernel_quantize_f16_4(
    device const float4* src [[buffer(0)]],
    device half4* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const int i = gid * 4;
    if (i >= n) return;

    float4 val = src[gid];
    dst[gid] = half4(val);
}

// Quantize F32 to Q4_0
// Each thread processes one block of 32 elements
kernel void kernel_quantize_q4_0(
    device const float* src [[buffer(0)]],
    device block_q4_0* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const int nb = n / QK;
    if (gid >= nb) return;

    const int i = gid * QK;
    device const float* x = src + i;

    // Compute max absolute value
    float amax = 0.0f;
    for (int j = 0; j < QK; j++) {
        amax = max(amax, abs(x[j]));
    }

    const float d = amax / 8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    // Quantize
    dst[gid].d = half(d);
    for (int j = 0; j < QK / 2; j++) {
        const float x0 = x[j] * id;
        const float x1 = x[QK/2 + j] * id;

        const uint8_t q0 = min(15, max(0, int8_t(round(x0 + 8.0f))));
        const uint8_t q1 = min(15, max(0, int8_t(round(x1 + 8.0f))));

        dst[gid].qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}

// Quantize F32 to Q8_0
// Each thread processes one block of 32 elements
kernel void kernel_quantize_q8_0(
    device const float* src [[buffer(0)]],
    device block_q8_0* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const int nb = n / QK;
    if (gid >= nb) return;

    const int i = gid * QK;
    device const float* x = src + i;

    // Compute max absolute value
    float amax = 0.0f;
    for (int j = 0; j < QK; j++) {
        amax = max(amax, abs(x[j]));
    }

    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    // Quantize
    dst[gid].d = half(d);
    for (int j = 0; j < QK; j++) {
        dst[gid].qs[j] = int8_t(round(x[j] * id));
    }
}

// Batch quantize F32 to F16
// For large batch sizes
kernel void kernel_quantize_f16_batch(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant int& n_rows [[buffer(2)]],
    constant int& n_cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int row = gid.y;
    const int col = gid.x;

    if (row >= n_rows || col >= n_cols) return;

    const int idx = row * n_cols + col;
    dst[idx] = half(src[idx]);
}

// Batch quantize F32 to Q4_0
// For large batch sizes
kernel void kernel_quantize_q4_0_batch(
    device const float* src [[buffer(0)]],
    device block_q4_0* dst [[buffer(1)]],
    constant int& n_rows [[buffer(2)]],
    constant int& n_cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int nb = n_cols / QK;
    const int row = gid.y;
    const int block_idx = gid.x;

    if (row >= n_rows || block_idx >= nb) return;

    const int i = row * n_cols + block_idx * QK;
    device const float* x = src + i;

    // Compute max absolute value
    float amax = 0.0f;
    for (int j = 0; j < QK; j++) {
        amax = max(amax, abs(x[j]));
    }

    const float d = amax / 8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    // Quantize
    const int dst_idx = row * nb + block_idx;
    dst[dst_idx].d = half(d);
    for (int j = 0; j < QK / 2; j++) {
        const float x0 = x[j] * id;
        const float x1 = x[QK/2 + j] * id;

        const uint8_t q0 = min(15, max(0, int8_t(round(x0 + 8.0f))));
        const uint8_t q1 = min(15, max(0, int8_t(round(x1 + 8.0f))));

        dst[dst_idx].qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}
