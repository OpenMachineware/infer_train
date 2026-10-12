// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GEMV kernels with Q8_0 quantized input
// Used for KV cache with IQ4_NL weights

#include <metal_stdlib>
using namespace metal;

#define QK 32

// IQ4_NL lookup table (non-linear quantization - only 16 values for 4-bit indices)
constant float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

struct BlockIQ4NL {
    half d;
    uint8_t qs[16];
};

struct BlockQ8_0 {
    half d;
    int8_t qs[32];
};

// ============================================================================
// Vector Dot Product: single row of IQ4_NL weights x Q8_0 input
// ============================================================================

kernel void vec_dot_iq4_nl_q8_0(
    device const BlockIQ4NL* x [[buffer(0)]],
    device const BlockQ8_0* y [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& nb [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]])
{
    float sumf = 0.0f;

    for (uint i = 0; i < nb; i++) {
        const device BlockIQ4NL& xb = x[i];
        const device BlockQ8_0& yb = y[i];

        float sumi = 0.0f;

        for (int j = 0; j < 16; j++) {
            uint8_t q = xb.qs[j];

            float v0 = kvalues_iq4nl_f[q & 0xf];
            float v1 = kvalues_iq4nl_f[(q >> 4) & 0xf];

            float q8_0 = float(yb.qs[j]);
            float q8_1 = float(yb.qs[j + 16]);

            sumi += v0 * q8_0 + v1 * q8_1;
        }

        float d = float(xb.d) * float(yb.d);
        sumf += d * sumi;
    }

    *result = sumf;
}

// ============================================================================
// Matrix-Vector: M rows × K columns (scalar version)
// Each thread processes one row
// ============================================================================

kernel void mv_iq4_nl_q8_0(
    device const BlockIQ4NL* weights [[buffer(0)]],
    device const BlockQ8_0* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& nb [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= m) return;

    float sumf = 0.0f;
    uint row_offset = tid * nb;

    for (uint i = 0; i < nb; i++) {
        const device BlockIQ4NL& wb = weights[row_offset + i];
        const device BlockQ8_0& ib = input[i];

        float sumi = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t q = wb.qs[j];
            float v0 = kvalues_iq4nl_f[q & 0xf];
            float v1 = kvalues_iq4nl_f[(q >> 4) & 0xf];
            float q8_0 = float(ib.qs[j]);
            float q8_1 = float(ib.qs[j + 16]);
            sumi += v0 * q8_0 + v1 * q8_1;
        }

        sumf += float(wb.d) * float(ib.d) * sumi;
    }

    output[tid] = sumf;
}

// ============================================================================
// Matrix-Vector: SIMD optimized version
// Each threadgroup (32 threads) processes one row
// ============================================================================

kernel void mv_iq4_nl_q8_0_simd(
    device const BlockIQ4NL* weights [[buffer(0)]],
    device const BlockQ8_0* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& nb [[buffer(4)]],
    uint tid [[thread_index_in_simdgroup]],
    uint row [[simdgroup_index_in_threadgroup]])
{
    if (row >= m) return;

    uint row_offset = row * nb;

    float sum = 0.0f;
    uint blocks_per_thread = (nb + 31) / 32;

    for (uint i = 0; i < blocks_per_thread; i++) {
        uint block_idx = tid * blocks_per_thread + i;
        if (block_idx >= nb) break;

        const device BlockIQ4NL& wb = weights[row_offset + block_idx];
        const device BlockQ8_0& ib = input[block_idx];

        for (int j = 0; j < 16; j++) {
            uint8_t q = wb.qs[j];
            float v0 = kvalues_iq4nl_f[q & 0xf];
            float v1 = kvalues_iq4nl_f[(q >> 4) & 0xf];
            float q8_0 = float(ib.qs[j]);
            float q8_1 = float(ib.qs[j + 16]);
            sum += v0 * q8_0 + v1 * q8_1;
        }
    }

    float total_sum = simd_sum(sum);

    if (tid == 0) {
        output[row] = total_sum;
    }
}
