// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Problem-size adaptive F16 GEMV kernels
// - short: K < 1024 (single thread per row, avoid sync overhead)
// - medium: 1024 <= K < 4096 (simdgroup cooperation, NR0=2)
// - large: K >= 4096 (multi-row batching + float4 vectorization, NR0=2)

#include <metal_stdlib>
using namespace metal;

#define SIMD_SIZE 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ============================================================================
// Short version: K < 1024
// One thread per row, avoid simdgroup synchronization overhead
// Good for small K where sync cost dominates
// ============================================================================

kernel void kernel_mul_mv_f16_f32_short(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x * 32 + tiisg;
    if (row >= m) return;

    device const half* row_weights = weights + row * k;
    device const float4* y4 = (device const float4*)input;
    device const half4* w4 = (device const half4*)row_weights;

    const int nb = k / 4;
    float sum = 0.0f;

    for (int i = tiisg; i < nb; i += 32) {
        float4 y4b[4] = { y4[i + 0], y4[i + 1], y4[i + 2], y4[i + 3] };
        float4 w4b[4] = { float4(w4[i*4 + 0]), float4(w4[i*4 + 1]),
                          float4(w4[i*4 + 2]), float4(w4[i*4 + 3]) };

        sum += w4b[0].x * y4b[0].x + w4b[0].y * y4b[0].y + w4b[0].z * y4b[0].z + w4b[0].w * y4b[0].w;
        sum += w4b[1].x * y4b[1].x + w4b[1].y * y4b[1].y + w4b[1].z * y4b[1].z + w4b[1].w * y4b[1].w;
        sum += w4b[2].x * y4b[2].x + w4b[2].y * y4b[2].y + w4b[2].z * y4b[2].z + w4b[2].w * y4b[2].w;
        sum += w4b[3].x * y4b[3].x + w4b[3].y * y4b[3].y + w4b[3].z * y4b[3].z + w4b[3].w * y4b[3].w;
    }

    for (int i = nb * 4 + tiisg; i < k; i += 32) {
        sum += float(row_weights[i]) * input[i];
    }

    output[row] = sum;
}

// ============================================================================
// Medium version: 1024 <= K < 4096 (NOT USED - same as large)
// ============================================================================

kernel void kernel_mul_mv_f16_f32_medium(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NR0 = 2;
    const int row0 = tgpig.x * NR0;
    if (row0 >= m) return;

    constexpr short NW = 32;
    constexpr short NB = 32;
    const int nb = k / NB;

    device const float4* y4 = (device const float4*)input;

    device const half4* ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4*)(weights + (row0 + row) * k);
    }

    float sumf[NR0] = { [ 0 ... NR0 - 1 ] = 0.0f };

    for (int ib = tiisg; ib < nb; ib += NW) {
        float4 y4b[4] = { y4[ib + 0], y4[ib + 1], y4[ib + 2], y4[ib + 3] };

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            float4 acc = 0.0f;
            for (int j = 0; j < 4; ++j) {
                acc += float4(ax4[row][ib*4 + j]) * y4b[j];
            }
            sumf[row] += acc[0] + acc[1] + acc[2] + acc[3];
        }
    }

    for (int i = nb*NB + tiisg; i < k; i += NW) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            sumf[row] += float(weights[(row0 + row)*k + i]) * input[i];
        }
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const int row_idx = row0 + row;
            if (row_idx < m) output[row_idx] = sumf[row];
        }
    }
}

// ============================================================================
// Large version: K >= 4096
// Multi-row batching (2 rows) + float4 vectorization
// Maximum throughput for large K values
// ============================================================================

kernel void kernel_mul_mv_f16_f32_large(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NR0 = 2;
    const int row0 = tgpig.x * NR0;
    if (row0 >= m) return;

    constexpr short NW = 32;
    constexpr short NB = 32;
    const int nb = k / NB;

    device const float4* y4 = (device const float4*)input;

    device const half4* ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4*)(weights + (row0 + row) * k);
    }

    float sumf[NR0] = { [ 0 ... NR0 - 1 ] = 0.0f };

    for (int ib = tiisg; ib < nb; ib += NW) {
        float4 y4b[4] = { y4[ib + 0], y4[ib + 1], y4[ib + 2], y4[ib + 3] };

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            float4 acc = 0.0f;
            for (int j = 0; j < 4; ++j) {
                acc += float4(ax4[row][ib*4 + j]) * y4b[j];
            }
            sumf[row] += acc[0] + acc[1] + acc[2] + acc[3];
        }
    }

    for (int i = nb*NB + tiisg; i < k; i += NW) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            sumf[row] += float(weights[(row0 + row)*k + i]) * input[i];
        }
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const int row_idx = row0 + row;
            if (row_idx < m) output[row_idx] = sumf[row];
        }
    }
}

// ============================================================================
// Dispatcher: choose kernel based on K
// ============================================================================
// In Rust: if k < 256 -> short, elif k < 2048 -> medium, else -> large
