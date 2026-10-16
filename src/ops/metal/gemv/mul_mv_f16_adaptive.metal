// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Problem-size adaptive F16 GEMV kernels
// Different kernels for different K ranges (matching llama.cpp approach):
// - short: K < 1024 (single thread per row, avoid sync overhead)
// - medium: 1024 <= K < 4096 (simdgroup cooperation, NR0=2)
// - large: K >= 4096 (multi-row batching + float4 vectorization, NR0=2)

#include <metal_stdlib>
using namespace metal;

#define SIMD_SIZE 32

// ============================================================================
// Short version: K < 1024
// One thread per row, avoid simdgroup synchronization overhead
// Good for small K where sync cost dominates
// ============================================================================

kernel void kernel_mul_mv_f16_f32_short(
    device const half* weights [[buffer(0)]],  // [M, K]
    device const float* input [[buffer(1)]],   // [K]
    device float* output [[buffer(2)]],        // [M]
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Each thread processes one row
    const int row = tgpig.x * 32 + tiisg;
    if (row >= m) return;

    // Simple dot product without simdgroup cooperation
    device const half* row_weights = weights + row * k;

    float sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        sum += float(row_weights[i]) * input[i];
    }

    output[row] = sum;
}

// ============================================================================
// Medium version: 1024 <= K < 4096
// Use simdgroup for parallel reduction with NR0=2 (2 rows per threadgroup)
// Better cache utilization for medium-sized K
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
    // Each threadgroup processes 2 rows (NR0=2)
    const int NR0 = 2;
    const int row0 = tgpig.x * NR0;

    if (row0 >= m) return;

    // Process both rows
    for (int ir = 0; ir < NR0; ir++) {
        const int row = row0 + ir;
        if (row >= m) continue;

        device const half* row_weights = weights + row * k;

        // Each thread handles elements at stride SIMD_SIZE
        float sum = 0.0f;
        for (uint i = tiisg; i < k; i += SIMD_SIZE) {
            sum += float(row_weights[i]) * input[i];
        }

        // Parallel reduction across simdgroup
        sum = simd_sum(sum);

        // Thread 0 writes result
        if (tiisg == 0) {
            output[row] = sum;
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
    // Process 2 rows per threadgroup
    const int row0 = tgpig.x * 2;
    const int NSG = 1;  // One simdgroup per threadgroup

    // Row indices
    const int rows[2] = {row0, row0 + 1};

    // Initialize accumulators for both rows
    float sum[2] = {0.0f, 0.0f};

    // Block size for processing
    const uint NB = 32;  // Process 32 elements per iteration
    const uint nb = k / NB;

    // Thread assignment: each thread handles one block
    const int ib = sgitg * SIMD_SIZE + tiisg;

    if (ib < nb) {
        // Load input block
        const int block_start = ib * NB;
        const int block_end = block_start + NB;

        // Process with float4 for vectorization
        for (int r = 0; r < 2; r++) {
            const int row = rows[r];
            if (row >= m) continue;

            device const half4* weights4 = (device const half4*)(weights + row * k + block_start);
            device const float4* input4 = (device const float4*)(input + block_start);

            // Process 8 float4s (32 elements)
            float partial_sum = 0.0f;
            for (uint j = 0; j < 8; j++) {
                partial_sum += dot(float4(weights4[j]), input4[j]);
            }
            sum[r] += partial_sum;
        }
    }

    // Handle remaining elements
    const int remainder_start = nb * NB;
    if (remainder_start < k && tiisg == 0) {
        for (int r = 0; r < 2; r++) {
            const int row = rows[r];
            if (row >= m) continue;

            for (uint i = remainder_start; i < k; i++) {
                sum[r] += float(weights[row * k + i]) * input[i];
            }
        }
    }

    // Reduce across threadgroup
    sum[0] = simd_sum(sum[0]);
    sum[1] = simd_sum(sum[1]);

    // Thread 0 writes results
    if (tiisg == 0) {
        if (row0 < m) output[row0] = sum[0];
        if (row0 + 1 < m) output[row0 + 1] = sum[1];
    }
}

// ============================================================================
// Dispatcher: choose kernel based on K
// ============================================================================
// In Rust: if k < 256 -> short, elif k < 2048 -> medium, else -> large
