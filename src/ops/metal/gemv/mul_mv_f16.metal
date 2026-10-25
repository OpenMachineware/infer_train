// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// F16 Matrix-Vector Multiplication kernels
// Self-written implementation for F16 weight matrix with F32 input

#include <metal_stdlib>
using namespace metal;

// Simple F16 GEMV: y = W @ x
// W is stored row-major as [M, K] in F16 format
// x is F32 vector of size K
// y is F32 vector of size M
// Each thread computes one output element

kernel void kernel_mul_mv_f16_f32(
    device const half* weights [[buffer(0)]],  // [M, K] in F16
    device const float* input [[buffer(1)]],    // [K] in F32
    device float* output [[buffer(2)]],         // [M] in F32
    constant uint& m [[buffer(3)]],             // M (output size)
    constant uint& k [[buffer(4)]],             // K (input size)
    uint tid [[thread_position_in_grid]])
{
    // Each thread computes one output element
    if (tid >= m) return;

    // W is stored row-major: W[i, j] is at position i * K + j
    device const half* row = weights + tid * k;

    float sum = 0.0f;

    // Simple loop: sum += W[tid, i] * x[i]
    for (uint i = 0; i < k; i++) {
        // Read half (F16) and convert to float
        float w_val = float(row[i]);
        sum += w_val * input[i];
    }

    output[tid] = sum;
}

// F16×F16 GEMV: y = W @ x
// W is stored row-major as [M, K] in F16 format
// x is F16 vector of size K
// y is F32 vector of size M
// Each thread computes one output element

kernel void kernel_mul_mv_f16_f16(
    device const half* weights [[buffer(0)]],  // [M, K] in F16
    device const half* input [[buffer(1)]],    // [K] in F16
    device float* output [[buffer(2)]],         // [M] in F32
    constant uint& m [[buffer(3)]],             // M (output size)
    constant uint& k [[buffer(4)]],             // K (input size)
    uint tid [[thread_position_in_grid]])
{
    // Each thread computes one output element
    if (tid >= m) return;

    // W is stored row-major: W[i, j] is at position i * K + j
    device const half* row = weights + tid * k;

    float sum = 0.0f;

    // Simple loop: sum += W[tid, i] * x[i]
    for (uint i = 0; i < k; i++) {
        // Read both as half and convert to float
        float w_val = float(row[i]);
        float x_val = float(input[i]);
        sum += w_val * x_val;
    }

    output[tid] = sum;
}

// Optimized F16 GEMV using simdgroup for better performance
// Uses simdgroup to process multiple elements in parallel

#define SIMD_SIZE 32

kernel void kernel_mul_mv_f16_f32_simd(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Each threadgroup handles one row
    const uint row = tgpig.x;
    if (row >= m) return;

    // Get pointer to this row's weights
    device const half* row_weights = weights + row * k;

    // Each thread in simdgroup processes a portion of the row
    float sum = 0.0f;

    // Process elements in chunks of SIMD_SIZE
    // Each thread handles elements at offset tiisg + i * SIMD_SIZE
    for (uint i = tiisg; i < k; i += SIMD_SIZE) {
        float w_val = float(row_weights[i]);
        sum += w_val * input[i];
    }

    // Reduce across simdgroup
    sum = simd_sum(sum);

    // Only thread 0 writes the result
    if (tiisg == 0) {
        output[row] = sum;
    }
}

// F16 GEMV with threadgroup shared memory for better cache usage
// Processes multiple rows per threadgroup

kernel void kernel_mul_mv_f16_f32_batch(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Process 4 rows per threadgroup (2 rows per simdgroup)
    const uint n_rows_per_tg = 4;
    const uint n_rows_per_sg = 2;

    const uint row_start = tgpig.x * n_rows_per_tg;
    const uint row_in_tg = sgitg * n_rows_per_sg;
    const uint row0 = row_start + row_in_tg;

    if (row0 >= m) return;

    // Process 2 rows per simdgroup
    for (uint r = 0; r < n_rows_per_sg; r++) {
        uint row = row0 + r;
        if (row >= m) continue;

        device const half* row_weights = weights + row * k;

        float sum = 0.0f;

        // Each thread processes elements at stride SIMD_SIZE
        for (uint i = tiisg; i < k; i += SIMD_SIZE) {
            float w_val = float(row_weights[i]);
            sum += w_val * input[i];
        }

        // Reduce across simdgroup
        sum = simd_sum(sum);

        // Thread 0 writes result
        if (tiisg == 0) {
            output[row] = sum;
        }
    }
}
