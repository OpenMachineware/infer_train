// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// FusedAddNorm Metal kernel - Add + RMSNorm in single dispatch
// Pattern from vLLM's fused_add_rms_norm kernel
// Algorithm: output = RMSNorm(lhs + rhs, weight, eps)

#include <metal_stdlib>
using namespace metal;

// F32 version - processes one row per threadgroup
kernel void kernel_fused_add_norm_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& dimension [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    device const float* row_lhs = lhs + row_idx * dimension;
    device const float* row_rhs = rhs + row_idx * dimension;
    device float* row_out = output + row_idx * dimension;

    // Step 1: Add and compute sum of squares
    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(dimension); i += group_size) {
        float val = row_lhs[i] + row_rhs[i];
        local_sum += val * val;
    }

    // Reduce within SIMD group
    float simd_result = simd_sum(local_sum);

    // Reduce across SIMD groups if needed
    if (group_size > 32) {
        if (simd_id == 0) {
            scratch[lane_in_simd] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane_in_simd == 0) {
            scratch[simd_id] = simd_result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simd_result = simd_sum(scratch[lane_in_simd]);
    }

    // Step 2: Compute RMS
    float mean_sq = simd_result / float(dimension);
    float rms = rsqrt(mean_sq + epsilon);

    // Step 3: Apply normalization and weight
    for (uint i = lane_id; i < uint(dimension); i += group_size) {
        row_out[i] = (row_lhs[i] + row_rhs[i]) * rms * weight[i];
    }
}

// F32 vectorized version (float4)
kernel void kernel_fused_add_norm_f32_4(
    device const float4* lhs [[buffer(0)]],
    device const float4* rhs [[buffer(1)]],
    device const float4* weight [[buffer(2)]],
    device float4* output [[buffer(3)]],
    constant int& dimension [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    const int vec_dim = dimension / 4;

    device const float4* row_lhs = lhs + row_idx * vec_dim;
    device const float4* row_rhs = rhs + row_idx * vec_dim;
    device float4* row_out = output + row_idx * vec_dim;

    // Add and compute sum of squares
    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(vec_dim); i += group_size) {
        float4 v = row_lhs[i] + row_rhs[i];
        local_sum += dot(v, v);
    }

    float simd_result = simd_sum(local_sum);

    if (group_size > 32) {
        if (simd_id == 0) {
            scratch[lane_in_simd] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane_in_simd == 0) {
            scratch[simd_id] = simd_result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simd_result = simd_sum(scratch[lane_in_simd]);
    }

    float mean_sq = simd_result / float(dimension);
    float rms = rsqrt(mean_sq + epsilon);

    for (uint i = lane_id; i < uint(vec_dim); i += group_size) {
        row_out[i] = (row_lhs[i] + row_rhs[i]) * rms * weight[i];
    }
}

// F16 version
kernel void kernel_fused_add_norm_f16(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& dimension [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    device const half* row_lhs = lhs + row_idx * dimension;
    device const half* row_rhs = rhs + row_idx * dimension;
    device half* row_out = output + row_idx * dimension;

    // Add and compute sum of squares
    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(dimension); i += group_size) {
        float val = float(row_lhs[i]) + float(row_rhs[i]);
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);

    if (group_size > 32) {
        if (simd_id == 0) {
            scratch[lane_in_simd] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane_in_simd == 0) {
            scratch[simd_id] = simd_result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simd_result = simd_sum(scratch[lane_in_simd]);
    }

    float mean_sq = simd_result / float(dimension);
    float rms = rsqrt(mean_sq + epsilon);

    for (uint i = lane_id; i < uint(dimension); i += group_size) {
        float sum = float(row_lhs[i]) + float(row_rhs[i]);
        row_out[i] = half(sum * rms * float(weight[i]));
    }
}

// F16 vectorized version (half4)
kernel void kernel_fused_add_norm_f16_4(
    device const half4* lhs [[buffer(0)]],
    device const half4* rhs [[buffer(1)]],
    device const half4* weight [[buffer(2)]],
    device half4* output [[buffer(3)]],
    constant int& dimension [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    const int vec_dim = dimension / 4;

    device const half4* row_lhs = lhs + row_idx * vec_dim;
    device const half4* row_rhs = rhs + row_idx * vec_dim;
    device half4* row_out = output + row_idx * vec_dim;

    // Add and compute sum of squares
    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(vec_dim); i += group_size) {
        float4 v = float4(row_lhs[i]) + float4(row_rhs[i]);
        local_sum += dot(v, v);
    }

    float simd_result = simd_sum(local_sum);

    if (group_size > 32) {
        if (simd_id == 0) {
            scratch[lane_in_simd] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane_in_simd == 0) {
            scratch[simd_id] = simd_result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simd_result = simd_sum(scratch[lane_in_simd]);
    }

    float mean_sq = simd_result / float(dimension);
    float rms = rsqrt(mean_sq + epsilon);

    for (uint i = lane_id; i < uint(vec_dim); i += group_size) {
        float4 sum = float4(row_lhs[i]) + float4(row_rhs[i]);
        float4 w = float4(weight[i]);
        row_out[i] = half4(sum * rms * w);
    }
}
