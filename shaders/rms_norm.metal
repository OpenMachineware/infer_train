// RMSNorm Metal kernel - Truly self-written implementation
// Algorithm: RMS = sqrt(mean(x^2) + eps), output = x/RMS * weight
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void kernel_rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& dimension [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    device const float* row_in = input + row_idx * dimension;
    device float* row_out = output + row_idx * dimension;

    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(dimension); i += group_size) {
        float val = row_in[i];
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
        row_out[i] = row_in[i] * rms * weight[i];
    }
}

kernel void kernel_rms_norm_f32_4(
    device const float4* input [[buffer(0)]],
    device const float4* weight [[buffer(1)]],
    device float4* output [[buffer(2)]],
    constant int& dimension [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    const int vec_dim = dimension / 4;

    device const float4* row_in = input + row_idx * vec_dim;
    device float4* row_out = output + row_idx * vec_dim;

    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(vec_dim); i += group_size) {
        float4 v = row_in[i];
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
        row_out[i] = row_in[i] * rms * weight[i];
    }
}

kernel void rms_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& dimension [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    device const half* row_in = input + row_idx * dimension;
    device half* row_out = output + row_idx * dimension;

    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(dimension); i += group_size) {
        float val = float(row_in[i]);
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
        row_out[i] = half(float(row_in[i]) * rms * float(weight[i]));
    }
}

kernel void rms_norm_f16_vec4(
    device const half4* input [[buffer(0)]],
    device const half4* weight [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant int& dimension [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    const int vec_dim = dimension / 4;

    device const half4* row_in = input + row_idx * vec_dim;
    device half4* row_out = output + row_idx * vec_dim;

    float local_sum = 0.0f;
    for (uint i = lane_id; i < uint(vec_dim); i += group_size) {
        float4 v = float4(row_in[i]);
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
        float4 v = float4(row_in[i]);
        float4 w = float4(weight[i]);
        row_out[i] = half4(v * rms * w);
    }
}
