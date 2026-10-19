// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// FusedNormLinear: RMSNorm + Linear projection in one kernel
// Combines: output = (input / sqrt(mean(input^2) + eps) * norm_weight) @ weights
// Reduces dispatch overhead and avoids writing normalized intermediate to global memory

#include <metal_stdlib>
using namespace metal;

// Fused RMSNorm + F16 GEMV kernel
// Each threadgroup computes one output element
// Threadgroup loads input, computes RMS, normalizes into shared memory, then computes dot product

kernel void kernel_fused_norm_linear_f16(
    device const float* input [[buffer(0)]],       // [K] input vector
    device const float* norm_weight [[buffer(1)]], // [K] RMSNorm scale weights
    device const half* weights [[buffer(2)]],      // [M, K] linear weights in F16
    device float* output [[buffer(3)]],            // [M] output vector
    constant uint& m [[buffer(4)]],                // M (output size)
    constant uint& k [[buffer(5)]],                // K (input size)
    constant float& epsilon [[buffer(6)]],         // RMSNorm epsilon
    threadgroup float* shared_input [[threadgroup(0)]], // Shared memory for normalized input
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    // Phase 1: Compute RMS across all threads in threadgroup
    float local_sum = 0.0f;
    for (uint i = lane_id; i < k; i += group_size) {
        float val = input[i];
        local_sum += val * val;
    }

    // Reduce within simdgroup
    float simd_result = simd_sum(local_sum);

    // For large threadgroups, use threadgroup reduction
    threadgroup float* reduction_buf = shared_input;
    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = simd_result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float total_sum = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                total_sum += reduction_buf[i];
            }
            reduction_buf[0] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean_sq = (group_size > 32) ? reduction_buf[0] / float(k) : simd_result / float(k);
    float rms = rsqrt(mean_sq + epsilon);

    // Phase 2: Normalize input and store in shared memory
    for (uint i = lane_id; i < k; i += group_size) {
        shared_input[i] = input[i] * rms * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Compute dot product with this row's weights
    device const half* row_weights = weights + row * k;

    float sum = 0.0f;
    for (uint i = lane_id; i < k; i += group_size) {
        sum += float(row_weights[i]) * shared_input[i];
    }

    // Reduce within simdgroup
    sum = simd_sum(sum);

    // For large threadgroups, reduce across simdgroups
    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float total = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                total += reduction_buf[i];
            }
            output[row] = total;
        }
    } else {
        if (lane_in_simd == 0) {
            output[row] = sum;
        }
    }
}

// Fused RMSNorm + F32 GEMV kernel (for F32 weights)
kernel void kernel_fused_norm_linear_f32(
    device const float* input [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* weights [[buffer(2)]],      // [M, K] linear weights in F32
    device float* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    // Phase 1: Compute RMS
    float local_sum = 0.0f;
    for (uint i = lane_id; i < k; i += group_size) {
        float val = input[i];
        local_sum += val * val;
    }

    float simd_result = simd_sum(local_sum);

    threadgroup float* reduction_buf = shared_input;
    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = simd_result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float total_sum = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                total_sum += reduction_buf[i];
            }
            reduction_buf[0] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean_sq = (group_size > 32) ? reduction_buf[0] / float(k) : simd_result / float(k);
    float rms = rsqrt(mean_sq + epsilon);

    // Phase 2: Normalize and store in shared memory
    for (uint i = lane_id; i < k; i += group_size) {
        shared_input[i] = input[i] * rms * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Compute dot product with F32 weights
    device const float* row_weights = weights + row * k;

    float sum = 0.0f;
    for (uint i = lane_id; i < k; i += group_size) {
        sum += row_weights[i] * shared_input[i];
    }

    sum = simd_sum(sum);

    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float total = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                total += reduction_buf[i];
            }
            output[row] = total;
        }
    } else {
        if (lane_in_simd == 0) {
            output[row] = sum;
        }
    }
}

// Vectorized FusedNormLinear using float4 for better memory throughput
kernel void kernel_fused_norm_linear_f16_4(
    device const float4* input [[buffer(0)]],       // [K/4] as float4
    device const float4* norm_weight [[buffer(1)]], // [K/4] as float4
    device const half4* weights [[buffer(2)]],      // [M, K/4] as half4
    device float* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    threadgroup float4* shared_input4 [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    const uint k4 = k / 4;

    // Phase 1: Compute RMS
    float local_sum = 0.0f;
    for (uint i = lane_id; i < k4; i += group_size) {
        float4 v = input[i];
        local_sum += dot(v, v);
    }

    float simd_result = simd_sum(local_sum);

    threadgroup float* reduction_buf = (threadgroup float*)shared_input4;
    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = simd_result;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float total_sum = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                total_sum += reduction_buf[i];
            }
            reduction_buf[0] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean_sq = (group_size > 32) ? reduction_buf[0] / float(k) : simd_result / float(k);
    float rms = rsqrt(mean_sq + epsilon);

    // Phase 2: Normalize and store
    for (uint i = lane_id; i < k4; i += group_size) {
        shared_input4[i] = input[i] * rms * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Compute dot product with half4 weights
    device const half4* row_weights = weights + row * k4;

    float4 sum4 = 0.0f;
    for (uint i = lane_id; i < k4; i += group_size) {
        float4 w = float4(row_weights[i]);
        float4 x = shared_input4[i];
        sum4 += w * x;
    }

    // Reduce each component
    float sum = simd_sum(sum4.x + sum4.y + sum4.z + sum4.w);

    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float total = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                total += reduction_buf[i];
            }
            output[row] = total;
        }
    } else {
        if (lane_in_simd == 0) {
            output[row] = sum;
        }
    }
}
