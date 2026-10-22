// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// FusedNormLinear: RMSNorm + Linear projection in one kernel
// Combines: output = (input / sqrt(mean(input^2) + eps) * norm_weight) @ weights
// Reduces dispatch overhead and avoids writing normalized intermediate to global memory

#include <metal_stdlib>
using namespace metal;

// Include block structure definitions for quantized formats
#define GGML_COMMON_DECL_METAL
#include "../include/blocks.h"

#define QK_K 256

// Helper function for Q4_K scale extraction (from kquant_mv.metal)
static inline uchar2 get_scale_min_k4(int j, int k, device const uchar * scales) {
    if (j < 4) {
        return uchar2(uchar(scales[j+0+k] & 63), uchar(scales[j+4+k] & 63));
    } else {
        return uchar2(
            uchar((scales[j+4+k] & 0x0F) | ((scales[j-4+k] & 0xC0) >> 2)),
            uchar((scales[j+4+k] >> 4) | ((scales[j-0+k] & 0xC0) >> 2))
        );
    }
}

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

// ============================================================================
// Fused RMSNorm + Q4_K GEMV kernel
// ============================================================================

kernel void kernel_fused_norm_linear_q4_k(
    device const float* input [[buffer(0)]],       // [K] input vector
    device const float* norm_weight [[buffer(1)]], // [K] RMSNorm scale weights
    device const char* weights [[buffer(2)]],      // [M, nb] Q4_K quantized weights
    device float* output [[buffer(3)]],            // [M] output vector
    constant uint& m [[buffer(4)]],                // M (output size)
    constant uint& k [[buffer(5)]],                // K (input size)
    constant float& epsilon [[buffer(6)]],         // RMSNorm epsilon
    constant uint& nb01 [[buffer(7)]],             // Byte stride between rows
    threadgroup float* shared_input [[threadgroup(0)]], // Shared memory for normalized input
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    const int nb = k / QK_K;  // Number of Q4_K blocks per row

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

    // Phase 3: Compute dot product with Q4_K weights
    device const char* row_ptr = weights + row * nb01;
    device const block_q4_K* blocks = (device const block_q4_K*)row_ptr;

    float sumf = 0.0f;

    // Each thread processes blocks at stride = group_size
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q4_K* block = blocks + ib;
        threadgroup const float* y = shared_input + ib * QK_K;

        // Super-block scales (convert FP16 to float)
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        // Process 8 sub-blocks of 32 elements
        for (int is = 0; is < 8; is++) {
            // Extract scale and min for this sub-block
            uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
            float sc = sc_min.x;
            float m = sc_min.y;

            float dl = d * sc;
            float ml = dmin * m;

            // 32 elements stored in 16 bytes (4 bits each)
            device const uchar* qs = block->qs + is * 16;

            // Process 16 bytes (32 elements)
            for (int kk = 0; kk < 16; kk++) {
                uchar byte = qs[kk];
                float v0 = (byte & 0x0F);
                float v1 = ((byte >> 4) & 0x0F);

                sumf += (dl * v0 - ml) * y[is * 32 + kk];
                sumf += (dl * v1 - ml) * y[is * 32 + 16 + kk];
            }
        }
    }

    // Reduce across all threads
    float sum_all = simd_sum(sumf);

    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = sum_all;
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
            output[row] = sum_all;
        }
    }
}

// ============================================================================
// Fused RMSNorm + Q5_K GEMV kernel
// ============================================================================

kernel void kernel_fused_norm_linear_q5_k(
    device const float* input [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const char* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& nb01 [[buffer(7)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    const int nb = k / QK_K;

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

    // Phase 2: Normalize
    for (uint i = lane_id; i < k; i += group_size) {
        shared_input[i] = input[i] * rms * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Compute dot product with Q5_K weights
    device const char* row_ptr = weights + row * nb01;
    device const block_q5_K* blocks = (device const block_q5_K*)row_ptr;

    float sumf = 0.0f;

    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q5_K* block = blocks + ib;
        threadgroup const float* y = shared_input + ib * QK_K;

        device const half* dh = (device const half*)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        // Process 8 sub-blocks of 32 elements
        for (int is = 0; is < 8; is++) {
            uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
            float sc = sc_min.x;
            float m = sc_min.y;

            float dl = d * sc;
            float ml = dmin * m;

            // Q5_K: low 4 bits in qs, high bit in qh
            device const uchar* qs = block->qs + is * 16;
            device const uchar* qh = block->qh + is * 4;

            for (int kk = 0; kk < 16; kk++) {
                uchar byte = qs[kk];
                uchar high_bits = qh[kk / 4];

                float v0 = (byte & 0x0F) + ((high_bits & 0x01) ? 16.0f : 0.0f);
                float v1 = ((byte >> 4) & 0x0F) + ((high_bits & 0x02) ? 16.0f : 0.0f);

                sumf += (dl * v0 - ml) * y[is * 32 + kk];
                sumf += (dl * v1 - ml) * y[is * 32 + 16 + kk];

                high_bits >>= 2;
            }
        }
    }

    float sum_all = simd_sum(sumf);

    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = sum_all;
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
            output[row] = sum_all;
        }
    }
}

// ============================================================================
// Fused RMSNorm + Q6_K GEMV kernel
// ============================================================================

kernel void kernel_fused_norm_linear_q6_k(
    device const float* input [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const char* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& nb01 [[buffer(7)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    const int nb = k / QK_K;

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

    // Phase 2: Normalize
    for (uint i = lane_id; i < k; i += group_size) {
        shared_input[i] = input[i] * rms * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Compute dot product with Q6_K weights
    device const char* row_ptr = weights + row * nb01;
    device const block_q6_K* blocks = (device const block_q6_K*)row_ptr;

    float sumf = 0.0f;

    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q6_K* block = blocks + ib;
        threadgroup const float* y = shared_input + ib * QK_K;

        // Q6_K: 16 sub-blocks of 16 elements each
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];

        for (int is = 0; is < 16; is++) {
            float sc = block->scales[is];

            // Q6_K: lower 4 bits in ql, upper 2 bits in qh
            device const uchar* ql = block->ql + is * 8;
            device const uchar* qh = block->qh + is * 4;

            for (int kk = 0; kk < 8; kk++) {
                uchar l = ql[kk];
                uchar h = qh[kk];

                float v0 = (l & 0x0F) + ((h & 0x03) << 4);
                float v1 = ((l >> 4) & 0x0F) + ((h >> 2) & 0x03) * 16.0f;

                sumf += d * sc * v0 * y[is * 16 + kk];
                sumf += d * sc * v1 * y[is * 16 + 8 + kk];
            }
        }
    }

    float sum_all = simd_sum(sumf);

    if (group_size > 32) {
        if (lane_in_simd == 0) {
            reduction_buf[simd_id] = sum_all;
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
            output[row] = sum_all;
        }
    }
}
