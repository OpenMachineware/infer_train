// Softmax Metal kernel - Truly self-written implementation
// Algorithm: Numerically stable softmax
//   1. Find max for stability
//   2. Compute exp(x - max)
//   3. Normalize by sum
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

constant float NEG_INF = 1e30f;

kernel void kernel_soft_max_f32_fast(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& num_heads [[buffer(3)]],
    threadgroup float* temp_mem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int head_idx = tgpig.x;
    const int batch_idx = tgpig.y;

    const uint64_t row_bytes = seq_len * 4;
    const uint64_t batch_bytes = num_heads * row_bytes;
    const uint64_t offset = batch_idx * batch_bytes + head_idx * row_bytes;

    device const float* row_scores = scores + offset / 4;
    device float* row_probs = probs + offset / 4;

    float local_max = NEG_INF;
    for (uint i = tpitg.x; i < uint(seq_len); i += tptg.x) {
        local_max = max(local_max, row_scores[i]);
    }

    float row_max = simd_max(local_max);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = NEG_INF;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_max = simd_max(temp_mem[tiisg]);
    }

    float local_sum = 0.0f;
    for (uint i = tpitg.x; i < uint(seq_len); i += tptg.x) {
        float exp_val = exp(row_scores[i] - row_max);
        local_sum += exp_val;
        row_probs[i] = exp_val;
    }

    float row_sum = simd_sum(local_sum);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_sum = simd_sum(temp_mem[tiisg]);
    }

    float scale = 1.0f / row_sum;
    for (uint i = tpitg.x; i < uint(seq_len); i += tptg.x) {
        row_probs[i] *= scale;
    }
}

kernel void kernel_soft_max_f32_4_fast(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& num_heads [[buffer(3)]],
    threadgroup float* temp_mem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int head_idx = tgpig.x;
    const int batch_idx = tgpig.y;

    const int vec_len = seq_len / 4;
    const uint64_t row_bytes = seq_len * 4;
    const uint64_t batch_bytes = num_heads * row_bytes;
    const uint64_t offset = batch_idx * batch_bytes + head_idx * row_bytes;

    device const float4* vec_scores = (device const float4*)(scores + offset / 4);
    device float4* vec_probs = (device float4*)(probs + offset / 4);

    float4 local_max4 = NEG_INF;
    for (uint i = tpitg.x; i < uint(vec_len); i += tptg.x) {
        local_max4 = fmax(local_max4, vec_scores[i]);
    }
    float local_max = max(max(local_max4[0], local_max4[1]),
                         max(local_max4[2], local_max4[3]));

    float row_max = simd_max(local_max);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = NEG_INF;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_max = simd_max(temp_mem[tiisg]);
    }

    float4 local_sum4 = 0.0f;
    for (uint i = tpitg.x; i < uint(vec_len); i += tptg.x) {
        float4 exp_val = exp(vec_scores[i] - row_max);
        local_sum4 += exp_val;
        vec_probs[i] = exp_val;
    }
    float local_sum = local_sum4[0] + local_sum4[1] + local_sum4[2] + local_sum4[3];

    float row_sum = simd_sum(local_sum);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_sum = simd_sum(temp_mem[tiisg]);
    }

    float scale = 1.0f / row_sum;
    for (uint i = tpitg.x; i < uint(vec_len); i += tptg.x) {
        vec_probs[i] *= scale;
    }
}

kernel void softmax_f16(
    device const half* scores [[buffer(0)]],
    device half* probs [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& num_heads [[buffer(3)]],
    threadgroup float* temp_mem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int head_idx = tgpig.x;
    const int batch_idx = tgpig.y;

    const uint64_t row_bytes = seq_len * 2;
    const uint64_t batch_bytes = num_heads * row_bytes;
    const uint64_t offset = batch_idx * batch_bytes + head_idx * row_bytes;

    device const half* row_scores = scores + offset / 2;
    device half* row_probs = probs + offset / 2;

    float local_max = NEG_INF;
    for (uint i = tpitg.x; i < uint(seq_len); i += tptg.x) {
        local_max = max(local_max, float(row_scores[i]));
    }

    float row_max = simd_max(local_max);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = NEG_INF;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_max = simd_max(temp_mem[tiisg]);
    }

    float local_sum = 0.0f;
    for (uint i = tpitg.x; i < uint(seq_len); i += tptg.x) {
        float exp_val = exp(float(row_scores[i]) - row_max);
        local_sum += exp_val;
        row_probs[i] = half(exp_val);
    }

    float row_sum = simd_sum(local_sum);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_sum = simd_sum(temp_mem[tiisg]);
    }

    float scale = 1.0f / row_sum;
    for (uint i = tpitg.x; i < uint(seq_len); i += tptg.x) {
        row_probs[i] = half(float(row_probs[i]) * scale);
    }
}

kernel void softmax_f16_vec4(
    device const half4* scores [[buffer(0)]],
    device half4* probs [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& num_heads [[buffer(3)]],
    threadgroup float* temp_mem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int head_idx = tgpig.x;
    const int batch_idx = tgpig.y;

    const int vec_len = seq_len / 4;
    const uint64_t vec_bytes = seq_len * 2;
    const uint64_t batch_bytes = num_heads * vec_bytes;
    const uint64_t offset = batch_idx * batch_bytes + head_idx * vec_bytes;

    device const half4* vec_scores = scores + offset / 8;
    device half4* vec_probs = probs + offset / 8;

    float4 local_max4 = NEG_INF;
    for (uint i = tpitg.x; i < uint(vec_len); i += tptg.x) {
        float4 v = float4(vec_scores[i]);
        local_max4 = fmax(local_max4, v);
    }
    float local_max = max(max(local_max4[0], local_max4[1]),
                         max(local_max4[2], local_max4[3]));

    float row_max = simd_max(local_max);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = NEG_INF;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_max = simd_max(temp_mem[tiisg]);
    }

    float4 local_sum4 = 0.0f;
    for (uint i = tpitg.x; i < uint(vec_len); i += tptg.x) {
        float4 exp_val = exp(float4(vec_scores[i]) - row_max);
        local_sum4 += exp_val;
        vec_probs[i] = half4(exp_val);
    }
    float local_sum = local_sum4[0] + local_sum4[1] + local_sum4[2] + local_sum4[3];

    float row_sum = simd_sum(local_sum);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            temp_mem[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            temp_mem[sgitg] = row_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        row_sum = simd_sum(temp_mem[tiisg]);
    }

    float scale = 1.0f / row_sum;
    for (uint i = tpitg.x; i < uint(vec_len); i += tptg.x) {
        vec_probs[i] = half4(float4(vec_probs[i]) * scale);
    }
}
