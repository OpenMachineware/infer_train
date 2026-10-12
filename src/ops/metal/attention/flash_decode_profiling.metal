// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode Profiling Kernel
// Measures time for each stage separately
//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

struct Q4_0Block {
    half d;
    uint8_t qs[16];
};

struct F16Block {
    half values[4];
};

template<typename T4>
inline void dequant_q4_0(device const Q4_0Block* block, int lane, thread T4& result) {
    const half d = block->d;
    const int group = lane % 8;
    const int is_high = group / 4;
    const int idx = group % 4;

    device const uint16_t* qs16 = (device const uint16_t*)block->qs;

    const half scale = is_high ? d / 16.0h : d;
    const half offset = -8.0h * d;
    const uint16_t mask = is_high ? 0x00F0 : 0x000F;

    for (int i = 0; i < 2; i++) {
        uint16_t val = qs16[2 * idx + i];
        result[2*i + 0] = scale * half(val & mask) + offset;
        result[2*i + 1] = (scale / 256.0h) * half((val >> 8) & mask) + offset;
    }
}

template<typename T4>
inline void dequant_f16(device const F16Block* block, int lane, thread T4& result) {
    for (int i = 0; i < 4; i++) {
        result[i] = block->values[i];
    }
}

// Stage 1: Q*K^T only
kernel void flash_decode_stage1_qk(
    constant uint& seq_k [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device const char* key_cache [[buffer(3)]],
    device float* scores [[buffer(4)]],  // Output: Q*K^T scores
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int NL = 8;
    constexpr int NE = 4;
    constexpr int C = 32;

    const int tx = tiisg % NL;
    const int ty = tiisg / NL;

    const int batch = gid.z;
    const int head = gid.y;

    threadgroup float4 s_query[32];

    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)
            (query + batch * HEAD_DIM + head * HEAD_DIM);
        s_query[tiisg] = q_ptr[tiisg];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ic = 0;  // Process first C items only for profiling
    float mqk[C / NE];

    #pragma unroll
    for (int cc = 0; cc < C / NE; cc++) {
        const int cache_idx = ic + NE * cc + ty;

        if (cache_idx < seq_k) {
            device const char* k_ptr = key_cache + cache_idx * HEAD_DIM + head * HEAD_DIM;
            mqk[cc] = 0.0f;

            #pragma unroll
            for (int ii = 0; ii < HEAD_DIM / (4 * NL); ii++) {
                const int elem_idx = ii * NL + tx;

                float4 k_values;
                dequant_f16((device const F16Block*)(k_ptr) + elem_idx, 0, k_values);

                mqk[cc] += dot(k_values, s_query[elem_idx]);
            }

            // Tree reduction
            mqk[cc] += simd_shuffle_down(mqk[cc], 4);
            mqk[cc] += simd_shuffle_down(mqk[cc], 2);
            mqk[cc] += simd_shuffle_down(mqk[cc], 1);
            mqk[cc] = simd_shuffle(mqk[cc], NL * ty);
        }
    }

    // Store scores
    const int score_idx = NE * tx + ty;
    if (score_idx < seq_k) {
        scores[head * seq_k + score_idx] = mqk[tx] * scale;
    }
}

// Stage 2: Softmax only (given scores, compute weights)
kernel void flash_decode_stage2_softmax(
    constant uint& seq_k [[buffer(0)]],
    device float* scores [[buffer(1)]],  // Input: Q*K^T scores, Output: weights
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int head = gid.y;

    threadgroup float s_scores[32];

    // Load scores
    if (tiisg < seq_k) {
        s_scores[tiisg] = scores[head * seq_k + tiisg];
    } else {
        s_scores[tiisg] = -INFINITY;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax
    float M = -INFINITY;
    float S = 0.0f;

    const float s = s_scores[tiisg];

    M = simd_max(max(M, s));

    const float ms = exp(-INFINITY - M);  // Simplified
    const float vs = exp(s - M);

    S = S * ms + simd_sum(vs);

    // Store weight
    if (tiisg < seq_k) {
        scores[head * seq_k + tiisg] = vs;
    }
}

// Stage 3: V accumulation only (given weights)
kernel void flash_decode_stage3_vacc(
    constant uint& seq_k [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const char* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int NL = 8;

    const int tx = tiisg % NL;
    const int head = gid.y;

    float4 O[HEAD_DIM / (4 * NL)];
    #pragma unroll
    for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
        O[j] = float4(0.0f);
    }

    const int ic = 0;
    const float weight = weights[head * seq_k + tiisg];
    const int cache_idx = ic + tiisg;

    if (cache_idx < seq_k) {
        device const char* v_ptr = value_cache + cache_idx * HEAD_DIM + head * HEAD_DIM;

        #pragma unroll
        for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
            const int elem_idx = j * NL + tx;

            float4 v_values;
            dequant_f16((device const F16Block*)(v_ptr) + elem_idx, 0, v_values);

            O[j] += weight * v_values;
        }
    }

    // Write output
    if (tiisg < HEAD_DIM / 4) {
        device float4* out_ptr = (device float4*)(output + head * HEAD_DIM);
        out_ptr[tiisg] = O[tiisg / NL];
    }
}
