// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Step-by-step performance testing
// Measure each step separately to find performance gaps
//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Step 1: Query Load Test
// Load query from global memory to shared memory
// ============================================================================

kernel void test_query_load_our(
    device const float* query [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;

    // Shared memory for query
    threadgroup float4 s_query[32];

    // Load query into shared memory
    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)query;
        s_query[tiisg] = q_ptr[tiisg];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write back to verify correctness
    if (tiisg < HEAD_DIM / 4) {
        device float4* out_ptr = (device float4*)output;
        out_ptr[tiisg] = s_query[tiisg];
    }
}

// ============================================================================
// Step 2: K Read Test (F16)
// Read Key cache from global memory
// ============================================================================

kernel void test_k_read_our(
    device const half* key_cache [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_k [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int C = 32;  // Cache items per iteration
    constexpr int NE = 4;  // Cache items per thread per iteration
    constexpr int NL = 32 / NE;  // 8

    const int tx = tiisg % NL;
    const int ty = tiisg / NL;

    const int head = gid.y;
    const int iwg = gid.x;

    // Pointer to key cache for this head
    device const half* k_head = key_cache + head * HEAD_DIM * seq_k;

    // Process first chunk
    const int ic = 0;

    // Read Key values (F16 direct read)
    float k_sum = 0.0f;

    if (ic + ty < seq_k) {
        device const half4* k_base = (device const half4*)(k_head + ic * HEAD_DIM);
        k_base += ty * (HEAD_DIM / 4) + tx;

        for (int ii = 0; ii < HEAD_DIM / (4 * NL); ii++) {
            half4 k_values = k_base[ii * NL];
            k_sum += float(k_values[0]) + float(k_values[1]) + float(k_values[2]) + float(k_values[3]);
        }
    }

    // Write result to output
    if (tiisg == 0) {
        output[head] = k_sum;
    }
}

// ============================================================================
// Step 3: QK Compute Test (F16)
// Compute Q*K^T dot product
// ============================================================================

kernel void test_qk_compute_our(
    device const float* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& seq_k [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int C = 32;
    constexpr int NE = 4;
    constexpr int NL = 8;

    const int tx = tiisg % NL;
    const int ty = tiisg / NL;

    const int head = gid.y;
    const int iwg = gid.x;

    // Step 1: Load query
    threadgroup float4 s_query[32];
    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)
            (query + head * HEAD_DIM);
        s_query[tiisg] = q_ptr[tiisg];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Q*K^T computation
    device const half* k_head = key_cache + head * HEAD_DIM * seq_k;

    float qk_sum = 0.0f;

    const int ic = iwg * C;  // Each workgroup starts at different position
    if (ic + ty < seq_k) {
        device const half4* k_base = (device const half4*)(k_head + ic * HEAD_DIM);
        k_base += ty * (HEAD_DIM / 4) + tx;

        float dot_result = 0.0f;
        for (int ii = 0; ii < HEAD_DIM / (4 * NL); ii++) {
            half4 k_values = k_base[ii * NL];
            dot_result += dot((float4)k_values, s_query[ii * NL + tx]);
        }

        // Tree reduction within ty group
        dot_result += simd_shuffle_down(dot_result, 4);
        dot_result += simd_shuffle_down(dot_result, 2);
        dot_result += simd_shuffle_down(dot_result, 1);
        dot_result = simd_shuffle(dot_result, NL * ty);

        qk_sum = dot_result;
    }

    // Write result
    if (tiisg == 0) {
        output[head * 512 + iwg] = qk_sum * scale;
    }
}

// ============================================================================
// Step 4: Score Storage + Online Softmax Test
// ============================================================================

kernel void test_softmax_our(
    device float* output [[buffer(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int C = 32;

    // Shared memory for scores
    threadgroup float s_scores[C];

    // Initialize with test values
    if (tiisg < C) {
        s_scores[tiisg] = float(tiisg) * 0.1f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax
    float M = -INFINITY;
    float S = 0.0f;

    for (int iter = 0; iter < 8; iter++) {
        const float s = s_scores[tiisg % C];

        const float m = M;
        M = simd_max(max(M, s));

        const float ms = exp(m - M);
        const float vs = exp(s - M);

        S = S * ms + simd_sum(vs);

        s_scores[tiisg % C] = vs;
    }

    // Write result
    if (tiisg == 0) {
        output[0] = M;
        output[1] = S;
    }
}

// ============================================================================
// Step 5: V Read + Accumulation Test
// ============================================================================

kernel void test_v_accum_our(
    device const half* value_cache [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_k [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int C = 32;
    constexpr int NE = 4;
    constexpr int NL = 8;

    const int tx = tiisg % NL;
    const int ty = tiisg / NL;

    const int head = gid.y;
    const int iwg = gid.x;

    // Output accumulator
    float4 O[HEAD_DIM / (4 * NL)];
    for (int i = 0; i < HEAD_DIM / (4 * NL); i++) {
        O[i] = float4(0.0f);
    }

    // Shared memory for weights
    threadgroup float s_weights[C];
    if (tiisg < C) {
        s_weights[tiisg] = 1.0f / C;  // Equal weights for test
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // V accumulation
    device const half* v_head = value_cache + head * HEAD_DIM * seq_k;

    const int ic = iwg * C;

    for (int v_iter = 0; v_iter < 8; v_iter++) {
        const float weight = s_weights[tiisg % C];
        const int cache_idx = ic + tiisg;

        if (cache_idx < seq_k) {
            device const half4* v_base = (device const half4*)(v_head + ic * HEAD_DIM);
            v_base += ty * (HEAD_DIM / 4) + tx;

            for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
                half4 v_values = v_base[j * NL];
                O[j] += weight * (float4)v_values;
            }
        }
    }

    // Write result
    if (tiisg < HEAD_DIM / 4) {
        device float4* out_ptr = (device float4*)(output + head * HEAD_DIM);
        out_ptr[tiisg] = O[tiisg / NL];
    }
}

// ============================================================================
// Complete test (all steps together)
// ============================================================================

kernel void test_complete_our(
    device const float* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_k [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int C = 32;
    constexpr int NE = 4;
    constexpr int NL = 8;
    constexpr int NWG = 16;

    const int tx = tiisg % NL;
    const int ty = tiisg / NL;

    const int head = gid.y;
    const int iwg = gid.x;

    // Step 1: Load query
    threadgroup float4 s_query[32];
    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)
            (query + head * HEAD_DIM);
        s_query[tiisg] = q_ptr[tiisg];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize state
    float M = -INFINITY;
    float S = 0.0f;
    float4 O[4];
    for (int i = 0; i < 4; i++) {
        O[i] = float4(0.0f);
    }

    threadgroup float s_scores[C];

    // Strided processing
    for (int ic0 = iwg; ic0 * C < seq_k; ic0 += NWG) {
        const int ic = ic0 * C;

        // Step 2: Q*K^T
        float mqk = 0.0f;

        if (ic + ty < seq_k) {
            device const half4* k_base = (device const half4*)(key_cache + head * HEAD_DIM * seq_k + ic * HEAD_DIM);
            k_base += ty * (HEAD_DIM / 4) + tx;

            for (int ii = 0; ii < HEAD_DIM / (4 * NL); ii++) {
                half4 k_values = k_base[ii * NL];
                mqk += dot((float4)k_values, s_query[ii * NL + tx]);
            }
        }

        // Step 3: Reduction
        mqk += simd_shuffle_down(mqk, 4);
        mqk += simd_shuffle_down(mqk, 2);
        mqk += simd_shuffle_down(mqk, 1);
        mqk = simd_shuffle(mqk, NL * ty);

        // Step 4: Score storage
        const int score_idx = NE * tx + ty;
        if (ic + score_idx < seq_k) {
            s_scores[score_idx] = mqk * scale;
        } else {
            s_scores[score_idx] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Online softmax
        const float s = s_scores[tiisg];

        const float m = M;
        M = simd_max(max(M, s));

        const float ms = exp(m - M);
        const float vs = exp(s - M);

        S = S * ms + simd_sum(vs);

        s_scores[tiisg] = vs;

        // Step 6: Scale previous output
        for (int j = 0; j < 4; j++) {
            O[j] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 7: V accumulation
        const float weight = s_scores[tiisg];
        const int cache_idx = ic + tiisg;

        if (cache_idx < seq_k) {
            device const half4* v_base = (device const half4*)(value_cache + head * HEAD_DIM * seq_k + ic * HEAD_DIM);
            v_base += ty * (HEAD_DIM / 4) + tx;

            for (int j = 0; j < 4; j++) {
                half4 v_values = v_base[j * NL];
                O[j] += weight * (float4)v_values;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 8: Finalize
    if (tiisg < HEAD_DIM / 4) {
        output[head * HEAD_DIM + tiisg * 4 + 0] = O[tiisg / 8][tiisg % 8 * 4 + 0] / S;
        output[head * HEAD_DIM + tiisg * 4 + 1] = O[tiisg / 8][tiisg % 8 * 4 + 1] / S;
        output[head * HEAD_DIM + tiisg * 4 + 2] = O[tiisg / 8][tiisg % 8 * 4 + 2] / S;
        output[head * HEAD_DIM + tiisg * 4 + 3] = O[tiisg / 8][tiisg % 8 * 4 + 3] / S;
    }
}
