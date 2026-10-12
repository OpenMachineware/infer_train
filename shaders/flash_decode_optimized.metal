// Flash Decode Vec Optimized for F16
// Uses continuous memory access pattern like llama.cpp
//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// F16 specific kernel with optimized memory access
kernel void flash_decode_vec_f16_optimized(
    constant uint& seq_k [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device const char* key_cache [[buffer(3)]],
    device const char* value_cache [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Thread layout
    constexpr int NL = 8;   // Threads per ty group
    constexpr int NE = 4;   // Cache items per group
    constexpr int C = 32;   // Total cache items per iteration

    const int tx = tiisg % NL;  // 0-7
    const int ty = tiisg / NL;  // 0-3

    const int batch = gid.z;
    const int head = gid.y;

    // Shared memory for query and scores
    threadgroup float4 s_query[32];
    threadgroup float s_scores[C];

    // Load query - all threads participate
    if (tiisg < 32) {
        device const float4* q_ptr = (device const float4*)
            (query + batch * 128 + head * 128);
        s_query[tiisg] = q_ptr[tiisg];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float M = -INFINITY;
    float S = 0.0f;
    float4 O[4];  // 4 float4s per thread (16 elements)
    for (int i = 0; i < 4; i++) O[i] = float4(0.0f);

    // Process all cache items
    const int nb_iterations = (seq_k + C - 1) / C;

    for (int iter = 0; iter < nb_iterations; iter++) {
        const int ic = iter * C;

        // Q*K^T
        float mqk[C / NE];
        for (int cc = 0; cc < C / NE; cc++) mqk[cc] = 0.0f;

        // Optimized K cache access: pre-adjust pointer
        device const half4* k_base = (device const half4*)
            (key_cache + ic * 128 + head * 128);

        for (int cc = 0; cc < C / NE; cc++) {
            const int cache_idx = ic + NE * cc + ty;

            if (cache_idx < seq_k) {
                // Key insight: adjust pointer for continuous access
                // k_base[ty * NS10 + tx] where NS10 = C * 128 / 4 = 1024
                // For each ty, we want to access different cache items
                // Each cache item has 128 floats = 32 float4s

                // Simple approach: direct index calculation
                device const half4* k_ptr = (device const half4*)
                    (key_cache + cache_idx * 128 + head * 128);

                // Each thread processes 4 float4s (16 elements)
                // tx=0: elements 0-15, tx=1: elements 1-16, etc.
                // To make it continuous: tx accesses tx, tx+8, tx+16, tx+24
                for (int ii = 0; ii < 4; ii++) {
                    const int idx = ii * NL + tx;  // 0, 8, 16, 24 for tx=0
                    const float4 k_val = float4(k_ptr[idx]);
                    mqk[cc] += dot(k_val, s_query[idx]);
                }
            } else {
                mqk[cc] = -INFINITY;
            }
        }

        // Reduce mqk
        for (int cc = 0; cc < C / NE; cc++) {
            // Tree reduction within ty group
            if (true) mqk[cc] += simd_shuffle_down(mqk[cc], 4);
            if (true) mqk[cc] += simd_shuffle_down(mqk[cc], 2);
            if (true) mqk[cc] += simd_shuffle_down(mqk[cc], 1);
            mqk[cc] = simd_shuffle(mqk[cc], NL * ty);
        }

        // Store scores
        const int score_idx = NE * tx + ty;
        if (score_idx < seq_k - ic && tx < NE) {
            s_scores[score_idx] = mqk[tx] * scale;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        const float s = s_scores[tiisg];
        const float m = M;
        M = simd_max(max(M, s));
        const float ms = exp(m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);

        s_scores[tiisg] = vs;

        // Scale output
        for (int j = 0; j < 4; j++) O[j] *= ms;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation
        const float weight = s_scores[tiisg];
        const int cache_idx = ic + tiisg;

        if (cache_idx < seq_k) {
            device const half4* v_ptr = (device const half4*)
                (value_cache + cache_idx * 128 + head * 128);

            for (int j = 0; j < 4; j++) {
                const int idx = j * NL + tx;
                O[j] += weight * float4(v_ptr[idx]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    if (tiisg < 32) {
        device float4* out_ptr = (device float4*)(output + head * 128);
        out_ptr[tiisg] = O[tiisg / NL];
    }
}
