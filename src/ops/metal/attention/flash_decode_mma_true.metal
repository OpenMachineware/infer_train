// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode Kernel
// Each simdgroup (32 threads) processes one head
// Sequential processing of KV positions with online softmax

#include <metal_stdlib>
using namespace metal;

kernel void flash_decode_mma_true_f16(
    constant uint& seq_k [[buffer(0)]],
    constant uint& kv_stride [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const half* key_cache [[buffer(4)]],
    device const half* value_cache [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int batch = gid.y;  // gid.y = batch index (0 to n_seqs-1)
    const int head = gid.x;   // gid.x = head index (0 to n_heads-1)

    constexpr short HEAD_DIM = 128;

    // Load Q into registers (each thread loads 16 elements)
    device const float4* q4 = (device const float4*)
        (query + batch * HEAD_DIM + head * HEAD_DIM);

    float4 q_frag[4];  // 16 elements per thread
    for (short i = 0; i < 4; i++) {
        q_frag[i] = q4[tiisg * 4 + i];
    }

    // Online softmax state
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator
    float4 O[4];
    for (short i = 0; i < 4; i++) {
        O[i] = float4(0.0f);
    }

    // Get K, V base pointers for this head
    device const half4* k_base = (device const half4*)
        (key_cache + head * kv_stride * HEAD_DIM);
    device const half4* v_base = (device const half4*)
        (value_cache + head * kv_stride * HEAD_DIM);

    // Process all KV positions
    for (uint pos = 0; pos < seq_k; pos++) {
        device const half4* k_h4 = k_base + pos * 32;  // 32 half4s per row
        device const half4* v_h4 = v_base + pos * 32;

        // Load K and V
        float4 k_frag[4];
        float4 v_frag[4];
        for (short i = 0; i < 4; i++) {
            k_frag[i] = float4(k_h4[tiisg * 4 + i]);
            v_frag[i] = float4(v_h4[tiisg * 4 + i]);
        }

        // Compute Q*K^T
        float qk = 0.0f;
        for (short i = 0; i < 4; i++) {
            qk += dot(q_frag[i], k_frag[i]);
        }

        // Reduce across all 32 threads
        float qk_total = simd_sum(qk);
        float score = qk_total * scale;

        // Online softmax update
        float m_prev = M;
        M = max(M, score);

        float exp_mdiff = exp(m_prev - M);
        float exp_sdiff = exp(score - M);

        S = S * exp_mdiff + exp_sdiff;

        // Update output
        for (short i = 0; i < 4; i++) {
            O[i] = O[i] * exp_mdiff + exp_sdiff * v_frag[i];
        }
    }

    // Final normalization
    float inv_S = S > 0.0f ? 1.0f / S : 0.0f;

    // Write output
    device float4* out_f4 = (device float4*)
        (output + batch * HEAD_DIM + head * HEAD_DIM);
    for (short i = 0; i < 4; i++) {
        out_f4[tiisg * 4 + i] = O[i] * inv_S;
    }
}
