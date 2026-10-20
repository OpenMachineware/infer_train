// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode - Simplified single-head version for correctness testing
//
// Design:
// - 1 threadgroup = 1 simdgroup = 1 head
// - 32 threads process head_dim elements in parallel
// - Each thread handles HEAD_DIM/32 = 4 elements
// - Process positions one at a time (not parallel over positions)

#include <metal_stdlib>
using namespace metal;

constexpr constant int HEAD_DIM = 128;
constexpr constant int ELEMS_PER_THREAD = HEAD_DIM / 32;  // 4 elements per thread

kernel void flash_decode_mma_f16(
    constant uint& seq_k [[buffer(0)]],
    constant uint& kv_stride [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    device const half* query [[buffer(3)]],       // [n_heads, head_dim]
    device const half* key_cache [[buffer(4)]],   // [n_kv_heads, kv_stride, head_dim]
    device const half* value_cache [[buffer(5)]], // [n_kv_heads, kv_stride, head_dim]
    device float* output [[buffer(6)]],           // [n_heads, head_dim]
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // One threadgroup per head
    const int head = gid.x;

    // GQA: map Q head to KV head
    // Assuming GQA ratio of 4 (16 Q heads / 4 KV heads)
    const int gqa_ratio = 4;
    const int kv_head = head / gqa_ratio;

    // Get pointers
    device const half* q_ptr = query + head * HEAD_DIM;
    device const half* k_head_ptr = key_cache + kv_head * kv_stride * HEAD_DIM;
    device const half* v_head_ptr = value_cache + kv_head * kv_stride * HEAD_DIM;

    // Load Q for this head
    // Each thread loads 4 elements (1 half4)
    device const half4* q_h4 = (device const half4*)(q_ptr + tiisg * ELEMS_PER_THREAD);
    float4 q_frag = (float4)q_h4[0];

    // Online softmax state
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator
    float4 O = float4(0.0f);

    // Process all KV positions sequentially
    for (uint pos = 0; pos < seq_k; pos++) {
        // Load K and V for this position
        device const half* k_pos = k_head_ptr + pos * HEAD_DIM;
        device const half* v_pos = v_head_ptr + pos * HEAD_DIM;
        device const half4* k_h4 = (device const half4*)(k_pos + tiisg * ELEMS_PER_THREAD);
        device const half4* v_h4 = (device const half4*)(v_pos + tiisg * ELEMS_PER_THREAD);

        float4 k_frag = (float4)k_h4[0];
        float4 v_frag = (float4)v_h4[0];

        // Compute partial Q*K^T
        float qk_partial = dot(q_frag, k_frag);

        // Reduce across all 32 threads using simd_sum
        float qk_total = simd_sum(qk_partial);
        float qk_score = qk_total * scale;

        // Online softmax update
        float m_prev = M;
        M = simd_max(max(M, qk_score));
        float exp_mdiff = exp(m_prev - M);
        float exp_sdiff = exp(qk_score - M);

        // CRITICAL FIX: S should not be summed across threads
        // exp_sdiff is already a scalar (same for all threads after broadcast)
        // We only need to update S once, not sum across 32 threads
        if (tiisg == 0) {
            S = S * exp_mdiff + exp_sdiff;
        }
        // Broadcast S to all threads
        S = simd_shuffle(S, 0);

        // Rescale and accumulate
        O *= exp_mdiff;
        O += exp_sdiff * v_frag;
    }

    // Final normalization
    O *= (1.0f / S);

    // Write output
    device float* out_ptr = output + head * HEAD_DIM;
    device float4* out_f4 = (device float4*)(out_ptr + tiisg * ELEMS_PER_THREAD);
    out_f4[0] = O;
}
