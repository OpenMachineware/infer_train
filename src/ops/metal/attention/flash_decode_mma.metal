// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// =============================================================================
// STATUS: EXPERIMENTAL - Work in progress for server mode (vLLM-style batching)
// =============================================================================
// This kernel is kept for future server-mode implementation where multiple
// decode positions can be batched together. Not currently used in production.
// For local inference, use flash_attn_quant.metal which supports all formats.
// =============================================================================

// Flash Decode using simdgroup operations
// Optimized for decode (single query per head)
//
// Design:
// - 1 threadgroup = 1 simdgroup = 4 heads
// - 32 threads total, 8 threads per head
// - Each thread handles 16 elements of head_dim (128/8=16)
// - Per-head online softmax with M, S state
// - Q*K^T: per-head reduction using simd_shuffle_xor
// - V accumulation: each thread accumulates its portion
//
// Memory layout:
// - Q: [n_heads, head_dim] - loaded once
// - K: [n_kv_heads, seq_len, head_dim] - loaded per position
// - V: [n_kv_heads, seq_len, head_dim] - loaded per position
//
// Thread layout (tiisg 0-31):
// - Threads 0-7: head 0
// - Threads 8-15: head 1
// - Threads 16-23: head 2
// - Threads 24-31: head 3

#include <metal_stdlib>
using namespace metal;

constexpr constant int HEAD_DIM = 128;
constexpr constant int THREADS_PER_HEAD = 8;
constexpr constant int HEADS_PER_SG = 4;
constexpr constant int ELEMS_PER_THREAD = HEAD_DIM / THREADS_PER_HEAD; // 16

// Per-head reduction: sum across 8 threads of a head
// Uses simd_shuffle_xor which is symmetric and respects 8-thread boundaries
inline float per_head_reduce(float value, int head_in_sg, int lane_in_head) {
    (void)head_in_sg;
    (void)lane_in_head;

    // XOR pattern: swaps pairs within each 8-thread group
    // xor 1: 0↔1, 2↔3, 4↔5, 6↔7, 8↔9, ...
    // xor 2: 0↔2, 1↔3, 4↔6, 5↔7, 8↔10, ...
    // xor 4: 0↔4, 1↔5, 2↔6, 3↔7, 8↔12, ...

    // After these steps, each thread has the sum of all 8 values
    // All threads participate in each step
    value += simd_shuffle_xor(value, 1);
    value += simd_shuffle_xor(value, 2);
    value += simd_shuffle_xor(value, 4);

    return value;
}

kernel void flash_decode_mma_f16(
    constant uint& seq_k [[buffer(0)]],
    constant uint& kv_stride [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& gqa_ratio [[buffer(7)]],  // n_heads / n_kv_heads
    device const half* query [[buffer(3)]],       // [n_heads, head_dim]
    device const half* key_cache [[buffer(4)]],   // [n_kv_heads, kv_stride, head_dim]
    device const half* value_cache [[buffer(5)]], // [n_kv_heads, kv_stride, head_dim]
    device float* output [[buffer(6)]],           // [n_heads, head_dim]
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Threadgroup processes 4 heads
    const int head_group = gid.x;
    const int head_base = head_group * HEADS_PER_SG;

    // Which head within this simdgroup (0-3)
    const int head_in_sg = tiisg / THREADS_PER_HEAD;
    // Which lane within this head (0-7)
    const int lane_in_head = tiisg % THREADS_PER_HEAD;
    // Global head index
    const int head = head_base + head_in_sg;

    // GQA: map Q head to KV head
    const int kv_head = head / gqa_ratio;

    // Get pointers
    device const half* q_ptr = query + head * HEAD_DIM;
    device const half* k_head_ptr = key_cache + kv_head * kv_stride * HEAD_DIM;
    device const half* v_head_ptr = value_cache + kv_head * kv_stride * HEAD_DIM;

    // Load Q for this head (each thread loads 16 elements = 4 half4)
    device const half4* q_h4 = (device const half4*)(q_ptr + lane_in_head * ELEMS_PER_THREAD);
    float4 q_frag[4];
    for (int i = 0; i < 4; i++) {
        q_frag[i] = (float4)q_h4[i];
    }

    // Online softmax state per head (each thread holds for its head)
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator
    float4 O[4];
    for (int i = 0; i < 4; i++) {
        O[i] = float4(0.0f);
    }

    // Process all KV positions
    for (uint pos = 0; pos < seq_k; pos++) {
        // Load K and V for this position
        device const half* k_pos = k_head_ptr + pos * HEAD_DIM;
        device const half* v_pos = v_head_ptr + pos * HEAD_DIM;
        device const half4* k_h4 = (device const half4*)(k_pos + lane_in_head * ELEMS_PER_THREAD);
        device const half4* v_h4 = (device const half4*)(v_pos + lane_in_head * ELEMS_PER_THREAD);

        float4 k_frag[4];
        float4 v_frag[4];
        for (int i = 0; i < 4; i++) {
            k_frag[i] = (float4)k_h4[i];
            v_frag[i] = (float4)v_h4[i];
        }

        // Compute partial Q*K^T
        float qk_partial = 0.0f;
        for (int i = 0; i < 4; i++) {
            qk_partial += dot(q_frag[i], k_frag[i]);
        }

        // Reduce within head using per-head reduction
        float qk_total = per_head_reduce(qk_partial, head_in_sg, lane_in_head);
        float qk_score = qk_total * scale;

        // Online softmax update
        float m_prev = M;
        M = max(M, qk_score);
        float exp_mdiff = exp(m_prev - M);
        float exp_sdiff = exp(qk_score - M);

        // Update S
        S = S * exp_mdiff + exp_sdiff;

        // Rescale and accumulate
        for (int i = 0; i < 4; i++) {
            O[i] *= exp_mdiff;
            O[i] += exp_sdiff * v_frag[i];
        }
    }

    // Final normalization
    float norm = 1.0f / S;
    for (int i = 0; i < 4; i++) {
        O[i] *= norm;
    }

    // Write output
    device float* out_ptr = output + head * HEAD_DIM;
    device float4* out_f4 = (device float4*)(out_ptr + lane_in_head * ELEMS_PER_THREAD);
    for (int i = 0; i < 4; i++) {
        out_f4[i] = O[i];
    }
}
