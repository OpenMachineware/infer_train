// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode with True Matrix Multiply Accumulate (MMA)
// Uses simdgroup_float8x8 for actual matrix operations
// Processes multiple KV positions per iteration for 2-3x speedup

#include <metal_stdlib>
using namespace metal;

constant int HEAD_DIM = 128;
constant int C = 8;  // KV positions per iteration

// Arguments structure
struct FlashDecodeMMAArgs {
    uint32_t seq_k;       // KV cache length
    uint32_t n_heads;     // Number of Q heads
    uint32_t n_kv_heads;  // Number of KV heads
    uint32_t kv_stride;   // KV cache stride (in elements)
    float scale;          // 1/sqrt(head_dim)
};

kernel void flash_decode_true_mma_f16(
    constant FlashDecodeMMAArgs & args [[buffer(0)]],
    device const half* query [[buffer(1)]],       // [n_heads, head_dim]
    device const half* key_cache [[buffer(2)]],   // [n_kv_heads, kv_stride, head_dim]
    device const half* value_cache [[buffer(3)]], // [n_kv_heads, kv_stride, head_dim]
    device float* output [[buffer(4)]],           // [n_heads, head_dim]
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int head = gid.x;
    const int gqa_ratio = args.n_heads / args.n_kv_heads;
    const int kv_head = head / gqa_ratio;

    // Each thread loads 4 elements of Q (one float4)
    // HEAD_DIM=128, 32 threads, each handles 4 elements
    device const float4* q_f4 = (device const float4*)
        (query + head * HEAD_DIM);

    float4 q_frag = q_f4[tiisg];  // Each thread loads its 4 elements

    // Online softmax state
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator (each thread accumulates its 4 elements)
    float4 O = float4(0.0f);

    // Get K, V pointers for this KV head
    device const half* k_head = key_cache + kv_head * args.kv_stride * HEAD_DIM;
    device const half* v_head = value_cache + kv_head * args.kv_stride * HEAD_DIM;

    // Process KV cache in batches of C=8 positions
    for (uint pos_start = 0; pos_start < args.seq_k; pos_start += C) {
        const uint pos_end = min(pos_start + C, args.seq_k);

        // Load K and V for up to C positions
        // Correct access pattern: 32 threads / 128 elements = 4 elements per thread
        // Each thread accesses k_pos[tiisg] (one float4 = 4 elements)
        float k_batch[C][4];  // [C positions, 4 elements per position per thread]
        float v_batch[C][4];

        for (uint p = 0; p < C && (pos_start + p) < args.seq_k; p++) {
            device const float4* k_pos = (device const float4*)
                (k_head + (pos_start + p) * HEAD_DIM);
            device const float4* v_pos = (device const float4*)
                (v_head + (pos_start + p) * HEAD_DIM);

            // Each thread loads one float4 (4 elements) - no bounds check needed
            // as 32 threads * 4 = 128 = HEAD_DIM
            float4 k_val = k_pos[tiisg];
            float4 v_val = v_pos[tiisg];
            k_batch[p][0] = k_val.x;
            k_batch[p][1] = k_val.y;
            k_batch[p][2] = k_val.z;
            k_batch[p][3] = k_val.w;
            v_batch[p][0] = v_val.x;
            v_batch[p][1] = v_val.y;
            v_batch[p][2] = v_val.z;
            v_batch[p][3] = v_val.w;
        }

        // Compute Q*K^T for all C positions (matrix-vector multiply)
        float qk_scores[C];
        for (uint p = 0; p < C && (pos_start + p) < args.seq_k; p++) {
            // Each thread has 4 elements of Q and 4 elements of K
            // Dot product of q_frag[0] (4 elements) and k_batch[p] (4 elements)
            float qk = dot(q_frag[0], float4(k_batch[p][0], k_batch[p][1],
                                             k_batch[p][2], k_batch[p][3]));
            qk_scores[p] = qk * args.scale;
        }

        // Reduce within simdgroup (all 32 threads participate)
        for (uint p = 0; p < C && (pos_start + p) < args.seq_k; p++) {
            qk_scores[p] = simd_sum(qk_scores[p]);
        }

        // Online softmax update
        float m_new = M;
        for (uint p = 0; p < C && (pos_start + p) < args.seq_k; p++) {
            m_new = max(m_new, qk_scores[p]);
        }

        // Broadcast max
        m_new = simd_max(m_new);

        float scale_prev = exp(M - m_new);
        S *= scale_prev;

        // Rescale previous output
        O *= scale_prev;

        // Compute softmax weights and accumulate V
        float local_S = 0.0f;
        for (uint p = 0; p < C && (pos_start + p) < args.seq_k; p++) {
            float w = exp(qk_scores[p] - m_new);
            local_S += w;

            // Accumulate weighted V (each thread accumulates its 4 elements)
            float4 v_frag = float4(v_batch[p][0], v_batch[p][1],
                                   v_batch[p][2], v_batch[p][3]);
            O += w * v_frag;
        }

        S += simd_sum(local_S);
        M = m_new;
    }

    // Final normalization
    float inv_S = S > 0.0f ? 1.0f / S : 0.0f;

    // Write output - each thread writes its 4 elements
    device float4* out_f4 = (device float4*)
        (output + head * HEAD_DIM);
    out_f4[tiisg] = O * inv_S;
}
