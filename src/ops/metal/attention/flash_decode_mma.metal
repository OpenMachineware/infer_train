// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode using simdgroup_matrix (MMA) operations
// Optimized for decode (single query per head)
//
// Design:
// - Threadgroup processes 4 heads using 1 simdgroup (32 threads)
// - Each thread handles 4 elements of head_dim (128/32=4)
// - Q*K^T: parallel reduction using simd_sum
// - V accumulation: each thread accumulates its portion
//
// Memory layout:
// - Q: [n_heads, head_dim] - loaded once
// - K: [seq_len, head_dim] - loaded per position
// - V: [seq_len, head_dim] - loaded per position

#include <metal_stdlib>
using namespace metal;

constexpr constant int HEAD_DIM = 128;
constexpr constant int THREADS_PER_HEAD = 8;  // Each head uses 8 threads
constexpr constant int HEADS_PER_SG = 4;      // 4 heads per simdgroup
constexpr constant int ELEMS_PER_THREAD = HEAD_DIM / THREADS_PER_HEAD;  // 16 elements per thread

kernel void flash_decode_mma_f16(
    constant uint& seq_k [[buffer(0)]],
    constant uint& kv_stride [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    device const half* query [[buffer(3)],       // [n_heads, head_dim]
    device const half* key_cache [[buffer(4)],   // [n_kv_heads, kv_stride, head_dim]
    device const half* value_cache [[buffer(5)], // [n_kv_heads, kv_stride, head_dim]
    device float* output [[buffer(6)],           // [n_heads, head_dim]
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Layout: each simdgroup (32 threads) handles 4 heads
    // - Thread groups 0-7: head 0
    // - Thread groups 8-15: head 1
    // - Thread groups 16-23: head 2
    // - Thread groups 24-31: head 3

    const int head_group = gid.x;  // 0 to (n_heads/4 - 1)
    const int batch = gid.y;
    const int head_base = head_group * HEADS_PER_SG;

    const int head_in_sg = tiisg / THREADS_PER_HEAD;  // 0-3: which head
    const int thread_in_head = tiisg % THREADS_PER_HEAD;  // 0-7: which thread in this head
    const int head = head_base + head_in_sg;

    // Load Q for all 4 heads (each thread loads 16 elements)
    // Q[head, elem] where elem = thread_in_head * 16 + i
    device const half* q_ptr = query + head * HEAD_DIM;
    device const half4* q_h4 = (device const half4*)(q_ptr + thread_in_head * ELEMS_PER_THREAD);

    float4 q_frag;  // 16 elements as 4 float4
    for (int i = 0; i < 4; i++) {
        q_frag[i] = (float4)q_h4[i];
    }

    // Online softmax state per head (each thread holds for its head)
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator
    float4 O[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

    // Get K/V pointers for this head's KV head (handle GQA)
    const int n_kv_heads = 4;  // Assuming 4 KV heads for 16 Q heads (GQA)
    const int kv_head = head / 4;  // Map Q head to KV head

    device const half* k_head_ptr = key_cache + kv_head * kv_stride * HEAD_DIM;
    device const half* v_head_ptr = value_cache + kv_head * kv_stride * HEAD_DIM;

    // Process all KV positions
    for (uint pos = 0; pos < seq_k; pos++) {
        // Load K and V for this position
        device const half* k_pos = k_head_ptr + pos * HEAD_DIM;
        device const half* v_pos = v_head_ptr + pos * HEAD_DIM;
        device const half4* k_h4 = (device const half4*)(k_pos + thread_in_head * ELEMS_PER_THREAD);
        device const half4* v_h4 = (device const half4*)(v_pos + thread_in_head * ELEMS_PER_THREAD);

        // Load K and V fragments
        float4 k_frag;
        float4 v_frag;
        for (int i = 0; i < 4; i++) {
            k_frag[i] = (float4)k_h4[i];
            v_frag[i] = (float4)v_h4[i];
        }

        // Compute partial Q*K^T (dot product portion)
        float qk_partial = 0.0f;
        for (int i = 0; i < 4; i++) {
            qk_partial += dot(q_frag[i], k_frag[i]);
        }

        // Reduce within each head (8 threads per head)
        // Thread layout for head_in_sg=0 (threads 0-7):
        //   thread_in_head: 0,1,2,3,4,5,6,7
        // For head_in_sg=1 (threads 8-15):
        //   thread_in_head: 0,1,2,3,4,5,6,7 (but tiisg: 8-15)
        //
        // Use tree reduction: each step halves the number of active threads
        // Step 1: threads 1,3,5,7 send to threads 0,2,4,6
        // Step 2: threads 2,6 send to threads 0,4
        // Step 3: thread 4 sends to thread 0

        // For each head, thread_in_head 0 is the leader
        // Other threads send their values to the leader

        // We need per-head reduction, not full simdgroup reduction
        // Use simd_shuffle_down with the leader as destination

        // Actually, the cleanest approach: use the broadcast-based pattern
        // Each step, half the threads send to the other half

        // Step 1: distance 4 - threads 4-7 send to threads 0-3 within each head
        // For head 0: thread 4->0, 5->1, 6->2, 7->3
        // For head 1: thread 12->8, 13->9, 14->10, 15->11
        // simd_shuffle_down by 4 works: thread 0 gets value from thread 4
        // But this doesn't match what we want (we want thread 4 to send to thread 0)

        // Correct approach: use simd_shuffle_up
        // Thread 4 sends its value "up" to thread 0
        float val_from_up = simd_shuffle_up(qk_partial, 4);
        // Now thread 0 gets value from thread 4 (within same head group)
        // But all threads get values from 4 threads above
        // For thread 0: gets from thread -4 (invalid)
        // For thread 4: gets from thread 0
        // This is backwards!

        // Let's use a different approach: gather at thread 0 of each head
        // Thread 0 of each head (tiisg = 0, 8, 16, 24) collects values from others

        // Simpler: each thread sends its value to the leader, leader sums
        if (thread_in_head != 0) {
            // Non-leader threads: send value to leader
            // Leader is at tiisg = head_in_sg * 8
            // Use simd_shuffle to send (no-op - just let leader read our value)
        }

        // Leader reads and sums
        float qk_sum = qk_partial;
        if (thread_in_head == 0) {
            // Sum values from threads 1-7 of this head
            for (int t = 1; t < 8; t++) {
                qk_sum += simd_shuffle(qk_partial, head_in_sg * 8 + t);
            }
        }

        float qk_score = qk_sum * scale;
        // Broadcast to all threads in this head
        qk_score = simd_shuffle(qk_score, head_in_sg * 8);

        // Online softmax update (all threads do this for their head)
        float m_new = max(M, qk_score);
        float exp_mdiff = exp(M - m_new);  // Scale factor for previous sum
        float exp_sdiff = exp(qk_score - m_new);  // New contribution

        S = S * exp_mdiff + exp_sdiff;
        M = m_new;

        // Compute attention weight for this position
        float weight = exp_sdiff / S;

        // Scale previous output accumulation
        for (int i = 0; i < 4; i++) {
            O[i] *= exp_mdiff;
        }

        // Accumulate weighted V
        for (int i = 0; i < 4; i++) {
            O[i] += weight * v_frag[i];
        }
    }

    // Normalize by sum
    // Actually, we already normalized during accumulation, so O is already normalized

    // Write output
    device float* out_ptr = output + head * HEAD_DIM;
    device float4* out_f4 = (device float4*)(out_ptr + thread_in_head * ELEMS_PER_THREAD);
    for (int i = 0; i < 4; i++) {
        out_f4[i] = O[i];
    }
}
