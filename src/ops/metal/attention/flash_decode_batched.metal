// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode with batched processing
// Processes multiple KV positions per iteration for better memory bandwidth
//
// Design:
// - Each simdgroup (32 threads) processes 1 head
// - Processes C=8 KV positions per iteration
// - Uses vectorized half4 loads for efficiency
// - Online softmax across batched positions
//
// Performance model:
// - Sequential processing: 1 position × seq_k iterations
// - Batched processing: 8 positions × (seq_k/8) iterations
// - 8x fewer loop iterations, better cache utilization

#include <metal_stdlib>
using namespace metal;

constexpr constant int HEAD_DIM = 128;
constexpr constant int BATCH_SIZE = 8;
constexpr constant int THREADS_PER_HEAD = 32;
constexpr constant int ELEMS_PER_THREAD = HEAD_DIM / THREADS_PER_HEAD;

struct FlashDecodeMMAArgs {
    uint seq_k;
    uint n_heads;
    uint n_kv_heads;
    uint kv_stride;
    float scale;
};

kernel void flash_decode_batched_f16(
    constant FlashDecodeMMAArgs& args [[buffer(0)]],
    device const half* query [[buffer(1)]],
    device const half* key_cache [[buffer(2)]],
    device const half* value_cache [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint seq_k = args.seq_k;
    const uint n_heads = args.n_heads;
    const uint n_kv_heads = args.n_kv_heads;
    const uint kv_stride = args.kv_stride;
    const float scale = args.scale;

    const uint head = gid.x;
    if (head >= n_heads) return;

    const uint gqa_ratio = n_heads / n_kv_heads;
    const uint kv_head = head / gqa_ratio;

    // Pointers
    device const half* q_ptr = query + head * HEAD_DIM;
    device const half* k_head_ptr = key_cache + kv_head * kv_stride * HEAD_DIM;
    device const half* v_head_ptr = value_cache + kv_head * kv_stride * HEAD_DIM;

    // Each thread loads 4 elements of Q (as 1 half4)
    const int elem_offset = tiisg * ELEMS_PER_THREAD;
    device const half4* q_h4 = (device const half4*)(q_ptr + elem_offset);
    const float4 q_frag = (float4)q_h4[0];

    // Online softmax state
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator
    float4 O = float4(0.0f);

    // Process KV positions in batches
    const uint n_batches = (seq_k + BATCH_SIZE - 1) / BATCH_SIZE;

    for (uint batch = 0; batch < n_batches; batch++) {
        const uint pos_base = batch * BATCH_SIZE;
        const uint batch_size = min((uint)BATCH_SIZE, seq_k - pos_base);

        // Load K and V for up to 8 positions
        float qk_scores[BATCH_SIZE];
        float4 v_batch[BATCH_SIZE];

        for (uint i = 0; i < batch_size; i++) {
            const uint pos = pos_base + i;

            // Load K at this position
            device const half* k_pos = k_head_ptr + pos * HEAD_DIM;
            device const half4* k_h4 = (device const half4*)(k_pos + elem_offset);
            const float4 k_frag = (float4)k_h4[0];

            // Compute Q*K for this thread's elements
            qk_scores[i] = dot(q_frag, k_frag);

            // Load V at this position
            device const half* v_pos = v_head_ptr + pos * HEAD_DIM;
            device const half4* v_h4 = (device const half4*)(v_pos + elem_offset);
            v_batch[i] = (float4)v_h4[0];
        }

        // Reduce Q*K scores across threads (per-position)
        // Each thread has partial dot product for its 4 elements
        // Sum across all 32 threads to get full Q*K for each position
        float qk_total[BATCH_SIZE];
        for (uint i = 0; i < batch_size; i++) {
            qk_total[i] = simd_sum(qk_scores[i]);
            qk_total[i] *= scale;
        }

        // Online softmax update for batch
        // Find max score in this batch
        float M_batch = M;
        for (uint i = 0; i < batch_size; i++) {
            M_batch = max(M_batch, qk_total[i]);
        }

        // Update M and S
        const float exp_M_diff = exp(M - M_batch);
        M = M_batch;

        float S_batch = S * exp_M_diff;
        for (uint i = 0; i < batch_size; i++) {
            S_batch += exp(qk_total[i] - M);
        }
        S = S_batch;

        // Rescale previous accumulator and add new weighted values
        O *= exp_M_diff;
        for (uint i = 0; i < batch_size; i++) {
            const float weight = exp(qk_total[i] - M);
            O += weight * v_batch[i];
        }
    }

    // Normalize
    const float norm = 1.0f / S;
    O *= norm;

    // Write output
    device float4* out_f4 = (device float4*)(output + head * HEAD_DIM + elem_offset);
    out_f4[0] = O;
}
