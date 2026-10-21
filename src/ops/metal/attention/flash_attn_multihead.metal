// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Multi-Head Flash Attention Decode Kernel
// Based on llama.cpp's tensor-based approach (fa_tensor.metal)
//
// Key parameters (matching llama.cpp):
// - Q = 32 (queries per threadgroup, one query per head for decode)
// - C = 64 (cache items per threadgroup)
// - NSG = 8 (simdgroups per threadgroup = 256 threads)
//
// Design:
// - Each threadgroup processes 32 heads
// - Each simdgroup (32 threads) handles 4 heads
// - Reduces dispatch overhead by 32x vs single-head approach
//
// Reference: llama.cpp-0.7.0/ggml/src/ggml-metal/kernels/fa_tensor.metal

#include <metal_stdlib>
using namespace metal;

// Constants matching llama.cpp's OP_FLASH_ATTN_EXT_TENSOR_*
constant short Q_PER_TG = 32; // queries per threadgroup
constant short C_PER_TG = 64; // cache items per threadgroup
constant short NSG = 8; // simdgroups per threadgroup
constant short NW = 32; // threads per simdgroup
constant short NT = NSG * NW; // total threads per threadgroup = 256

// For decode: 1 query per head, so Q = n_heads processed together
// Each simdgroup handles Q/NSG = 4 heads
// NQ = 4 heads per simdgroup

template <short DK, short DV>
kernel void kernel_flash_attn_multihead(
    constant uint &seq_k [[buffer(0)]], // KV cache length
    constant uint &kv_stride [[buffer(1)]], // KV cache stride (max_seq)
    constant float &scale [[buffer(2)]], // Attention scale (1/sqrt(DK))
    device const float *query [[buffer(3)]], // Query [n_heads, DK]
    device const half *key [[buffer(4)]], // Key cache [n_heads, max_seq, DK]
    device const half *value [[buffer(5)]], // Value cache [n_heads, max_seq, DV]
    device float *output [[buffer(6)]], // Output [n_heads, DV]
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Grid layout: [n_heads/Q, n_seqs, 1]
    // For decode with n_heads=32, n_seqs=1: grid = [1, 1, 1]
    // tgpig.x = head_group (0 for first 32 heads)
    // tgpig.y = batch index

    const short batch = tgpig.y;
    const short head_group = tgpig.x; // Which group of Q heads

    // Each simdgroup handles Q/NSG = 4 heads
    constexpr short NQ = Q_PER_TG / NSG; // 4 heads per simdgroup
    const short local_head_start = sgitg * NQ; // 0, 4, 8, 12, ...

    // Load queries for this simdgroup's heads
    // Query layout: [n_heads, DK]
    // Each thread loads part of each query

    // For DK=128, each thread loads DK/4/32 = 1 float4 per query
    // Total: 4 queries * 1 float4 = 4 float4s per thread
    float4 q_local[NQ];

    for (short q = 0; q < NQ; q++)
    {
        const short head = head_group * Q_PER_TG + local_head_start + q;
        device const float4 *q_ptr = (device const float4 *)(query + head * DK);

        // Each thread loads one float4 from each query
        // For DK=128: 128/4 = 32 float4s total, 32 threads load 1 each
        if (tiisg < DK / 4)
        {
            q_local[q] = q_ptr[tiisg];
        }
    }

    // Online softmax state for each head this simdgroup handles
    float M[NQ];
    float S[NQ];

    // Output accumulator for each head (DV/4 float4s per head)
    float4 O[NQ * DV / 4];

    // Initialize
    for (short q = 0; q < NQ; q++)
    {
        M[q] = -INFINITY;
        S[q] = 0.0f;
    }

    for (short i = 0; i < NQ * DV / 4; i++)
    {
        O[i] = float4(0.0f);
    }

    // Process KV cache in chunks of C items
    const uint n_iterations = (seq_k + C_PER_TG - 1) / C_PER_TG;

    // K/V cache layout: [n_heads, max_seq, DK/DV]
    // For each head: K at key + head * kv_stride * DK

    for (uint iter = 0; iter < n_iterations; iter++)
    {
        const uint kv_start = iter * C_PER_TG;
        const uint kv_end = min(kv_start + C_PER_TG, seq_k);

        // Process C cache items
        // Each thread handles C/NW items
        // For C=64, NW=32: 2 items per thread

        for (short q = 0; q < NQ; q++)
        {
            const short head = head_group * Q_PER_TG + local_head_start + q;

            // Compute Q*K^T for this head
            // Process C/NW positions per thread
            for (uint c_offset = 0; c_offset < C_PER_TG / NW; c_offset++)
            {
                const uint c = tiisg + c_offset * NW;
                const uint pos = kv_start + c;

                if (pos >= kv_end)
                {
                    continue;
                }

                // Load K for this position
                device const half *k_ptr = key + head * kv_stride * DK + pos * DK;

                // Compute dot product: Q · K
                float qk = 0.0f;
                for (short i = 0; i < DK / 4; i++)
                {
                    // Load K (half4) and convert to float4
                    half4 k_h4 = *((device const half4 *)(k_ptr + i * 4));
                    float4 k_f4 = float4(k_h4);

                    // Dot with Q (only this thread's portion)
                    qk += dot(q_local[q], k_f4);
                }

                // Reduce within simdgroup
                // All threads contribute to the dot product
                float qk_total = simd_sum(qk);
                float score = qk_total * scale;

                // Online softmax update
                float m_prev = M[q];
                M[q] = max(M[q], score);

                float exp_mdiff = exp(m_prev - M[q]);
                float exp_sdiff = exp(score - M[q]);

                S[q] = S[q] * exp_mdiff + exp_sdiff;

                // Load V and update output
                device const half *v_ptr = value + head * kv_stride * DV + pos * DV;

                // Load V (half4) and update O
                for (short i = 0; i < DV / 4; i++)
                {
                    half4 v_h4 = *((device const half4 *)(v_ptr + i * 4));
                    float4 v_f4 = float4(v_h4);

                    // This thread updates its portion of the output
                    O[q * DV / 4 + i] = O[q * DV / 4 + i] * exp_mdiff + exp_sdiff * v_f4;
                }
            }
        }

        // Synchronize after each iteration
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (short q = 0; q < NQ; q++)
    {
        const short head = head_group * Q_PER_TG + local_head_start + q;
        const float inv_S = S[q] > 0.0f ? 1.0f / S[q] : 0.0f;

        device float4 *out_ptr = (device float4 *)(output + head * DV);
        for (short i = 0; i < DV / 4; i++)
        {
            out_ptr[i] = O[q * DV / 4 + i] * inv_S;
        }
    }
}

// Template instantiations for common head dimensions
template [[host_name("kernel_flash_attn_multihead_d64")]] kernel void kernel_flash_attn_multihead<64, 64>(
    constant uint &,
    constant uint &,
    constant float &,
    device const float *,
    device const half *,
    device const half *,
    device float *,
    uint3,
    ushort,
    ushort);

template [[host_name("kernel_flash_attn_multihead_d128")]] kernel void kernel_flash_attn_multihead<128, 128>(
    constant uint &,
    constant uint &,
    constant float &,
    device const float *,
    device const half *,
    device const half *,
    device float *,
    uint3,
    ushort,
    ushort);
