// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode Kernel with NWG parallelization
// Multiple workgroups process different chunks of KV positions in parallel
// Final reduction combines partial results using online softmax formula

#include <metal_stdlib>
using namespace metal;

// First pass: each workgroup processes a chunk of positions
kernel void flash_decode_nwg_pass1_f16(
    constant uint& seq_k [[buffer(0)]],
    constant uint& kv_stride [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const half* key_cache [[buffer(4)]],
    device const half* value_cache [[buffer(5)]],
    device float* partial_O [[buffer(6)]],  // [n_wg, n_heads, head_dim]
    device float* partial_S [[buffer(7)]],  // [n_wg, n_heads]
    device float* partial_M [[buffer(8)]],  // [n_wg, n_heads]
    constant uint& n_wg [[buffer(9)]],
    constant uint& n_heads [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int batch = gid.y;
    const int head = gid.x;
    const int wg = gid.z;  // workgroup index (0 to n_wg-1)

    constexpr short HEAD_DIM = 128;
    constexpr short C = 32;  // positions per iteration per workgroup

    // Calculate chunk range for this workgroup
    // Each workgroup processes: positions [wg*C, (wg+1)*C), then [wg*C + n_wg*C, wg*C + 2*n_wg*C), etc.
    uint pos_start = wg * C;
    uint pos_stride = n_wg * C;

    // Load Q into registers
    device const float4* q4 = (device const float4*)
        (query + batch * HEAD_DIM + head * HEAD_DIM);

    float4 q_frag[4];
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

    // Get K, V base pointers
    device const half4* k_base = (device const half4*)
        (key_cache + head * kv_stride * HEAD_DIM);
    device const half4* v_base = (device const half4*)
        (value_cache + head * kv_stride * HEAD_DIM);

    // Process assigned positions
    for (uint pos = pos_start; pos < seq_k; pos += pos_stride) {
        device const half4* k_h4 = k_base + pos * 32;
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

        float qk_total = simd_sum(qk);
        float score = qk_total * scale;

        // Online softmax update
        float m_prev = M;
        M = max(M, score);

        float exp_mdiff = exp(m_prev - M);
        float exp_sdiff = exp(score - M);

        S = S * exp_mdiff + exp_sdiff;

        for (short i = 0; i < 4; i++) {
            O[i] = O[i] * exp_mdiff + exp_sdiff * v_frag[i];
        }
    }

    // Write partial results
    // Layout: partial_O[wg, head, dim]
    device float4* out_f4 = (device float4*)
        (partial_O + (wg * n_heads + head) * HEAD_DIM);
    for (short i = 0; i < 4; i++) {
        out_f4[tiisg * 4 + i] = O[i];
    }

    // Write partial S and M
    if (tiisg == 0) {
        uint idx = wg * n_heads + head;
        partial_S[idx] = S;
        partial_M[idx] = M;
    }
}

// Second pass: reduction across workgroups
kernel void flash_decode_nwg_pass2_f16(
    device float* partial_O [[buffer(0)]],
    device float* partial_S [[buffer(1)]],
    device float* partial_M [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& n_wg [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int batch = gid.y;
    const int head = gid.x;

    constexpr short HEAD_DIM = 128;

    // Load partial results from all workgroups
    float global_M = -INFINITY;
    for (uint wg = 0; wg < n_wg; wg++) {
        uint idx = wg * n_heads + head;
        global_M = max(global_M, partial_M[idx]);
    }

    // Combine partial results using online softmax formula
    float global_S = 0.0f;
    float4 global_O[4];
    for (short i = 0; i < 4; i++) {
        global_O[i] = float4(0.0f);
    }

    for (uint wg = 0; wg < n_wg; wg++) {
        uint idx = wg * n_heads + head;
        float wg_M = partial_M[idx];
        float wg_S = partial_S[idx];

        float scale = exp(wg_M - global_M);
        global_S += wg_S * scale;

        device float4* wg_O = (device float4*)
            (partial_O + (wg * n_heads + head) * HEAD_DIM);
        for (short i = 0; i < 4; i++) {
            global_O[i] += scale * wg_O[tiisg * 4 + i];
        }
    }

    // Final normalization
    float inv_S = global_S > 0.0f ? 1.0f / global_S : 0.0f;

    // Write final output
    device float4* out_f4 = (device float4*)
        (output + batch * HEAD_DIM + head * HEAD_DIM);
    for (short i = 0; i < 4; i++) {
        out_f4[tiisg * 4 + i] = global_O[i] * inv_S;
    }
}
