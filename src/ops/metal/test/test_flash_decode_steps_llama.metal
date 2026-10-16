// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// ============================================================================

kernel void test_query_load_llama_style(
    device const float* query [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int NW = 32;  // N_SIMDWIDTH
    constexpr int PK4 = HEAD_DIM / 4;  // Padded head_dim in float4s
    constexpr int DK4 = HEAD_DIM / 4;

    // Shared memory for query
    threadgroup float4 sq4[32];

    // Each thread loads elements at stride NW
    device const float4* q4 = (device const float4*)query;

    for (short i = tiisg; i < PK4; i += NW) {
        if (i < DK4) {
            sq4[i] = q4[i];
        } else {
            sq4[i] = float4(0.0f);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write back to verify
    if (tiisg < HEAD_DIM / 4) {
        device float4* out_ptr = (device float4*)output;
        out_ptr[tiisg] = sq4[tiisg];
    }
}

// ============================================================================
// ============================================================================

kernel void test_qk_compute_llama_style(
    device const float* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& seq_k [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int HEAD_DIM = 128;
    constexpr int C = 32;   // Cache items per iteration
    constexpr int NL = 32 / NE;  // 8
    constexpr int NW = 32;
    constexpr int DK4 = HEAD_DIM / 4;

    const short tx = tiisg % NL;  // 0-7
    const short ty = tiisg / NL;  // 0-3

    const int head = gid.y;
    const int iwg = gid.x;  // Workgroup index

    threadgroup float4 sq4[32];

    device const float4* q4 = (device const float4*)(query + head * HEAD_DIM);
    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = q4[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ic = iwg * C;  // Start position for this workgroup

    device const half4* pk4 = (device const half4*)(key_cache + head * HEAD_DIM * seq_k + ic * HEAD_DIM);

    float mqk = 0.0f;  // For Q=1, just single value

    if (ic + ty < seq_k) {
        for (short ii = 0; ii < DK4/NL; ++ii) {
            // For cc=0 (first iteration):
            const half4 k_elem = pk4[ii*NL];
            mqk += dot((float4)k_elem, sq4[ii*NL + tx]);
        }
    }

    // Lines 354-371: conditional compilation
    if (NE <= 1) {
        mqk += simd_shuffle_down(mqk, 16);
    }
    if (NE <= 2) {
        mqk += simd_shuffle_down(mqk,  8);
    }
    if (NE <= 4) {
        mqk += simd_shuffle_down(mqk,  4);
    }
    if (NE <= 8) {
        mqk += simd_shuffle_down(mqk,  2);
    }
    if (NE <= 16) {
        mqk += simd_shuffle_down(mqk,  1);
    }

    mqk = simd_shuffle(mqk, NL*ty);

    // Write result
    if (tiisg == 0) {
        output[head * 512 + iwg] = mqk * scale;
    }
}

// ============================================================================
// ============================================================================

kernel void test_v_accum_llama_style(
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
    constexpr int NW = 32;
    constexpr int DV4 = HEAD_DIM / 4;

    const short tx = tiisg % NL;
    const short ty = tiisg / NL;

    const int head = gid.y;
    const int iwg = gid.x;

    for (short ii = 0; ii < DV4/NL; ++ii) {
        lo[ii] = 0.0f;
    }

    // Shared memory for weights (simulating softmax output)
    threadgroup float ss[C];
    if (tiisg < C) {
        ss[tiisg] = 1.0f / C;  // Equal weights for test
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ic = iwg * C;

    device const half4* pv4 = (device const half4*)(value_cache + head * HEAD_DIM * seq_k + ic * HEAD_DIM);

    if (ic + ty < seq_k) {
        for (short cc = 0; cc < C/NE; ++cc) {
            for (short ii = 0; ii < DV4/NL; ++ii) {
                const half4 v_elem = pv4[cc*NE*NS20/4 + ii*NL];
                lo[ii] += v_elem[0] * ss[cc*NE + ty];  // Simplified for Q=1
                lo[ii] += v_elem[1] * ss[cc*NE + ty];
                lo[ii] += v_elem[2] * ss[cc*NE + ty];
                lo[ii] += v_elem[3] * ss[cc*NE + ty];
            }
        }
    }

    if (NE > 1) {
        for (short ii = 0; ii < DV4/NL; ++ii) {
            lo[ii] += simd_shuffle_down(lo[ii], 16);
            lo[ii] += simd_shuffle_down(lo[ii],  8);
            lo[ii] += simd_shuffle_down(lo[ii],  4);
            lo[ii] += simd_shuffle_down(lo[ii],  2);
            lo[ii] += simd_shuffle_down(lo[ii],  1);
        }
    }

    // Write result
    if (tiisg < DV4) {
        output[head * HEAD_DIM + tiisg * 4] = lo[tiisg / 8];
    }
}

// ============================================================================
// ============================================================================

kernel void test_complete_llama_style(
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
    constexpr int NW = 32;
    constexpr int DK4 = HEAD_DIM / 4;
    constexpr int DV4 = HEAD_DIM / 4;
    constexpr int NS10 = HEAD_DIM;
    constexpr int NS20 = HEAD_DIM;
    constexpr int NWG = 16;

    const short tx = tiisg % NL;
    const short ty = tiisg / NL;

    const int head = gid.y;
    const int iwg = gid.x;

    threadgroup float4 sq4[32];
    device const float4* q4 = (device const float4*)(query + head * HEAD_DIM);
    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = q4[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float S = 0.0f;
    float M = -FLT_MAX/2;

    // Output accumulator
    float lo[DV4/NL];
    for (short ii = 0; ii < DV4/NL; ++ii) {
        lo[ii] = 0.0f;
    }

    // Shared memory for scores
    threadgroup float ss[C];

    for (int ic0 = iwg; ic0 * C < seq_k; ic0 += NWG) {
        const int ic = ic0 * C;

        // Q*K^T
        device const half4* pk4 = (device const half4*)(key_cache + head * HEAD_DIM * seq_k + ic * HEAD_DIM);
        pk4 += ty * NS10/4 + tx;

        float mqk = 0.0f;

        if (ic + ty < seq_k) {
            for (short ii = 0; ii < DK4/NL; ++ii) {
                const half4 k_elem = pk4[ii*NL];
                mqk += dot((float4)k_elem, sq4[ii*NL + tx]);
            }
        }

        // Reduction
        mqk += simd_shuffle_down(mqk, 4);
        mqk += simd_shuffle_down(mqk, 2);
        mqk += simd_shuffle_down(mqk, 1);
        mqk = simd_shuffle(mqk, NL*ty);

        // Store score
        const int score_idx = NE * tx + ty;
        if (ic + score_idx < seq_k) {
            ss[score_idx] = mqk * scale;
        } else {
            ss[score_idx] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float s = ss[tiisg];

        const float m = M;
        M = simd_max(max(M, s));

        const float ms = exp(m - M);
        const float vs = exp(s - M);

        S = S * ms + simd_sum(vs);

        ss[tiisg] = vs;

        // Scale previous output
        for (short ii = 0; ii < DV4/NL; ++ii) {
            lo[ii] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float weight = ss[tiisg];

        device const half4* pv4 = (device const half4*)(value_cache + head * HEAD_DIM * seq_k + ic * HEAD_DIM);
        pv4 += ty * NS20/4 + tx;

        if (ic + tiisg < seq_k) {
            for (short ii = 0; ii < DV4/NL; ++ii) {
                const half4 v_elem = pv4[ii*NL];
                lo[ii] += weight * (float)v_elem[0];
                lo[ii] += weight * (float)v_elem[1];
                lo[ii] += weight * (float)v_elem[2];
                lo[ii] += weight * (float)v_elem[3];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output (normalized)
    if (tiisg < DV4) {
        const float norm = 1.0f / S;
        device float4* out_ptr = (device float4*)(output + head * HEAD_DIM);
        // Simplified output - just broadcast
        out_ptr[tiisg] = float4(lo[tiisg/8] * norm);
    }
}
