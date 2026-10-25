// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// =============================================================================
// STATUS: EXPERIMENTAL - Work in progress for server mode (vLLM-style batching)
// =============================================================================
// This kernel implements prefill Flash Attention with MMA operations.
// Kept for future server-mode implementation. For local inference prefill,
// use flash_attn_quant.metal which has full format support.
// =============================================================================

// Flash Decode with Matrix Multiply Accumulate (MMA) operations
// Based on llama.cpp's approach with simdgroup_float8x8

#include <metal_stdlib>
using namespace metal;

// Function constants for kernel specialization
constant bool FC_has_mask    [[function_constant(0)]];
constant bool FC_has_sinks   [[function_constant(1)]];
constant bool FC_has_bias    [[function_constant(2)]];
constant int  FC_ns10        [[function_constant(20)]]; // K stride
constant int  FC_ns20        [[function_constant(21)]]; // V stride
constant int  FC_nsg         [[function_constant(22)]]; // number of simdgroups

#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Arguments structure matching llama.cpp
struct FlashAttnExtArgs {
    uint32_t ne01;  // n_heads
    uint32_t ne02;  // n_head_groups (for GQA)
    uint32_t ne03;  // n_batch
    uint32_t ne11;  // seq_len (KV cache length)
    uint64_t nb01;  // Q head stride
    uint64_t nb02;  // Q group stride
    uint64_t nb03;  // Q batch stride
    uint64_t nb11;  // K head stride
    uint64_t nb12;  // K group stride
    uint64_t nb13;  // K batch stride
    uint64_t nb21;  // V head stride
    uint64_t nb22;  // V group stride
    uint64_t nb23;  // V batch stride
    uint64_t nb31;  // output head stride
    uint64_t nb32;  // output group stride
    uint64_t nb33;  // output batch stride
    int32_t  ne_12_2; // GQA ratio for dim 2
    int32_t  ne_12_3; // GQA ratio for dim 3
    float    scale;
};

// ============================================================================
// Flash Decode with MMA for F16 KV cache
// Processes multiple KV positions per iteration using matrix operations
// ============================================================================

template<
    typename q_t,     // query type in registers
    typename q4_t,    // query type for half4
    typename q8x8_t,  // query matrix type (simdgroup_float8x8)
    typename k_t,     // key type in registers
    typename k4x4_t,  // key type for half4x4
    typename k8x8_t,  // key matrix type
    typename v_t,     // value type in registers
    typename v4x4_t,  // value type for half4x4
    typename v8x8_t,  // value matrix type
    typename qk_t,    // Q*K type
    typename qk8x8_t, // Q*K matrix type
    typename s_t,     // softmax type
    typename s2_t,    // softmax type for half2
    typename s8x8_t,  // softmax matrix type
    typename o_t,     // output type
    typename o4_t,    // output type for half4
    typename o8x8_t,  // output matrix type
    short DK,         // K head dimension
    short DV,         // V head dimension
    short C,          // KV positions per iteration (8 or 16)
    short NSG>        // number of simdgroups (1)
kernel void kernel_flash_attn_ext_mma(
        constant FlashAttnExtArgs & args [[buffer(0)]],
        device const char * q_data [[buffer(1)]],
        device const char * k_data [[buffer(2)]],
        device const char * v_data [[buffer(3)]],
        device       char * dst    [[buffer(4)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    // Thread layout for decode (single query position)
    const short iq1 = tgpig.x; // head index
    const short iq2 = tgpig.y % args.ne02; // head group index
    const short iq3 = tgpig.z; // batch index

    // GQA: map Q head to KV head
    const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    // Get Q pointer
    device const char * q = q_data + iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03;

    // Get K, V pointers
    device const char * k = k_data + ikv2 * args.nb12 + ikv3 * args.nb13;
    device const char * v = v_data + ikv2 * args.nb22 + ikv3 * args.nb23;

    // Load Q into shared memory (once for all KV positions)
    constexpr short DK4  = DK / 4;
    constexpr short DK8  = DK / 8;
    constexpr short DK16 = DK / 16;
    constexpr short DV4  = DV / 4;
    constexpr short DV8  = DV / 8;
    constexpr short DV16 = DV / 16;
    constexpr short PV   = DV;  // Padded V dimension
    constexpr short PV4  = PV / 4;
    constexpr short PV8  = PV / 8;
    constexpr short NW   = N_SIMDWIDTH;

    // Each thread loads part of Q
    threadgroup q_t sq[DK];
    device const q4_t * q4 = (device const q4_t *) q;

    FOR_UNROLL(short i = 0; i < DK4; i += NW) {
        short idx = i + tiisg;
        if (idx < DK4) {
            ((threadgroup q4_t *)sq)[idx] = q4[idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float M = -FLT_MAX / 2;
    float S = 0.0f;

    // Output accumulator
    threadgroup o_t so[PV];
    FOR_UNROLL(short i = 0; i < PV4; i += NW) {
        short idx = i + tiisg;
        if (idx < PV4) {
            ((threadgroup o4_t *)so)[idx] = o4_t(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process KV cache in blocks of C positions
    constexpr short NC = C / 8;  // Number of 8x8 matrices

    for (uint ic = 0; ic < args.ne11; ic += C) {
        // Q*K^T using MMA
        // Load C positions of K
        // Each 8x8 matrix represents 8 KV positions x 8 elements of head_dim

        threadgroup s_t ss[C];  // Attention scores

        qk8x8_t mqk[NC];
        FOR_UNROLL(short nc = 0; nc < NC; ++nc) {
            mqk[nc] = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);
        }

        // Matrix multiply: Q (1xDK) * K^T (DKxC) -> (1xC)
        // We load K in 8x8 blocks and multiply with corresponding Q blocks
        device const k_t * pk = (device const k_t *) (k + ic * args.nb11);
        threadgroup const q_t * pq = sq;

        FOR_UNROLL(short i = 0; i < DK8; i += 2) {
            simdgroup_barrier(mem_flags::mem_none);

            k8x8_t mk[2];
            q8x8_t mq[2];

            // Load Q (1x8 elements per matrix)
            simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
            simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);

            // Load K^T (8 positions x 8 elements per matrix)
            // Note: K is stored as [seq, head_dim], we want [head_dim, seq]
            simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, FC_ns10, 0, true);
            simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, FC_ns10, 0, true);

            simdgroup_barrier(mem_flags::mem_none);

            // MMA: mqk += mq * mk^T
            FOR_UNROLL(short nc = 0; nc < NC; ++nc) {
                simdgroup_multiply_accumulate(mqk[nc], mq[0], mk[nc], mqk[nc]);
                simdgroup_multiply_accumulate(mqk[nc], mq[1], mk[nc], mqk[nc]);
            }

            pk += 8 * FC_ns10;
        }

        // Store Q*K scores
        FOR_UNROLL(short nc = 0; nc < NC; ++nc) {
            simdgroup_store(mqk[nc], ss + 8 * nc, C, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax across C positions
        float m_new = M;

        // Find max in this block
        FOR_UNROLL(short i = 0; i < C && (ic + i) < args.ne11; ++i) {
            float s = ((threadgroup s_t *)ss)[i] * args.scale;
            m_new = max(m_new, s);
        }

        // Broadcast max across simdgroup
        m_new = simd_max(m_new);

        // Rescale previous accumulation
        float scale_prev = exp(M - m_new);
        S *= scale_prev;

        FOR_UNROLL(short i = 0; i < PV4; i += NW) {
            short idx = i + tiisg;
            if (idx < PV4) {
                ((threadgroup o4_t *)so)[idx] *= scale_prev;
            }
        }

        // Compute softmax weights and accumulate V
        float local_S = 0.0f;

        FOR_UNROLL(short i = 0; i < C && (ic + i) < args.ne11; ++i) {
            float s = ((threadgroup s_t *)ss)[i] * args.scale;
            float w = exp(s - m_new);
            local_S += w;
            ((threadgroup s_t *)ss)[i] = w;
        }

        S += simd_sum(local_S);

        // O = O + weights * V
        // This is another MMA: O (1xDV) += ss (CxC) * V (CxDV)
        // We can process this as multiple 8x8 matrix operations

        device const v_t * pv = (device const v_t *) (v + ic * args.nb21);

        o8x8_t mo[PV8 / NSG];
        {
            auto sot = (threadgroup o8x8_t *) so + sgitg * (PV8 / NSG);
            FOR_UNROLL(short ii = 0; ii < PV8 / NSG; ++ii) {
                simdgroup_load(mo[ii], (threadgroup o_t *) sot, PV, 0, false);
                sot += NSG;
            }
        }

        FOR_UNROLL(short cc = 0; cc < NC; ++cc) {
            s8x8_t ms;
            simdgroup_load(ms, ss + 8 * cc, C, 0, false);

            FOR_UNROLL(short ii = 0; ii < PV8 / NSG / 2; ++ii) {
                v8x8_t mv[2];
                simdgroup_load(mv[0], pv + 0 * NSG + 16 * ii * NSG, FC_ns20, 0, false);
                simdgroup_load(mv[1], pv + 8 * NSG + 16 * ii * NSG, FC_ns20, 0, false);

                simdgroup_multiply_accumulate(mo[2 * ii + 0], ms, mv[0], mo[2 * ii + 0]);
                simdgroup_multiply_accumulate(mo[2 * ii + 1], ms, mv[1], mo[2 * ii + 1]);
            }

            pv += 8 * FC_ns20;
        }

        // Store accumulated O
        {
            auto sot = (threadgroup o8x8_t *) so + sgitg * (PV8 / NSG);
            FOR_UNROLL(short ii = 0; ii < PV8 / NSG; ++ii) {
                simdgroup_store(mo[ii], (threadgroup o_t *) sot, PV, 0, false);
                sot += NSG;
            }
        }

        M = m_new;
    }

    // Final normalization and write output
    float inv_S = S > 0.0f ? 1.0f / S : 0.0f;

    device char * dst_ptr = dst + iq1 * args.nb31 + iq2 * args.nb32 + iq3 * args.nb33;
    device float * out_f32 = (device float *) dst_ptr;

    FOR_UNROLL(short i = 0; i < DV4; i += NW) {
        short idx = i + tiisg;
        if (idx < DV4) {
            threadgroup o4_t val = ((threadgroup o4_t *)so)[idx];
            device float4 * out_f4 = (device float4 *) out_f32;
            out_f4[idx] = (float4) val * inv_S;
        }
    }
}

// Instantiations for common dimensions
template [[host_name("kernel_flash_attn_ext_mma_f16_dk128_dv128_c8")]]
kernel void kernel_flash_attn_ext_mma<
    float, float4, simdgroup_float8x8,    // Q types
    float, float4x4, simdgroup_float8x8,  // K types
    float, float4x4, simdgroup_float8x8,  // V types
    float, simdgroup_float8x8,            // QK types
    float, float2, simdgroup_float8x8,    // Softmax types
    float, float4, simdgroup_float8x8,    // Output types
    128, 128, 8, 1                         // Dimensions
>(
    constant FlashAttnExtArgs & args,
    device const char * q_data,
    device const char * k_data,
    device const char * v_data,
    device       char * dst,
    uint3   tgpig,
    ushort  tiisg,
    ushort  sgitg);

template [[host_name("kernel_flash_attn_ext_mma_f16_dk128_dv128_c16")]]
kernel void kernel_flash_attn_ext_mma<
    float, float4, simdgroup_float8x8,
    float, float4x4, simdgroup_float8x8,
    float, float4x4, simdgroup_float8x8,
    float, simdgroup_float8x8,
    float, float2, simdgroup_float8x8,
    float, float4, simdgroup_float8x8,
    128, 128, 16, 1
>(
    constant FlashAttnExtArgs & args,
    device const char * q_data,
    device const char * k_data,
    device const char * v_data,
    device       char * dst,
    uint3   tgpig,
    ushort  tiisg,
    ushort  sgitg);
