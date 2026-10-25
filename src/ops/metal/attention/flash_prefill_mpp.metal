// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// =============================================================================
// STATUS: EXPERIMENTAL - Work in progress for server mode
// =============================================================================
// Uses Metal Performance Primitives for tensor operations. Requires MPP framework.
// Not currently used. For prefill, use flash_attn_quant.metal with full format support.
// =============================================================================

// High-performance Flash Prefill using MPP tensor operations
// Processes Q queries per threadgroup, C=256 cache items per iteration

#include <metal_stdlib>
using namespace metal;

#ifdef GGML_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct FlashPrefillArgs {
    uint32_t batch;
    uint32_t heads;
    uint32_t seq_q;
    uint32_t seq_k;
    uint32_t head_dim;
    float scale;
    uint32_t q_stride_b;
    uint32_t q_stride_h;
    uint32_t q_stride_s;
    uint32_t k_stride_b;  // In elements
    uint32_t k_stride_h;
    uint32_t k_stride_s;
    uint32_t v_stride_b;
    uint32_t v_stride_h;
    uint32_t v_stride_s;
    uint32_t o_stride_b;
    uint32_t o_stride_h;
    uint32_t o_stride_s;
};

// F16 prefill using MPP tiles
template<short DK, short DV, short Q = 32, short C = 256, short NSG = 4>
kernel void flash_prefill_mpp_f16(
    constant FlashPrefillArgs & args [[buffer(0)]],
    device const half * q [[buffer(1)]],
    device const half * k [[buffer(2)]],
    device const half * v [[buffer(3)]],
    device half * output [[buffer(4)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NW = 32;  // SIMD width
    constexpr short NT = NW * NSG;
    constexpr short NQ = Q / NSG;  // queries per simdgroup
    constexpr short NC = C / NW;   // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const int iq1 = tgpig.x * Q;

    const short tiitg = sgitg * NW + tiisg;

    // Shared memory layout
    threadgroup half  * sq = (threadgroup half  *) shmem;              // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q * DK);      // [Q, C]  scores
    threadgroup half  * sp = (threadgroup half  *) (ss + Q * C);       // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q * C);       // [Q]     scale factors

    // Load queries into shared memory with scale folded in
    device const half * q_base = q + b * args.q_stride_b + h * args.q_stride_h;

    for (int i = tiitg; i < Q * DK / 4; i += NT) {
        const int j = i / (DK / 4);

        half4 q4 = 0.0h;
        if (iq1 + j < args.seq_q) {
            device const half4 * q_ptr = (device const half4 *)(q_base + (iq1 + j) * args.q_stride_s);
            q4 = q_ptr[i % (DK / 4)];
            q4 *= (half)args.scale;
        }

        ((threadgroup half4 *) sq)[i] = q4;
    }

    // Initialize softmax state
    float M[NQ];
    float S[NQ];

    for (short jj = 0; jj < NQ; jj++) {
        M[jj] = -FLT_MAX / 2;
        S[jj] = 0.0f;
    }

    // Setup MPP tensors
    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C, Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C, Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    device const half * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const half * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    auto tv0 = tensor((device half *) v_base, dextents<int32_t, 2>(DV, args.seq_k), array<int, 2>({1, (int)args.v_stride_s}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    // Initialize output accumulator
    for (short i = 0; i < co.get_capacity(); i++) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process KV cache in chunks
    for (int ic0 = 0, ic = 0; ic < args.seq_k; ic0++, ic += C) {
        // Q*K^T
        {
            auto tk = tensor((device half *) (k_base + (uint64_t)ic * args.k_stride_s),
                           dextents<int32_t, 2>(DK, min((int)C, (int)(args.seq_k - ic))),
                           array<int, 2>({1, (int)args.k_stride_s}));
            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        for (short jj = 0; jj < NQ; jj++) {
            const short j = jj * NSG + sgitg;

            if (iq1 + j >= args.seq_q) continue;

            float s[NC];

            for (short ii = 0; ii < NC; ii++) {
                int idx = ic + ii * NW + tiisg;
                if (idx < args.seq_k) {
                    s[ii] = ss[j * C + ii * NW + tiisg];
                } else {
                    s[ii] = -FLT_MAX / 2;
                }
            }

            // Find max
            float m = M[jj];
            for (short ii = 0; ii < NC; ii++) {
                m = max(m, s[ii]);
            }
            m = simd_max(m);

            // Lazy rescaling
            float ms = 1.0f;
            if (m > M[jj] + 8.0f) {
                ms = exp(M[jj] - m);
                M[jj] = m;
            }

            // Compute probabilities
            float sum = 0.0f;
            for (short ii = 0; ii < NC; ii++) {
                int idx = ic + ii * NW + tiisg;
                if (idx < args.seq_k) {
                    half p = (half)exp(s[ii] - M[jj]);
                    sp[j * C + ii * NW + tiisg] = p;
                    sum += (float)p;
                } else {
                    sp[j * C + ii * NW + tiisg] = 0.0h;
                }
            }

            S[jj] = S[jj] * ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale O and accumulate P*V
        {
            // Apply scale to O
            for (short i = 0; i < co.get_capacity(); i++) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }

            auto tv = tensor((device half *) (v_base + (uint64_t)ic * args.v_stride_s),
                           dextents<int32_t, 2>(DV, min((int)C, (int)(args.seq_k - ic))),
                           array<int, 2>({1, (int)args.v_stride_s}));
            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    device half * o_base = output + b * args.o_stride_b + h * args.o_stride_h;

    for (short jj = 0; jj < NQ; jj++) {
        const short j = jj * NSG + sgitg;

        if (iq1 + j >= args.seq_q) continue;

        for (short d = tiisg; d < DV; d += NW) {
            float val = co[d + j * DV] / S[jj];
            o_base[(iq1 + j) * args.o_stride_s + d] = (half)val;
        }
    }
}

// Template instantiations for common head dimensions
template [[host_name("flash_prefill_mpp_f16_d128")]]
kernel void flash_prefill_mpp_f16<128, 128, 32, 256, 4>(
    constant FlashPrefillArgs &, device const half *, device const half *, device const half *, device half *,
    threadgroup char *, uint3, ushort, ushort);

#endif // GGML_METAL_HAS_TENSOR
