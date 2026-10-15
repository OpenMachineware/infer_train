// Optimized SSM Scan kernel for Mamba architecture
// Uses shared memory batching like llama.cpp for better performance
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <metal_stdlib>
using namespace metal;

constant constexpr short N_SIMDWIDTH = 32;

// SSM Scan with shared memory batching
// One threadgroup per (d_inner, head, seq)
// Each thread handles one state element
// Process tokens in batches using shared memory
template <ushort d_state>
kernel void kernel_ssm_scan_batched(
    device const float* src0 [[buffer(0)]],  // s0 (initial state)
    device const float* src1 [[buffer(1)]],  // x (input)
    device const float* src2 [[buffer(2)]],  // dt (delta time)
    device const float* src3 [[buffer(3)]],  // A (decay parameter)
    device const float* src4 [[buffer(4)]],  // B (input projection)
    device const float* src5 [[buffer(5)]],  // C (output projection)
    device float* dst [[buffer(6)]],         // y (output) + s_out
    constant uint& d_inner [[buffer(7)]],
    constant uint& n_tok [[buffer(8)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgptg [[simdgroups_per_threadgroup]])
{
    // Shared memory layout:
    // [0..sgptg*N_SIMDWIDTH-1]: partial sums for reduction
    // [sgptg*N_SIMDWIDTH..sgptg*N_SIMDWIDTH+sgptg-1]: pre-computed x_dt values
    // [sgptg*N_SIMDWIDTH+sgptg..sgptg*N_SIMDWIDTH+2*sgptg-1]: pre-computed dA values
    threadgroup float* shared_sums = shared;
    threadgroup float* shared_x_dt = shared + sgptg * N_SIMDWIDTH;
    threadgroup float* shared_dA = shared + sgptg * N_SIMDWIDTH + sgptg;

    shared_sums[tpitg.x] = 0.0f;

    const int32_t i0 = tpitg.x;  // State element index
    const int32_t i1 = tgpig.x;  // d_inner index
    const int32_t ir = tgpig.y;  // Head index
    const int32_t i3 = tgpig.z;  // Sequence index

    const int32_t nc = d_state;
    const int32_t n_t = n_tok;

    // Data layout:
    // x: [d_inner, n_head, n_tok] with stride n_head * n_tok
    // dt: [d_inner, n_head, n_tok] with stride n_head * n_tok
    // A: [d_inner, n_head, d_state]
    // B: [d_state, n_tok]
    // C: [d_state, n_tok]
    // s0: [d_inner, n_head, d_state]
    // y: [d_inner, n_head, n_tok]
    // s_out: [d_inner, n_head, d_state]

    // Get pointers for this d_inner and head
    device const float* x_ptr = src1 + i1 * n_tok + ir * d_inner * n_tok;
    device const float* dt_ptr = src2 + i1 * n_tok + ir * d_inner * n_tok;
    device const float* A_ptr = src3 + i1 * d_state + ir * d_inner * d_state;
    device const float* B_ptr = src4;  // Shared across all heads
    device const float* C_ptr = src5;  // Shared across all heads
    device const float* s0_ptr = src0 + i1 * d_state + ir * d_inner * d_state;
    device float* s_out_ptr = dst + d_inner * n_tok + i1 * d_state + ir * d_inner * d_state;
    device float* y_ptr = dst + i1 * n_tok + ir * d_inner * n_tok;

    const int32_t i = i0 + i1 * nc;

    // Initialize state
    float s0 = s0_ptr[i0];
    float s = 0.0f;

    // Load A value for this state element
    const float A0 = A_ptr[i0];

    // Process tokens in batches of sgptg
    for (int i2 = 0; i2 < n_t; i2 += sgptg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pre-compute x_dt and dA for this batch of tokens
        // Only first sgptg threads do the loads and expensive math
        if (i0 < sgptg && i2 + i0 < n_t) {
            device const float* x_t = x_ptr + i0;
            device const float* dt_t = dt_ptr + i0;

            const float dt0 = dt_t[0];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_x_dt[i0] = x_t[0] * dtsp;
            shared_dA[i0] = dtsp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process each token in the batch
        for (int t = 0; t < sgptg && i2 + t < n_t; t++) {
            const float x_dt = shared_x_dt[t];
            const float dA = exp(shared_dA[t] * A0);

            // Update state: s = s0 * dA + B * x_dt
            s = (s0 * dA) + (B_ptr[i0 * n_tok + i2 + t] * x_dt);

            // Compute output contribution: s * C
            const float sumf = simd_sum(s * C_ptr[i0 * n_tok + i2 + t]);

            if (tiisg == 0) {
                shared_sums[t * N_SIMDWIDTH + sgitg] = sumf;
            }

            // Recurse
            s0 = s;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Final reduction for y
        const float sumf = simd_sum(shared_sums[sgitg * N_SIMDWIDTH + tiisg]);

        if (tiisg == 0 && i2 + sgitg < n_t) {
            y_ptr[i2 + sgitg] = sumf;
        }
    }

    // Store final state
    s_out_ptr[i0] = s;
}

// Kernel for d_state=16 (Mamba-1)
kernel void ssm_scan_d16_f32(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device const float* src2 [[buffer(2)]],
    device const float* src3 [[buffer(3)]],
    device const float* src4 [[buffer(4)]],
    device const float* src5 [[buffer(5)]],
    device float* dst [[buffer(6)]],
    constant uint& d_inner [[buffer(7)]],
    constant uint& n_tok [[buffer(8)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgptg [[simdgroups_per_threadgroup]])
{
    kernel_ssm_scan_batched<16>(
        src0, src1, src2, src3, src4, src5, dst,
        d_inner, n_tok, shared,
        tgpig, tpitg, sgitg, tiisg, sgptg
    );
}
