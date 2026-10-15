// SSD (State Space Duality) Optimized Kernel
// Uses Metal simdgroup matrix multiply-accelerate (MMA) for parallel scan
// Algorithm design based on llama.cpp, self-written implementation
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <metal_stdlib>
using namespace metal;

// Constants matching llama.cpp
constant constexpr short SSD_CS  = 64;  // Chunk size (tokens per chunk)
constant constexpr short SSD_TC  = 8;   // Tile size (8x8 matrix tiles)
constant constexpr short SSD_HD  = 64;  // Head dimension for Mamba-2
constant constexpr short SSD_NSG = 4;   // Number of SIMD groups per threadgroup

#ifndef N_SIMDWIDTH_DEFINED
#define N_SIMDWIDTH_DEFINED
constant constexpr short N_SIMDWIDTH = 32;
#endif

// SSD Scan with simdgroup_float8x8 matrix operations
// One threadgroup per (head, sequence)
// Tokens processed in chunks of 64
// Uses C*B^T matrix multiplication for parallel state accumulation
kernel void kernel_ssm_scan_ssd_mma(
    device const float* src0 [[buffer(0)]],  // s0 (initial state)
    device const float* src1 [[buffer(1)]],  // x (input)
    device const float* src2 [[buffer(2)]],  // dt (delta time)
    device const float* src3 [[buffer(3)]],  // A (decay parameter)
    device const float* src4 [[buffer(4)]],  // B (input projection)
    device const float* src5 [[buffer(5)]],  // C (output projection)
    device const float* src6 [[buffer(6)]],  // ids (sequence ids, optional)
    device float* dst [[buffer(7)]],         // y (output) + s_out (final state)
    constant uint& d_state [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& n_head [[buffer(10)]],
    constant uint& n_seq_tokens [[buffer(11)]],
    constant uint& n_seqs [[buffer(12)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr short CS = SSD_CS;
    constexpr short TC = SSD_TC;
    constexpr short HD = SSD_HD;
    constexpr short NSG = SSD_NSG;

    // Shared memory layout (matching llama.cpp):
    // acs/exp(acs)/state-decay vectors, dtX[CS][HD], four private SAM row tiles [8][CS],
    // and two 8x8 scratch tiles per simdgroup. Total: 26.75 KiB.
    threadgroup float* shared_acs = shared;
    threadgroup float* shared_exp_acs = shared + CS;
    threadgroup float* shared_state_decay = shared + 2 * CS;
    threadgroup float* shared_dtx = shared + 3 * CS;
    threadgroup float* shared_sam = shared + 3 * CS + CS * HD;
    threadgroup float* sam_rows = shared_sam + sgitg * TC * CS;
    threadgroup float* shared_tile = shared_sam + NSG * TC * CS;
    threadgroup float* tile0 = shared_tile + sgitg * 2 * TC * TC;
    threadgroup float* tile1 = tile0 + TC * TC;

    const int32_t ir = tgpig.y;  // Current head
    const int32_t i3 = tgpig.z;  // Current sequence

    const int32_t nc = d_state;
    const int32_t nr = head_dim;
    const int32_t n_t = n_seq_tokens;

    // Data layout (matching llama.cpp):
    // x: [n_t, n_head, head_dim] with stride ns12 = n_head * head_dim
    // dt: [n_t, n_head] with stride ns21 = n_head
    // A: [n_head] scalar per head
    // B: [n_t, d_state] with stride d_state
    // C: [n_t, d_state] with stride d_state
    // s0: [n_head, d_state, head_dim]
    // y: [n_t, n_head, head_dim]
    // s_out: [n_head, d_state, head_dim]

    // Compute strides
    const int32_t ns12 = n_head * head_dim;  // x stride per token
    const int32_t ns21 = n_head;              // dt stride per token
    const int32_t ns42 = d_state;             // B stride per token
    const int32_t ns52 = d_state;             // C stride per token

    // Get base pointers for this head
    device const float* x_ptr = src1 + ir * head_dim;
    device const float* dt_ptr = src2 + ir;
    device const float* A_ptr = src3 + ir;
    device const float* B_ptr = src4;
    device const float* C_ptr = src5;
    device const float* s0_ptr = src0 + ir * d_state * head_dim;
    device float* s_out_ptr = dst + n_t * n_head * head_dim + ir * d_state * head_dim;
    device float* y_ptr = dst + ir * head_dim;  // For i3=0

    const float A_val = A_ptr[0];

    // Process tokens in chunks of CS
    for (int32_t t0 = 0; t0 < n_t; t0 += CS) {
        // Step 1: Pre-compute dtx = x * softplus(dt) for all tokens in chunk
        // Parallel load across all threads (matching llama.cpp)
        for (int32_t idx = tiitg; idx < CS * HD; idx += NSG * N_SIMDWIDTH) {
            const int32_t t = idx / HD;
            const int32_t c = idx % HD;
            const float dt_val = dt_ptr[(t0 + t) * ns21];
            const float dt_sp = (dt_val <= 20.0f) ? log(1.0f + exp(dt_val)) : dt_val;
            shared_dtx[idx] = x_ptr[(t0 + t) * ns12 + c] * dt_sp;
        }

        // Step 2: Compute cumulative acs = cumsum(dt * A)
        if (tiitg < CS) {
            const float dt_val = dt_ptr[(t0 + tiitg) * ns21];
            const float dt_sp = (dt_val <= 20.0f) ? log(1.0f + exp(dt_val)) : dt_val;
            shared_acs[tiitg] = dt_sp * A_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cumulative sum (only thread 0 does this, matching llama.cpp)
        if (tiitg == 0) {
            float acc = 0.0f;
            for (short t = 0; t < CS; ++t) {
                acc += shared_acs[t];
                shared_acs[t] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Compute exp_acs and state_decay (matching llama.cpp)
        if (tiitg < CS) {
            shared_exp_acs[tiitg] = exp(shared_acs[tiitg]);
            shared_state_decay[tiitg] = exp(shared_acs[CS - 1] - shared_acs[tiitg]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Get state pointer (s0 for first chunk, s_out for subsequent)
        device const float* state = (t0 == 0) ? s0_ptr : s_out_ptr;

        // Step 4: Build SAM (State Accumulator Matrix) using MMA
        // SAM[i,j] = C[i] * B[j] * exp(acs[i] - acs[j]) for j <= i
        // Each SIMD group builds one 8x64 row tile (matching llama.cpp)

        for (short ib = sgitg; ib < CS / TC; ib += NSG) {
            // For each column block jb <= ib (lower triangular)
            for (short jb = 0; jb <= ib; ++jb) {
                // Compute C * B^T using MMA
                simdgroup_float8x8 cb = make_filled_simdgroup_matrix<float, 8>(0.0f);

                // Sum over d_state dimension
                for (int32_t k0 = 0; k0 < nc; k0 += TC) {
                    simdgroup_float8x8 mc, mb;
                    // Load C tile: [8 tokens, 8 states]
                    simdgroup_load(mc, C_ptr + (t0 + ib * TC) * ns52 + k0, ns52);
                    // Load B tile: [8 tokens, 8 states] (transposed)
                    simdgroup_load(mb, B_ptr + (t0 + jb * TC) * ns42 + k0, ns42, 0, true);
                    simdgroup_multiply_accumulate(cb, mc, mb, cb);
                }

                // Store to SAM
                threadgroup float* sam = sam_rows + jb * TC;
                simdgroup_store(cb, sam, CS);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // Apply decay scaling: SAM[i,j] *= exp(acs[i] - acs[j]) for j <= i
                // (matching llama.cpp)
                for (short e = tiisg; e < TC * TC; e += N_SIMDWIDTH) {
                    const short ri = e / TC;
                    const short rj = e % TC;
                    const short i = ib * TC + ri;
                    const short j = jb * TC + rj;
                    sam[ri * CS + rj] = (j <= i) ?
                        sam[ri * CS + rj] * exp(shared_acs[i] - shared_acs[j]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Step 5: Compute y = SAM @ dtx + exp_acs * (C @ state)
            for (short ch = 0; ch < HD / TC; ++ch) {
                simdgroup_float8x8 y_diag = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_float8x8 y_inter = make_filled_simdgroup_matrix<float, 8>(0.0f);

                // y_diag = sum over jb: SAM @ dtx
                for (short jb = 0; jb <= ib; ++jb) {
                    simdgroup_float8x8 sam, mdtx;
                    simdgroup_load(sam, sam_rows + jb * TC, CS);
                    simdgroup_load(mdtx, shared_dtx + jb * TC * HD + ch * TC, HD);
                    simdgroup_multiply_accumulate(y_diag, sam, mdtx, y_diag);
                }

                // y_inter = C @ state
                for (int32_t k0 = 0; k0 < nc; k0 += TC) {
                    simdgroup_float8x8 mc, ms;
                    simdgroup_load(mc, C_ptr + (t0 + ib * TC) * ns52 + k0, ns52);
                    simdgroup_load(ms, state + ch * TC * nc + k0, nc, 0, true);
                    simdgroup_multiply_accumulate(y_inter, mc, ms, y_inter);
                }

                // Store y_diag and y_inter
                simdgroup_store(y_diag, tile0, TC);
                simdgroup_store(y_inter, tile1, TC);
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // y = y_diag + exp_acs * y_inter (matching llama.cpp)
                for (short e = tiisg; e < TC * TC; e += N_SIMDWIDTH) {
                    const short ri = e / TC;
                    const short ci = e % TC;
                    const int32_t token = t0 + ib * TC + ri;
                    y_ptr[token * n_head * head_dim + ch * TC + ci] =
                        tile0[e] + shared_exp_acs[ib * TC + ri] * tile1[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Step 6: Update state for next chunk (matching llama.cpp exactly)
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        const float chunk_decay = exp(shared_acs[CS - 1]);
        for (int32_t idx = tiitg; idx < nc * HD; idx += NSG * N_SIMDWIDTH) {
            const int32_t ci = idx / nc;
            const int32_t si = idx % nc;

            // Unrolled inner loop for better instruction scheduling
            float state_c0 = 0.0f, state_c1 = 0.0f, state_c2 = 0.0f, state_c3 = 0.0f;
            for (short t = 0; t < CS; t += 4) {
                const float decay0 = shared_state_decay[t];
                const float decay1 = shared_state_decay[t + 1];
                const float decay2 = shared_state_decay[t + 2];
                const float decay3 = shared_state_decay[t + 3];

                const float b0 = B_ptr[(t0 + t) * ns42 + si];
                const float b1 = B_ptr[(t0 + t + 1) * ns42 + si];
                const float b2 = B_ptr[(t0 + t + 2) * ns42 + si];
                const float b3 = B_ptr[(t0 + t + 3) * ns42 + si];

                const float dtx0 = shared_dtx[t * HD + ci];
                const float dtx1 = shared_dtx[(t + 1) * HD + ci];
                const float dtx2 = shared_dtx[(t + 2) * HD + ci];
                const float dtx3 = shared_dtx[(t + 3) * HD + ci];

                state_c0 += decay0 * b0 * dtx0;
                state_c1 += decay1 * b1 * dtx1;
                state_c2 += decay2 * b2 * dtx2;
                state_c3 += decay3 * b3 * dtx3;
            }
            s_out_ptr[idx] = chunk_decay * state[idx] + (state_c0 + state_c1 + state_c2 + state_c3);
        }

        // Ensure state is visible before next chunk
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
