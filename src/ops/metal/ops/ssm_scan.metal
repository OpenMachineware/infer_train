// Optimized SSM Scan kernel for Mamba architecture
// Simple memory layout:
//   x:  [d_inner, n_tok] - row-major
//   dt: [d_inner, n_tok] - row-major
//   A:  [d_inner, d_state] - row-major
//   B:  [d_state, n_tok] - row-major
//   C:  [d_state, n_tok] - row-major
//   y:  [d_inner, n_tok] - row-major
//   s_out: [d_inner, d_state] - row-major (stored after y in dst)
//
// Batching optimization: process sgptg tokens per iteration
// Each SIMD group handles one d_inner element
// Threads within SIMD group handle different state elements

#include <metal_stdlib>
using namespace metal;

#ifndef N_SIMDWIDTH_DEFINED
#define N_SIMDWIDTH_DEFINED
constant constexpr short N_SIMDWIDTH = 32;
#endif

kernel void ssm_scan_d16_f32(
    device const float* src0 [[buffer(0)]],  // s0 (initial state) [d_inner, d_state]
    device const float* src1 [[buffer(1)]],  // x (input) [d_inner, n_tok]
    device const float* src2 [[buffer(2)]],  // dt (delta time) [d_inner, n_tok]
    device const float* src3 [[buffer(3)]],  // A (decay parameter) [d_inner, d_state]
    device const float* src4 [[buffer(4)]],  // B (input projection) [d_state, n_tok]
    device const float* src5 [[buffer(5)]],  // C (output projection) [d_state, n_tok]
    device float* dst [[buffer(6)]],         // y (output) [d_inner, n_tok] + s_out [d_inner, d_state]
    constant uint& d_state [[buffer(7)]],
    constant uint& d_inner [[buffer(8)]],
    constant uint& n_tok [[buffer(9)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgptg [[simdgroups_per_threadgroup]])
{
    // Shared memory layout (per SIMD group for independence):
    // [0..sgptg*N_SIMDWIDTH-1]: unused (was for partial sums)
    // [sgptg*N_SIMDWIDTH..sgptg*N_SIMDWIDTH+sgptg*sgptg-1]: pre-computed x_dt values (sgptg tokens * sgptg SIMD groups)
    // [sgptg*N_SIMDWIDTH+sgptg*sgptg..]: pre-computed dA values (sgptg tokens * sgptg SIMD groups)
    threadgroup float* shared_x_dt = shared + sgptg * N_SIMDWIDTH;
    threadgroup float* shared_dA = shared + sgptg * N_SIMDWIDTH + sgptg * sgptg;

    const int32_t nc = d_state;          // = 16
    const int32_t nr = d_inner;
    const int32_t n_t = n_tok;

    // Thread assignment:
    // - sgitg: which d_inner element within the threadgroup [0, sgptg-1]
    // - tiisg: which state element [0, 31], only [0, nc-1] are valid
    const int32_t d_idx = tgpig.x * sgptg + sgitg;  // global d_inner index
    const int32_t state_idx = tiisg;

    // Bounds check
    if (d_idx >= nr) {
        return;
    }

    // Compute position in state array: [d_inner, d_state]
    const int32_t state_array_idx = state_idx + d_idx * nc;

    // Load initial state (if provided and within valid state range)
    float s = 0.0f;
    if (src0 != nullptr && state_idx < nc) {
        s = src0[state_array_idx];
    }

    // Load A value for this state element
    float A0 = 0.0f;
    if (state_idx < nc) {
        A0 = src3[state_array_idx];
    }

    // Base pointers for this d_inner element
    device const float* x_base = src1 + d_idx * n_tok;
    device const float* dt_base = src2 + d_idx * n_tok;
    device float* y_base = dst + d_idx * n_tok;
    device float* s_out_base = dst + nr * n_tok + d_idx * nc;

    // Process tokens in batches of sgptg
    for (int32_t i2 = 0; i2 < n_t; i2 += sgptg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pre-compute x_dt and dA for this batch of tokens
        // Each SIMD group loads its own data independently
        // tiisg < sgptg condition ensures we only use sgptg threads for loading
        if (tiisg < sgptg && i2 + tiisg < n_t) {
            const float x_t = x_base[i2 + tiisg];
            const float dt_t = dt_base[i2 + tiisg];

            const float dt_sp = dt_t <= 20.0f ? log(1.0f + exp(dt_t)) : dt_t;
            // Store per SIMD group: index by [sgitg][tiisg]
            shared_x_dt[sgitg * sgptg + tiisg] = x_t * dt_sp;
            shared_dA[sgitg * sgptg + tiisg] = dt_sp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process each token in the batch
        for (int t = 0; t < sgptg && i2 + t < n_t; t++) {
            // Load from per-SIMD-group storage
            const float x_dt = shared_x_dt[sgitg * sgptg + t];
            const float dt_sp = shared_dA[sgitg * sgptg + t];

            // Decay
            const float decay = exp(dt_sp * A0);

            // Load B and C for this state element and token
            float B_t = 0.0f;
            float C_t = 0.0f;
            if (state_idx < nc) {
                B_t = src4[state_idx * n_tok + i2 + t];
                C_t = src5[state_idx * n_tok + i2 + t];
            }

            // State update: s = s * decay + B * x_dt
            s = s * decay + B_t * x_dt;

            // Output contribution: y += s * C
            const float y_contrib = s * C_t;

            // Reduce across state elements (threads in same SIMD group)
            const float y_sum = simd_sum(y_contrib);

            // Thread 0 (state_idx=0) writes output directly
            if (tiisg == 0) {
                y_base[i2 + t] = y_sum;
            }
        }
    }

    // Write final state
    if (state_idx < nc) {
        s_out_base[state_idx] = s;
    }
}

// SSD (State Space Duality) kernel for Mamba-2
// Memory layout: token-major for x, B, C; channel-major for state
// x: [n_tok, HD] where HD=64
// dt: [n_tok]
// A: scalar (single value)
// B, C: [n_tok, d_state]
// s0, s_out: [HD, d_state]
// y: [n_tok, HD]
kernel void ssm_scan_ssd_f32(
    device const float* x [[buffer(0)]],
    device const float* dt [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* B [[buffer(3)]],
    device const float* C [[buffer(4)]],
    device const float* s0 [[buffer(5)]],
    device float* y [[buffer(6)]],
    device float* s_out [[buffer(7)]],
    constant uint& d_state [[buffer(8)]],
    constant uint& n_tok [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    const uint HD = 64;
    const uint ch = gid;

    if (ch >= HD) {
        return;
    }

    const float A_val = A[0];  // Single scalar

    // Load initial state for this channel
    thread float state[256];  // Max d_state
    for (uint n = 0; n < d_state; n++) {
        state[n] = s0 ? s0[ch * d_state + n] : 0.0f;
    }

    // Sequential scan over tokens
    for (uint t = 0; t < n_tok; t++) {
        float x_val = x[t * HD + ch];
        float dt_val = dt[t];

        // Softplus on dt
        float dt_sp = (dt_val <= 20.0f) ? log(1.0f + exp(dt_val)) : dt_val;

        // x * dt
        float x_dt = x_val * dt_sp;

        // Compute output: y = sum(state * C)
        float y_val = 0.0f;

        // Update state and compute output
        for (uint n = 0; n < d_state; n++) {
            // Decay
            float decay = exp(dt_sp * A_val);

            // Load B and C for this state element and token
            float B_val = B[t * d_state + n];
            float C_val = C[t * d_state + n];

            // State update: s = s * decay + B * x * dt
            state[n] = state[n] * decay + B_val * x_dt;

            // Output: y += s * C
            y_val += state[n] * C_val;
        }

        // Store output
        y[t * HD + ch] = y_val;
    }

    // Store final state
    for (uint n = 0; n < d_state; n++) {
        s_out[ch * d_state + n] = state[n];
    }
}
