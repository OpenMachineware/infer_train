// Optimized SSM Scan kernel for Mamba architecture
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <metal_stdlib>
using namespace metal;

// SSM Scan: Recurrent state update for state space models
//
// Algorithm:
//   For each time step t:
//     1. Apply softplus to dt: dt_sp = log(1 + exp(dt)) for dt <= 20, else dt
//     2. Compute decay: dA = exp(dt_sp * A)
//     3. Update state: s[t] = s[t-1] * dA + B[t] * x[t] * dt_sp
//     4. Compute output: y[t] = sum(s[t] * C[t])

// Device function for SSM scan (not a kernel)
template <ushort d_state>
void ssm_scan_opt_impl(
    device const float* x,
    device const float* dt,
    device const float* A,
    device const float* B,
    device const float* C,
    device const float* s0,
    device float* y,
    device float* s_out,
    constant uint& d_inner,
    constant uint& n_tok,
    uint d_idx,
    uint state_idx)
{
    // gid.x = d_inner index
    // gid.y = state element index (within simd group)

    if (d_idx >= d_inner) {
        return;
    }

    // Strides
    const uint x_stride = n_tok;
    const uint a_stride = d_state;
    const uint bc_stride = n_tok;

    // Base pointers for this d_inner
    device const float* x_ptr = x + d_idx * x_stride;
    device const float* dt_ptr = dt + d_idx * x_stride;
    device const float* A_ptr = A + d_idx * a_stride;
    device float* y_ptr = y + d_idx * n_tok;
    device float* s_ptr = s_out + d_idx * d_state;

    // Initialize state (this thread holds one state element)
    float state = 0.0f;
    if (s0 != nullptr && state_idx < d_state) {
        state = s0[d_idx * d_state + state_idx];
    }

    // Load A value for this state element (constant across time)
    float A_val = 0.0f;
    if (state_idx < d_state) {
        A_val = A_ptr[state_idx];
    }

    // Process each token
    for (uint t = 0; t < n_tok; t++) {
        // Load x and dt for this token (shared across all state elements)
        float x_t = x_ptr[t];
        float dt_t = dt_ptr[t];

        // Softplus on dt
        float dt_sp = (dt_t <= 20.0f) ? log(1.0f + exp(dt_t)) : dt_t;

        // Compute x * dt (shared)
        float x_dt = x_t * dt_sp;

        // Compute decay (shared)
        float decay = exp(dt_sp * A_val);

        // Load B and C for this state element and token
        float B_t = 0.0f;
        float C_t = 0.0f;
        if (state_idx < d_state) {
            B_t = B[state_idx * bc_stride + t];
            C_t = C[state_idx * bc_stride + t];
        }

        // Update state
        state = state * decay + B_t * x_dt;

        // Compute output contribution: state * C
        float y_contrib = state * C_t;

        // Parallel reduction across state dimension using SIMD
        // Only threads in the same SIMD group (same d_idx, different state_idx) participate
        float y_sum = simd_sum(y_contrib);

        // Thread 0 (state_idx=0) writes output
        if (state_idx == 0) {
            y_ptr[t] = y_sum;
        }
    }

    // Write final state
    if (state_idx < d_state) {
        s_ptr[state_idx] = state;
    }
}

// Kernel for d_state=16 (Mamba-1)
kernel void ssm_scan_d16_f32(
    device const float* x [[buffer(0)]],
    device const float* dt [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* B [[buffer(3)]],
    device const float* C [[buffer(4)]],
    device const float* s0 [[buffer(5)]],
    device float* y [[buffer(6)]],
    device float* s_out [[buffer(7)]],
    constant uint& d_inner [[buffer(8)]],
    constant uint& n_tok [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d_idx = gid.x;
    const uint state_idx = gid.y;
    ssm_scan_opt_impl<16>(x, dt, A, B, C, s0, y, s_out, d_inner, n_tok, d_idx, state_idx);
}
