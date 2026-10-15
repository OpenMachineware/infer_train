// SSM Scan kernel for Mamba architecture
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
//
// Input shapes:
//   x: [d_inner, n_tok] - Input sequence
//   dt: [d_inner, n_tok] - Delta (step size) per position
//   A: [d_inner, d_state] - State transition matrix
//   B: [d_state, n_tok] - Input modulation
//   C: [d_state, n_tok] - Output modulation
//   s0: [d_inner, d_state] - Initial state (optional, zeros if null)
//
// Output shapes:
//   y: [d_inner, n_tok] - Output sequence
//   s: [d_inner, d_state] - Final state

// Single-head SSM scan (Mamba-1 style: d_state = 16)
// Each thread processes one d_inner element, unrolled over d_state
template <ushort d_state>
void ssm_scan_f32_impl(
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
    uint d_idx)
{
    if (d_idx >= d_inner) {
        return;
    }

    // Strides (row-major)
    const uint x_stride = n_tok;
    const uint dt_stride = n_tok;
    const uint A_stride = d_state;
    const uint BC_stride = n_tok;  // B and C have shape [d_state, n_tok]

    // Pointers for this d_inner element
    const uint x_base = d_idx * x_stride;
    const uint dt_base = d_idx * dt_stride;
    const uint A_base = d_idx * A_stride;
    const uint y_base = d_idx * n_tok;
    const uint s_base = d_idx * d_state;

    // Load A values (constant across time)
    float regA[d_state];
    for (ushort n = 0; n < d_state; n++) {
        regA[n] = A[A_base + n];
    }

    // Load initial state (or zeros)
    float state[d_state];
    for (ushort n = 0; n < d_state; n++) {
        state[n] = (s0 != nullptr) ? s0[s_base + n] : 0.0f;
    }

    // Scan over time
    for (uint t = 0; t < n_tok; t++) {
        // Load x and dt for this position
        float x_t = x[x_base + t];
        float dt_t = dt[dt_base + t];

        // Softplus on dt
        float dt_sp = (dt_t <= 20.0f) ? log(1.0f + exp(dt_t)) : dt_t;

        // x * dt
        float x_dt = x_t * dt_sp;

        // Accumulate output
        float y_t = 0.0f;

        // Update state and compute output
        for (ushort n = 0; n < d_state; n++) {
            // Load B and C for this (state, time)
            float B_t = B[n * BC_stride + t];
            float C_t = C[n * BC_stride + t];

            // Decay: exp(dt_sp * A)
            float decay = exp(dt_sp * regA[n]);

            // State update: s = s * decay + B * x * dt
            state[n] = state[n] * decay + B_t * x_dt;

            // Output: y += s * C
            y_t += state[n] * C_t;
        }

        // Store output
        y[y_base + t] = y_t;
    }

    // Store final state
    for (ushort n = 0; n < d_state; n++) {
        s_out[s_base + n] = state[n];
    }
}

// Specialized kernel for d_state=16 (Mamba-1)
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
    uint gid [[thread_position_in_grid]])
{
    ssm_scan_f32_impl<16>(x, dt, A, B, C, s0, y, s_out, d_inner, n_tok, gid);
}
