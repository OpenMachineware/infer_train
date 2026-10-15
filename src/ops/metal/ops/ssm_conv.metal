// SSM Conv1D kernel for Mamba architecture
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <metal_stdlib>
using namespace metal;

// SSM Conv1D: 1D convolution for state space models
// Input x: [d_inner, n_tok] (token-major for cache efficiency)
// Weight w: [d_conv, d_inner] (transposed for coalesced access)
// Bias: [d_inner]
// Output y: [d_inner, n_tok]
//
// For each position i:
//   y[i] = sum(x[i:i+d_conv] * w) + bias
//
// d_conv = 4 for Qwen3.8-27B (ssm_conv1d.weight shape=[4, 10240])

template <typename T, ushort d_conv>
kernel void ssm_conv1d_f32(
    device const float* x [[buffer(0)]],       // [d_inner, n_tok + d_conv - 1]
    device const float* w [[buffer(1)]],       // [d_inner, d_conv]
    device const float* bias [[buffer(2)]],    // [d_inner] or nullptr
    device float* y [[buffer(3)]],             // [d_inner, n_tok]
    constant uint& d_inner [[buffer(4)]],
    constant uint& n_tok [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint d_idx = gid.y;   // d_inner dimension
    const uint t_idx = gid.x;   // token dimension

    if (d_idx >= d_inner || t_idx >= n_tok) {
        return;
    }

    // Load weights into registers (small, d_conv elements)
    float weights[d_conv];
    for (ushort j = 0; j < d_conv; j++) {
        weights[j] = w[d_idx * d_conv + j];
    }

    // Compute convolution
    float sum = 0.0f;
    for (ushort j = 0; j < d_conv; j++) {
        sum += x[d_idx * (n_tok + d_conv - 1) + t_idx + j] * weights[j];
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[d_idx];
    }

    // Store result
    y[d_idx * n_tok + t_idx] = sum;
}

// Specialized kernel for d_conv=4 (Qwen3.8)
kernel void ssm_conv1d_d4_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& d_inner [[buffer(4)]],
    constant uint& n_tok [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d_idx = gid.y;
    const uint t_idx = gid.x;

    if (d_idx >= d_inner || t_idx >= n_tok) {
        return;
    }

    const uint x_stride = n_tok + 3;  // n_tok + d_conv - 1

    // Load 4 weights
    float w0 = w[d_idx * 4 + 0];
    float w1 = w[d_idx * 4 + 1];
    float w2 = w[d_idx * 4 + 2];
    float w3 = w[d_idx * 4 + 3];

    // Convolution with 4-element kernel
    float x0 = x[d_idx * x_stride + t_idx + 0];
    float x1 = x[d_idx * x_stride + t_idx + 1];
    float x2 = x[d_idx * x_stride + t_idx + 2];
    float x3 = x[d_idx * x_stride + t_idx + 3];

    float sum = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;

    if (bias != nullptr) {
        sum += bias[d_idx];
    }

    y[d_idx * n_tok + t_idx] = sum;
}

// Batch version for multiple sequences
kernel void ssm_conv1d_batch_f32(
    device const float* x [[buffer(0)]],       // [n_seq, d_inner, n_tok + d_conv - 1]
    device const float* w [[buffer(1)]],       // [d_inner, d_conv]
    device const float* bias [[buffer(2)]],    // [d_inner]
    device float* y [[buffer(3)]],             // [n_seq, d_inner, n_tok]
    constant uint& d_inner [[buffer(4)]],
    constant uint& n_tok [[buffer(5)]],
    constant uint& n_seq [[buffer(6)]],
    constant uint& d_conv_val [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint s_idx = gid.z;  // sequence
    const uint d_idx = gid.y;  // d_inner
    const uint t_idx = gid.x;  // token

    if (s_idx >= n_seq || d_idx >= d_inner || t_idx >= n_tok) {
        return;
    }

    const uint x_stride = n_tok + d_conv_val - 1;
    const uint x_offset = s_idx * d_inner * x_stride + d_idx * x_stride + t_idx;
    const uint w_offset = d_idx * d_conv_val;
    const uint y_offset = s_idx * d_inner * n_tok + d_idx * n_tok + t_idx;

    float sum = 0.0f;
    for (uint j = 0; j < d_conv_val; j++) {
        sum += x[x_offset + j] * w[w_offset + j];
    }

    if (bias != nullptr) {
        sum += bias[d_idx];
    }

    y[y_offset] = sum;
}
