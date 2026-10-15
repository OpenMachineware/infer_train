// SSM Conv1D kernel for Mamba architecture
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <metal_stdlib>
using namespace metal;

// Function constant IDs for SSM Conv
// Using 1000+ to avoid conflicts with other kernels
constant ushort FC_SSM_CONV_BS   [[function_constant(1000)]];
constant ushort FC_SSM_CONV_NC   [[function_constant(1001)]];
constant bool   FC_SSM_CONV_SILU [[function_constant(1002)]];

// Batched kernel using function constants for compile-time specialization
// One threadgroup per (d_inner row, token batch)
// Each thread handles one token
template <ushort BATCH_SIZE, ushort NC, bool USE_SILU>
kernel void ssm_conv1d_batched_impl(
    device const float* x [[buffer(0)]],       // [d_inner, n_tok + d_conv - 1]
    device const float* w [[buffer(1)]],       // [d_inner, d_conv]
    device const float* bias [[buffer(2)]],    // [d_inner] or nullptr
    device float* y [[buffer(3)]],             // [d_inner, n_tok]
    constant uint& d_inner [[buffer(4)]],
    constant uint& n_tok [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]])
{
    const uint d_idx = tgpig.x;
    const uint t_batch = tgpig.y;
    const uint t_idx = t_batch * BATCH_SIZE + tiitg;

    if (d_idx >= d_inner || t_idx >= n_tok) {
        return;
    }

    const uint x_stride = n_tok + NC - 1;

    // Load convolution weights
    float sum = 0.0f;
    for (ushort i = 0; i < NC; ++i) {
        sum += x[d_idx * x_stride + t_idx + i] * w[d_idx * NC + i];
    }

    // Optional SiLU activation
    if (USE_SILU) {
        sum = sum / (1.0f + fast::exp(-sum));
    }

    if (bias != nullptr) {
        sum += bias[d_idx];
    }

    y[d_idx * n_tok + t_idx] = sum;
}

// Specialized instantiation for d_conv=4, batch_size=256, no SiLU
typedef decltype(ssm_conv1d_batched_impl<256, 4, false>) ssm_conv1d_batched_t;

template [[host_name("ssm_conv1d_d4_batched_256")]]
kernel ssm_conv1d_batched_t ssm_conv1d_batched_impl<256, 4, false>;

template [[host_name("ssm_conv1d_d4_batched_128")]]
kernel ssm_conv1d_batched_t ssm_conv1d_batched_impl<128, 4, false>;

template [[host_name("ssm_conv1d_d4_batched_64")]]
kernel ssm_conv1d_batched_t ssm_conv1d_batched_impl<64, 4, false>;

template [[host_name("ssm_conv1d_d4_batched_32")]]
kernel ssm_conv1d_batched_t ssm_conv1d_batched_impl<32, 4, false>;

template [[host_name("ssm_conv1d_d4_batched_16")]]
kernel ssm_conv1d_batched_t ssm_conv1d_batched_impl<16, 4, false>;

template [[host_name("ssm_conv1d_d4_batched_8")]]
kernel ssm_conv1d_batched_t ssm_conv1d_batched_impl<8, 4, false>;

template [[host_name("ssm_conv1d_d4_batched_2")]]
kernel ssm_conv1d_batched_t ssm_conv1d_batched_impl<2, 4, false>;

// Non-batched kernel for decode (n_tok=1)
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

    const uint x_stride = n_tok + 3;

    // Convolution with d_conv=4
    float sum = x[d_idx * x_stride + t_idx + 0] * w[d_idx * 4 + 0]
              + x[d_idx * x_stride + t_idx + 1] * w[d_idx * 4 + 1]
              + x[d_idx * x_stride + t_idx + 2] * w[d_idx * 4 + 2]
              + x[d_idx * x_stride + t_idx + 3] * w[d_idx * 4 + 3];

    if (bias != nullptr) {
        sum += bias[d_idx];
    }

    y[d_idx * n_tok + t_idx] = sum;
}
