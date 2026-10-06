// Convolution Metal kernels - Self-written implementation
// Supports 2D convolution with F32/F16 weights
// Algorithm: Direct convolution with padding handling
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

struct Conv2DConfig {
    int batch_size;
    int input_channels;
    int output_channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
};

// Standard 2D convolution - each thread computes one output element
kernel void conv_2d_f32_f32(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Conv2DConfig& cfg [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Decode output position from global ID
    const int total_outputs = cfg.batch_size * cfg.output_channels * cfg.output_height * cfg.output_width;
    if (gid.x >= uint(total_outputs)) {
        return;
    }

    int remaining = gid.x;
    const int ow = remaining % cfg.output_width;
    remaining /= cfg.output_width;
    const int oh = remaining % cfg.output_height;
    remaining /= cfg.output_height;
    const int oc = remaining % cfg.output_channels;
    remaining /= cfg.output_channels;
    const int batch = remaining;

    // Compute input window position
    const int in_y_start = oh * cfg.stride_h - cfg.pad_h;
    const int in_x_start = ow * cfg.stride_w - cfg.pad_w;

    float accumulator = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < cfg.input_channels; ic++) {
        // Loop over kernel height
        for (int kh = 0; kh < cfg.kernel_h; kh++) {
            const int in_y = in_y_start + kh * cfg.dilation_h;
            if (in_y < 0 || in_y >= cfg.input_height) {
                continue;
            }

            // Loop over kernel width
            for (int kw = 0; kw < cfg.kernel_w; kw++) {
                const int in_x = in_x_start + kw * cfg.dilation_w;
                if (in_x < 0 || in_x >= cfg.input_width) {
                    continue;
                }

                // Input index: [batch, ic, in_y, in_x]
                const int input_idx = batch * cfg.input_channels * cfg.input_height * cfg.input_width
                                    + ic * cfg.input_height * cfg.input_width
                                    + in_y * cfg.input_width
                                    + in_x;

                // Weight index: [oc, ic, kh, kw]
                const int weight_idx = oc * cfg.input_channels * cfg.kernel_h * cfg.kernel_w
                                     + ic * cfg.kernel_h * cfg.kernel_w
                                     + kh * cfg.kernel_w
                                     + kw;

                accumulator += input[input_idx] * weights[weight_idx];
            }
        }
    }

    // Output index: [batch, oc, oh, ow]
    const int output_idx = batch * cfg.output_channels * cfg.output_height * cfg.output_width
                         + oc * cfg.output_height * cfg.output_width
                         + oh * cfg.output_width
                         + ow;

    output[output_idx] = accumulator;
}

// 2D convolution with F16 weights
kernel void conv_2d_f16_f32(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Conv2DConfig& cfg [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int total_outputs = cfg.batch_size * cfg.output_channels * cfg.output_height * cfg.output_width;
    if (gid.x >= uint(total_outputs)) {
        return;
    }

    int remaining = gid.x;
    const int ow = remaining % cfg.output_width;
    remaining /= cfg.output_width;
    const int oh = remaining % cfg.output_height;
    remaining /= cfg.output_height;
    const int oc = remaining % cfg.output_channels;
    remaining /= cfg.output_channels;
    const int batch = remaining;

    const int in_y_start = oh * cfg.stride_h - cfg.pad_h;
    const int in_x_start = ow * cfg.stride_w - cfg.pad_w;

    float accumulator = 0.0f;

    for (int ic = 0; ic < cfg.input_channels; ic++) {
        for (int kh = 0; kh < cfg.kernel_h; kh++) {
            const int in_y = in_y_start + kh * cfg.dilation_h;
            if (in_y < 0 || in_y >= cfg.input_height) {
                continue;
            }

            for (int kw = 0; kw < cfg.kernel_w; kw++) {
                const int in_x = in_x_start + kw * cfg.dilation_w;
                if (in_x < 0 || in_x >= cfg.input_width) {
                    continue;
                }

                const int input_idx = batch * cfg.input_channels * cfg.input_height * cfg.input_width
                                    + ic * cfg.input_height * cfg.input_width
                                    + in_y * cfg.input_width
                                    + in_x;

                const int weight_idx = oc * cfg.input_channels * cfg.kernel_h * cfg.kernel_w
                                     + ic * cfg.kernel_h * cfg.kernel_w
                                     + kh * cfg.kernel_w
                                     + kw;

                accumulator += input[input_idx] * float(weights[weight_idx]);
            }
        }
    }

    const int output_idx = batch * cfg.output_channels * cfg.output_height * cfg.output_width
                         + oc * cfg.output_height * cfg.output_width
                         + oh * cfg.output_width
                         + ow;

    output[output_idx] = accumulator;
}

struct DepthwiseConv2DConfig {
    int batch_size;
    int channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_h;
    int pad_w;
};

// Depthwise convolution - each thread computes one output element
kernel void conv_2d_depthwise_f32_f32(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant DepthwiseConv2DConfig& cfg [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int total_outputs = cfg.batch_size * cfg.channels * cfg.output_height * cfg.output_width;
    if (gid.x >= uint(total_outputs)) {
        return;
    }

    int remaining = gid.x;
    const int ow = remaining % cfg.output_width;
    remaining /= cfg.output_width;
    const int oh = remaining % cfg.output_height;
    remaining /= cfg.output_height;
    const int c = remaining % cfg.channels;
    remaining /= cfg.channels;
    const int batch = remaining;

    const int in_y_start = oh * cfg.stride_h - cfg.pad_h;
    const int in_x_start = ow * cfg.stride_w - cfg.pad_w;

    float accumulator = 0.0f;

    for (int kh = 0; kh < cfg.kernel_h; kh++) {
        const int in_y = in_y_start + kh * cfg.dilation_h;
        if (in_y < 0 || in_y >= cfg.input_height) {
            continue;
        }

        for (int kw = 0; kw < cfg.kernel_w; kw++) {
            const int in_x = in_x_start + kw * cfg.dilation_w;
            if (in_x < 0 || in_x >= cfg.input_width) {
                continue;
            }

            // Input index: [batch, c, in_y, in_x]
            const int input_idx = batch * cfg.channels * cfg.input_height * cfg.input_width
                                + c * cfg.input_height * cfg.input_width
                                + in_y * cfg.input_width
                                + in_x;

            // Weight index: [c, kh, kw] (one filter per channel)
            const int weight_idx = c * cfg.kernel_h * cfg.kernel_w
                                 + kh * cfg.kernel_w
                                 + kw;

            accumulator += input[input_idx] * weights[weight_idx];
        }
    }

    const int output_idx = batch * cfg.channels * cfg.output_height * cfg.output_width
                         + c * cfg.output_height * cfg.output_width
                         + oh * cfg.output_width
                         + ow;

    output[output_idx] = accumulator;
}

// Depthwise convolution with F16 weights
kernel void conv_2d_depthwise_f16_f32(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant DepthwiseConv2DConfig& cfg [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int total_outputs = cfg.batch_size * cfg.channels * cfg.output_height * cfg.output_width;
    if (gid.x >= uint(total_outputs)) {
        return;
    }

    int remaining = gid.x;
    const int ow = remaining % cfg.output_width;
    remaining /= cfg.output_width;
    const int oh = remaining % cfg.output_height;
    remaining /= cfg.output_height;
    const int c = remaining % cfg.channels;
    remaining /= cfg.channels;
    const int batch = remaining;

    const int in_y_start = oh * cfg.stride_h - cfg.pad_h;
    const int in_x_start = ow * cfg.stride_w - cfg.pad_w;

    float accumulator = 0.0f;

    for (int kh = 0; kh < cfg.kernel_h; kh++) {
        const int in_y = in_y_start + kh * cfg.dilation_h;
        if (in_y < 0 || in_y >= cfg.input_height) {
            continue;
        }

        for (int kw = 0; kw < cfg.kernel_w; kw++) {
            const int in_x = in_x_start + kw * cfg.dilation_w;
            if (in_x < 0 || in_x >= cfg.input_width) {
                continue;
            }

            const int input_idx = batch * cfg.channels * cfg.input_height * cfg.input_width
                                + c * cfg.input_height * cfg.input_width
                                + in_y * cfg.input_width
                                + in_x;

            const int weight_idx = c * cfg.kernel_h * cfg.kernel_w
                                 + kh * cfg.kernel_w
                                 + kw;

            accumulator += input[input_idx] * float(weights[weight_idx]);
        }
    }

    const int output_idx = batch * cfg.channels * cfg.output_height * cfg.output_width
                         + c * cfg.output_height * cfg.output_width
                         + oh * cfg.output_width
                         + ow;

    output[output_idx] = accumulator;
}