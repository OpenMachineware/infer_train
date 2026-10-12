// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Convolution Metal kernels - Self-written implementation
// Supports 2D convolution with F32/F16 weights
// Algorithm: Direct convolution with padding handling
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Standard 2D convolution - each thread computes one output element
kernel void conv_2d_f32_f32(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& input_channels [[buffer(4)]],
    constant int& output_channels [[buffer(5)]],
    constant int& input_height [[buffer(6)]],
    constant int& input_width [[buffer(7)]],
    constant int& output_height [[buffer(8)]],
    constant int& output_width [[buffer(9)]],
    constant int& kernel_h [[buffer(10)]],
    constant int& kernel_w [[buffer(11)]],
    constant int& stride_h [[buffer(12)]],
    constant int& stride_w [[buffer(13)]],
    constant int& dilation_h [[buffer(14)]],
    constant int& dilation_w [[buffer(15)]],
    constant int& pad_h [[buffer(16)]],
    constant int& pad_w [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Decode output position from global ID
    const int total_outputs = batch_size * output_channels * output_height * output_width;
    if (int(gid.x) >= total_outputs) {
        return;
    }

    int remaining = int(gid.x);
    const int ow = remaining % output_width;
    remaining /= output_width;
    const int oh = remaining % output_height;
    remaining /= output_height;
    const int oc = remaining % output_channels;
    remaining /= output_channels;
    const int batch = remaining;

    // Compute input window position
    const int in_y_start = oh * stride_h - pad_h;
    const int in_x_start = ow * stride_w - pad_w;

    float accumulator = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < input_channels; ic++) {
        // Loop over kernel height
        for (int kh = 0; kh < kernel_h; kh++) {
            const int in_y = in_y_start + kh * dilation_h;
            if (in_y < 0 || in_y >= input_height) {
                continue;
            }

            // Loop over kernel width
            for (int kw = 0; kw < kernel_w; kw++) {
                const int in_x = in_x_start + kw * dilation_w;
                if (in_x < 0 || in_x >= input_width) {
                    continue;
                }

                // Input index: [batch, ic, in_y, in_x]
                const int input_idx = batch * input_channels * input_height * input_width
                                    + ic * input_height * input_width
                                    + in_y * input_width
                                    + in_x;

                // Weight index: [oc, ic, kh, kw]
                const int weight_idx = oc * input_channels * kernel_h * kernel_w
                                     + ic * kernel_h * kernel_w
                                     + kh * kernel_w
                                     + kw;

                accumulator += input[input_idx] * weights[weight_idx];
            }
        }
    }

    // Output index: [batch, oc, oh, ow]
    const int output_idx = batch * output_channels * output_height * output_width
                         + oc * output_height * output_width
                         + oh * output_width
                         + ow;

    output[output_idx] = accumulator;
}

// 2D convolution with F16 weights
kernel void conv_2d_f16_f32(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& input_channels [[buffer(4)]],
    constant int& output_channels [[buffer(5)]],
    constant int& input_height [[buffer(6)]],
    constant int& input_width [[buffer(7)]],
    constant int& output_height [[buffer(8)]],
    constant int& output_width [[buffer(9)]],
    constant int& kernel_h [[buffer(10)]],
    constant int& kernel_w [[buffer(11)]],
    constant int& stride_h [[buffer(12)]],
    constant int& stride_w [[buffer(13)]],
    constant int& dilation_h [[buffer(14)]],
    constant int& dilation_w [[buffer(15)]],
    constant int& pad_h [[buffer(16)]],
    constant int& pad_w [[buffer(17)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int total_outputs = batch_size * output_channels * output_height * output_width;
    if (int(gid.x) >= total_outputs) {
        return;
    }

    int remaining = int(gid.x);
    const int ow = remaining % output_width;
    remaining /= output_width;
    const int oh = remaining % output_height;
    remaining /= output_height;
    const int oc = remaining % output_channels;
    remaining /= output_channels;
    const int batch = remaining;

    const int in_y_start = oh * stride_h - pad_h;
    const int in_x_start = ow * stride_w - pad_w;

    float accumulator = 0.0f;

    for (int ic = 0; ic < input_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            const int in_y = in_y_start + kh * dilation_h;
            if (in_y < 0 || in_y >= input_height) {
                continue;
            }

            for (int kw = 0; kw < kernel_w; kw++) {
                const int in_x = in_x_start + kw * dilation_w;
                if (in_x < 0 || in_x >= input_width) {
                    continue;
                }

                const int input_idx = batch * input_channels * input_height * input_width
                                    + ic * input_height * input_width
                                    + in_y * input_width
                                    + in_x;

                const int weight_idx = oc * input_channels * kernel_h * kernel_w
                                     + ic * kernel_h * kernel_w
                                     + kh * kernel_w
                                     + kw;

                accumulator += input[input_idx] * float(weights[weight_idx]);
            }
        }
    }

    const int output_idx = batch * output_channels * output_height * output_width
                         + oc * output_height * output_width
                         + oh * output_width
                         + ow;

    output[output_idx] = accumulator;
}

// Depthwise convolution - each thread computes one output element
kernel void conv_2d_depthwise_f32_f32(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& channels [[buffer(4)]],
    constant int& input_height [[buffer(5)]],
    constant int& input_width [[buffer(6)]],
    constant int& output_height [[buffer(7)]],
    constant int& output_width [[buffer(8)]],
    constant int& kernel_h [[buffer(9)]],
    constant int& kernel_w [[buffer(10)]],
    constant int& stride_h [[buffer(11)]],
    constant int& stride_w [[buffer(12)]],
    constant int& dilation_h [[buffer(13)]],
    constant int& dilation_w [[buffer(14)]],
    constant int& pad_h [[buffer(15)]],
    constant int& pad_w [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int total_outputs = batch_size * channels * output_height * output_width;
    if (int(gid.x) >= total_outputs) {
        return;
    }

    int remaining = int(gid.x);
    const int ow = remaining % output_width;
    remaining /= output_width;
    const int oh = remaining % output_height;
    remaining /= output_height;
    const int c = remaining % channels;
    remaining /= channels;
    const int batch = remaining;

    const int in_y_start = oh * stride_h - pad_h;
    const int in_x_start = ow * stride_w - pad_w;

    float accumulator = 0.0f;

    for (int kh = 0; kh < kernel_h; kh++) {
        const int in_y = in_y_start + kh * dilation_h;
        if (in_y < 0 || in_y >= input_height) {
            continue;
        }

        for (int kw = 0; kw < kernel_w; kw++) {
            const int in_x = in_x_start + kw * dilation_w;
            if (in_x < 0 || in_x >= input_width) {
                continue;
            }

            // Input index: [batch, c, in_y, in_x]
            const int input_idx = batch * channels * input_height * input_width
                                + c * input_height * input_width
                                + in_y * input_width
                                + in_x;

            // Weight index: [c, kh, kw] (one filter per channel)
            const int weight_idx = c * kernel_h * kernel_w
                                 + kh * kernel_w
                                 + kw;

            accumulator += input[input_idx] * weights[weight_idx];
        }
    }

    const int output_idx = batch * channels * output_height * output_width
                         + c * output_height * output_width
                         + oh * output_width
                         + ow;

    output[output_idx] = accumulator;
}

// Depthwise convolution with F16 weights
kernel void conv_2d_depthwise_f16_f32(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& channels [[buffer(4)]],
    constant int& input_height [[buffer(5)]],
    constant int& input_width [[buffer(6)]],
    constant int& output_height [[buffer(7)]],
    constant int& output_width [[buffer(8)]],
    constant int& kernel_h [[buffer(9)]],
    constant int& kernel_w [[buffer(10)]],
    constant int& stride_h [[buffer(11)]],
    constant int& stride_w [[buffer(12)]],
    constant int& dilation_h [[buffer(13)]],
    constant int& dilation_w [[buffer(14)]],
    constant int& pad_h [[buffer(15)]],
    constant int& pad_w [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int total_outputs = batch_size * channels * output_height * output_width;
    if (int(gid.x) >= total_outputs) {
        return;
    }

    int remaining = int(gid.x);
    const int ow = remaining % output_width;
    remaining /= output_width;
    const int oh = remaining % output_height;
    remaining /= output_height;
    const int c = remaining % channels;
    remaining /= channels;
    const int batch = remaining;

    const int in_y_start = oh * stride_h - pad_h;
    const int in_x_start = ow * stride_w - pad_w;

    float accumulator = 0.0f;

    for (int kh = 0; kh < kernel_h; kh++) {
        const int in_y = in_y_start + kh * dilation_h;
        if (in_y < 0 || in_y >= input_height) {
            continue;
        }

        for (int kw = 0; kw < kernel_w; kw++) {
            const int in_x = in_x_start + kw * dilation_w;
            if (in_x < 0 || in_x >= input_width) {
                continue;
            }

            const int input_idx = batch * channels * input_height * input_width
                                + c * input_height * input_width
                                + in_y * input_width
                                + in_x;

            const int weight_idx = c * kernel_h * kernel_w
                                 + kh * kernel_w
                                 + kw;

            accumulator += input[input_idx] * float(weights[weight_idx]);
        }
    }

    const int output_idx = batch * channels * output_height * output_width
                         + c * output_height * output_width
                         + oh * output_width
                         + ow;

    output[output_idx] = accumulator;
}
