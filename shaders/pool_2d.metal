// Pooling Metal kernels - Self-written implementation
// Supports 2D pooling operations (max, avg) and 1D variants
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Max pooling - each thread computes one output element
kernel void pool_2d_max_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& input_height [[buffer(2)]],
    constant int& input_width [[buffer(3)]],
    constant int& output_height [[buffer(4)]],
    constant int& output_width [[buffer(5)]],
    constant int& kernel_h [[buffer(6)]],
    constant int& kernel_w [[buffer(7)]],
    constant int& stride_h [[buffer(8)]],
    constant int& stride_w [[buffer(9)]],
    constant int& pad_h [[buffer(10)]],
    constant int& pad_w [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= output_height * output_width) {
        return;
    }

    const int out_y = int(gid) / output_width;
    const int out_x = int(gid) % output_width;

    const int in_y_start = out_y * stride_h - pad_h;
    const int in_x_start = out_x * stride_w - pad_w;

    float max_val = -INFINITY;

    for (int ky = 0; ky < kernel_h; ky++) {
        const int in_y = in_y_start + ky;
        if (in_y < 0 || in_y >= input_height) {
            continue;
        }

        for (int kx = 0; kx < kernel_w; kx++) {
            const int in_x = in_x_start + kx;
            if (in_x < 0 || in_x >= input_width) {
                continue;
            }

            const float val = input[in_y * input_width + in_x];
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    output[out_y * output_width + out_x] = max_val;
}

// Average pooling - each thread computes one output element
kernel void pool_2d_avg_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& input_height [[buffer(2)]],
    constant int& input_width [[buffer(3)]],
    constant int& output_height [[buffer(4)]],
    constant int& output_width [[buffer(5)]],
    constant int& kernel_h [[buffer(6)]],
    constant int& kernel_w [[buffer(7)]],
    constant int& stride_h [[buffer(8)]],
    constant int& stride_w [[buffer(9)]],
    constant int& pad_h [[buffer(10)]],
    constant int& pad_w [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= output_height * output_width) {
        return;
    }

    const int out_y = int(gid) / output_width;
    const int out_x = int(gid) % output_width;

    const int in_y_start = out_y * stride_h - pad_h;
    const int in_x_start = out_x * stride_w - pad_w;

    float sum = 0.0f;
    int count = 0;

    for (int ky = 0; ky < kernel_h; ky++) {
        const int in_y = in_y_start + ky;
        if (in_y < 0 || in_y >= input_height) {
            continue;
        }

        for (int kx = 0; kx < kernel_w; kx++) {
            const int in_x = in_x_start + kx;
            if (in_x < 0 || in_x >= input_width) {
                continue;
            }

            sum += input[in_y * input_width + in_x];
            count++;
        }
    }

    output[out_y * output_width + out_x] = (count > 0) ? (sum / count) : 0.0f;
}

// 1D max pooling
kernel void pool_1d_max_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& input_length [[buffer(2)]],
    constant int& output_length [[buffer(3)]],
    constant int& kernel_size [[buffer(4)]],
    constant int& stride [[buffer(5)]],
    constant int& pad [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= output_length) {
        return;
    }

    const int out_pos = int(gid);
    const int in_start = out_pos * stride - pad;

    float max_val = -INFINITY;

    for (int k = 0; k < kernel_size; k++) {
        const int in_pos = in_start + k;
        if (in_pos >= 0 && in_pos < input_length) {
            const float val = input[in_pos];
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    output[out_pos] = max_val;
}

// 1D average pooling
kernel void pool_1d_avg_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& input_length [[buffer(2)]],
    constant int& output_length [[buffer(3)]],
    constant int& kernel_size [[buffer(4)]],
    constant int& stride [[buffer(5)]],
    constant int& pad [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= output_length) {
        return;
    }

    const int out_pos = int(gid);
    const int in_start = out_pos * stride - pad;

    float sum = 0.0f;
    int count = 0;

    for (int k = 0; k < kernel_size; k++) {
        const int in_pos = in_start + k;
        if (in_pos >= 0 && in_pos < input_length) {
            sum += input[in_pos];
            count++;
        }
    }

    output[out_pos] = (count > 0) ? (sum / count) : 0.0f;
}