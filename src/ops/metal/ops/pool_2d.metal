// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Pooling Metal kernels - Self-written implementation
// Each thread processes one output pixel, scans input window
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))

// Parameter struct - format matches llama.cpp for API compatibility
struct Pool2DArgs {
    int k0;   int k1;   int s0;   int s1;
    int p0;   int p1;
    long IH;  long IW;  long OH;  long OW; long np;
};

// Max pooling: each thread computes one output element
kernel void pool_2d_max_f32(
    constant Pool2DArgs& args [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint thread_id [[thread_position_in_grid]]
) {
    // Early exit for threads beyond output size
    if (thread_id >= args.np) {
        return;
    }

    // Decode linear thread ID to (batch, out_h, out_w)
    int linear_idx = int(thread_id);
    int output_hw = int(args.OH) * int(args.OW);
    int batch_idx = linear_idx / output_hw;
    int row_col_idx = linear_idx % output_hw;
    int out_row = row_col_idx / int(args.OW);
    int out_col = row_col_idx % int(args.OW);

    // Compute input window bounds with padding
    int win_top = out_row * args.s1 - args.p1;
    int win_left = out_col * args.s0 - args.p0;

    int read_start_row = MAX_VAL(0, win_top);
    int read_end_row = MIN_VAL(int(args.IH), win_top + args.k1);
    int read_start_col = MAX_VAL(0, win_left);
    int read_end_col = MIN_VAL(int(args.IW), win_left + args.k0);

    // Pointer to input channel
    int input_offset = batch_idx * int(args.IH) * int(args.IW);
    device const float* in_ptr = input + input_offset;

    // Scan window to find maximum
    float max_value = -INFINITY;
    for (int r = read_start_row; r < read_end_row; r++) {
        for (int c = read_start_col; c < read_end_col; c++) {
            int idx = r * int(args.IW) + c;
            float val = in_ptr[idx];
            max_value = MAX_VAL(max_value, val);
        }
    }

    // Write output
    int output_offset = batch_idx * output_hw;
    output[output_offset + out_row * int(args.OW) + out_col] = max_value;
}

// Average pooling: each thread computes one output element
kernel void pool_2d_avg_f32(
    constant Pool2DArgs& args [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= args.np) {
        return;
    }

    int linear_idx = int(thread_id);
    int output_hw = int(args.OH) * int(args.OW);
    int batch_idx = linear_idx / output_hw;
    int row_col_idx = linear_idx % output_hw;
    int out_row = row_col_idx / int(args.OW);
    int out_col = row_col_idx % int(args.OW);

    int win_top = out_row * args.s1 - args.p1;
    int win_left = out_col * args.s0 - args.p0;

    int read_start_row = MAX_VAL(0, win_top);
    int read_end_row = MIN_VAL(int(args.IH), win_top + args.k1);
    int read_start_col = MAX_VAL(0, win_left);
    int read_end_col = MIN_VAL(int(args.IW), win_left + args.k0);

    int input_offset = batch_idx * int(args.IH) * int(args.IW);
    device const float* in_ptr = input + input_offset;

    float sum = 0.0f;
    for (int r = read_start_row; r < read_end_row; r++) {
        for (int c = read_start_col; c < read_end_col; c++) {
            int idx = r * int(args.IW) + c;
            sum += in_ptr[idx];
        }
    }

    int output_offset = batch_idx * output_hw;
    output[output_offset + out_row * int(args.OW) + out_col] = sum / (args.k0 * args.k1);
}

// 1D pooling
struct Pool1DArgs {
    int IW, OW, k0, s0, p0, np;
};

kernel void pool_1d_max_f32(
    constant Pool1DArgs& args [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (int(thread_id) >= args.np) return;

    int out_col = int(thread_id) % args.OW;
    int row_idx = int(thread_id) / args.OW;

    int win_start = out_col * args.s0 - args.p0;
    int read_start = MAX_VAL(0, win_start);
    int read_end = MIN_VAL(args.IW, win_start + args.k0);

    int row_offset = row_idx * args.IW;
    device const float* row_ptr = input + row_offset;

    float max_val = -INFINITY;
    for (int i = read_start; i < read_end; i++) {
        float v = row_ptr[i];
        if (v > max_val) max_val = v;
    }

    output[row_idx * args.OW + out_col] = max_val;
}

kernel void pool_1d_avg_f32(
    constant Pool1DArgs& args [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (int(thread_id) >= args.np) return;

    int out_col = int(thread_id) % args.OW;
    int row_idx = int(thread_id) / args.OW;

    int win_start = out_col * args.s0 - args.p0;
    int read_start = MAX_VAL(0, win_start);
    int read_end = MIN_VAL(args.IW, win_start + args.k0);

    int row_offset = row_idx * args.IW;
    device const float* row_ptr = input + row_offset;

    float sum = 0.0f;
    int count = 0;
    for (int i = read_start; i < read_end; i++) {
        sum += row_ptr[i];
        count++;
    }

    output[row_idx * args.OW + out_col] = count > 0 ? sum / count : 0.0f;
}
