// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple wrapper to expose llama.cpp vec_dot functions for benchmarking
// Compile with:
// clang -c -O3 -mcpu=apple-m1 -I../llama.cpp-0.5.0/ggml/src -I../llama.cpp-0.5.0/ggml/src/ggml-cpu -I../llama.cpp-0.5.0/ggml/src/ggml-cpu/arch/arm -I../llama.cpp-0.5.0/ggml/include llama_vec_dot_wrapper.c -o llama_vec_dot_wrapper.o
// ar rcs libllama_vec_dot.a llama_vec_dot_wrapper.o

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "quants.h"
#include "ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

// Expose llama.cpp vec_dot functions with simple interface
// These will be called from Rust for direct performance comparison

void llama_vec_dot_q4_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q4_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q5_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q5_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q6_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q6_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q2_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q2_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q3_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q3_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

// Q0/Q1 formats
void llama_vec_dot_q4_0_q8_0(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q4_0_q8_0(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q8_0_q8_0(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q8_0_q8_0(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q4_1_q8_1(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q4_1_q8_1(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q5_0_q8_0(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q5_0_q8_0(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q5_1_q8_1(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q5_1_q8_1(n, s, 0, vx, 0, vy, 0, 1);
}

// IQ formats
void llama_vec_dot_iq4_nl_q8_0(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq4_nl_q8_0(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq3_xxs_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq3_xxs_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq3_s_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq3_s_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq2_xxs_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq2_xxs_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq2_xs_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq2_xs_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq2_s_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq2_s_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq1_s_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq1_s_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq1_m_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq1_m_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_iq4_xs_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_iq4_xs_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

// TQ formats
void llama_vec_dot_tq2_0_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_tq2_0_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_tq1_0_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_tq1_0_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

// Include vec.h for SILU, Softmax, GELU
#include "vec.h"

// Activation functions
void llama_silu_f32(int n, float * y, const float * x) {
    ggml_vec_silu_f32(n, y, x);
}

void llama_gelu_f32(int n, float * y, const float * x) {
    ggml_vec_gelu_f32(n, y, x);
}

void llama_gelu_quick_f32(int n, float * y, const float * x) {
    ggml_vec_gelu_quick_f32(n, y, x);
}

float llama_softmax_f32(int n, float * y, const float * x) {
    // Find max first
    float max = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max) max = x[i];
    }
    return ggml_vec_soft_max_f32(n, y, x, max);
}

// Binary operations
void llama_add_f32(int n, float * z, const float * x, const float * y) {
    ggml_vec_add_f32(n, z, x, y);
}

void llama_sub_f32(int n, float * z, const float * x, const float * y) {
    ggml_vec_sub_f32(n, z, x, y);
}

void llama_mul_f32(int n, float * z, const float * x, const float * y) {
    ggml_vec_mul_f32(n, z, x, y);
}

void llama_div_f32(int n, float * z, const float * x, const float * y) {
    ggml_vec_div_f32(n, z, x, y);
}

// RMS_NORM - extracted from llama.cpp ops.cpp
void llama_rms_norm_f32(
    int n_rows,
    int hidden_dim,
    float * dst,
    const float * x,
    const float * w,
    float eps
) {
    for (int row = 0; row < n_rows; row++) {
        const float * x_row = x + row * hidden_dim;
        float * dst_row = dst + row * hidden_dim;

        // Compute sum of squares
        double sum_sq = 0.0;
        for (int i = 0; i < hidden_dim; i++) {
            sum_sq += (double)(x_row[i] * x_row[i]);
        }

        // Compute scale
        float scale = 1.0f / sqrtf((float)(sum_sq / hidden_dim) + eps);

        // Apply normalization and weight
        for (int i = 0; i < hidden_dim; i++) {
            dst_row[i] = x_row[i] * scale * w[i];
        }
    }
}

// ROPE_NEOX - simplified version for benchmark
void llama_rope_neox_f32(
    int n_dims,
    int n_heads,
    float * dst,
    const float * src,
    int pos,
    float freq_base
) {
    int head_dim = n_dims;
    int n_dims_half = n_dims / 2;
    float theta_scale = powf(freq_base, -2.0f / n_dims);

    for (int h = 0; h < n_heads; h++) {
        float theta = pos;
        const float * src_head = src + h * head_dim;
        float * dst_head = dst + h * head_dim;

        for (int ic = 0; ic < n_dims_half; ic++) {
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            float x0 = src_head[ic];
            float x1 = src_head[ic + n_dims_half];

            dst_head[ic] = x0 * cos_theta - x1 * sin_theta;
            dst_head[ic + n_dims_half] = x0 * sin_theta + x1 * cos_theta;

            theta *= theta_scale;
        }
    }
}

// SCALE
void llama_scale_f32(int n, float * dst, const float * src, float scale) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i] * scale;
    }
}

// ADD1: dst = src + value
void llama_add1_f32(int n, float * dst, const float * src, float value) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i] + value;
    }
}

// DUP/CPY: memcpy
void llama_dup_f32(int n, float * dst, const float * src) {
    memcpy(dst, src, n * sizeof(float));
}
