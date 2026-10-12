// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

typedef uint16_t ggml_fp16_t;
#include "native/fp16_attention.c"

int main() {
    int num_heads = 32, head_dim = 128, seq_len = 1024;

    // FP32 Q like Rust
    float* q = aligned_alloc(64, num_heads * head_dim * sizeof(float));
    ggml_fp16_t* k = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* v = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
    float* vkq32 = aligned_alloc(64, head_dim * sizeof(float));
    ggml_fp16_t* q_f16 = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));

    for (int i = 0; i < num_heads * head_dim; i++) q[i] = (rand() % 1000) / 1000.0;
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) { k[i] = (rand() % 1000) / 1000.0; v[i] = (rand() % 1000) / 1000.0; }

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int w = 0; w < 20; w++) {
        decode_attention_f32_q_f16_kv(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq32, q_f16, NULL, 0);
    }

    double total_ns = 0;
    for (int iter = 0; iter < 100; iter++) {
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        decode_attention_f32_q_f16_kv(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq32, q_f16, NULL, 0);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        total_ns += (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }

    printf("FP32 Q + FP16 K/V: %.2f ms, %.2f GFLOPS\n",
           total_ns / 100 / 1e6, 2.0 * num_heads * seq_len * head_dim / (total_ns / 100) * 1e9 / 1e9);

    return 0;
}
