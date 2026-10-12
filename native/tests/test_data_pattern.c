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

    float* q = aligned_alloc(64, num_heads * head_dim * sizeof(float));
    ggml_fp16_t* k = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* v = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
    float* vkq32 = aligned_alloc(64, head_dim * sizeof(float));
    ggml_fp16_t* q_f16 = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));

    // Random Q like Rust
    for (int i = 0; i < num_heads * head_dim; i++) q[i] = (rand() % 2000 - 1000) / 1000.0;  // -1.0 to 1.0

    // Constant K, V like Rust (0.5 and 0.3)
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) {
        k[i] = (ggml_fp16_t)((float16_t)0.5f);
        v[i] = (ggml_fp16_t)((float16_t)0.3f);
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int w = 0; w < 10; w++) {
        decode_attention_f32_q_f16_kv(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq32, q_f16, NULL, 0);
    }

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int iter = 0; iter < 20; iter++) {
        decode_attention_f32_q_f16_kv(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq32, q_f16, NULL, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed_ns = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    double flops = 2.0 * num_heads * seq_len * head_dim;
    double gflops = flops / (elapsed_ns / 20.0) * 1e9 / 1e9;

    printf("Random Q, const K/V: %.2f ms, %.2f GFLOPS\n", elapsed_ns / 20 / 1e6, gflops);

    return 0;
}
