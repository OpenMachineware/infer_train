// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Direct F16 attention benchmark - call our C implementation directly
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include "fp16_attention.c"

#define NUM_ITER 100

int main(int argc, char** argv) {
    int num_heads = 32;
    int head_dim = 128;
    int seq_len = 512;

    if (argc >= 4) {
        num_heads = atoi(argv[1]);
        head_dim = atoi(argv[2]);
        seq_len = atoi(argv[3]);
    }

    printf("F16 attention benchmark: num_heads=%d, head_dim=%d, seq_len=%d\n",
           num_heads, head_dim, seq_len);

    // Allocate buffers
    float* q = (float*)malloc(num_heads * head_dim * sizeof(float));
    ggml_fp16_t* k = (ggml_fp16_t*)malloc(seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* v = (ggml_fp16_t*)malloc(seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    float* output = (float*)malloc(num_heads * head_dim * sizeof(float));

    // Work buffers
    float* vkq32 = (float*)malloc(head_dim * sizeof(float));
    ggml_fp16_t* q_f16 = (ggml_fp16_t*)malloc(head_dim * sizeof(ggml_fp16_t));

    // Initialize with random data
    for (int i = 0; i < num_heads * head_dim; i++) {
        q[i] = 0.1f * (float)(i % 100);
    }
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) {
        k[i] = (ggml_fp16_t)((float16_t)(0.1f * (float)(i % 100)));
        v[i] = (ggml_fp16_t)((float16_t)(0.1f * (float)(i % 100)));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    // Warmup
    decode_attention_f32_q_f16_kv(q, k, v, output, num_heads, head_dim, seq_len, scale,
                                  vkq32, q_f16, NULL, 0);

    // Benchmark
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < NUM_ITER; iter++) {
        decode_attention_f32_q_f16_kv(q, k, v, output, num_heads, head_dim, seq_len, scale,
                                      vkq32, q_f16, NULL, 0);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    double elapsed_ms = elapsed_ns / 1e6 / NUM_ITER;

    // FLOPS: 2 * num_heads * seq_len * head_dim * (dot + mad + softmax)
    // dot: 2 * head_dim, mad: 2 * head_dim, softmax: ~10 per position
    double flops = 2.0 * num_heads * seq_len * head_dim * 2 * NUM_ITER;
    double gflops = flops / elapsed_ns;

    printf("seq_len=%5d: %6.2f ms, %7.2f GFLOPS\n", seq_len, elapsed_ms, gflops);

    free(q);
    free(k);
    free(v);
    free(output);
    free(vkq32);
    free(q_f16);

    return 0;
}
