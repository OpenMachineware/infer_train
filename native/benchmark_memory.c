// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark memory access patterns for attention
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

typedef __fp16 float16_t;

#define NUM_HEADS 32
#define HEAD_DIM 128
#define WARMUP 3
#define ITERATIONS 100

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Simulate attention memory access pattern
static volatile float16_t dummy_result;  // Prevent optimization

static void simulate_attention_access(
    const float16_t* k,
    const float16_t* v,
    int num_heads,
    int head_dim,
    int seq_len,
    void* buffer
) {
    float16x8_t* sum = (float16x8_t*)buffer;

    for (int h = 0; h < num_heads; h++) {
        // Zero accumulator
        sum[0] = vdupq_n_f16(0.0f);

        // Access K and V for all positions
        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v + (s * num_heads + h) * head_dim;

            // Simulate vec_dot and vec_mad
            float16x8_t kval = vld1q_f16(k_row);
            float16x8_t vval = vld1q_f16(v_row);
            sum[0] = vfmaq_f16(sum[0], kval, vval);
        }
    }

    // Write to volatile to prevent optimization
    dummy_result = vget_lane_f16(vget_low_f16(sum[0]), 0);
}

int main(void) {
    // Test different seq_len values
    int seq_lens[] = {128, 256, 512, 1024, 2048, 4096};
    int num_tests = sizeof(seq_lens) / sizeof(seq_lens[0]);

    for (int t = 0; t < num_tests; t++) {
        int seq_len = seq_lens[t];

        // Allocate KV cache
        size_t kv_size = (size_t)seq_len * NUM_HEADS * HEAD_DIM * sizeof(float16_t);
        float16_t* k = (float16_t*)aligned_alloc(64, kv_size);
        float16_t* v = (float16_t*)aligned_alloc(64, kv_size);

        // Initialize with random values
        for (size_t i = 0; i < seq_len * NUM_HEADS * HEAD_DIM; i++) {
            k[i] = (float16_t)(rand() / (float)RAND_MAX);
            v[i] = (float16_t)(rand() / (float)RAND_MAX);
        }

        // Work buffer
        float16x8_t buffer[16];

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            simulate_attention_access(k, v, NUM_HEADS, HEAD_DIM, seq_len, buffer);
        }

        // Measure
        double start = get_time_ms();
        for (int i = 0; i < ITERATIONS; i++) {
            simulate_attention_access(k, v, NUM_HEADS, HEAD_DIM, seq_len, buffer);
        }
        double end = get_time_ms();

        // Calculate metrics
        double time_ms = (end - start) / ITERATIONS;
        size_t bytes_accessed = 2 * (size_t)seq_len * NUM_HEADS * HEAD_DIM * sizeof(float16_t) * ITERATIONS;
        double bandwidth_gb_s = (bytes_accessed / (end - start)) / 1e6;

        printf("seq_len=%4d: time=%6.2f ms, bandwidth=%6.2f GB/s\n",
               seq_len, time_ms, bandwidth_gb_s);

        free(k);
        free(v);
    }

    return 0;
}
