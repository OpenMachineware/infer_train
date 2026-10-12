// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test memory access pattern impact
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

typedef uint16_t ggml_fp16_t;

#define NUM_ITER 100

// Test: Position-major access (8 KB stride between positions)
void test_pos_major(ggml_fp16_t* vkq, ggml_fp16_t* v_cache, int head_dim, int num_heads, int seq_len) {
    float weight = 0.5f;
    float16x8_t vx = vdupq_n_f16((float16_t)weight);
    float16_t* vkq_ptr = (float16_t*)vkq;

    for (int s = 0; s < seq_len; s++) {
        // 8 KB stride: s * num_heads * head_dim
        float16_t* v_row = (float16_t*)(v_cache + s * num_heads * head_dim);

        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t ax = vld1q_f16(v_row + d);
            float16x8_t ay = vld1q_f16(vkq_ptr + d);
            ay = vfmaq_f16(ay, ax, vx);
            vst1q_f16(vkq_ptr + d, ay);
        }
    }
}

// Test: Head-major access (256 byte stride between positions)
void test_head_major(ggml_fp16_t* vkq, ggml_fp16_t* v_cache, int head_dim, int seq_len) {
    float weight = 0.5f;
    float16x8_t vx = vdupq_n_f16((float16_t)weight);
    float16_t* vkq_ptr = (float16_t*)vkq;

    for (int s = 0; s < seq_len; s++) {
        // 256 byte stride: s * head_dim
        float16_t* v_row = (float16_t*)(v_cache + s * head_dim);

        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t ax = vld1q_f16(v_row + d);
            float16x8_t ay = vld1q_f16(vkq_ptr + d);
            ay = vfmaq_f16(ay, ax, vx);
            vst1q_f16(vkq_ptr + d, ay);
        }
    }
}

int main() {
    int head_dim = 128;
    int num_heads = 32;

    printf("Memory access pattern test\n");
    printf("Position-major: 8 KB stride between consecutive positions\n");
    printf("Head-major: 256 byte stride between consecutive positions\n\n");

    for (int seq_len = 128; seq_len <= 4096; seq_len *= 2) {
        printf("=== seq_len=%d ===\n", seq_len);

        // Position-major layout: [seq_len][num_heads][head_dim]
        size_t size_pos = seq_len * num_heads * head_dim * sizeof(ggml_fp16_t);
        ggml_fp16_t* v_pos = aligned_alloc(64, size_pos);

        // Head-major layout: [num_heads][seq_len][head_dim]
        size_t size_head = num_heads * seq_len * head_dim * sizeof(ggml_fp16_t);
        ggml_fp16_t* v_head = aligned_alloc(64, size_head);

        ggml_fp16_t* vkq = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));

        // Initialize
        for (size_t i = 0; i < seq_len * num_heads * head_dim; i++) {
            v_pos[i] = (i % 1000) / 1000.0;
            v_head[i] = (i % 1000) / 1000.0;
        }

        // Warmup
        for (int w = 0; w < 5; w++) {
            test_pos_major(vkq, v_pos, head_dim, num_heads, seq_len);
            test_head_major(vkq, v_head, head_dim, seq_len);
        }

        // Test position-major (8 KB stride)
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        for (int iter = 0; iter < NUM_ITER; iter++) {
            test_pos_major(vkq, v_pos, head_dim, num_heads, seq_len);
        }
        clock_gettime(CLOCK_MONOTONIC, &t2);
        double time_pos = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
        double time_per_iter_pos = time_pos / NUM_ITER / seq_len;

        // Test head-major (256 byte stride)
        clock_gettime(CLOCK_MONOTONIC, &t1);
        for (int iter = 0; iter < NUM_ITER; iter++) {
            test_head_major(vkq, v_head, head_dim, seq_len);
        }
        clock_gettime(CLOCK_MONOTONIC, &t2);
        double time_head = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
        double time_per_iter_head = time_head / NUM_ITER / seq_len;

        printf("Position-major: %.1f ns/call (%.1f ms total)\n", time_per_iter_pos, time_pos / NUM_ITER / 1e6);
        printf("Head-major:     %.1f ns/call (%.1f ms total)\n", time_per_iter_head, time_head / NUM_ITER / 1e6);
        printf("Speedup: %.2fx\n\n", time_pos / time_head);

        free(v_pos);
        free(v_head);
        free(vkq);
    }

    return 0;
}
