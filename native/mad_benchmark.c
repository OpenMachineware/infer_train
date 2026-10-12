// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// 专门测试 vec_mad 性能
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

typedef __fp16 float16_t;
typedef uint16_t ggml_fp16_t;

static inline uint64_t get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// 我们的实现
void our_vec_mad(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    for (int i = 0; i < n; i += 8) {
        float16x8_t vy = vld1q_f16((const float16_t*)y + i);
        float16x8_t vx = vld1q_f16((const float16_t*)x + i);
        vy = vfmaq_n_f16(vy, vx, v);
        vst1q_f16((float16_t*)y + i, vy);
    }
}

// llama.cpp 风格 (4路展开)
void llama_vec_mad(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    const int np = (n & ~31);
    float16x8_t vx = vdupq_n_f16((float16_t)v);

    for (int i = 0; i < np; i += 32) {
        float16x8_t ax[4], ay[4];
        for (int j = 0; j < 4; j++) {
            ax[j] = vld1q_f16((const float16_t*)(x + i + j * 8));
            ay[j] = vld1q_f16((const float16_t*)(y + i + j * 8));
            ay[j] = vfmaq_f16(ay[j], ax[j], vx);
            vst1q_f16((float16_t*)(y + i + j * 8), ay[j]);
        }
    }
}

int main(void) {
    int n = 128;
    int iterations = 1000000;

    ggml_fp16_t* y = aligned_alloc(64, n * sizeof(ggml_fp16_t));
    ggml_fp16_t* x = aligned_alloc(64, n * sizeof(ggml_fp16_t));

    for (int i = 0; i < n; i++) {
        x[i] = (rand() % 1000) / 1000.0;
        y[i] = (rand() % 1000) / 1000.0;
    }

    float v = 0.5f;

    // Warmup
    for (int i = 0; i < 1000; i++) {
        our_vec_mad(y, x, v, n);
        llama_vec_mad(y, x, v, n);
    }

    // Measure ours
    uint64_t t0 = get_ns();
    for (int i = 0; i < iterations; i++) {
        our_vec_mad(y, x, v, n);
    }
    uint64_t our_ns = get_ns() - t0;

    // Measure llama style
    t0 = get_ns();
    for (int i = 0; i < iterations; i++) {
        llama_vec_mad(y, x, v, n);
    }
    uint64_t llama_ns = get_ns() - t0;

    printf("vec_mad benchmark (%d iterations, n=%d):\n", iterations, n);
    printf("  Our:   %lu ns total, %.2f ns/call\n", our_ns, (double)our_ns/iterations);
    printf("  Llama: %lu ns total, %.2f ns/call\n", llama_ns, (double)llama_ns/iterations);
    printf("  Gap:   %+.1f%%\n", 100.0*(our_ns - llama_ns)/llama_ns);

    free(y); free(x);
    return 0;
}
