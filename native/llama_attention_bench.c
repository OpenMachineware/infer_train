// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark llama.cpp style F16 attention
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

typedef __fp16 float16_t;
typedef uint16_t ggml_fp16_t;

#define NUM_HEADS 32
#define HEAD_DIM 128
#define WARMUP 3
#define ITERATIONS 20

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// FP16 dot product matching llama.cpp's ggml_vec_dot_f16
static inline float vec_dot_f16_llama(const float16_t* x, const float16_t* y, int n) {
    float16x8_t sum[4] = { vdupq_n_f16(0.0f), vdupq_n_f16(0.0f),
                           vdupq_n_f16(0.0f), vdupq_n_f16(0.0f) };

    const int np = n & ~31;  // Process 32 at a time
    for (int i = 0; i < np; i += 32) {
        sum[0] = vfmaq_f16(sum[0], vld1q_f16(x + i),      vld1q_f16(y + i));
        sum[1] = vfmaq_f16(sum[1], vld1q_f16(x + i + 8),  vld1q_f16(y + i + 8));
        sum[2] = vfmaq_f16(sum[2], vld1q_f16(x + i + 16), vld1q_f16(y + i + 16));
        sum[3] = vfmaq_f16(sum[3], vld1q_f16(x + i + 24), vld1q_f16(y + i + 24));
    }

    // Reduce
    sum[0] = vaddq_f16(sum[0], sum[1]);
    sum[2] = vaddq_f16(sum[2], sum[3]);
    sum[0] = vaddq_f16(sum[0], sum[2]);

    // Final reduction to FP32
    float32x4_t s = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum[0])),
                               vcvt_high_f32_f16(sum[0]));
    return vaddvq_f32(s);
}

// FP16 vec_mad matching llama.cpp's ggml_vec_mad_f16
static inline void vec_mad_f16_llama(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    float16x8_t vx = vdupq_n_f16((float16_t)v);
    float16x8_t sum[4];

    const int np = n & ~31;
    for (int i = 0; i < np; i += 32) {
        sum[0] = vld1q_f16((const float16_t*)(y + i));
        sum[1] = vld1q_f16((const float16_t*)(y + i + 8));
        sum[2] = vld1q_f16((const float16_t*)(y + i + 16));
        sum[3] = vld1q_f16((const float16_t*)(y + i + 24));

        sum[0] = vfmaq_f16(sum[0], vld1q_f16((const float16_t*)(x + i)),      vx);
        sum[1] = vfmaq_f16(sum[1], vld1q_f16((const float16_t*)(x + i + 8)),  vx);
        sum[2] = vfmaq_f16(sum[2], vld1q_f16((const float16_t*)(x + i + 16)), vx);
        sum[3] = vfmaq_f16(sum[3], vld1q_f16((const float16_t*)(x + i + 24)), vx);

        vst1q_f16((float16_t*)(y + i),      sum[0]);
        vst1q_f16((float16_t*)(y + i + 8),  sum[1]);
        vst1q_f16((float16_t*)(y + i + 16), sum[2]);
        vst1q_f16((float16_t*)(y + i + 24), sum[3]);
    }
}

static volatile float dummy_result;

// Full attention loop matching llama.cpp
static void attention_llama_style(
    const float16_t* q,
    const float16_t* k,
    const float16_t* v,
    float* output,
    int num_heads,
    int head_dim,
    int seq_len,
    float scale,
    void* work
) {
    ggml_fp16_t* VKQ = (ggml_fp16_t*)work;

    for (int h = 0; h < num_heads; h++) {
        const float16_t* q_head = q + h * head_dim;
        float* output_head = output + h * head_dim;

        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));

        float M = -INFINITY;
        float S = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v + (s * num_heads + h) * head_dim;

            float score = vec_dot_f16_llama(q_head, k_row, head_dim) * scale;

            if (score > M) {
                float ms = expf(M - score);
                S *= ms;
                for (int d = 0; d < head_dim; d += 8) {
                    float16x8_t vkq = vld1q_f16((const float16_t*)(VKQ + d));
                    vst1q_f16((float16_t*)(VKQ + d), vmulq_n_f16(vkq, ms));
                }
                M = score;
            }

            float vs = expf(score - M);
            S += vs;
            vec_mad_f16_llama(VKQ, (const ggml_fp16_t*)v_row, vs, head_dim);
        }

        // Normalize and output
        float inv_S = 1.0f / S;
        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t acc = vld1q_f16((const float16_t*)(VKQ + d));
            float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
            float32x4_t acc1 = vcvt_high_f32_f16(acc);
            vst1q_f32(output_head + d, vmulq_n_f32(acc0, inv_S));
            vst1q_f32(output_head + d + 4, vmulq_n_f32(acc1, inv_S));
        }
    }

    dummy_result = output[0];
}

int main(void) {
    int seq_lens[] = {128, 256, 512, 1024, 2048};
    int num_tests = sizeof(seq_lens) / sizeof(seq_lens[0]);

    for (int t = 0; t < num_tests; t++) {
        int seq_len = seq_lens[t];

        // Allocate
        size_t kv_size = (size_t)seq_len * NUM_HEADS * HEAD_DIM * sizeof(float16_t);
        float16_t* q = (float16_t*)aligned_alloc(64, NUM_HEADS * HEAD_DIM * sizeof(float16_t));
        float16_t* k = (float16_t*)aligned_alloc(64, kv_size);
        float16_t* v = (float16_t*)aligned_alloc(64, kv_size);
        float* output = (float*)aligned_alloc(64, NUM_HEADS * HEAD_DIM * sizeof(float));
        void* work = aligned_alloc(64, HEAD_DIM * sizeof(float16_t));

        // Initialize
        for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) {
            q[i] = (float16_t)((rand() % 2000 - 1000) / 1000.0);
        }
        for (size_t i = 0; i < (size_t)seq_len * NUM_HEADS * HEAD_DIM; i++) {
            k[i] = (float16_t)((rand() % 2000 - 1000) / 1000.0);
            v[i] = (float16_t)((rand() % 2000 - 1000) / 1000.0);
        }

        float scale = 1.0f / sqrtf((float)HEAD_DIM);

        // Warmup
        for (int w = 0; w < WARMUP; w++) {
            attention_llama_style(q, k, v, output, NUM_HEADS, HEAD_DIM, seq_len, scale, work);
        }

        // Measure
        double start = get_time_ms();
        for (int i = 0; i < ITERATIONS; i++) {
            attention_llama_style(q, k, v, output, NUM_HEADS, HEAD_DIM, seq_len, scale, work);
        }
        double end = get_time_ms();

        double time_ms = (end - start) / ITERATIONS;
        double flops = 2.0 * NUM_HEADS * seq_len * HEAD_DIM;
        double gflops = flops / (time_ms * 1e6);

        printf("seq_len=%4d: time=%6.2f ms, %6.2f GFLOPS\n", seq_len, time_ms, gflops);

        free(q); free(k); free(v); free(output); free(work);
    }

    return 0;
}
