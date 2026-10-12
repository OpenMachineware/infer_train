// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// 简化版插桩 - 只测量关键步骤
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

typedef __fp16 float16_t;
typedef uint16_t ggml_fp16_t;

static inline uint64_t get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    int num_heads = 32, head_dim = 128;
    int seq_lens[] = {128, 512, 1024, 2048};

    printf("=== Step-by-step Profiling ===\n\n");

    for (int t = 0; t < 4; t++) {
        int seq_len = seq_lens[t];

        // Allocate
        size_t kv_size = (size_t)seq_len * num_heads * head_dim * sizeof(ggml_fp16_t);
        float* q_f32 = aligned_alloc(64, num_heads * head_dim * sizeof(float));
        ggml_fp16_t* k = aligned_alloc(64, kv_size);
        ggml_fp16_t* v = aligned_alloc(64, kv_size);
        float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
        float16_t* VKQ = aligned_alloc(64, head_dim * sizeof(float16_t));
        float16_t* Q_f16 = aligned_alloc(64, head_dim * sizeof(float16_t));

        // Init
        for (int i = 0; i < num_heads * head_dim; i++) q_f32[i] = (rand() % 2000 - 1000) / 1000.0;
        for (size_t i = 0; i < (size_t)seq_len * num_heads * head_dim; i++) {
            k[i] = (rand() % 1000) / 1000.0;
            v[i] = (rand() % 1000) / 1000.0;
        }

        float scale = 1.0f / sqrtf((float)head_dim);

        // Profile each step
        uint64_t total_dot = 0, total_mad = 0, total_softmax = 0;
        uint64_t total_qconv = 0, total_output = 0;

        for (int h = 0; h < num_heads; h++) {
            const float* q_head = q_f32 + h * head_dim;
            float* out_head = output + h * head_dim;

            // Q -> FP16
            uint64_t t0 = get_ns();
            for (int d = 0; d < head_dim; d += 8) {
                float32x4_t q0 = vld1q_f32(q_head + d);
                float32x4_t q1 = vld1q_f32(q_head + d + 4);
                vst1q_f16(Q_f16 + d, vcvt_high_f16_f32(vcvt_f16_f32(q0), q1));
            }
            total_qconv += get_ns() - t0;

            // Clear
            memset(VKQ, 0, head_dim * sizeof(float16_t));

            float M = -INFINITY;
            float S = 0.0f;

            for (int s = 0; s < seq_len; s++) {
                const float16_t* k_row = (const float16_t*)k + (s * num_heads + h) * head_dim;
                const float16_t* v_row = (const float16_t*)v + (s * num_heads + h) * head_dim;

                // vec_dot
                t0 = get_ns();
                float16x8_t sum0 = vdupq_n_f16(0.0f);
                float16x8_t sum1 = vdupq_n_f16(0.0f);
                float16x8_t sum2 = vdupq_n_f16(0.0f);
                float16x8_t sum3 = vdupq_n_f16(0.0f);
                for (int i = 0; i < head_dim; i += 32) {
                    sum0 = vfmaq_f16(sum0, vld1q_f16(Q_f16 + i),      vld1q_f16(k_row + i));
                    sum1 = vfmaq_f16(sum1, vld1q_f16(Q_f16 + i + 8),  vld1q_f16(k_row + i + 8));
                    sum2 = vfmaq_f16(sum2, vld1q_f16(Q_f16 + i + 16), vld1q_f16(k_row + i + 16));
                    sum3 = vfmaq_f16(sum3, vld1q_f16(Q_f16 + i + 24), vld1q_f16(k_row + i + 24));
                }
                sum0 = vaddq_f16(sum0, sum1);
                sum2 = vaddq_f16(sum2, sum3);
                sum0 = vaddq_f16(sum0, sum2);
                float32x4_t sf = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum0)),
                                           vcvt_high_f32_f16(sum0));
                float score = vaddvq_f32(sf) * scale;
                total_dot += get_ns() - t0;

                // softmax
                t0 = get_ns();
                float old_max = M;
                if (score > M) {
                    float correction = expf(old_max - score);
                    S *= correction;
                    M = score;

                    // scale VKQ
                    float16x8_t vc = vdupq_n_f16((float16_t)correction);
                    for (int i = 0; i < head_dim; i += 8) {
                        float16x8_t vy = vld1q_f16(VKQ + i);
                        vst1q_f16(VKQ + i, vmulq_f16(vy, vc));
                    }
                    score = 1.0f;
                } else {
                    score = expf(score - M);
                }
                S += score;
                total_softmax += get_ns() - t0;

                // vec_mad
                t0 = get_ns();
                float16x8_t vw = vdupq_n_f16((float16_t)score);
                for (int i = 0; i < head_dim; i += 8) {
                    float16x8_t vy = vld1q_f16(VKQ + i);
                    float16x8_t vx = vld1q_f16(v_row + i);
                    vst1q_f16(VKQ + i, vfmaq_f16(vy, vx, vw));
                }
                total_mad += get_ns() - t0;
            }

            // Output
            t0 = get_ns();
            float inv_S = 1.0f / S;
            for (int d = 0; d < head_dim; d += 8) {
                float16x8_t acc = vld1q_f16(VKQ + d);
                float32x4_t a0 = vcvt_f32_f16(vget_low_f16(acc));
                float32x4_t a1 = vcvt_high_f32_f16(acc);
                vst1q_f32(out_head + d, vmulq_n_f32(a0, inv_S));
                vst1q_f32(out_head + d + 4, vmulq_n_f32(a1, inv_S));
            }
            total_output += get_ns() - t0;
        }

        printf("seq_len=%d:\n", seq_len);
        printf("  Q->FP16:   %6lu us (%.1f%%)\n", total_qconv/1000,
               100.0*total_qconv/(total_qconv+total_dot+total_softmax+total_mad+total_output));
        printf("  vec_dot:   %6lu us (%.1f%%)  [%.1f ns/call]\n", total_dot/1000,
               100.0*total_dot/(total_qconv+total_dot+total_softmax+total_mad+total_output),
               (double)total_dot/(num_heads*seq_len));
        printf("  softmax:   %6lu us (%.1f%%)\n", total_softmax/1000,
               100.0*total_softmax/(total_qconv+total_dot+total_softmax+total_mad+total_output));
        printf("  vec_mad:   %6lu us (%.1f%%)  [%.1f ns/call]\n", total_mad/1000,
               100.0*total_mad/(total_qconv+total_dot+total_softmax+total_mad+total_output),
               (double)total_mad/(num_heads*seq_len));
        printf("  Output:    %6lu us (%.1f%%)\n", total_output/1000,
               100.0*total_output/(total_qconv+total_dot+total_softmax+total_mad+total_output));

        uint64_t total = total_qconv + total_dot + total_softmax + total_mad + total_output;
        double gflops = 2.0 * num_heads * seq_len * head_dim / (total / 1e9) / 1e9;
        printf("  TOTAL:     %6lu us = %.1f GFLOPS\n\n", total/1000, gflops);

        free(q_f32); free(k); free(v); free(output); free(VKQ); free(Q_f16);
    }

    return 0;
}
