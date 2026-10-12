// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// 两边算法插桩对比
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
#include <inttypes.h>

typedef __fp16 float16_t;
typedef uint16_t ggml_fp16_t;

#define GGML_FP16_TO_FP32(x) ((float)(*((float16_t*)&(x))))

// 计时器
static inline uint64_t get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// 统计结构
typedef struct {
    uint64_t q_conv_ns;
    uint64_t clear_ns;
    uint64_t dot_ns;
    uint64_t softmax_ns;
    uint64_t scale_ns;
    uint64_t mad_ns;
    uint64_t output_ns;
    int scale_count;
} Stats;

static inline float vec_dot_f16(const float16_t* q, const float16_t* k, int n) {
    float16x8_t sum0 = vdupq_n_f16(0.0f);
    float16x8_t sum1 = vdupq_n_f16(0.0f);
    float16x8_t sum2 = vdupq_n_f16(0.0f);
    float16x8_t sum3 = vdupq_n_f16(0.0f);
    int i = 0;

    for (; i + 32 <= n; i += 32) {
        sum0 = vfmaq_f16(sum0, vld1q_f16(q + i),      vld1q_f16(k + i));
        sum1 = vfmaq_f16(sum1, vld1q_f16(q + i + 8),  vld1q_f16(k + i + 8));
        sum2 = vfmaq_f16(sum2, vld1q_f16(q + i + 16), vld1q_f16(k + i + 16));
        sum3 = vfmaq_f16(sum3, vld1q_f16(q + i + 24), vld1q_f16(k + i + 24));
    }

    sum0 = vaddq_f16(sum0, sum1);
    sum2 = vaddq_f16(sum2, sum3);
    sum0 = vaddq_f16(sum0, sum2);

    for (; i + 8 <= n; i += 8) {
        sum0 = vfmaq_f16(sum0, vld1q_f16(q + i), vld1q_f16(k + i));
    }

    float32x4_t s = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum0)),
                               vcvt_high_f32_f16(sum0));
    float result = vaddvq_f32(s);
    return result;
}

static inline void vec_scale_f16(ggml_fp16_t* y, float v, int n) {
    for (int i = 0; i < n; i += 8) {
        float16x8_t vy = vld1q_f16((const float16_t*)y + i);
        vy = vmulq_n_f16(vy, v);
        vst1q_f16((float16_t*)y + i, vy);
    }
}

static inline void vec_mad_f16(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    for (int i = 0; i < n; i += 8) {
        float16x8_t vy = vld1q_f16((const float16_t*)y + i);
        float16x8_t vx = vld1q_f16((const float16_t*)x + i);
        vy = vfmaq_n_f16(vy, vx, v);
        vst1q_f16((float16_t*)y + i, vy);
    }
}

// 我们的实现（带插桩）
void our_attention_instrumented(const float* q_f32, const ggml_fp16_t* k, const ggml_fp16_t* v,
                                float* output, int num_heads, int head_dim, int seq_len, float scale,
                                void* work, ggml_fp16_t* q_f16_buf, Stats* stats) {
    const float16_t* k_ptr = (const float16_t*)k;
    const float16_t* v_ptr = (const float16_t*)v;
    float* VKQ = (float*)work;
    ggml_fp16_t* Q_f16 = q_f16_buf;

    memset(stats, 0, sizeof(Stats));

    for (int h = 0; h < num_heads; h++) {
        const float* q_head = q_f32 + h * head_dim;
        float* output_head = output + h * head_dim;

        uint64_t t0 = get_ns();
        for (int d = 0; d < head_dim; d += 8) {
            float32x4_t q0 = vld1q_f32(q_head + d);
            float32x4_t q1 = vld1q_f32(q_head + d + 4);
            vst1q_f16((float16_t*)(Q_f16 + d), vcvt_high_f16_f32(vcvt_f16_f32(q0), q1));
        }
        stats->q_conv_ns += get_ns() - t0;

        t0 = get_ns();
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));
        stats->clear_ns += get_ns() - t0;

        float M = -INFINITY;
        float S = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;

            t0 = get_ns();
            float score = vec_dot_f16((const float16_t*)Q_f16, k_row, head_dim) * scale;
            stats->dot_ns += get_ns() - t0;

            t0 = get_ns();
            float old_max = M;
            if (score > M) M = score;

            float weight;
            if (score > old_max) {
                float correction = expf(old_max - M);
                S *= correction;
                weight = 1.0f;
                stats->softmax_ns += get_ns() - t0;

                t0 = get_ns();
                vec_scale_f16((ggml_fp16_t*)VKQ, correction, head_dim);
                stats->scale_ns += get_ns() - t0;
                stats->scale_count++;
            } else {
                weight = expf(score - M);
                stats->softmax_ns += get_ns() - t0;
            }

            S += weight;

            t0 = get_ns();
            vec_mad_f16((ggml_fp16_t*)VKQ, (const ggml_fp16_t*)v_row, weight, head_dim);
            stats->mad_ns += get_ns() - t0;
        }

        t0 = get_ns();
        float inv_sum = 1.0f / S;
        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t acc = vld1q_f16((const float16_t*)VKQ + d);
            float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
            float32x4_t acc1 = vcvt_high_f32_f16(acc);
            acc0 = vmulq_n_f32(acc0, inv_sum);
            acc1 = vmulq_n_f32(acc1, inv_sum);
            vst1q_f32(output_head + d, acc0);
            vst1q_f32(output_head + d + 4, acc1);
        }
        stats->output_ns += get_ns() - t0;
    }
}

// llama.cpp风格实现（带插桩）- inline vec_dot
void llama_attention_instrumented(const float* q_f32, const ggml_fp16_t* k, const ggml_fp16_t* v,
                                  float* output, int num_heads, int head_dim, int seq_len, float scale,
                                  void* work, ggml_fp16_t* q_f16_buf, Stats* stats) {
    ggml_fp16_t * VKQ = (ggml_fp16_t *)work;

    memset(stats, 0, sizeof(Stats));

    for (int h = 0; h < num_heads; h++) {
        const float* q_head = q_f32 + h * head_dim;
        float* out_head = output + h * head_dim;

        uint64_t t0 = get_ns();
        for (int d = 0; d < head_dim; d += 8) {
            float32x4_t q0 = vld1q_f32(q_head + d);
            float32x4_t q1 = vld1q_f32(q_head + d + 4);
            vst1q_f16((__fp16*)(q_f16_buf + d), vcvt_high_f16_f32(vcvt_f16_f32(q0), q1));
        }
        stats->q_conv_ns += get_ns() - t0;

        t0 = get_ns();
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));
        stats->clear_ns += get_ns() - t0;

        float M = -INFINITY;
        float S = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const ggml_fp16_t * k_row = k + (s * num_heads + h) * head_dim;
            const ggml_fp16_t * v_row = v + (s * num_heads + h) * head_dim;

            // Inline vec_dot (llama.cpp style)
            t0 = get_ns();
            float score;
            ggml_fp16_t * q_f16 = q_f16_buf;
            const int np = (head_dim & ~31);
            float16x8_t sum[4] = { vdupq_n_f16(0.0f), vdupq_n_f16(0.0f),
                                   vdupq_n_f16(0.0f), vdupq_n_f16(0.0f) };
            for (int i = 0; i < np; i += 32) {
                sum[0] = vfmaq_f16(sum[0], vld1q_f16((const __fp16*)(q_f16 + i)),
                                   vld1q_f16((const __fp16*)(k_row + i)));
                sum[1] = vfmaq_f16(sum[1], vld1q_f16((const __fp16*)(q_f16 + i + 8)),
                                   vld1q_f16((const __fp16*)(k_row + i + 8)));
                sum[2] = vfmaq_f16(sum[2], vld1q_f16((const __fp16*)(q_f16 + i + 16)),
                                   vld1q_f16((const __fp16*)(k_row + i + 16)));
                sum[3] = vfmaq_f16(sum[3], vld1q_f16((const __fp16*)(q_f16 + i + 24)),
                                   vld1q_f16((const __fp16*)(k_row + i + 24)));
            }
            sum[0] = vaddq_f16(sum[0], sum[1]);
            sum[2] = vaddq_f16(sum[2], sum[3]);
            sum[0] = vaddq_f16(sum[0], sum[2]);
            const float32x4_t t0_f = vcvt_f32_f16(vget_low_f16(sum[0]));
            const float32x4_t t1_f = vcvt_f32_f16(vget_high_f16(sum[0]));
            score = vaddvq_f32(vaddq_f32(t0_f, t1_f));
            score *= scale;
            stats->dot_ns += get_ns() - t0;

            t0 = get_ns();
            if (score > M) {
                float ms = expf(M - score);
                S *= ms;
                M = score;
                stats->softmax_ns += get_ns() - t0;

                t0 = get_ns();
                float16x8_t vx = vdupq_n_f16((__fp16)ms);
                for (int i = 0; i < head_dim; i += 8) {
                    float16x8_t vy = vld1q_f16((const __fp16*)(VKQ + i));
                    vst1q_f16((__fp16*)(VKQ + i), vmulq_f16(vy, vx));
                }
                stats->scale_ns += get_ns() - t0;
                stats->scale_count++;
            } else {
                float vs = expf(score - M);
                S += vs;
                stats->softmax_ns += get_ns() - t0;

                t0 = get_ns();
                float16x8_t vv = vdupq_n_f16((__fp16)vs);
                for (int i = 0; i < head_dim; i += 8) {
                    float16x8_t ax = vld1q_f16((const __fp16*)(v_row + i));
                    float16x8_t ay = vld1q_f16((const __fp16*)(VKQ + i));
                    vst1q_f16((__fp16*)(VKQ + i), vfmaq_f16(ay, ax, vv));
                }
                stats->mad_ns += get_ns() - t0;
            }
        }

        t0 = get_ns();
        float inv_S = 1.0f / S;
        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t acc = vld1q_f16((const __fp16*)(VKQ + d));
            float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
            float32x4_t acc1 = vcvt_high_f32_f16(acc);
            vst1q_f32(out_head + d, vmulq_n_f32(acc0, inv_S));
            vst1q_f32(out_head + d + 4, vmulq_n_f32(acc1, inv_S));
        }
        stats->output_ns += get_ns() - t0;
    }
}

int main(void) {
    int num_heads = 32, head_dim = 128;
    int seq_lens[] = {128, 512, 1024, 2048};

    printf("=== Instrumented Comparison (ns per step) ===\n\n");

    for (int t = 0; t < 4; t++) {
        int seq_len = seq_lens[t];

        size_t kv_size = (size_t)seq_len * num_heads * head_dim * sizeof(ggml_fp16_t);
        float* q_f32 = aligned_alloc(64, num_heads * head_dim * sizeof(float));
        ggml_fp16_t* k = aligned_alloc(64, kv_size);
        ggml_fp16_t* v = aligned_alloc(64, kv_size);
        float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
        void* work = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));
        ggml_fp16_t* q_f16_buf = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));

        for (int i = 0; i < num_heads * head_dim; i++) {
            q_f32[i] = (float)((rand() % 2000 - 1000) / 1000.0);
        }
        for (size_t i = 0; i < (size_t)seq_len * num_heads * head_dim; i++) {
            k[i] = (rand() % 1000) / 1000.0;
            v[i] = (rand() % 1000) / 1000.0;
        }

        float scale = 1.0f / sqrtf((float)head_dim);
        Stats our_stats, llama_stats;

        // Warmup
        our_attention_instrumented(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf, &our_stats);
        llama_attention_instrumented(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf, &llama_stats);

        // Measure
        our_attention_instrumented(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf, &our_stats);
        llama_attention_instrumented(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf, &llama_stats);

        printf("seq_len=%d (per call):\n", seq_len);
        printf("  Step              Ours(ns)   Llama(ns)     Gap\n");
        printf("  Q->FP16:       %10" PRIu64 "  %10" PRIu64 "   %+.1f%%\n",
               our_stats.q_conv_ns/num_heads, llama_stats.q_conv_ns/num_heads,
               100.0*(int64_t)(our_stats.q_conv_ns - llama_stats.q_conv_ns)/(int64_t)llama_stats.q_conv_ns);
        printf("  Clear:         %10" PRIu64 "  %10" PRIu64 "   %+.1f%%\n",
               our_stats.clear_ns/num_heads, llama_stats.clear_ns/num_heads,
               100.0*(int64_t)(our_stats.clear_ns - llama_stats.clear_ns)/(int64_t)llama_stats.clear_ns);
        printf("  vec_dot:       %10" PRIu64 "  %10" PRIu64 "   %+.1f%%\n",
               our_stats.dot_ns/(num_heads*seq_len), llama_stats.dot_ns/(num_heads*seq_len),
               100.0*(int64_t)(our_stats.dot_ns - llama_stats.dot_ns)/(int64_t)llama_stats.dot_ns);
        printf("  softmax:       %10" PRIu64 "  %10" PRIu64 "   %+.1f%%\n",
               our_stats.softmax_ns/(num_heads*seq_len), llama_stats.softmax_ns/(num_heads*seq_len),
               100.0*(int64_t)(our_stats.softmax_ns - llama_stats.softmax_ns)/(int64_t)llama_stats.softmax_ns);
        printf("  vec_scale:     %10" PRIu64 "  %10" PRIu64 "   %+.1f%%  (count: %d)\n",
               our_stats.scale_ns/(num_heads), llama_stats.scale_ns/(num_heads),
               100.0*(int64_t)(our_stats.scale_ns - llama_stats.scale_ns)/(int64_t)llama_stats.scale_ns,
               our_stats.scale_count);
        printf("  vec_mad:       %10" PRIu64 "  %10" PRIu64 "   %+.1f%%\n",
               our_stats.mad_ns/(num_heads*seq_len), llama_stats.mad_ns/(num_heads*seq_len),
               100.0*(int64_t)(our_stats.mad_ns - llama_stats.mad_ns)/(int64_t)llama_stats.mad_ns);
        printf("  Output:        %10" PRIu64 "  %10" PRIu64 "   %+.1f%%\n",
               our_stats.output_ns/num_heads, llama_stats.output_ns/num_heads,
               100.0*(int64_t)(our_stats.output_ns - llama_stats.output_ns)/(int64_t)llama_stats.output_ns);

        uint64_t our_total = our_stats.q_conv_ns + our_stats.clear_ns + our_stats.dot_ns +
                            our_stats.softmax_ns + our_stats.scale_ns + our_stats.mad_ns + our_stats.output_ns;
        uint64_t llama_total = llama_stats.q_conv_ns + llama_stats.clear_ns + llama_stats.dot_ns +
                              llama_stats.softmax_ns + llama_stats.scale_ns + llama_stats.mad_ns + llama_stats.output_ns;
        printf("  TOTAL:         %10" PRIu64 "  %10" PRIu64 "   %+.1f%%\n\n", our_total, llama_total,
               100.0*(int64_t)(our_total - llama_total)/(int64_t)llama_total);

        free(q_f32); free(k); free(v); free(output); free(work); free(q_f16_buf);
    }

    return 0;
}
