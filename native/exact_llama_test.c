// Exact copy of llama.cpp benchmark code
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

#define GGML_FP16_TO_FP32(x) (((union { uint16_t h; float f; }) { .h = (x) }).f)
typedef uint16_t ggml_fp16_t;

void ggml_vec_dot_f16(int n, float * s, ggml_fp16_t * x, ggml_fp16_t * y) {
    const int np = (n & ~31);
    float16x8_t sum[4] = { vdupq_n_f16(0.0f), vdupq_n_f16(0.0f), vdupq_n_f16(0.0f), vdupq_n_f16(0.0f) };
    for (int i = 0; i < np; i += 32) {
        sum[0] = vfmaq_f16(sum[0], vld1q_f16((const __fp16*)(x + i)), vld1q_f16((const __fp16*)(y + i)));
        sum[1] = vfmaq_f16(sum[1], vld1q_f16((const __fp16*)(x + i + 8)), vld1q_f16((const __fp16*)(y + i + 8)));
        sum[2] = vfmaq_f16(sum[2], vld1q_f16((const __fp16*)(x + i + 16)), vld1q_f16((const __fp16*)(y + i + 16)));
        sum[3] = vfmaq_f16(sum[3], vld1q_f16((const __fp16*)(x + i + 24)), vld1q_f16((const __fp16*)(y + i + 24)));
    }
    sum[0] = vaddq_f16(sum[0], sum[1]);
    sum[2] = vaddq_f16(sum[2], sum[3]);
    sum[0] = vaddq_f16(sum[0], sum[2]);
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16(sum[0]));
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(sum[0]));
    *s = vaddvq_f32(vaddq_f32(t0, t1));
    for (int i = np; i < n; ++i) *s += GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]);
}

void ggml_vec_mad_f16(int n, ggml_fp16_t * y, ggml_fp16_t * x, float v) {
    const int np = (n & ~31);
    float16x8_t vx = vdupq_n_f16((__fp16)v);
    for (int i = 0; i < np; i += 32) {
        float16x8_t ax[4], ay[4];
        for (int j = 0; j < 4; j++) {
            ax[j] = vld1q_f16((const __fp16*)(x + i + j * 8));
            ay[j] = vld1q_f16((const __fp16*)(y + i + j * 8));
            ay[j] = vfmaq_f16(ay[j], ax[j], vx);
            vst1q_f16((__fp16*)(y + i + j * 8), ay[j]);
        }
    }
    for (int i = np; i < n; ++i) y[i] = GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i]) * v;
}

void ggml_vec_scale_f16(int n, ggml_fp16_t * y, float v) {
    const int np = (n & ~7);
    float16x8_t vx = vdupq_n_f16((__fp16)v);
    for (int i = 0; i < np; i += 8) {
        float16x8_t vy = vld1q_f16((const __fp16*)(y + i));
        vy = vmulq_f16(vy, vx);
        vst1q_f16((__fp16*)(y + i), vy);
    }
    for (int i = np; i < n; ++i) y[i] = GGML_FP16_TO_FP32(y[i]) * v;
}

volatile float dummy = 0.0f;

void attention_llama_ggml(const ggml_fp16_t *q, const ggml_fp16_t *k, const ggml_fp16_t *v,
                          float *output, int num_heads, int head_dim, int seq_len, float scale,
                          void * work) {
    ggml_fp16_t * VKQ = (ggml_fp16_t *)work;
    for (int h = 0; h < num_heads; h++) {
        const ggml_fp16_t * q_head = q + h * head_dim;
        float * out_head = output + h * head_dim;
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));
        float M = -INFINITY;
        float S = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            const ggml_fp16_t * k_row = k + (s * num_heads + h) * head_dim;
            const ggml_fp16_t * v_row = v + (s * num_heads + h) * head_dim;
            float score;
            ggml_vec_dot_f16(head_dim, &score, (ggml_fp16_t *)q_head, (ggml_fp16_t *)k_row);
            score *= scale;
            if (score > M) {
                float ms = expf(M - score);
                S *= ms;
                ggml_vec_scale_f16(head_dim, VKQ, ms);
                M = score;
            }
            float vs = expf(score - M);
            S += vs;
            ggml_vec_mad_f16(head_dim, VKQ, (ggml_fp16_t *)v_row, vs);
        }
        float inv_S = 1.0f / S;
        for (int d = 0; d < head_dim; d++) out_head[d] = GGML_FP16_TO_FP32(VKQ[d]) * inv_S;
    }
    dummy = output[0];
}

int main() {
    int num_heads = 32, head_dim = 128, seq_len = 1024;
    
    ggml_fp16_t* q = aligned_alloc(64, num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* k = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* v = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
    void* work = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));
    
    for (int i = 0; i < num_heads * head_dim; i++) q[i] = (rand() % 1000) / 1000.0;
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) { k[i] = (rand() % 1000) / 1000.0; v[i] = (rand() % 1000) / 1000.0; }
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int w = 0; w < 20; w++) attention_llama_ggml(q, k, v, output, num_heads, head_dim, seq_len, scale, work);
    
    double total_ns = 0;
    for (int iter = 0; iter < 100; iter++) {
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        attention_llama_ggml(q, k, v, output, num_heads, head_dim, seq_len, scale, work);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        total_ns += (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }
    
    printf("Average: %.2f ms\n", total_ns / 100 / 1e6);
    printf("GFLOPS:  %.2f\n", 2.0 * num_heads * seq_len * head_dim / (total_ns / 100) * 1e9 / 1e9);
    
    return 0;
}
