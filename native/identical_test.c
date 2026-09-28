#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

typedef uint16_t ggml_fp16_t;
#define GGML_FP16_TO_FP32(x) (((union { uint16_t h; float f; }) { .h = (x) }).f)

static inline float vec_dot_f16(const __fp16* q, const __fp16* k, int n) {
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
    float32x4_t s = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum0)), vcvt_high_f32_f16(sum0));
    float result = vaddvq_f32(s);
    for (; i < n; i++) {
        result += (float)q[i] * (float)k[i];
    }
    return result;
}

static inline void vec_mad_f16(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    const float16x8_t vx = vdupq_n_f16((__fp16)v);
    const __fp16* x_ptr = (const __fp16*)x;
    __fp16* y_ptr = (__fp16*)y;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        float16x8_t ax0 = vld1q_f16(x_ptr + i);     float16x8_t ay0 = vld1q_f16(y_ptr + i);
        ay0 = vfmaq_f16(ay0, ax0, vx);              vst1q_f16(y_ptr + i, ay0);
        float16x8_t ax1 = vld1q_f16(x_ptr + i + 8); float16x8_t ay1 = vld1q_f16(y_ptr + i + 8);
        ay1 = vfmaq_f16(ay1, ax1, vx);              vst1q_f16(y_ptr + i + 8, ay1);
        float16x8_t ax2 = vld1q_f16(x_ptr + i + 16);float16x8_t ay2 = vld1q_f16(y_ptr + i + 16);
        ay2 = vfmaq_f16(ay2, ax2, vx);              vst1q_f16(y_ptr + i + 16, ay2);
        float16x8_t ax3 = vld1q_f16(x_ptr + i + 24);float16x8_t ay3 = vld1q_f16(y_ptr + i + 24);
        ay3 = vfmaq_f16(ay3, ax3, vx);              vst1q_f16(y_ptr + i + 24, ay3);
    }
    for (; i + 8 <= n; i += 8) {
        float16x8_t ax = vld1q_f16(x_ptr + i); float16x8_t ay = vld1q_f16(y_ptr + i);
        ay = vfmaq_f16(ay, ax, vx);            vst1q_f16(y_ptr + i, ay);
    }
    for (; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i]) * v;
    }
}

static inline void vec_scale_f16(ggml_fp16_t* y, float v, int n) {
    const float16x8_t vx = vdupq_n_f16((__fp16)v);
    __fp16* y_ptr = (__fp16*)y;
    for (int i = 0; i + 8 <= n; i += 8) {
        float16x8_t ay = vld1q_f16(y_ptr + i);
        ay = vmulq_f16(ay, vx);
        vst1q_f16(y_ptr + i, ay);
    }
}

volatile float dummy = 0.0f;

void our_attention(const ggml_fp16_t* q, const ggml_fp16_t* k, const ggml_fp16_t* v,
                   float* output, int num_heads, int head_dim, int seq_len, float scale,
                   void* vkq16) {
    const __fp16* q_ptr = (const __fp16*)q;
    const __fp16* k_ptr = (const __fp16*)k;
    const __fp16* v_ptr = (const __fp16*)v;
    
    for (int h = 0; h < num_heads; h++) {
        const __fp16* q_head = q_ptr + h * head_dim;
        float* output_head = output + h * head_dim;
        memset(vkq16, 0, head_dim * sizeof(ggml_fp16_t));
        float M = -INFINITY;
        float S = 0.0f;
        
        for (int s = 0; s < seq_len; s++) {
            const __fp16* k_row = k_ptr + (s * num_heads + h) * head_dim;
            const __fp16* v_row = v_ptr + (s * num_heads + h) * head_dim;
            float score = vec_dot_f16(q_head, k_row, head_dim) * scale;
            
            float old_max = M;
            if (score > M) M = score;
            float weight;
            if (score > old_max) {
                float correction = expf(old_max - M);
                S *= correction;
                vec_scale_f16(vkq16, correction, head_dim);
                weight = 1.0f;
            } else {
                weight = expf(score - M);
            }
            S += weight;
            vec_mad_f16(vkq16, (const ggml_fp16_t*)v_row, weight, head_dim);
        }
        
        float inv_S = 1.0f / S;
        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t acc = vld1q_f16((const __fp16*)vkq16 + d);
            float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
            float32x4_t acc1 = vcvt_high_f32_f16(acc);
            acc0 = vmulq_n_f32(acc0, inv_S);
            acc1 = vmulq_n_f32(acc1, inv_S);
            vst1q_f32(output_head + d, acc0);
            vst1q_f32(output_head + d + 4, acc1);
        }
    }
    dummy = output[0];
}

int main() {
    int num_heads = 32, head_dim = 128, seq_len = 1024;
    
    ggml_fp16_t* q = aligned_alloc(64, num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* k = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* v = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
    void* vkq = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));
    
    for (int i = 0; i < num_heads * head_dim; i++) q[i] = (rand() % 1000) / 1000.0;
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) { k[i] = (rand() % 1000) / 1000.0; v[i] = (rand() % 1000) / 1000.0; }
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int w = 0; w < 20; w++) our_attention(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq);
    
    double total_ns = 0;
    for (int iter = 0; iter < 100; iter++) {
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        our_attention(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        total_ns += (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    }
    
    printf("Average: %.2f ms\n", total_ns / 100 / 1e6);
    printf("GFLOPS:  %.2f\n", 2.0 * num_heads * seq_len * head_dim / (total_ns / 100) * 1e9 / 1e9);
    
    return 0;
}
