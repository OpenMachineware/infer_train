// Our exact attention implementation, compiled directly
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

typedef __fp16 float16_t;
typedef uint16_t ggml_fp16_t;

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
    
    for (; i < n; i++) {
        result += (float)q[i] * (float)k[i];
    }
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

volatile float dummy_result = 0.0f;

void our_attention(const float* q_f32, const ggml_fp16_t* k, const ggml_fp16_t* v,
                   float* output, int num_heads, int head_dim, int seq_len, float scale,
                   void* work, ggml_fp16_t* q_f16_buf) {
    const float16_t* k_ptr = (const float16_t*)k;
    const float16_t* v_ptr = (const float16_t*)v;
    float* VKQ = (float*)work;
    ggml_fp16_t* Q_f16 = q_f16_buf;
    
    for (int h = 0; h < num_heads; h++) {
        const float* q_head = q_f32 + h * head_dim;
        float* output_head = output + h * head_dim;
        
        // Convert Q to FP16 (once per head)
        for (int d = 0; d < head_dim; d += 8) {
            float32x4_t q0 = vld1q_f32(q_head + d);
            float32x4_t q1 = vld1q_f32(q_head + d + 4);
            vst1q_f16((float16_t*)(Q_f16 + d), vcvt_high_f16_f32(vcvt_f16_f32(q0), q1));
        }
        
        // Zero FP16 accumulator
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));
        
        float M = -INFINITY;
        float S = 0.0f;
        
        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;
            
            float score = vec_dot_f16((const float16_t*)Q_f16, k_row, head_dim) * scale;
            
            float old_max = M;
            if (score > M) {
                M = score;
            }
            
            float weight;
            if (score > old_max) {
                float correction = expf(old_max - M);
                S *= correction;
                vec_scale_f16((ggml_fp16_t*)VKQ, correction, head_dim);
                weight = 1.0f;
            } else {
                weight = expf(score - M);
            }
            
            S += weight;
            vec_mad_f16((ggml_fp16_t*)VKQ, (const ggml_fp16_t*)v_row, weight, head_dim);
        }
        
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
    }
    
    dummy_result = output[0];
}

int main(void) {
    int num_heads = 32, head_dim = 128;
    int seq_lens[] = {128, 256, 512, 1024, 2048};
    
    for (int t = 0; t < 5; t++) {
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
        
        for (int w = 0; w < 5; w++) {
            our_attention(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf);
        }
        
        struct timespec ts1, ts2;
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        for (int i = 0; i < 20; i++) {
            our_attention(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf);
        }
        clock_gettime(CLOCK_MONOTONIC, &ts2);
        
        double time_ns = (ts2.tv_sec - ts1.tv_sec) * 1e9 + (ts2.tv_nsec - ts1.tv_nsec);
        double time_ms = time_ns / 20.0 / 1e6;
        double flops = 2.0 * num_heads * seq_len * head_dim;
        double gflops = flops / (time_ns / 20.0) * 1e9 / 1e9;
        
        printf("seq_len=%4d: %6.2f ms, %6.2f GFLOPS\n", seq_len, time_ms, gflops);
        
        free(q_f32); free(k); free(v); free(output); free(work); free(q_f16_buf);
    }
    
    return 0;
}
