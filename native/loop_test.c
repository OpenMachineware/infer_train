#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

typedef uint16_t ggml_fp16_t;

#define N 128
#define NUM_ITER 1000000

// Our loop style
float vec_dot_our(const __fp16* q, const __fp16* k, int n) {
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
    float32x4_t s = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum0)), vcvt_high_f32_f16(sum0));
    return vaddvq_f32(s);
}

// llama.cpp loop style
void vec_dot_llama(int n, float * s, ggml_fp16_t * x, ggml_fp16_t * y) {
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
}

int main() {
    __fp16* a = aligned_alloc(64, N * sizeof(__fp16));
    __fp16* b = aligned_alloc(64, N * sizeof(__fp16));
    for (int i = 0; i < N; i++) { a[i] = i * 0.01; b[i] = i * 0.02; }

    volatile float result = 0;

    // Test our style
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int i = 0; i < NUM_ITER; i++) {
        result += vec_dot_our(a, b, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double time_our = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);

    // Test llama style
    float s;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int i = 0; i < NUM_ITER; i++) {
        vec_dot_llama(N, &s, (ggml_fp16_t*)a, (ggml_fp16_t*)b);
        result += s;
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double time_llama = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);

    printf("Our style:   %.2f ns\n", time_our / NUM_ITER);
    printf("llama style: %.2f ns\n", time_llama / NUM_ITER);
    printf("Ratio: %.2fx\n", time_our / time_llama);

    return 0;
}
