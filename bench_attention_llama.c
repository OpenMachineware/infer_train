#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define NUM_ITER 20
#define WARMUP 10
#define NUM_HEADS 32
#define HEAD_DIM 128
#define SEQ_LEN 512

typedef uint16_t ggml_fp16_t;
volatile float dummy = 0.0f;

void vec_dot_f16(int n, float * s, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    const int np = (n & ~31);
    float16x8_t sum[4] = { vdupq_n_f16(0.0f) };
    for (int i = 0; i < np; i += 32) {
        sum[0] = vfmaq_f16(sum[0], vld1q_f16((const __fp16*)(x + i)), vld1q_f16((const __fp16*)(y + i)));
        sum[1] = vfmaq_f16(sum[1], vld1q_f16((const __fp16*)(x + i + 8)), vld1q_f16((const __fp16*)(y + i + 8)));
        sum[2] = vfmaq_f16(sum[2], vld1q_f16((const __fp16*)(x + i + 16)), vld1q_f16((const __fp16*)(y + i + 16)));
        sum[3] = vfmaq_f16(sum[3], vld1q_f16((const __fp16*)(x + i + 24)), vld1q_f16((const __fp16*)(y + i + 24)));
    }
    sum[0] = vaddq_f16(sum[0], sum[1]); sum[2] = vaddq_f16(sum[2], sum[3]); sum[0] = vaddq_f16(sum[0], sum[2]);
    *s = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(sum[0])), vcvt_f32_f16(vget_high_f16(sum[0]))));
}

void vec_mad_f16(int n, ggml_fp16_t * y, const ggml_fp16_t * x, float v) {
    const int np = (n & ~31);
    float16x8_t vx = vdupq_n_f16((__fp16)v);
    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; j++) {
            float16x8_t ax = vld1q_f16((const __fp16*)(x + i + j * 8));
            float16x8_t ay = vld1q_f16((const __fp16*)(y + i + j * 8));
            ay = vfmaq_f16(ay, ax, vx);
            vst1q_f16((__fp16*)(y + i + j * 8), ay);
        }
    }
}

void vec_scale_f16(int n, ggml_fp16_t * y, float v) {
    const int np = (n & ~7);
    float16x8_t vx = vdupq_n_f16((__fp16)v);
    for (int i = 0; i < np; i += 8) {
        float16x8_t vy = vld1q_f16((const __fp16*)(y + i));
        vst1q_f16((__fp16*)(y + i), vmulq_f16(vy, vx));
    }
}

void attention(const ggml_fp16_t *q, const ggml_fp16_t *k, const ggml_fp16_t *v, float *output) {
    ggml_fp16_t VKQ[HEAD_DIM];
    float scale = 1.0f / 11.313708498984761f;
    
    for (int h = 0; h < NUM_HEADS; h++) {
        const ggml_fp16_t *q_head = q + h * HEAD_DIM;
        float *out_head = output + h * HEAD_DIM;
        memset(VKQ, 0, HEAD_DIM * sizeof(ggml_fp16_t));
        float M = -1e30f, S = 0.0f;
        
        for (int s = 0; s < SEQ_LEN; s++) {
            const ggml_fp16_t *k_row = k + (s * NUM_HEADS + h) * HEAD_DIM;
            const ggml_fp16_t *v_row = v + (s * NUM_HEADS + h) * HEAD_DIM;
            float score;
            vec_dot_f16(HEAD_DIM, &score, q_head, k_row);
            score *= scale;
            if (score > M) {
                float ms = expf(M - score);
                S *= ms;
                vec_scale_f16(HEAD_DIM, VKQ, ms);
                M = score;
            }
            float vs = expf(score - M);
            S += vs;
            vec_mad_f16(HEAD_DIM, VKQ, v_row, vs);
        }
        float inv_S = 1.0f / S;
        for (int d = 0; d < HEAD_DIM; d++) out_head[d] = (float)VKQ[d] * inv_S;
    }
    dummy = output[0];
}

int main() {
    size_t size = (size_t)SEQ_LEN * NUM_HEADS * HEAD_DIM * sizeof(ggml_fp16_t);
    ggml_fp16_t *q = aligned_alloc(64, NUM_HEADS * HEAD_DIM * sizeof(ggml_fp16_t));
    ggml_fp16_t *k = aligned_alloc(64, size);
    ggml_fp16_t *v = aligned_alloc(64, size);
    float *output = aligned_alloc(64, NUM_HEADS * HEAD_DIM * sizeof(float));
    
    for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) q[i] = 0x3800;
    for (size_t i = 0; i < (size_t)SEQ_LEN * NUM_HEADS * HEAD_DIM; i++) { k[i] = 0x3400; v[i] = 0x3000; }
    
    for (int w = 0; w < WARMUP; w++) attention(q, k, v, output);
    
    struct timespec ts1, ts2;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    for (int i = 0; i < NUM_ITER; i++) attention(q, k, v, output);
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    
    double ms = ((ts2.tv_sec - ts1.tv_sec) * 1000.0 + (ts2.tv_nsec - ts1.tv_nsec) / 1000000.0) / NUM_ITER;
    double gflops = 2.0 * NUM_HEADS * SEQ_LEN * HEAD_DIM * NUM_ITER / ((ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) / 1e9) / 1e9;
    printf("C: %.2f ms, %.2f GFLOPS\n", ms, gflops);
    return 0;
}
