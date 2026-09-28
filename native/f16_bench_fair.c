// Fair F16 attention benchmark - match llama.cpp's benchmark exactly
// Takes FP16 Q (like llama.cpp), no FP32→FP16 conversion inside
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

typedef uint16_t ggml_fp16_t;
#define GGML_FP16_TO_FP32(x) (((union { uint16_t h; float f; }) { .h = (x) }).f)

// Include our implementation but use the FP16 native version
#include "fp16_attention.c"

#define NUM_ITER 20

// Our attention taking FP16 Q (like llama.cpp)
void our_attention_f16_q(
    const ggml_fp16_t* q,
    const ggml_fp16_t* k,
    const ggml_fp16_t* v,
    float* output,
    int num_heads,
    int head_dim,
    int seq_len,
    float scale
) {
    const float16_t* q_ptr = (const float16_t*)q;
    const float16_t* k_ptr = (const float16_t*)k;
    const float16_t* v_ptr = (const float16_t*)v;
    
    // Allocate work buffer for FP16 VKQ
    ggml_fp16_t VKQ[256];  // Max head_dim
    
    for (int h = 0; h < num_heads; h++) {
        const float16_t* q_head = q_ptr + h * head_dim;
        float* output_head = output + h * head_dim;

        // Zero FP16 accumulator
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));

        float M = -INFINITY;
        float S = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;

            float score = vec_dot_f16(q_head, k_row, head_dim) * scale;

            float old_max = M;
            if (score > M) {
                M = score;
            }

            float weight;
            if (score > old_max) {
                float correction = expf(old_max - M);
                S *= correction;
                vec_scale_f16(VKQ, correction, head_dim);
                weight = 1.0f;
            } else {
                weight = expf(score - M);
            }

            S += weight;
            vec_mad_f16(VKQ, (const ggml_fp16_t*)v_row, weight, head_dim);
        }

        // Convert FP16 VKQ to FP32 and normalize
        if (S != 0.0f) {
            float S_inv = 1.0f / S;
            for (int d = 0; d < head_dim; d += 8) {
                float16x8_t acc = vld1q_f16((const float16_t*)VKQ + d);
                float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
                float32x4_t acc1 = vcvt_high_f32_f16(acc);
                acc0 = vmulq_n_f32(acc0, S_inv);
                acc1 = vmulq_n_f32(acc1, S_inv);
                vst1q_f32(output_head + d, acc0);
                vst1q_f32(output_head + d + 4, acc1);
            }
        }
    }
}

int main(int argc, char** argv) {
    int num_heads = 32;
    int head_dim = 128;
    int seq_len = 512;
    
    if (argc >= 4) {
        num_heads = atoi(argv[1]);
        head_dim = atoi(argv[2]);
        seq_len = atoi(argv[3]);
    }
    
    printf("F16 attention benchmark (FP16 Q): num_heads=%d, head_dim=%d, seq_len=%d\n", 
           num_heads, head_dim, seq_len);
    
    // Allocate buffers - FP16 Q like llama.cpp
    ggml_fp16_t* q = (ggml_fp16_t*)aligned_alloc(64, num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* k = (ggml_fp16_t*)aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* v = (ggml_fp16_t*)aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    float* output = (float*)aligned_alloc(64, num_heads * head_dim * sizeof(float));
    
    // Initialize with random data (like llama.cpp)
    for (int i = 0; i < num_heads * head_dim; i++) {
        q[i] = (rand() % 1000) / 1000.0;
    }
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) {
        k[i] = (rand() % 1000) / 1000.0;
        v[i] = (rand() % 1000) / 1000.0;
    }
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Warmup
    for (int w = 0; w < 5; w++) {
        our_attention_f16_q(q, k, v, output, num_heads, head_dim, seq_len, scale);
    }
    
    // Benchmark
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < NUM_ITER; iter++) {
        our_attention_f16_q(q, k, v, output, num_heads, head_dim, seq_len, scale);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    double elapsed_ms = elapsed_ns / NUM_ITER / 1e6;
    
    double flops = 2.0 * num_heads * seq_len * head_dim;
    double gflops = flops / (elapsed_ns / NUM_ITER) * 1e9 / 1e9;
    
    printf("seq_len=%5d: %6.2f ms, %7.2f GFLOPS\n", seq_len, elapsed_ms, gflops);
    
    free(q);
    free(k);
    free(v);
    free(output);
    
    return 0;
}