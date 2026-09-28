#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

typedef uint16_t ggml_fp16_t;
#define GGML_FP16_TO_FP32(x) (((union { uint16_t h; float f; }) { .h = (x) }).f)
#include "fp16_attention.c"

#define NUM_ITER 100
#define WARMUP 20

int main() {
    int num_heads = 32, head_dim = 128, seq_len = 1024;

    ggml_fp16_t* q = aligned_alloc(64, num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* k = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    ggml_fp16_t* v = aligned_alloc(64, seq_len * num_heads * head_dim * sizeof(ggml_fp16_t));
    float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
    ggml_fp16_t* vkq = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));

    for (int i = 0; i < num_heads * head_dim; i++) q[i] = (rand() % 1000) / 1000.0;
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) { k[i] = (rand() % 1000) / 1000.0; v[i] = (rand() % 1000) / 1000.0; }

    float scale = 1.0f / sqrtf((float)head_dim);

    // Warmup
    for (int w = 0; w < WARMUP; w++) {
        decode_attention_f16_native(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq);
    }

    // Measure with per-iteration timing
    double total_ns = 0;
    double min_ns = 1e18, max_ns = 0;

    for (int iter = 0; iter < NUM_ITER; iter++) {
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        decode_attention_f16_native(q, k, v, output, num_heads, head_dim, seq_len, scale, vkq);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        double ns = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
        total_ns += ns;
        if (ns < min_ns) min_ns = ns;
        if (ns > max_ns) max_ns = ns;
    }

    printf("Average: %.2f ms (%.1f min, %.1f max)\n", total_ns / NUM_ITER / 1e6, min_ns / 1e6, max_ns / 1e6);
    printf("GFLOPS:  %.2f\n", 2.0 * num_heads * seq_len * head_dim / (total_ns / NUM_ITER) * 1e9 / 1e9);

    free(q); free(k); free(v); free(output); free(vkq);
    return 0;
}
