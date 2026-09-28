// Simple Q4_K Metal MV benchmark using ggml C API directly
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "llama.cpp-0.4.1/ggml/include/ggml.h"

// Q4_K block structure matching llama.cpp
#define QK_K 256

typedef struct {
    ggml_fp16_t d;      // Super-block scale
    ggml_fp16_t dmin;   // Super-block min scale
    uint8_t scales[12]; // Scales and mins
    uint8_t qs[QK_K/2]; // 4-bit quants
} block_q4_K;

void generate_q4_k_weights(block_q4_K* weights, int m, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < 12; j++) {
            weights[i].scales[j] = (i * 17 + j * 13) % 64;
        }
        for (int j = 0; j < QK_K/2; j++) {
            weights[i].qs[j] = (i * 23 + j * 17) % 256;
        }
        weights[i].d = ggml_fp32_to_fp16(0.5f + (i % 10) * 0.1f);
        weights[i].dmin = ggml_fp32_to_fp16(0.1f + (i % 5) * 0.05f);
    }
}

void generate_f32_input(float* input, int k) {
    for (int i = 0; i < k; i++) {
        input[i] = (i % 10) * 0.1f;
    }
}

int main(int argc, char** argv) {
    int m = 1024;
    int k = 4096;

    if (argc >= 3) {
        m = atoi(argv[1]);
        k = atoi(argv[2]);
    }

    printf("Testing Q4_K MV: M=%d, K=%d\n", m, k);

    int nb = k / QK_K;

    // Allocate and generate data
    block_q4_K* weights = (block_q4_K*)malloc(m * nb * sizeof(block_q4_K));
    float* input = (float*)malloc(k * sizeof(float));
    float* output = (float*)malloc(m * sizeof(float));

    generate_q4_k_weights(weights, m, k);
    generate_f32_input(input, k);

    // Use llama.cpp's reference implementation
    // For now, just test that we can compute the result correctly

    // CPU reference implementation would go here
    // We need to use the ggml API directly

    printf("Data generated successfully\n");
    printf("Weights size: %zu bytes\n", m * nb * sizeof(block_q4_K));
    printf("Input size: %zu bytes\n", k * sizeof(float));

    // Cleanup
    free(weights);
    free(input);
    free(output);

    return 0;
}
