// Generate Q4_K test data using llama.cpp's quantization
// This ensures the data format is correct

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml/include/ggml.h"

int main() {
    printf("=== Generate Q4_K Test Data ===\n");

    // Test dimensions
    const int M = 128;  // Small for quick test
    const int K = 4096;
    const int N = 32;

    // Allocate float data
    float *src = (float*)malloc(K * sizeof(float));
    float *input = (float*)malloc(K * N * sizeof(float));

    // Fill with simple pattern
    for (int i = 0; i < K; i++) {
        src[i] = (i % 10) * 0.1f;  // 0.0 to 0.9
    }
    for (int i = 0; i < K * N; i++) {
        input[i] = (i % 10) * 0.1f;
    }

    // Quantize to Q4_K
    const int nb = K / 256;
    block_q4_K *q4k = (block_q4_K*)malloc(M * nb * sizeof(block_q4_K));

    for (int row = 0; row < M; row++) {
        quantize_row_q4_K_ref(src, q4k + row * nb, K);
    }

    printf("Generated %d Q4_K blocks\n", M * nb);

    // Save to files
    FILE *f_weights = fopen("test_q4k_weights.bin", "wb");
    fwrite(q4k, sizeof(block_q4_K), M * nb, f_weights);
    fclose(f_weights);

    // Save input as FP16
    FILE *f_input = fopen("test_q4k_input.bin", "wb");
    for (int i = 0; i < K * N; i++) {
        ggml_fp16_t fp16 = GGML_FP32_TO_FP16(input[i]);
        fwrite(&fp16, sizeof(ggml_fp16_t), 1, f_input);
    }
    fclose(f_input);

    printf("Saved to test_q4k_weights.bin and test_q4k_input.bin\n");

    // Compute reference output
    float *output = (float*)malloc(M * N * sizeof(float));

    // Dequantize and compute
    float *deq = (float*)malloc(K * sizeof(float));
    for (int row = 0; row < M; row++) {
        dequantize_row_q4_K(q4k + row * nb, deq, K);

        for (int col = 0; col < N; col++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += deq[k] * input[col * K + k];  // Column-major B
            }
            output[row * N + col] = sum;
        }
    }

    // Save reference output
    FILE *f_output = fopen("test_q4k_output.bin", "wb");
    fwrite(output, sizeof(float), M * N, f_output);
    fclose(f_output);

    printf("Saved reference output to test_q4k_output.bin\n");
    printf("First 5 values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", output[i]);
    }
    printf("\n");

    free(src);
    free(input);
    free(q4k);
    free(deq);
    free(output);

    return 0;
}
