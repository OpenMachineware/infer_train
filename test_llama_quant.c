// Test llama.cpp IQ2_XXS quantization directly
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Minimal includes from llama.cpp
typedef struct {
    unsigned short d; // delta
    unsigned short qs[32]; // packed grid indices and signs
} block_iq2_xxs;

#define QK_K 256
#define KMAX_Q 3

// External functions from llama.cpp
extern void quantize_row_iq2_xxs_ref(const float * GGML_RESTRICT x, block_iq2_xxs * GGML_RESTRICT y, int64_t k);
extern void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

int main() {
    // Create test data
    float test_data[QK_K];
    for (int i = 0; i < QK_K; i++) {
        test_data[i] = (i % 2 == 0) ? 0.5f : 1.0f;
    }

    printf("Original data (first 32):\n");
    for (int i = 0; i < 32; i++) {
        printf("%.3f ", test_data[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }

    // Quantize
    block_iq2_xxs block;
    quantize_row_iq2_xxs_ref(test_data, &block, QK_K);

    printf("\nQuantized block:\n");
    printf("d = %f\n", (float)block.d / 31.0f);

    // Dequantize
    float dequant[QK_K];
    dequantize_row_iq2_xxs(&block, dequant, QK_K);

    printf("\nDequantized data (first 32):\n");
    for (int i = 0; i < 32; i++) {
        printf("%.3f ", dequant[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }

    printf("\nComparison:\n");
    float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < QK_K; i++) {
        float diff = fabsf(test_data[i] - dequant[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    printf("Max diff = %.6f at index %d\n", max_diff, max_idx);
    printf("Original[%d] = %.6f, Dequant[%d] = %.6f\n",
           max_idx, test_data[max_idx], max_idx, dequant[max_idx]);

    return 0;
}