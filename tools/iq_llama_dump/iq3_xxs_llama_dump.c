// 直接调用 llama.cpp 的 quantize_iq3_xxs 函数
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint8_t qs[3*(QK_K/8)];
} block_iq3_xxs;

// llama.cpp 函数声明
void ggml_quantize_init(int type);
size_t quantize_iq3_xxs(const float *src, void *dst, int64_t nrows, int64_t n_per_row, const float *imatrix);

#define GGML_TYPE_IQ3_XXS 18

int main() {
    printf("=== 直接调用 llama.cpp IQ3_XXS 量化 ===\n\n");

    ggml_quantize_init(GGML_TYPE_IQ3_XXS);

    // Test with different seed
    float test_data[QK_K];
    for (int i = 0; i < QK_K; i++) {
        test_data[i] = fmodf(i * 0.137f + 123.0f, 2.0f) - 1.0f;
    }

    float quant_weights[QK_K];
    for (int i = 0; i < QK_K; i++) quant_weights[i] = 1.0f;

    block_iq3_xxs block;
    size_t result = quantize_iq3_xxs(test_data, &block, 1, QK_K, quant_weights);

    printf("Result size: %zu\n", result);
    printf("Block:\n");
    printf("  d = 0x%04x\n", block.d);
    printf("  qs[0:96] =\n");
    for (int i = 0; i < 96; i++) {
        printf("0x%02x, ", block.qs[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }

    return 0;
}
