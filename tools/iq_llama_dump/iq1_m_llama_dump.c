// 直接调用 llama.cpp 的 quantize_iq1_m 函数
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint8_t qs[QK_K/8];
    uint8_t qh[QK_K/16];
    uint8_t scales[QK_K/32];
} block_iq1_m;

// llama.cpp 函数声明
void ggml_quantize_init(int type);
size_t quantize_iq1_m(const float *src, void *dst, int64_t nrows, int64_t n_per_row, const float *imatrix);

#define GGML_TYPE_IQ1_M 29

int main() {
    printf("=== 直接调用 llama.cpp IQ1_M 量化 ===\n\n");

    ggml_quantize_init(GGML_TYPE_IQ1_M);

    float test_data[QK_K] = {
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812, 0.934245, 0.814190,
        -0.321567, 0.456789, -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
        0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678, 0.567890, -0.234567,
    };

    float quant_weights[QK_K];
    for (int i = 0; i < QK_K; i++) quant_weights[i] = 1.0f;

    block_iq1_m block;
    size_t result = quantize_iq1_m(test_data, &block, 1, QK_K, quant_weights);

    printf("Result size: %zu\n", result);
    printf("Block:\n");
    printf("  qs[0:16] = ");
    for (int i = 0; i < 16; i++) printf("0x%02x ", block.qs[i]);
    printf("\n");
    printf("  qh[0:8] = ");
    for (int i = 0; i < 8; i++) printf("0x%02x ", block.qh[i]);
    printf("\n");
    printf("  scales[0:8] = ");
    for (int i = 0; i < 8; i++) printf("0x%02x ", block.scales[i]);
    printf("\n");

    return 0;
}
