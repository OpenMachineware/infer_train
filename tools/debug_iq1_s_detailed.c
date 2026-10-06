// Debug IQ1_S dequantization step-by-step
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

// FP16 to FP32
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;

    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        exponent = 1;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3ff;
        exponent += 112;
    } else if (exponent == 31) {
        return sign ? -INFINITY : INFINITY;
    } else {
        exponent += 112;
    }

    uint32_t f = (sign << 31) | (exponent << 23) | (mantissa << 13);
    return *(float*)&f;
}

// Load complete grid from file
static uint32_t iq1s_grid_gpu[2048];

int load_grid(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return -1;

    char line[1024];
    int idx = 0;

    while (fgets(line, sizeof(line), f) && idx < 2048) {
        char* p = line;
        while (p && idx < 2048) {
            char* hex = strstr(p, "0x");
            if (!hex) break;
            iq1s_grid_gpu[idx++] = strtoul(hex, NULL, 16);
            p = hex + 10;
        }
    }

    fclose(f);
    return idx;
}

int main() {
    printf("=== IQ1_S Dequantization Step-by-Step ===\n\n");

    // Load grid
    int n = load_grid("/Users/jia/Desktop/infer_train/shaders/iq_grid_tables.h");
    printf("Loaded %d grid entries\n\n", n);

    if (n < 2048) {
        printf("ERROR: Grid incomplete!\n");
        return 1;
    }

    // Block data from llama.cpp dump
    uint16_t d = 0x2a94;
    uint8_t qs[32] = {
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
    };
    uint16_t qh[8] = {0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f};

    float d_f32 = fp16_to_fp32(d);
    printf("Scale d = 0x%04x -> %.10f\n\n", d, d_f32);

    // Dequantize first 32 elements (ib32=0, il=0 and il=1)
    printf("Dequantizing first 32 elements:\n");

    float dequant_values[32];

    for (int il = 0; il < 2; il++) {
        int ib32 = 0;

        // From dequantize_iq1_s:
        float dl = d_f32 * (2 * ((qh[ib32] >> 12) & 7) + 1);
        float ml = dl * (qh[ib32] & 0x8000 ? -1 - 0.125f : -1 + 0.125f);
        uint16_t h = qh[ib32] >> (6 * il);

        uint8_t qs0 = qs[2*il + 0];
        uint8_t qs1 = qs[2*il + 1];

        uint16_t grid_idx1 = qs0 | ((h << 8) & 0x700);
        uint16_t grid_idx2 = qs1 | ((h << 5) & 0x700);

        printf("  il=%d: h=0x%04x, grid_idx1=%d (0x%03x), grid_idx2=%d (0x%03x)\n",
               il, h, grid_idx1, grid_idx1, grid_idx2, grid_idx2);

        // Grid values
        uint32_t grid1_val = iq1s_grid_gpu[grid_idx1];
        uint32_t grid2_val = iq1s_grid_gpu[grid_idx2];

        printf("    grid1[%d] = 0x%08x, grid2[%d] = 0x%08x\n",
               grid_idx1, grid1_val, grid_idx2, grid2_val);

        // Dequantize 16 values
        for (int i = 0; i < 4; i++) {
            uint8_t byte1 = (grid1_val >> (i * 8)) & 0xff;
            uint8_t byte2 = (grid2_val >> (i * 8)) & 0xff;

            // Low nibble
            float val1_low = dl * (byte1 & 0xf) + ml;
            float val1_high = dl * (byte1 >> 4) + ml;
            float val2_low = dl * (byte2 & 0xf) + ml;
            float val2_high = dl * (byte2 >> 4) + ml;

            int base = il * 16 + i * 4;
            dequant_values[base + 0] = val1_low;
            dequant_values[base + 1] = val1_high;
            dequant_values[base + 2] = val2_low;
            dequant_values[base + 3] = val2_high;

            printf("    [%d-%d]: %.6f, %.6f, %.6f, %.6f\n",
                   base, base+3, val1_low, val1_high, val2_low, val2_high);
        }
    }

    printf("\nFirst 32 dequantized values:\n");
    for (int i = 0; i < 32; i++) {
        printf("  [%2d]: %.6f\n", i, dequant_values[i]);
    }

    // Input test data
    float input[QK_K] = {
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812, 0.934245, 0.814190,
        -0.321567, 0.456789, -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
        0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678, 0.567890, -0.234567,
    };

    // Compute dot product
    printf("\nDot product of first 32 elements:\n");
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        sum += dequant_values[i] * input[i];
        printf("  [%2d]: %.6f * %.6f = %.6f, sum=%.6f\n",
               i, dequant_values[i], input[i], dequant_values[i] * input[i], sum);
    }

    printf("\nDot product of first 32: %.6f\n", sum);

    return 0;
}
