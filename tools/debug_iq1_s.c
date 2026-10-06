// Debug IQ1_S dequantization
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define QK_K 256

// Simplified FP16 to FP32 conversion
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal
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

// Extract first 50 entries from iq1s_grid_gpu (full grid has 2048 entries)
// Each entry is 4 bytes (uint32_t) containing 8 nibbles
static const uint32_t iq1s_grid_gpu_sample[] = {
    0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101, 0x00020000,
    0x00020002, 0x00020200, 0x00020202, 0x01000101, 0x01010001, 0x01010100, 0x01010102, 0x01020101,
    0x02000000, 0x02000002, 0x02000200, 0x02000202, 0x02010101, 0x02020000, 0x02020002, 0x02020200,
    0x02020202, 0x00000110, 0x00000111, 0x00010011, 0x00010110, 0x00010112, 0x00010211, 0x00010212,
    0x00020111, 0x01000011, 0x01000112, 0x01000211, 0x01010012, 0x01010111, 0x01010212, 0x01020011,
    0x01020110, 0x01020112, 0x01020210, 0x02000111, 0x02010011, 0x02010110, 0x02010112, 0x02020111,
};

int main() {
    printf("=== IQ1_S Dequantization Debug ===\n\n");

    // Block data from dump tool
    uint16_t d = 0x2a94;
    uint8_t qs[32] = {
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
        0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
    };
    uint16_t qh[8] = {0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f, 0xf46f};

    // Convert scale
    float d_f32 = fp16_to_fp32(d);
    printf("Scale d = 0x%04x -> %f\n", d, d_f32);

    // Process first 32-element sub-block (ib32=0, il=0)
    printf("\nProcessing ib32=0, il=0:\n");

    int ib32 = 0;
    int il = 0;

    // From dequantize_iq1_s:
    const float dl = d_f32 * (2 * ((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - 0.125f : -1 + 0.125f);

    printf("  qh[0] = 0x%04x\n", qh[0]);
    printf("  (qh[0] >> 12) & 7 = %d\n", (qh[0] >> 12) & 7);
    printf("  dl = %f * (2 * %d + 1) = %f\n", d_f32, (qh[0] >> 12) & 7, dl);
    printf("  qh[0] & 0x8000 = 0x%04x\n", qh[0] & 0x8000);
    printf("  ml = %f * (-1 %s 0.125) = %f\n", dl, (qh[0] & 0x8000) ? "-" : "+", ml);

    const uint16_t h = qh[ib32] >> (6 * il);
    printf("  h = qh[0] >> 0 = 0x%04x\n", h);

    // Grid indices
    uint16_t grid_idx1 = qs[0] | ((h << 8) & 0x700);
    uint16_t grid_idx2 = qs[1] | ((h << 5) & 0x700);

    printf("  grid_idx1 = qs[0] | ((h << 8) & 0x700) = 0x%02x | 0x%03x = 0x%03x (%d)\n",
           qs[0], (h << 8) & 0x700, grid_idx1, grid_idx1);
    printf("  grid_idx2 = qs[1] | ((h << 5) & 0x700) = 0x%02x | 0x%03x = 0x%03x (%d)\n",
           qs[1], (h << 5) & 0x700, grid_idx2, grid_idx2);

    // Note: grid_idx1 = 0x69f = 1695, grid_idx2 = 0x6e5 = 1765
    // These are too large for our sample table, but let's see what the pattern looks like

    printf("\nFirst 16 dequantized values (from grid):\n");
    printf("  (Using sample grid, actual indices are %d, %d)\n", grid_idx1, grid_idx2);

    // For index 0 in sample
    uint32_t grid_val = iq1s_grid_gpu_sample[0];
    printf("  Grid[0] = 0x%08x\n", grid_val);
    for (int i = 0; i < 4; i++) {
        uint8_t nibble = (grid_val >> (i * 8)) & 0xff;
        printf("    Byte %d: 0x%02x -> nibbles: %d, %d\n", i, nibble, nibble >> 4, nibble & 0xf);
    }

    return 0;
}
