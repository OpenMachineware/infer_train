// Compare Q5_K quantization error ranges
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} block_q5_k;

// Dequantize a single block to F32
void dequant_block_q5_k(block_q5_k * block, float * out) {
    // Decode d and dmin (simplified - assume f16 is stored as u16 bits)
    float d = 1.0f;  // Simplified
    float dmin = 0.5f;

    // Decode scales from 12-byte format
    uint32_t utmp[4];
    memcpy(utmp, block->scales, 12);

    const uint32_t KMASK1 = 0x3f3f3f3f;
    const uint32_t KMASK2 = 0x0f0f0f0f;
    const uint32_t KMASK3 = 0x03030303;

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    uint32_t uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    uint8_t scales[8], mins[4];
    for (int i = 0; i < 8; i++) scales[i] = (utmp[i/4] >> ((i%4)*8)) & 0x3F;
    for (int i = 0; i < 4; i++) mins[i] = (utmp[2] >> (i*8)) & 0x3F;

    // Dequantize 256 elements
    for (int iter = 0; iter < 4; iter++) {
        float scale0 = scales[iter * 2];
        float scale1 = scales[iter * 2 + 1];
        float minv = mins[iter / 2];

        for (int k = 0; k < 16; k++) {
            uint8_t qs0 = block->qs[iter * 32 + k];
            uint8_t qs1 = block->qs[iter * 32 + 16 + k];

            // 5-bit quantized values: 4 bits + 1 bit high
            float q0 = (qs0 & 0x0F) + ((block->qh[k] & 1) ? 16.0f : 0.0f);
            float q1 = (qs1 & 0x0F) + ((block->qh[16+k] & 1) ? 16.0f : 0.0f);
            float q2 = ((qs0 >> 4) & 0x0F) + ((block->qh[k] & 2) ? 16.0f : 0.0f);
            float q3 = ((qs1 >> 4) & 0x0F) + ((block->qh[16+k] & 2) ? 16.0f : 0.0f);

            // Dequantized value: d * scale * q - dmin * min
            out[iter * 64 + k] = d * scale0 * q0 - dmin * minv;
            out[iter * 64 + 16 + k] = d * scale0 * q1 - dmin * minv;
            out[iter * 64 + 32 + k] = d * scale1 * q2 - dmin * minv;
            out[iter * 64 + 48 + k] = d * scale1 * q3 - dmin * minv;
        }
    }
}

int main() {
    // Test quantization error bounds
    printf("Q5_K Quantization Error Analysis\n");
    printf("================================\n\n");

    printf("5-bit quantization: range [0, 31], step size = 1\n");
    printf("With 6-bit scales: range [0, 63]\n");
    printf("Theoretical max error: 0.5 * scale (rounding error per element)\n");
    printf("For worst case scale=63: ~31.5 per element\n");
    printf("But GEMV accumulates across 256 elements...\n\n");

    printf("For dot product of 256 elements:\n");
    printf("  Max single element error: ~31.5\n");
    printf("  If errors add up: 31.5 * 256 = 8064\n");
    printf("  With random signs: sqrt(256) * 31.5 ≈ 504\n");
    printf("  Typical: < 100\n\n");

    printf("But 1.19 error for dot product with random Y values:\n");
    printf("  This is actually quite good!\n");
    printf("  It means average element error is ~1.19/sqrt(256) ≈ 0.075\n");
    printf("  Which is much better than worst case\n\n");

    printf("However, I should verify with actual test data...\n");

    return 0;
}
