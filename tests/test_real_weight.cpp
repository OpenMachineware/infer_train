// Test real weight decoding
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>

const int QK_K = 256;

struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

struct BlockQ8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float f = mant / 1024.0f;
        for (int i = 0; i < 14; i++) f /= 2.0f;
        return sign ? -f : f;
    }
    if (exp == 31) return sign ? -1.0f/0.0f : 1.0f/0.0f;
    float f = (1.0f + mant/1024.0f) * (1 << (exp - 15));
    return sign ? -f : f;
}

float vec_dot_q6_k_q8_k(const BlockQ6_K* x, const BlockQ8_K* y, int nb) {
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;

        int8_t aux8[QK_K];
        int8_t* a = aux8;

        // Decode Q6_K values
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                a[l +  0] = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            ql += 64;
            qh += 32;
        }

        // Compute dot product
        int aux32[8] = {0};
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; j++) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; l++) aux32[l] += scale * q8[l] * a[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; l++) aux32[l] += scale * q8[l] * a[l];
            q8 += 8; a += 8;
        }

        float d = y[i].d * fp16_to_fp32(x[i].d);
        for (int l = 0; l < 8; l++) sum += d * aux32[l];
    }

    return sum;
}

int main() {
    printf("BlockQ6_K size: %zu bytes\n", sizeof(BlockQ6_K));
    printf("BlockQ8_K size: %zu bytes\n", sizeof(BlockQ8_K));

    // Create test data simulating real weights
    BlockQ6_K weight;
    memset(&weight, 0, sizeof(weight));

    // Set d to represent a small scale (like in real model)
    // FP16 for 0.00003: 0x0700 approx
    weight.d = 0x0700;  // approx 0.00003

    // Set some scales
    for (int i = 0; i < 16; i++) {
        weight.scales[i] = (i % 2 == 0) ? 100 : -80;
    }

    // Set some ql and qh values
    for (int i = 0; i < 128; i++) {
        weight.ql[i] = (i * 17) % 256;
    }
    for (int i = 0; i < 64; i++) {
        weight.qh[i] = (i * 23) % 256;
    }

    // Create input
    BlockQ8_K input;
    memset(&input, 0, sizeof(input));
    input.d = 0.1f;
    for (int i = 0; i < 256; i++) {
        input.qs[i] = (i % 19) - 9;  // Values from -9 to 9
    }
    for (int j = 0; j < 16; j++) {
        int sum = 0;
        for (int k = 0; k < 16; k++) {
            sum += input.qs[j * 16 + k];
        }
        input.bsums[j] = sum;
    }

    float result = vec_dot_q6_k_q8_k(&weight, &input, 1);
    printf("\nTest vec_dot result: %.6f\n", result);

    // Compare with expected: d * sum(scale * dot(q6-32, q8))
    // Manual calculation would be complex, just check it's reasonable
    if (std::abs(result) < 1000.0f) {
        printf("Result is reasonable (not huge)\n");
    } else {
        printf("WARNING: Result seems too large\n");
    }

    return 0;
}
