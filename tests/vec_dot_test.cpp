// Test Q5_K vec_dot correctness
// Compare single block result between Rust and C++
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// FP16 conversion (GGML style)
static inline uint32_t fp32_to_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static inline float fp32_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

#define QK_K 256

struct block_q5_K {
    uint8_t qs[128];     // 4-bit quantized values
    uint8_t qh[32];      // high bits
    uint8_t scales[12];  // scales and mins encoded
    uint16_t d;          // super-block scale
    uint16_t dmin;       // super-block min scale
};

struct block_q8_K {
    float d;             // scale
    int8_t qs[256];      // quantized values
    int16_t bsums[16];   // block sums
};

float vec_dot_q5_k_q8_k_generic(const block_q5_K * x, const block_q8_K * y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    int32_t aux32[8];
    memset(aux32, 0, 8*sizeof(int32_t));

    const uint8_t * GGML_RESTRICT q4 = x->qs;
    const uint8_t * GGML_RESTRICT hm = x->qh;
    const  int8_t * GGML_RESTRICT q8 = y->qs;

    int8_t * GGML_RESTRICT a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < QK_K/64; ++j) {
        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
        for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
        a += 32; m <<= 1;
        for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
        for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
        a += 32; m <<= 1;
        q4 += 32;
    }
    memcpy(utmp, x->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < QK_K/16; ++j) sumi += y->bsums[j] * mins[j/2];

    a = aux8;
    int is = 0;
    for (int j = 0; j < QK_K/32; ++j) {
        int32_t scale = scales[is++];
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
    }

    const float d = fp16_to_fp32(x->d) * y->d;
    float sumf = 0;
    for (int l = 0; l < 8; ++l) sumf += d * aux32[l];
    const float dmin = fp16_to_fp32(x->dmin) * y->d;
    sumf -= dmin * sumi;

    return sumf;
}

int main() {
    // Test with specific values
    block_q5_K q5 = {};
    block_q8_K q8 = {};

    // Initialize with test values
    q5.d = 0x3c00;  // 1.0 in FP16
    q5.dmin = 0x0000;  // 0.0
    memset(q5.scales, 1, 12);  // All scales = 1
    memset(q5.qs, 0x88, 128);  // Mid-range values
    memset(q5.qh, 0x00, 32);   // No high bits

    q8.d = 1.0f;
    for (int i = 0; i < 256; i++) q8.qs[i] = 1;  // All ones
    for (int i = 0; i < 16; i++) q8.bsums[i] = 16;  // Block sum = 16 * 1 = 16

    float result = vec_dot_q5_k_q8_k_generic(&q5, &q8);
    printf("Test 1: All ones\n");
    printf("Result: %.6f\n", result);

    // Expected: each 4-bit value = 8, so 8 * 1 * scale * d
    // With 256 elements, sum = 256 * 8 = 2048
    // But we need to account for the actual calculation

    return 0;
}
