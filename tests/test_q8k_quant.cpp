// Test Q8_K quantization
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

const int QK_K = 256;

struct BlockQ8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

void quantize_row_q8_k(const float* x, BlockQ8_K* y, int n) {
    int nb = n / QK_K;

    for (int i = 0; i < nb; i++) {
        const float* x_block = &x[i * QK_K];

        // Find max
        float amax = 0.0f;
        float max = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float abs_val = std::abs(x_block[j]);
            if (abs_val > amax) {
                amax = abs_val;
                max = x_block[j];
            }
        }

        if (amax == 0.0f) {
            y[i].d = 0.0f;
            for (int j = 0; j < 256; j++) y[i].qs[j] = 0;
            for (int j = 0; j < 16; j++) y[i].bsums[j] = 0;
            continue;
        }

        float iscale = -127.0f / max;

        for (int j = 0; j < QK_K; j++) {
            int v = std::round(iscale * x_block[j]);
            y[i].qs[j] = std::max(-128, std::min(127, v));
        }

        for (int j = 0; j < 16; j++) {
            int sum = 0;
            for (int k = 0; k < 16; k++) {
                sum += y[i].qs[j * 16 + k];
            }
            y[i].bsums[j] = sum;
        }

        y[i].d = 1.0f / iscale;
    }
}

int main() {
    // Test with some values
    float x[256];
    for (int i = 0; i < 256; i++) {
        x[i] = 0.1f * (i % 10 - 5);  // Values from -0.5 to 0.4
    }

    BlockQ8_K y;
    quantize_row_q8_k(x, &y, 256);

    printf("Input range: [%.2f, %.2f]\n", -0.5f, 0.4f);
    printf("Q8_K.d = %.6f\n", y.d);
    printf("Q8_K.qs[0..8] = [");
    for (int i = 0; i < 8; i++) {
        printf("%d", y.qs[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
    printf("Q8_K.bsums[0..4] = [");
    for (int i = 0; i < 4; i++) {
        printf("%d", y.bsums[i]);
        if (i < 3) printf(", ");
    }
    printf("]\n");

    // Verify: dequantize and compare
    float max_err = 0.0f;
    for (int i = 0; i < 256; i++) {
        float dq = y.d * y.qs[i];
        float err = std::abs(dq - x[i]);
        max_err = std::max(max_err, err);
    }
    printf("Max dequant error: %.6f\n", max_err);

    return 0;
}
