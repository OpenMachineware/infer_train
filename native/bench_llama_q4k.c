// llama.cpp K-quant vec_dot benchmark
// 直接调用llama.cpp函数，无FFI开销

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arm_neon.h>

#define QK_K 256

// Block structures matching llama.cpp
typedef struct {
    uint8_t scales[12];
    uint8_t qs[128];
    uint16_t d;
    uint16_t dmin;
} block_q4_K;

typedef struct {
    int8_t qs[256];
    float d;
    int16_t bsums[16];
} block_q8_K;

// Timing helper
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// CPU warmup - multi-round to stabilize frequency
void warmup_cpu(void) {
    for (int round = 0; round < 3; round++) {
        volatile double sum = 0.0;
        for (int i = 0; i < 100000000; i++) {
            sum = sum * 0.9999 + 1.0;
        }
        if (round < 2) {
            usleep(2000000); // 2 seconds between rounds
        }
    }
}

// Generate test data with seed
void generate_q4k_blocks(block_q4_K *x, int nb, int seed) {
    for (int i = 0; i < nb; i++) {
        x[i].d = 15360; // FP16 1.0
        x[i].dmin = 0;
        for (int j = 0; j < 12; j++) {
            x[i].scales[j] = ((j + 1) % 4) + 1;
        }
        for (int j = 0; j < 128; j++) {
            x[i].qs[j] = (i + j + seed) & 0xFF;
        }
    }
}

void generate_q8k_blocks(block_q8_K *y, int nb, int seed) {
    for (int i = 0; i < nb; i++) {
        y[i].d = 1.0f + (i % 10) * 0.1f;
        for (int j = 0; j < 256; j++) {
            y[i].qs[j] = ((j % 64) - 32);
        }
        for (int j = 0; j < 16; j++) {
            y[i].bsums[j] = (i + j + seed) % 256;
        }
    }
}

// External llama.cpp functions
extern void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

int main() {
    printf("llama.cpp Q4_K vec_dot Benchmark\n");
    printf("================================\n\n");

    printf("Warming up CPU...\n");
    warmup_cpu();
    printf("Warmup complete\n\n");

    const int seed = 42; // Same seed as Rust benchmark
    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int iterations[] = {50000, 50000, 50000, 10000, 10000};

    printf("Format | K     | GFLOPS  | Time (ms)\n");
    printf("-------|-------|---------|----------\n");

    for (int s = 0; s < num_sizes; s++) {
        int k = sizes[s];
        int nb = k / QK_K;
        int iter = iterations[s];

        block_q4_K *x = malloc(nb * sizeof(block_q4_K));
        block_q8_K *y = malloc(nb * sizeof(block_q8_K));

        generate_q4k_blocks(x, nb, seed);
        generate_q8k_blocks(y, nb, seed);

        float result = 0.0f;

        uint64_t start = get_time_ns();
        for (int i = 0; i < iter; i++) {
            ggml_vec_dot_q4_K_q8_K(k, &result, 0, x, 0, y, 0, 1);
        }
        uint64_t end = get_time_ns();

        double time_ns = (end - start);
        double flops = 2.0 * k * iter;
        double gflops = flops / (time_ns / 1e9) / 1e9;
        double time_ms = time_ns / 1e6 / iter;

        printf("Q4_K   | %5d | %7.1f | %.4f\n", k, gflops, time_ms);

        free(x);
        free(y);
    }

    printf("\nNote: Test data generated with seed=%d (same as Rust benchmark)\n", seed);
    return 0;
}
