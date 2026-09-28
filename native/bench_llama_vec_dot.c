// Benchmark llama.cpp vec_dot functions for K-quants
// Compile: see compile.sh in this directory

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arm_neon.h>

#define QK_K 256
#define QK8_K 256

// Block structures matching llama.cpp
typedef struct {
    uint8_t scales[3*QK_K/64];
    uint8_t qs[QK_K/2];
    uint8_t d;
    uint8_t dmin;
} block_q4_K;

typedef struct {
    uint8_t qs[QK_K/2];
    uint8_t qh[QK_K/8];
    uint8_t scales[3*QK_K/64];
    uint8_t d;
    uint8_t dmin;
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    uint8_t d;
    uint8_t dmin;
} block_q6_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    uint8_t scales[QK_K/16];
    uint8_t d;
    uint8_t dmin;
} block_q2_K;

typedef struct {
    uint8_t qh[QK_K/4];
    uint8_t qs[QK_K/2];
    uint8_t scales[12];
    uint8_t d;
} block_q3_K;

typedef struct {
    int8_t qs[QK_K];
    float d;
} block_q8_K;

// Timing helper
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// CPU warmup
void warmup_cpu(void) {
    volatile double sum = 0.0;
    for (int i = 0; i < 50000000; i++) {
        sum = sum * 0.9999 + 1.0;
    }
    usleep(500000); // 500ms
}

// Generate test data
void generate_q4_K_data(block_q4_K *x, block_q8_K *y, int nb) {
    for (int i = 0; i < nb; i++) {
        memset(&x[i], 0, sizeof(block_q4_K));
        memset(&y[i], 0, sizeof(block_q8_K));

        x[i].d = 128;
        x[i].dmin = 0;
        for (int j = 0; j < 12; j++) {
            x[i].scales[j] = 1;
        }
        for (int j = 0; j < QK_K/2; j++) {
            x[i].qs[j] = (i + j) & 0xFF;
        }

        y[i].d = 1.0f;
        for (int j = 0; j < QK_K; j++) {
            y[i].qs[j] = j % 64 - 32;
        }
    }
}

void generate_q5_K_data(block_q5_K *x, block_q8_K *y, int nb) {
    for (int i = 0; i < nb; i++) {
        memset(&x[i], 0, sizeof(block_q5_K));
        memset(&y[i], 0, sizeof(block_q8_K));

        x[i].d = 128;
        x[i].dmin = 0;
        for (int j = 0; j < 12; j++) {
            x[i].scales[j] = 1;
        }
        for (int j = 0; j < QK_K/2; j++) {
            x[i].qs[j] = (i + j) & 0xFF;
        }
        for (int j = 0; j < QK_K/8; j++) {
            x[i].qh[j] = (i * j) & 0xFF;
        }

        y[i].d = 1.0f;
        for (int j = 0; j < QK_K; j++) {
            y[i].qs[j] = j % 64 - 32;
        }
    }
}

void generate_q6_K_data(block_q6_K *x, block_q8_K *y, int nb) {
    for (int i = 0; i < nb; i++) {
        memset(&x[i], 0, sizeof(block_q6_K));
        memset(&y[i], 0, sizeof(block_q8_K));

        x[i].d = 128;
        x[i].dmin = 0;
        for (int j = 0; j < QK_K/16; j++) {
            x[i].scales[j] = 1;
        }
        for (int j = 0; j < QK_K/2; j++) {
            x[i].ql[j] = (i + j) & 0xFF;
        }
        for (int j = 0; j < QK_K/4; j++) {
            x[i].qh[j] = (i * j) & 0xFF;
        }

        y[i].d = 1.0f;
        for (int j = 0; j < QK_K; j++) {
            y[i].qs[j] = j % 64 - 32;
        }
    }
}

// External llama.cpp functions
extern void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q5_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

int main() {
    printf("llama.cpp K-quant vec_dot Benchmark\n");
    printf("===================================\n\n");

    warmup_cpu();

    const int nb = 16; // 16 blocks = 4096 elements
    const int n = nb * QK_K;
    const int iterations = 10000;

    float result = 0.0f;

    // Q4_K benchmark
    block_q4_K *x4 = malloc(nb * sizeof(block_q4_K));
    block_q8_K *y4 = malloc(nb * sizeof(block_q8_K));
    generate_q4_K_data(x4, y4, nb);

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_q8_K(n, &result, 0, x4, 0, y4, 0, 1);
    }
    uint64_t end = get_time_ns();

    double time_us = (end - start) / 1000.0 / iterations;
    double flops = 2.0 * n * iterations;
    double gflops = flops / ((end - start) / 1e9) / 1e9;

    printf("Q4_K: %.3f us/iter, %.1f GFLOPS, result=%.3f\n", time_us, gflops, result);

    free(x4);
    free(y4);

    // Q5_K benchmark
    block_q5_K *x5 = malloc(nb * sizeof(block_q5_K));
    block_q8_K *y5 = malloc(nb * sizeof(block_q8_K));
    generate_q5_K_data(x5, y5, nb);

    start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q5_K_q8_K(n, &result, 0, x5, 0, y5, 0, 1);
    }
    end = get_time_ns();

    time_us = (end - start) / 1000.0 / iterations;
    flops = 2.0 * n * iterations;
    gflops = flops / ((end - start) / 1e9) / 1e9;

    printf("Q5_K: %.3f us/iter, %.1f GFLOPS, result=%.3f\n", time_us, gflops, result);

    free(x5);
    free(y5);

    // Q6_K benchmark
    block_q6_K *x6 = malloc(nb * sizeof(block_q6_K));
    block_q8_K *y6 = malloc(nb * sizeof(block_q8_K));
    generate_q6_K_data(x6, y6, nb);

    start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q6_K_q8_K(n, &result, 0, x6, 0, y6, 0, 1);
    }
    end = get_time_ns();

    time_us = (end - start) / 1000.0 / iterations;
    flops = 2.0 * n * iterations;
    gflops = flops / ((end - start) / 1e9) / 1e9;

    printf("Q6_K: %.3f us/ %.1f GFLOPS, result=%.3f\n", time_us, gflops, result);

    free(x6);
    free(y6);

    return 0;
}
