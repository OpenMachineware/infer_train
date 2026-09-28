// Simplified test: use ggml API directly
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "llama.cpp-0.4.1/ggml/include/ggml.h"

#define QK 32

void test_cpu_mv() {
    printf("Testing CPU MV performance\n");

    int m = 256, k = 4096;
    int nb = k / QK;

    // Allocate memory
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create tensors (use Q4_0 for simplicity since IQ4_NL might not be available in simple build)
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    struct ggml_tensor* y = ggml_mul_mat(ctx, w, x);

    // Initialize with dummy data
    memset(w->data, 0, ggml_nbytes(w));
    memset(x->data, 1, ggml_nbytes(x));

    // Create graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    // Warmup
    ggml_graph_compute(ctx, graph);

    // Benchmark
    int n_iter = 100;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < n_iter; i++) {
        ggml_graph_compute(ctx, graph);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
    double avg_us = elapsed / n_iter;

    printf("CPU MV %dx%d: %.2f µs/iter\n", m, k, avg_us);
    printf("GFLOPS: %.2f\n", (2.0 * m * k) / (avg_us * 1e-6) / 1e9);

    ggml_free(ctx);
}

int main() {
    printf("GGML version: %s\n", ggml_version());
    test_cpu_mv();
    return 0;
}
