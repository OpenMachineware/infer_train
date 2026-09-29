// llama.cpp GEMM template
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-metal.h"
#include "ggml-backend.h"

uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s <type>\nTypes: q2_k, q3_k, q4_k, q5_k, q6_k\n", argv[0]); return 1; }

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) { printf("Failed to init Metal\n"); return 1; }

    const int m = 4096, k = 4096, n = 64;
    ggml_type type = GGML_TYPE_Q4_K;

    if (strcmp(argv[1], "q2_k") == 0) type = GGML_TYPE_Q2_K;
    else if (strcmp(argv[1], "q3_k") == 0) type = GGML_TYPE_Q3_K;
    else if (strcmp(argv[1], "q4_k") == 0) type = GGML_TYPE_Q4_K;
    else if (strcmp(argv[1], "q5_k") == 0) type = GGML_TYPE_Q5_K;
    else if (strcmp(argv[1], "q6_k") == 0) type = GGML_TYPE_Q6_K;

    struct ggml_init_params params = { .mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, k, m);
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    float* data = (float*)ggml_backend_buffer_get_base(buffer);
    for (int i = 0; i < k * n; i++) data[i] = 1.0f;
    ggml_backend_tensor_set(input, data, 0, k * n * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    double total_ns = 0;
    for (int i = 0; i < 10; i++) {
        uint64_t start = get_time_ns();
        ggml_backend_graph_compute(backend, graph);
        uint64_t end = get_time_ns();
        total_ns += (end - start);
    }

    double gflops = 2.0 * m * k * n / (total_ns / 10.0 / 1e9) / 1e9;
    printf("%s GEMM: %.2f GFLOPS\n", argv[1], gflops);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
