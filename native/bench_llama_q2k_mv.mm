// llama.cpp Q2_K MV test
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ggml.h"
#include "ggml-metal.h"
#include "ggml-backend.h"

uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main() {
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) { printf("Failed to init Metal\n"); return 1; }

    const int m = 4096, k = 4096;

    struct ggml_init_params params = { .mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_K, k, m);
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weights, input);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    // Set data
    float* data = (float*)ggml_backend_buffer_get_base(buffer);
    for (int i = 0; i < k; i++) data[i] = 1.0f;
    ggml_backend_tensor_set(input, data, 0, k * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    double total_ns = 0;
    for (int i = 0; i < 10; i++) {
        uint64_t start = get_time_ns();
        ggml_backend_graph_compute(backend, graph);
        uint64_t end = get_time_ns();
        total_ns += (end - start);
    }

    double gflops = 2.0 * m * k / (total_ns / 10.0 / 1e9) / 1e9;
    printf("Q2_K MV: %.2f GFLOPS\n", gflops);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
