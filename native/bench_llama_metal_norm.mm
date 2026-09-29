// llama.cpp Metal RMS Norm test - Using ggml-metal backend
// Compile: clang++ -O2 -o bench_llama_metal_norm bench_llama_metal_norm.mm \
//   -I../llama.cpp-0.5.0/ggml/include \
//   -L../llama.cpp-0.5.0/build-metal/bin \
//   -lggml-metal -lggml-cpu -lggml-base -lggml \
//   -framework Metal -framework Foundation

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
    printf("llama.cpp Metal RMS Norm Test (via ggml-metal backend)\n");
    printf("=======================================================\n\n");

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        printf("Error: Failed to initialize Metal backend\n");
        return 1;
    }

    const int size = 4096;

    struct ggml_init_params params = { .mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context * ctx = ggml_init(params);

    // Create tensors
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    struct ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    struct ggml_tensor * output = ggml_rms_norm(ctx, input, 1e-5f);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    // Set data
    float* data = (float*)ggml_backend_buffer_get_base(buffer);
    for (int i = 0; i < size; i++) { data[i] = 1.0f; }
    for (int i = 0; i < size; i++) { data[size + i] = 1.0f; }

    ggml_backend_tensor_set(input, data, 0, size * sizeof(float));
    ggml_backend_tensor_set(weight, data + size, 0, size * sizeof(float));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    printf("Running benchmark...\n");
    double total_ns = 0;
    for (int i = 0; i < 10; i++) {
        uint64_t start = get_time_ns();
        ggml_backend_graph_compute(backend, graph);
        uint64_t end = get_time_ns();
        total_ns += (end - start);
    }

    double avg_us = total_ns / 10.0 / 1000.0;
    printf("\nllama.cpp Metal RMS Norm size=4096: %.2f µs\n", avg_us);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
