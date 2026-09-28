// Test llama.cpp Metal MV kernel performance
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include "llama.cpp-0.4.1/ggml/include/ggml.h"
#include "llama.cpp-0.4.1/ggml/include/ggml-metal.h"

#define QK 32

typedef struct {
    ggml_fp16_t d;
    uint8_t qs[QK/2];
} block_q4_0;

void generate_q4_0_weights(block_q4_0* weights, int m, int k) {
    int nb = k / QK;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < QK/2; j++) {
            weights[i].qs[j] = (i * 17 + j * 13) % 256;
        }
        weights[i].d = ggml_fp32_to_fp16(0.5f + (i % 10) * 0.1f);
    }
}

void generate_f32_input(float* input, int k) {
    for (int i = 0; i < k; i++) {
        input[i] = (i % 10) * 0.1f;
    }
}

int main(int argc, char** argv) {
    int m = 1024;
    int k = 4096;

    if (argc >= 3) {
        m = atoi(argv[1]);
        k = atoi(argv[2]);
    }

    printf("Testing MV: M=%d, K=%d\n", m, k);

    int nb = k / QK;

    // Allocate and generate data
    std::vector<block_q4_0> weights(m * nb);
    std::vector<float> input(k);
    std::vector<float> output(m);

    generate_q4_0_weights(weights.data(), m, k);
    generate_f32_input(input.data(), k);

    // Initialize Metal backend
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "Failed to initialize Metal backend\n");
        return 1;
    }

    // Create context
    struct ggml_init_params params = {
        .mem_size = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create tensors (use Q4_0 for simplicity since it has Metal support)
    struct ggml_tensor* w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);
    struct ggml_tensor* x_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    struct ggml_tensor* y_tensor = ggml_mul_mat(ctx, w_tensor, x_tensor);

    // Allocate on GPU
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    // Copy data to GPU
    ggml_backend_tensor_set(w_tensor, weights.data(), 0, m * nb * sizeof(block_q4_0));
    ggml_backend_tensor_set(x_tensor, input.data(), 0, k * sizeof(float));

    // Create graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y_tensor);

    // Warmup
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(y_tensor, output.data(), 0, m * sizeof(float));

    // Benchmark
    int n_iter = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, graph);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = duration.count() / (double)n_iter;

    printf("llama.cpp Metal MV %dx%d: %.2f µs/iter\n", m, k, avg_us);
    printf("GFLOPS: %.2f\n", (2.0 * m * k) / (avg_us * 1e-6) / 1e9);

    // Sample output
    printf("Sample output[0]: %f\n", output[0]);

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
