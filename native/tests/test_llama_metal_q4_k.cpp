// Test llama.cpp Metal Q4_K MV kernel performance
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include "llama.cpp-0.4.1/ggml/include/ggml.h"
#include "llama.cpp-0.4.1/ggml/include/ggml-metal.h"

#define QK_K 256

// Q4_K block structure matching llama.cpp
typedef struct {
    ggml_fp16_t d;      // Super-block scale
    ggml_fp16_t dmin;   // Super-block min scale
    uint8_t scales[12]; // Scales and mins
    uint8_t qs[QK_K/2]; // 4-bit quants
} block_q4_K;

void generate_q4_k_weights(block_q4_K* weights, int m, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < 12; j++) {
            weights[i].scales[j] = (i * 17 + j * 13) % 64;
        }
        for (int j = 0; j < QK_K/2; j++) {
            weights[i].qs[j] = (i * 23 + j * 17) % 256;
        }
        weights[i].d = ggml_fp32_to_fp16(0.5f + (i % 10) * 0.1f);
        weights[i].dmin = ggml_fp32_to_fp16(0.1f + (i % 5) * 0.05f);
    }
}

void generate_f32_input(float* input, int k) {
    for (int i = 0; i < k; i++) {
        input[i] = (i % 10) * 0.1f;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <M> <K>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);

    printf("Testing Q4_K MV: M=%d, K=%d\n", m, k);

    int nb = k / QK_K;

    // Allocate and generate data
    std::vector<block_q4_K> weights(m * nb);
    std::vector<float> input(k);
    std::vector<float> output(m);

    generate_q4_k_weights(weights.data(), m, k);
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

    // Create tensors
    struct ggml_tensor* w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, k, m);
    struct ggml_tensor* x_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    struct ggml_tensor* y_tensor = ggml_mul_mat(ctx, w_tensor, x_tensor);

    // Allocate on GPU
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    // Copy data to GPU
    ggml_backend_tensor_set(w_tensor, weights.data(), 0, m * nb * sizeof(block_q4_K));
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

    printf("llama.cpp Metal Q4_K MV %dx%d: %.2f µs/iter\n", m, k, avg_us);
    printf("GFLOPS: %.2f\n", (2.0 * m * k) / (avg_us * 1e-6) / 1e9);

    // Sample output
    printf("Sample output[0]: %f\n", output[0]);

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
