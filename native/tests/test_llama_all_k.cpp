// Test llama.cpp Metal Q2_K/Q5_K MV kernel performance
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include "llama.cpp-0.4.1/ggml/include/ggml.h"
#include "llama.cpp-0.4.1/ggml/include/ggml-metal.h"

#define QK_K 256

typedef struct {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[16];
    uint8_t qs[QK_K/4];
} block_q2_K;

typedef struct {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

void generate_q2_k_weights(block_q2_K* weights, int m, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < 16; j++) {
            weights[i].scales[j] = (i * 17 + j * 13) % 64;
        }
        for (int j = 0; j < QK_K/4; j++) {
            weights[i].qs[j] = (i * 23 + j * 17) % 256;
        }
        weights[i].d = ggml_fp32_to_fp16(0.5f + (i % 10) * 0.1f);
        weights[i].dmin = ggml_fp32_to_fp16(0.1f + (i % 5) * 0.05f);
    }
}

void generate_q5_k_weights(block_q5_K* weights, int m, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < 12; j++) {
            weights[i].scales[j] = (i * 17 + j * 13) % 64;
        }
        for (int j = 0; j < QK_K/8; j++) {
            weights[i].qh[j] = (i * 23 + j * 17) % 256;
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

void test_q2_k(int m, int k) {
    int nb = k / QK_K;

    std::vector<block_q2_K> weights(m * nb);
    std::vector<float> input(k);
    std::vector<float> output(m);

    generate_q2_k_weights(weights.data(), m, k);
    generate_f32_input(input.data(), k);

    ggml_backend_t backend = ggml_backend_metal_init();
    struct ggml_init_params params = {
        .mem_size = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_K, k, m);
    struct ggml_tensor* x_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    struct ggml_tensor* y_tensor = ggml_mul_mat(ctx, w_tensor, x_tensor);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
    ggml_backend_tensor_set(w_tensor, weights.data(), 0, m * nb * sizeof(block_q2_K));
    ggml_backend_tensor_set(x_tensor, input.data(), 0, k * sizeof(float));

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y_tensor);

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(y_tensor, output.data(), 0, m * sizeof(float));

    int n_iter = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, graph);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = duration.count() / (double)n_iter;

    printf("llama Q2_K %dx%d: %.2f µs, %.2f GFLOPS\n", m, k, avg_us, (2.0 * m * k) / (avg_us * 1e-6) / 1e9);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

void test_q5_k(int m, int k) {
    int nb = k / QK_K;

    std::vector<block_q5_K> weights(m * nb);
    std::vector<float> input(k);
    std::vector<float> output(m);

    generate_q5_k_weights(weights.data(), m, k);
    generate_f32_input(input.data(), k);

    ggml_backend_t backend = ggml_backend_metal_init();
    struct ggml_init_params params = {
        .mem_size = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, k, m);
    struct ggml_tensor* x_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    struct ggml_tensor* y_tensor = ggml_mul_mat(ctx, w_tensor, x_tensor);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
    ggml_backend_tensor_set(w_tensor, weights.data(), 0, m * nb * sizeof(block_q5_K));
    ggml_backend_tensor_set(x_tensor, input.data(), 0, k * sizeof(float));

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y_tensor);

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(y_tensor, output.data(), 0, m * sizeof(float));

    int n_iter = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, graph);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = duration.count() / (double)n_iter;

    printf("llama Q5_K %dx%d: %.2f µs, %.2f GFLOPS\n", m, k, avg_us, (2.0 * m * k) / (avg_us * 1e-6) / 1e9);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <Q2_K|Q5_K> <M>\n", argv[0]);
        return 1;
    }

    const char* type = argv[1];
    int m = atoi(argv[2]);
    int k = 4096;

    if (strcmp(type, "Q2_K") == 0) {
        test_q2_k(m, k);
    } else if (strcmp(type, "Q5_K") == 0) {
        test_q5_k(m, k);
    } else {
        printf("Unknown type: %s\n", type);
        return 1;
    }

    return 0;
}
