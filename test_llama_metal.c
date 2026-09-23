// Test llama.cpp Metal MV performance
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../llama.cpp-0.4.1/ggml/include/ggml.h"
#include "../llama.cpp-0.4.1/ggml/include/ggml-metal.h"

#define QK 32

typedef struct {
    ggml_fp16_t d;
    uint8_t qs[16];
} block_iq4_nl;

typedef struct {
    ggml_fp16_t d;
    int8_t qs[32];
} block_q8_0;

void generate_iq4_nl_weights(block_iq4_nl* weights, int m, int k) {
    int nb = k / QK;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < 16; j++) {
            weights[i].qs[j] = (i * 17 + j * 13) % 256;
        }
        weights[i].d = ggml_fp32_to_fp16(0.5f + (i % 10) * 0.1f);
    }
}

void generate_q8_0_input(block_q8_0* input, int k) {
    int nb = k / QK;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < 32; j++) {
            input[i].qs[j] = (i * 23 + j * 7) % 256 - 128;
        }
        input[i].d = ggml_fp32_to_fp16(1.0f + (i % 10) * 0.1f);
    }
}

double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char** argv) {
    int m = 256;
    int k = 4096;

    if (argc >= 3) {
        m = atoi(argv[1]);
        k = atoi(argv[2]);
    }

    printf("Testing MV: M=%d, K=%d\n", m, k);

    int nb = k / QK;

    // Allocate memory
    block_iq4_nl* weights = (block_iq4_nl*)malloc(m * nb * sizeof(block_iq4_nl));
    block_q8_0* input = (block_q8_0*)malloc(nb * sizeof(block_q8_0));
    float* output = (float*)malloc(m * sizeof(float));

    generate_iq4_nl_weights(weights, m, k);
    generate_q8_0_input(input, k);

    // Initialize Metal
    ggml_backend_t backend = ggml_backend_metal_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to initialize Metal backend\n");
        return 1;
    }

    // Create context
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create tensors
    struct ggml_tensor* w_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, k, m);
    struct ggml_tensor* x_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, k);
    struct ggml_tensor* y_tensor = ggml_mul_mat(ctx, w_tensor, x_tensor);

    // Allocate on GPU
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_metal_buffer_type());

    // Copy data to GPU
    ggml_backend_tensor_set(w_tensor, weights, 0, m * nb * sizeof(block_iq4_nl));
    ggml_backend_tensor_set(x_tensor, input, 0, nb * sizeof(block_q8_0));

    // Create graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y_tensor);

    // Warmup
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_tensor_get(y_tensor, output, 0, m * sizeof(float));

    // Benchmark
    int n_iter = 100;
    double start = get_time_us();

    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, graph);
    }

    double end = get_time_us();
    double avg_us = (end - start) / n_iter;

    printf("llama.cpp Metal MV %dx%d: %.2f µs/iter\n", m, k, avg_us);
    printf("GFLOPS: %.2f\n", (2.0 * m * k) / (avg_us * 1e-6) / 1e9);

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    free(weights);
    free(input);
    free(output);

    return 0;
}
