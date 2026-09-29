// llama.cpp Metal MV test - Using ggml-metal backend
// Compile: clang++ -O2 -o bench_llama_metal_ggml bench_llama_metal_ggml.mm \
//   -I../llama.cpp-0.5.0/ggml/include \
//   -L../llama.cpp-0.5.0/build-metal \
//   -lggml-metal -lggml-base -framework Metal -framework Foundation

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-metal.h"
#include "ggml-backend.h"

#define QK_K 256

typedef struct {
    uint8_t scales[12];
    uint8_t qs[128];
    uint16_t d;
    uint16_t dmin;
} block_q4_K;

uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void generate_q4k_blocks(block_q4_K *x, int nb, int seed) {
    for (int i = 0; i < nb; i++) {
        x[i].d = 15360;
        x[i].dmin = 0;
        for (int j = 0; j < 12; j++) x[i].scales[j] = ((j + 1) % 4) + 1;
        for (int j = 0; j < 128; j++) x[i].qs[j] = ((i + j + seed) & 0xFF);
    }
}

int main() {
    printf("llama.cpp Metal MV Test - Q4_K (via ggml-metal backend)\n");
    printf("========================================================\n\n");

    // Initialize ggml Metal backend
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        printf("Error: Failed to initialize Metal backend\n");
        return 1;
    }
    printf("Metal backend initialized\n");

    // Test parameters
    const int m = 4096;
    const int k = 4096;
    const int nb = m * (k / QK_K);
    const int seed = 42;

    // Create ggml context
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("Error: Failed to create ggml context\n");
        return 1;
    }

    // Create tensors
    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, k, m);
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    struct ggml_tensor * output = ggml_mul_mat(ctx, weights, input);

    // Allocate buffers
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buffer) {
        printf("Error: Failed to allocate buffer\n");
        return 1;
    }
    printf("Buffers allocated\n");

    // Generate test data
    block_q4_K* weights_data = (block_q4_K*)ggml_backend_buffer_get_base(buffer);
    generate_q4k_blocks(weights_data, nb, seed);

    float* input_data = (float*)(weights_data + nb);
    for (int i = 0; i < k; i++) input_data[i] = 1.0f;

    // Set tensor data
    ggml_backend_tensor_set(weights, weights_data, 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));

    // Create computation graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    printf("\nRunning benchmark...\n");
    double total_ns = 0;
    for (int i = 0; i < 10; i++) {
        uint64_t start = get_time_ns();
        ggml_backend_graph_compute(backend, graph);
        uint64_t end = get_time_ns();
        total_ns += (end - start);
    }

    double avg_ns = total_ns / 10.0;
    double gflops = 2.0 * m * k / (avg_ns / 1e9) / 1e9;
    printf("\nllama.cpp Metal Q4_K MV M=4096 K=4096: %.2f GFLOPS (avg %.2f ms)\n", gflops, avg_ns / 1e6);

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
