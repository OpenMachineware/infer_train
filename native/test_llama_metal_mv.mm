// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// llama.cpp Metal MV test - Q4_K
// Compile: clang++ -O2 -o test_llama_metal_mv test_llama_metal_mv.mm \
//   -I../llama.cpp-0.5.0 -L../llama.cpp-0.5.0/build-metal -lggml-metal -lggml-base -framework Metal

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

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
        x[i].d = 15360;  // FP16 1.0
        x[i].dmin = 0;
        for (int j = 0; j < 12; j++) {
            x[i].scales[j] = ((j + 1) % 4) + 1;
        }
        for (int j = 0; j < 128; j++) {
            x[i].qs[j] = ((i + j + seed) & 0xFF);
        }
    }
}

int main() {
    printf("llama.cpp Metal MV Test - Q4_K\n");
    printf("===============================\n\n");

    const int m = 4096;
    const int k = 4096;
    const int nb = m * (k / QK_K);
    const int seed = 42;

    // Generate test data
    std::vector<block_q4_K> weights(nb);
    generate_q4k_blocks(weights.data(), nb, seed);

    std::vector<float> input(k, 1.0f);
    std::vector<float> output(m);

    // TODO: Call llama.cpp Metal API
    // For now, this is a placeholder showing the interface we need

    printf("Test data prepared: M=%d, K=%d, blocks=%d\n", m, k, nb);
    printf("\nTo get actual performance:\n");
    printf("1. Use ggml_backend_metal_init() to create Metal backend\n");
    printf("2. Create tensors with ggml_new_tensor_2d()\n");
    printf("3. Compute with ggml_backend_graph_compute()\n\n");

    printf("Alternative: Run llama.cpp's built-in benchmark:\n");
    printf("  cd llama.cpp-0.5.0/build-metal\n");
    printf("  ./bin/llama-bench -m q4_k -p 4096 -n 1\n");

    return 0;
}
