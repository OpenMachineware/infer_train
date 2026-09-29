// Simple llama.cpp Metal MV benchmark for Q4_K
// Compile: gcc -o test_llama_metal_mv test_llama_metal_mv.c -I../llama.cpp-0.5.0 -L../llama.cpp-0.5.0/build-metal -lggml-metal -lggml

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
        for (int j = 0; j < 12; j++) {
            x[i].scales[j] = ((j + 1) % 4) + 1;
        }
        for (int j = 0; j < 128; j++) {
            x[i].qs[j] = ((i + j + seed) & 0xFF);
        }
    }
}

int main() {
    printf("llama.cpp Metal MV Benchmark - Q4_K\n");
    printf("====================================\n\n");

    // This is a stub - actual Metal testing requires linking with ggml-metal
    printf("Note: This is a placeholder for Metal benchmark.\n");
    printf("To get accurate baseline, run llama.cpp's test-backend-ops perf mode.\n\n");

    printf("Recommended command:\n");
    printf("  cd llama.cpp-0.5.0/build-metal\n");
    printf("  ./bin/test-backend-ops perf -b MTL0 -o 'MUL_MAT.*q4_k'\n\n");

    printf("Or use the Rust benchmark for direct comparison:\n");
    printf("  cargo run --release --bin bench_unified_mv\n");

    return 0;
}
