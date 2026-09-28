// Simple wrapper to expose llama.cpp vec_dot functions for benchmarking
// Compile with:
// clang -c -O3 -mcpu=apple-m1 -I../llama.cpp-0.5.0/ggml/src -I../llama.cpp-0.5.0/ggml/src/ggml-cpu -I../llama.cpp-0.5.0/ggml/src/ggml-cpu/arch/arm -I../llama.cpp-0.5.0/ggml/include llama_vec_dot_wrapper.c -o llama_vec_dot_wrapper.o
// ar rcs libllama_vec_dot.a llama_vec_dot_wrapper.o

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../llama.cpp-0.5.0/ggml/src/ggml-cpu/quants.h"
#include "../../llama.cpp-0.5.0/ggml/src/ggml-cpu/ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

// Expose llama.cpp vec_dot functions with simple interface
// These will be called from Rust for direct performance comparison

void llama_vec_dot_q4_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q4_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q5_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q5_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q6_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q6_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q2_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q2_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

void llama_vec_dot_q3_K_q8_K(
    int n,
    float * s,
    const void * vx,
    const void * vy
) {
    ggml_vec_dot_q3_K_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}
