// C++ wrapper for llama.cpp flash attention
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdio>

// llama.cpp headers
extern "C" {
#include "ggml.h"
#include "ggml-metal.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
}

// Simple benchmark interface
extern "C" {

struct FlashAttnResult {
    double time_us;
    int status;
    char error_msg[256];
};

// One-time initialization
static ggml_backend_t g_backend = nullptr;
static ggml_context * g_ctx = nullptr;

int init_llama_backend() {
    if (g_backend) return 0;

    g_backend = ggml_backend_metal_init();
    if (!g_backend) {
        fprintf(stderr, "Failed to init Metal backend\n");
        return -1;
    }

    struct ggml_init_params params = {
        256*1024*1024, // mem_size
        nullptr,       // mem_buffer
        true,          // no_alloc
    };
    g_ctx = ggml_init(params);
    if (!g_ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        return -2;
    }

    return 0;
}

ggml_type get_ggml_type(const char * format) {
    if (strcmp(format, "f16") == 0) return GGML_TYPE_F16;
    if (strcmp(format, "bf16") == 0) return GGML_TYPE_BF16;
    if (strcmp(format, "q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(format, "q4_1") == 0) return GGML_TYPE_Q4_1;
    if (strcmp(format, "q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(format, "q5_1") == 0) return GGML_TYPE_Q5_1;
    if (strcmp(format, "q8_0") == 0) return GGML_TYPE_Q8_0;
    if (strcmp(format, "q2_K") == 0 || strcmp(format, "q2_k") == 0) return GGML_TYPE_Q2_K;
    if (strcmp(format, "q3_K") == 0 || strcmp(format, "q3_k") == 0) return GGML_TYPE_Q3_K;
    if (strcmp(format, "q4_K") == 0 || strcmp(format, "q4_k") == 0) return GGML_TYPE_Q4_K;
    if (strcmp(format, "q5_K") == 0 || strcmp(format, "q5_k") == 0) return GGML_TYPE_Q5_K;
    if (strcmp(format, "q6_K") == 0 || strcmp(format, "q6_k") == 0) return GGML_TYPE_Q6_K;
    return GGML_TYPE_F16;
}

FlashAttnResult run_llama_flash_attn(
    const char * format,
    const float * q_data,
    const void * k_data,
    const void * v_data,
    float * out_data,
    int num_heads,
    int head_dim,
    int seq_k,
    int rounds
) {
    FlashAttnResult result = {0, 0, ""};

    if (init_llama_backend() != 0) {
        result.status = -1;
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to init Metal backend");
        return result;
    }

    ggml_type kv_type = get_ggml_type(format);

    // Create tensors (decode: seq_q = 1)
    // Q: [head_dim, 1, num_heads, 1] (ne0, ne1, ne2, ne3)
    // K: [head_dim, seq_k, 1, 1]     (num_kv_heads = 1 for simplicity)
    // V: [head_dim, seq_k, 1, 1]
    ggml_tensor * q = ggml_new_tensor_4d(g_ctx, GGML_TYPE_F32, head_dim, 1, num_heads, 1);

    // For flash attention, K/V need ne[2] to divide Q's ne[2]
    // We use ne[2]=1 for K/V (single KV head), which divides any num_heads
    ggml_tensor * k = ggml_new_tensor_4d(g_ctx, kv_type, head_dim, seq_k, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(g_ctx, kv_type, head_dim, seq_k, 1, 1);

    if (!q || !k || !v) {
        result.status = -2;
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create tensors");
        return result;
    }

    // Create graph
    ggml_cgraph * gf = ggml_new_graph(g_ctx);

    // Flash attention
    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor * out = ggml_flash_attn_ext(g_ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);

    if (!out) {
        result.status = -3;
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create flash attention op");
        return result;
    }

    ggml_build_forward_expand(gf, out);

    // Allocate buffers
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(g_ctx, ggml_backend_get_default_buffer_type(g_backend));

    // Copy data to device
    ggml_backend_tensor_set(q, q_data, 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data, 0, ggml_nbytes(v));

    // Warmup
    if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
        result.status = -4;
        snprintf(result.error_msg, sizeof(result.error_msg), "Graph compute failed");
        ggml_backend_buffer_free(buf);
        return result;
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rounds; i++) {
        ggml_backend_graph_compute(g_backend, gf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // Get output
    ggml_backend_tensor_get(out, out_data, 0, ggml_nbytes(out));

    result.time_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / rounds;

    // Cleanup - keep context alive for reuse
    ggml_backend_buffer_free(buf);

    return result;
}

void cleanup_llama_backend() {
    if (g_ctx) {
        ggml_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_backend) {
        ggml_backend_free(g_backend);
        g_backend = nullptr;
    }
}

} // extern "C"
