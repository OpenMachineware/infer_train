// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Minimal llama.cpp decode benchmark

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_tokens]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int n_decode = argc > 2 ? atoi(argv[2]) : 20;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // All layers on GPU

    llama_model * model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    printf("=== llama.cpp Decode Benchmark ===\n");
    printf("Model: %s\n", model_path);
    printf("Decode tokens: %d\n\n", n_decode);

    // Prefill
    printf("Prefill (10 tokens)...\n");
    std::vector<llama_token> prefill_tokens;
    for (int i = 1; i <= 10; i++) {
        prefill_tokens.push_back(i);
    }

    llama_batch batch = llama_batch_get_one(prefill_tokens.data(), prefill_tokens.size());
    auto t0 = std::chrono::high_resolution_clock::now();
    llama_decode(ctx, batch);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  Time: %ld ms\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

    // Decode
    printf("Decoding %d tokens...\n", n_decode);
    std::vector<llama_token> decode_tokens;
    double total_decode_ms = 0.0;

    for (int i = 0; i < n_decode; i++) {
        llama_token token = 11 + i;
        decode_tokens.clear();
        decode_tokens.push_back(token);

        batch = llama_batch_get_one(decode_tokens.data(), decode_tokens.size());
        auto start = std::chrono::high_resolution_clock::now();
        llama_decode(ctx, batch);
        auto end = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        total_decode_ms += ms;

        if (i < 5 || i >= n_decode - 5) {
            printf("  Token %d: %ld ms\n", i + 1, ms);
        }
    }

    printf("\nAverage decode time: %.1f ms\n", total_decode_ms / n_decode);
    printf("Throughput: %.1f tokens/sec\n", 1000.0 / (total_decode_ms / n_decode));

    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
