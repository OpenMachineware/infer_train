// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// Universal operator test framework - llama.cpp version
// Build: make
// Usage:
//   1. View output: ./op_test_cpp --op rms_norm --verbose
//   2. Benchmark: ./op_test_cpp --op rms_norm --bench 1000

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

// GGML headers
extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "quants.h"
}

// Simple command line parser
struct Args {
    const char* op = "rms_norm";
    bool verbose = false;
    int bench = 0;
    int size = 1024;
    uint64_t seed = 42;
};

void parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--op") == 0 && i + 1 < argc) {
            args.op = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            args.verbose = true;
        } else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            args.bench = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            args.size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = strtoull(argv[++i], nullptr, 10);
        }
    }
}

std::vector<float> generate_input(int size, uint64_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = dis(gen);
    }
    return result;
}

void test_rms_norm(const Args& args) {
    int size = args.size;
    auto input = generate_input(size, args.seed);
    auto weight = generate_input(size, args.seed + 1);
    std::vector<float> output(size);

    // Compute RMS
    float sum_sq = 0;
    for (int i = 0; i < size; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = sqrtf(sum_sq / size + 1e-5f);
    float scale = 1.0f / rms;

    if (args.verbose) {
        printf("=== RMS Norm ===\n");
        printf("Size: %d\n", size);
        printf("Input RMS: %.6f\n", rms);
        printf("Scale: %.6f\n", scale);
        printf("Input [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", input[i], i < 4 ? ", " : "]\n");
        }
        printf("Weight [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", weight[i], i < 4 ? ", " : "]\n");
        }
    }

    // Execute RMS norm
    const float eps = 1e-5f;
    ggml_vec_norm_f32(size, &scale, input.data());
    scale = 1.0f / (scale + eps);

    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale * weight[i];
    }

    if (args.verbose) {
        printf("Output [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", output[i], i < 4 ? ", " : "]\n");
        }

        float out_min = *std::min_element(output.begin(), output.end());
        float out_max = *std::max_element(output.begin(), output.end());
        printf("Output range: [%.4f, %.4f]\n", out_min, out_max);

        float out_sum_sq = 0;
        for (int i = 0; i < size; i++) {
            out_sum_sq += output[i] * output[i];
        }
        float out_rms = sqrtf(out_sum_sq / size);
        printf("Output RMS (should be ~1.0 with weight=1): %.4f\n", out_rms);
    }

    // Benchmark
    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < args.bench; iter++) {
            ggml_vec_norm_f32(size, &scale, input.data());
            scale = 1.0f / (scale + eps);
            for (int i = 0; i < size; i++) {
                output[i] = input[i] * scale * weight[i];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %ld us\n", args.bench, elapsed.count());
        printf("Per iteration: %.2f us (%.2f GB/s)\n", us_per_iter,
               (size * 4.0 * 3.0) / us_per_iter / 1000.0);
    }
}

void test_silu(const Args& args) {
    int size = args.size;
    auto input = generate_input(size, args.seed);
    std::vector<float> output(size);

    if (args.verbose) {
        printf("=== SiLU Activation ===\n");
        printf("Size: %d\n", size);
        printf("Input [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", input[i], i < 4 ? ", " : "]\n");
        }
    }

    // SiLU: x * sigmoid(x)
    auto silu = [](float x) -> float {
        return x / (1.0f + expf(-x));
    };

    for (int i = 0; i < size; i++) {
        output[i] = silu(input[i]);
    }

    if (args.verbose) {
        printf("Output [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", output[i], i < 4 ? ", " : "]\n");
        }
        float out_min = *std::min_element(output.begin(), output.end());
        float out_max = *std::max_element(output.begin(), output.end());
        printf("Output range: [%.4f, %.4f]\n", out_min, out_max);
    }

    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < args.bench; iter++) {
            for (int i = 0; i < size; i++) {
                output[i] = silu(input[i]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %ld us\n", args.bench, elapsed.count());
        printf("Per iteration: %.2f us\n", us_per_iter);
    }
}

void test_quantize_q8_k(const Args& args) {
    const int QK_K = 256;
    int size = args.size;
    int nb = (size + QK_K - 1) / QK_K;
    int total_size = nb * QK_K;

    auto input = generate_input(total_size, args.seed);
    // Expand range
    for (auto& v : input) {
        v *= 10.0f;
    }

    std::vector<block_q8_K> output(nb);

    if (args.verbose) {
        printf("=== Quantize Q8_K ===\n");
        printf("Size: %d (%d blocks)\n", total_size, nb);
        float in_min = *std::min_element(input.begin(), input.end());
        float in_max = *std::max_element(input.begin(), input.end());
        printf("Input range: [%.4f, %.4f]\n", in_min, in_max);
        printf("Input [0..5]: [");
        for (int i = 0; i < 5; i++) {
            printf("%.4f%s", input[i], i < 4 ? ", " : "]\n");
        }
    }

    // Execute quantization
    quantize_row_q8_K_ref(input.data(), output.data(), total_size);

    if (args.verbose) {
        printf("\nFirst block:\n");
        printf("  d = %.6f\n", output[0].d);
        printf("  qs [0..10]: [");
        for (int i = 0; i < 10; i++) {
            printf("%d%s", (int)output[0].qs[i], i < 9 ? ", " : "]\n");
        }
        printf("  bsums [0..4]: [");
        for (int i = 0; i < 4; i++) {
            printf("%d%s", (int)output[0].bsums[i], i < 3 ? ", " : "]\n");
        }

        // Verification
        printf("\nVerification:\n");
        for (int i = 0; i < 5; i++) {
            float deq = output[0].d * output[0].qs[i];
            printf("  qs[%d] = %d, dequant = %.4f, original = %.4f\n",
                   i, (int)output[0].qs[i], deq, input[i]);
        }
    }

    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < args.bench; iter++) {
            quantize_row_q8_K_ref(input.data(), output.data(), total_size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %ld us\n", args.bench, elapsed.count());
        printf("Per iteration: %.2f us (%.2f GB/s)\n", us_per_iter,
               (total_size * 4.0) / us_per_iter / 1000.0);
    }
}

void test_vec_dot_q5_k(const Args& args) {
    const int QK_K = 256;
    int nb = 4;
    int total_size = nb * QK_K;

    auto input = generate_input(total_size, args.seed);
    for (auto& v : input) {
        v *= 5.0f;
    }

    // Quantize input to Q8_K
    std::vector<block_q8_K> input_q8k(nb);
    quantize_row_q8_K_ref(input.data(), input_q8k.data(), total_size);

    // Create dummy Q5_K weights
    std::vector<block_q5_K> weights(nb);
    memset(weights.data(), 0, nb * sizeof(block_q5_K));

    for (auto& b : weights) {
        b.d = GGML_FP32_TO_FP16(0.001f);
        b.dmin = GGML_FP32_TO_FP16(0.01f);
        for (int i = 0; i < 128; i++) {
            b.qs[i] = 0x88;
        }
        for (int i = 0; i < 12; i++) {
            b.scales[i] = 32;
        }
    }

    if (args.verbose) {
        printf("=== vec_dot Q5_K x Q8_K ===\n");
        printf("Input size: %d (%d blocks)\n", total_size, nb);
        printf("Input [0..5]: [");
        for (int i = 0; i < 5; i++) {
            printf("%.4f%s", input[i], i < 4 ? ", " : "]\n");
        }
        printf("Q8_K block 0: d = %.6f\n", input_q8k[0].d);
        printf("Q8_K block 0 qs [0..5]: [");
        for (int i = 0; i < 5; i++) {
            printf("%d%s", (int)input_q8k[0].qs[i], i < 4 ? ", " : "]\n");
        }
        printf("Q5_K block 0: d = %.6f, dmin = %.6f\n",
               GGML_FP16_TO_FP32(weights[0].d), GGML_FP16_TO_FP32(weights[0].dmin));
    }

    // Compute dot product
    float result;
    ggml_vec_dot_q5_K_q8_k(total_size, &result, 0, weights.data(), 0, input_q8k.data(), 0, 1);

    if (args.verbose) {
        printf("\nResult: %.6f\n", result);
    }

    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = 0;
        for (int iter = 0; iter < args.bench; iter++) {
            ggml_vec_dot_q5_K_q8_k(total_size, &result, 0, weights.data(), 0, input_q8k.data(), 0, 1);
            sum += result;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %ld us\n", args.bench, elapsed.count());
        printf("Per iteration: %.2f us (%.2f GOPS)\n", us_per_iter,
               (total_size * 2.0) / us_per_iter / 1000.0);
        printf("(sum = %f to prevent optimization)\n", sum);
    }
}

void test_vec_dot_q6_k(const Args& args) {
    const int QK_K = 256;
    int nb = 4;
    int total_size = nb * QK_K;

    auto input = generate_input(total_size, args.seed);
    for (auto& v : input) {
        v *= 5.0f;
    }

    std::vector<block_q8_K> input_q8k(nb);
    quantize_row_q8_K_ref(input.data(), input_q8k.data(), total_size);

    std::vector<block_q6_K> weights(nb);
    memset(weights.data(), 0, nb * sizeof(block_q6_K));

    for (auto& b : weights) {
        b.d = GGML_FP32_TO_FP16(0.001f);
        for (int i = 0; i < 128; i++) {
            b.ql[i] = 32;
        }
    }

    if (args.verbose) {
        printf("=== vec_dot Q6_K x Q8_K ===\n");
        printf("Input size: %d (%d blocks)\n", total_size, nb);
        printf("Q8_K block 0: d = %.6f\n", input_q8k[0].d);
        printf("Q6_K block 0: d = %.6f\n", GGML_FP16_TO_FP32(weights[0].d));
    }

    float result;
    ggml_vec_dot_q6_K_q8_k(total_size, &result, 0, weights.data(), 0, input_q8k.data(), 0, 1);

    if (args.verbose) {
        printf("Result: %.6f\n", result);
    }

    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = 0;
        for (int iter = 0; iter < args.bench; iter++) {
            ggml_vec_dot_q6_K_q8_k(total_size, &result, 0, weights.data(), 0, input_q8k.data(), 0, 1);
            sum += result;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %ld us\n", args.bench, elapsed.count());
        printf("Per iteration: %.2f us\n", us_per_iter);
    }
}

void test_vec_dot_q4_k(const Args& args) {
    const int QK_K = 256;
    int nb = 4;
    int total_size = nb * QK_K;

    auto input = generate_input(total_size, args.seed);
    for (auto& v : input) {
        v *= 5.0f;
    }

    std::vector<block_q8_K> input_q8k(nb);
    quantize_row_q8_K_ref(input.data(), input_q8k.data(), total_size);

    std::vector<block_q4_K> weights(nb);
    memset(weights.data(), 0, nb * sizeof(block_q4_K));

    for (auto& b : weights) {
        b.d = GGML_FP32_TO_FP16(0.001f);
        b.dmin = GGML_FP32_TO_FP16(0.01f);
        for (int i = 0; i < 144; i++) {
            b.qs[i] = 0x88;
        }
        for (int i = 0; i < 12; i++) {
            b.scales[i] = 32;
        }
    }

    if (args.verbose) {
        printf("=== vec_dot Q4_K x Q8_K ===\n");
        printf("Input size: %d (%d blocks)\n", total_size, nb);
        printf("Q8_K block 0: d = %.6f\n", input_q8k[0].d);
        printf("Q4_K block 0: d = %.6f\n", GGML_FP16_TO_FP32(weights[0].d));
    }

    float result;
    ggml_vec_dot_q4_K_q8_k(total_size, &result, 0, weights.data(), 0, input_q8k.data(), 0, 1);

    if (args.verbose) {
        printf("Result: %.6f\n", result);
    }

    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = 0;
        for (int iter = 0; iter < args.bench; iter++) {
            ggml_vec_dot_q4_K_q8_k(total_size, &result, 0, weights.data(), 0, input_q8k.data(), 0, 1);
            sum += result;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %ld us\n", args.bench, elapsed.count());
        printf("Per iteration: %.2f us\n", us_per_iter);
    }
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);

    if (strcmp(args.op, "rms_norm") == 0) {
        test_rms_norm(args);
    } else if (strcmp(args.op, "silu") == 0) {
        test_silu(args);
    } else if (strcmp(args.op, "quantize_q8_k") == 0) {
        test_quantize_q8_k(args);
    } else if (strcmp(args.op, "vec_dot_q5_k") == 0) {
        test_vec_dot_q5_k(args);
    } else if (strcmp(args.op, "vec_dot_q6_k") == 0) {
        test_vec_dot_q6_k(args);
    } else if (strcmp(args.op, "vec_dot_q4_k") == 0) {
        test_vec_dot_q4_k(args);
    } else {
        fprintf(stderr, "Unknown operator: %s\n", args.op);
        fprintf(stderr, "Supported: rms_norm, silu, quantize_q8_k, vec_dot_q5_k, vec_dot_q6_k, vec_dot_q4_k\n");
        return 1;
    }

    return 0;
}
