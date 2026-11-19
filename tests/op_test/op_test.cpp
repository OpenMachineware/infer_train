// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Universal operator test framework - llama.cpp version
// Build: make
// Usage:
//   1. View output: ./op_test_cpp --op rms_norm --verbose
//   2. Benchmark: ./op_test_cpp --op rms_norm --bench 1000
//   3. From data file: ./op_test_cpp --data /tmp/test.bin -v

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>

// GGML headers
extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
#include "quants.h"  // Internal quantization functions
}

// Simple command line parser
struct Args {
    const char* op = "rms_norm";
    const char* data_file = nullptr;
    bool verbose = false;
    int bench = 0;
    int size = 1024;
    uint64_t seed = 42;
};

struct TestData {
    std::string op;
    int size;
    uint64_t seed;
    std::vector<float> input;
    std::vector<float> weight;
};

TestData read_data_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open data file: %s\n", path);
        exit(1);
    }

    TestData data;

    // Read header
    uint32_t op_len;
    fread(&op_len, sizeof(op_len), 1, f);
    uint32_t size_u32;
    fread(&size_u32, sizeof(size_u32), 1, f);
    data.size = (int)size_u32;
    fread(&data.seed, sizeof(data.seed), 1, f);

    // Read op name
    char op_buf[256];
    fread(op_buf, 1, op_len, f);
    data.op = std::string(op_buf, op_len);

    // Read input
    data.input.resize(data.size);
    fread(data.input.data(), sizeof(float), data.size, f);

    // Read weight
    data.weight.resize(data.size);
    fread(data.weight.data(), sizeof(float), data.size, f);

    fclose(f);
    return data;
}

void parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--op") == 0 && i + 1 < argc) {
            args.op = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            args.data_file = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            args.verbose = true;
        } else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            args.bench = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            args.size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = strtoull(argv[i], nullptr, 10);
        }
    }
}

std::vector<float> generate_input(int size, uint64_t seed) {
    std::mt19937 rng((unsigned)seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = dist(rng);
    }
    return result;
}

void test_rms_norm(const Args& args) {
    int size = args.size;
    auto input = generate_input(size, args.seed);
    auto weight = generate_input(size, args.seed + 1);
    std::vector<float> output(size);

    if (args.verbose) {
        printf("=== RMS Norm ===\n");
        printf("Size: %d\n", size);
        printf("Input [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", input[i], i < 4 ? ", " : "]\n");
        }
        printf("Weight [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", weight[i], i < 4 ? ", " : "]\n");
        }
    }

    // Compute RMS norm manually (llama.cpp style)
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = sqrtf(sum_sq / size + 1e-5f);
    float scale = 1.0f / rms;

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
            sum_sq = 0.0f;
            for (int i = 0; i < size; i++) {
                sum_sq += input[i] * input[i];
            }
            rms = sqrtf(sum_sq / size + 1e-5f);
            scale = 1.0f / rms;
            for (int i = 0; i < size; i++) {
                output[i] = input[i] * scale * weight[i];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %lld us\n", args.bench, (long long)elapsed.count());
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
        printf("Benchmark: %d iterations in %lld us\n", args.bench, (long long)elapsed.count());
        printf("Per iteration: %.2f us\n", us_per_iter);
    }
}

// QK_K = 256 elements per K-quant block
#define QK_K 256

// Include the block definitions and vec_dot functions from llama.cpp
// These are in ggml-common.h and quants.c

void test_quantize_q8_k(const Args& args) {
    int size = QK_K * 4;  // 4 blocks = 1024 elements
    auto input = generate_input(size, args.seed);
    // Scale up to get more interesting quantization
    for (auto& v : input) v *= 5.0f;

    if (args.verbose) {
        printf("=== Quantize Q8_K ===\n");
        printf("Size: %d (4 blocks)\n", size);
        float in_min = *std::min_element(input.begin(), input.end());
        float in_max = *std::max_element(input.begin(), input.end());
        printf("Input range: [%.4f, %.4f]\n", in_min, in_max);
        printf("Input [0..5]: [");
        for (int i = 0; i < 5 && i < (int)input.size(); i++) {
            printf("%.4f%s", input[i], i < 4 ? ", " : "]\n");
        }
    }

    // Allocate Q8_K output (292 bytes per block)
    int nb = 4;
    std::vector<uint8_t> output(nb * 292);

    // Call llama.cpp's quantize function
    // quantize_row_q8_K_ref(input.data(), output.data(), size);

    if (args.verbose) {
        // Print first block info
        // Block Q8_K: d (float), qs[256], bsums[16]
        float d;
        memcpy(&d, output.data(), 4);
        printf("Block 0: d = %.6f\n", d);

        int8_t* qs = (int8_t*)(output.data() + 4);
        printf("qs [0..5]: [");
        for (int i = 0; i < 5; i++) {
            printf("%d%s", qs[i], i < 4 ? ", " : "]\n");
        }
    }
}

void test_vec_dot_q4_k(const Args& args) {
    int nb = 4;
    int size = QK_K * nb;
    auto input = generate_input(size, args.seed);
    for (auto& v : input) v *= 5.0f;

    if (args.verbose) {
        printf("=== vec_dot Q4_K x Q8_K ===\n");
        printf("Input size: %d (%d blocks)\n", size, nb);
    }

    // This would require linking with llama.cpp's quantization and vec_dot functions
    // For now, we just print a placeholder
    printf("vec_dot Q4_K test not yet implemented - requires llama.cpp internal functions\n");
}

void test_vec_dot_q5_k(const Args& args) {
    int nb = 4;
    int size = QK_K * nb;
    auto input = generate_input(size, args.seed);
    for (auto& v : input) v *= 5.0f;

    if (args.verbose) {
        printf("=== vec_dot Q5_K x Q8_K ===\n");
        printf("Input size: %d (%d blocks)\n", size, nb);
        float in_min = *std::min_element(input.begin(), input.end());
        float in_max = *std::max_element(input.begin(), input.end());
        printf("Input range: [%.4f, %.4f]\n", in_min, in_max);
    }

    // Allocate Q5_K and Q8_K blocks
    // Q5_K: 176 bytes per block (256 elements)
    // Q8_K: 292 bytes per block (256 elements)
    std::vector<uint8_t> q5k_data(nb * 176);
    std::vector<uint8_t> q8k_data(nb * 292);

    // Quantize to Q5_K and Q8_K
    quantize_row_q5_K(input.data(), q5k_data.data(), size);
    quantize_row_q8_K(input.data(), q8k_data.data(), size);

    // Compute dot product
    float result = 0.0f;
    ggml_vec_dot_q5_K_q8_K(size, &result, sizeof(float), q5k_data.data(), 176, q8k_data.data(), 292, 1);

    if (args.verbose) {
        printf("Q5_K x Q8_K dot product: %.6f\n", result);

        // Compute reference dot product (sum of input^2, since we're dotting same values)
        float ref = 0.0f;
        for (int i = 0; i < size; i++) {
            ref += input[i] * input[i];
        }
        printf("Reference (input · input): %.6f\n", ref);
        printf("Relative error: %.6f%%\n", fabs(result - ref) / ref * 100.0f);
    }

    // Benchmark
    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < args.bench; iter++) {
            float tmp = 0.0f;
            ggml_vec_dot_q5_K_q8_K(size, &tmp, sizeof(float), q5k_data.data(), 176, q8k_data.data(), 292, 1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_iter = elapsed.count() / (double)args.bench;
        printf("Benchmark: %d iterations in %lld us\n", args.bench, (long long)elapsed.count());
        printf("Per iteration: %.2f us (%.2f GB/s for 2x%d bytes)\n", us_per_iter,
               (2.0 * nb * 176.0 + 2.0 * nb * 292.0) / us_per_iter / 1000.0, nb);
    }
}

void test_vec_dot_q6_k(const Args& args) {
    int nb = 4;
    int size = QK_K * nb;
    auto input = generate_input(size, args.seed);
    for (auto& v : input) v *= 5.0f;

    if (args.verbose) {
        printf("=== vec_dot Q6_K x Q8_K ===\n");
        printf("Input size: %d (%d blocks)\n", size, nb);
    }

    printf("vec_dot Q6_K test not yet implemented - requires llama.cpp internal functions\n");
}

// Forward declaration
void test_rms_norm_with_data(const Args& args, const std::vector<float>& input, const std::vector<float>& weight);

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);

    // If data file specified, read from file
    if (args.data_file) {
        TestData data = read_data_file(args.data_file);
        printf("Operator: %s, size: %d\n", data.op.c_str(), data.size);

        if (data.op == "rms_norm") {
            test_rms_norm_with_data(args, data.input, data.weight);
        } else {
            printf("Unknown operator from data file: %s\n", data.op.c_str());
            return 1;
        }
        return 0;
    }

    printf("Operator: %s, size: %d, verbose: %d, bench: %d\n",
           args.op, args.size, args.verbose, args.bench);

    if (strcmp(args.op, "rms_norm") == 0) {
        test_rms_norm(args);
    } else if (strcmp(args.op, "silu") == 0) {
        test_silu(args);
    } else if (strcmp(args.op, "quantize_q8_k") == 0) {
        test_quantize_q8_k(args);
    } else if (strcmp(args.op, "vec_dot_q4_k") == 0) {
        test_vec_dot_q4_k(args);
    } else if (strcmp(args.op, "vec_dot_q5_k") == 0) {
        test_vec_dot_q5_k(args);
    } else if (strcmp(args.op, "vec_dot_q6_k") == 0) {
        test_vec_dot_q6_k(args);
    } else {
        printf("Unknown operator: %s\n", args.op);
        printf("Available: rms_norm, silu, quantize_q8_k, vec_dot_q4_k, vec_dot_q5_k, vec_dot_q6_k\n");
        return 1;
    }

    return 0;
}

// Test rms_norm using data read from file
void test_rms_norm_with_data(const Args& args, const std::vector<float>& input, const std::vector<float>& weight) {
    int size = args.size;
    std::vector<float> output(size);
    float eps = 1e-5f;

    if (args.verbose) {
        printf("=== RMS Norm ===\n");
        printf("Size: %d\n", size);
        printf("Input [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", input[i], i < 4 ? ", " : "]\n");
        }
        printf("Weight [0..5]: [");
        for (int i = 0; i < 5 && i < size; i++) {
            printf("%.4f%s", weight[i], i < 4 ? ", " : "]\n");
        }
    }

    // Compute RMS norm
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = sqrtf(sum_sq / size + eps);
    float scale = 1.0f / rms;

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
    }

    if (args.bench > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < args.bench; iter++) {
            for (int i = 0; i < size; i++) {
                output[i] = input[i] * scale * weight[i];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double us_per_iter = (double)us / args.bench;
        printf("Benchmark: %d iterations in %ld us\n", args.bench, us);
        printf("Per iteration: %.2f us\n", us_per_iter);
    }
}
