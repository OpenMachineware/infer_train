// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// llama.cpp Metal Q4_K MV - for Instruments Time Profiler
// Compile: clang++ -O3 -framework Metal -framework Foundation -I../llama.cpp-0.5.0/ggml/include -L../llama.cpp-0.5.0/build/ggml/src -lggml -lggml-metal -o llama_instrumented llama_instrumented.cpp
// Run: Instruments -t "Time Profiler" ./llama_instrumented

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <thread>
#include <chrono>

#define QK_K 256

// Q4_K block structure matching llama.cpp
typedef struct {
    uint16_t d;      // Super-block scale
    uint16_t dmin;   // Super-block min scale
    uint8_t scales[12]; // Scales and mins
    uint8_t qs[QK_K/2]; // 4-bit quants
} block_q4_K;

void generate_q4_k_weights(block_q4_K* weights, int m, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < 12; j++) {
            weights[i].scales[j] = (i * 17 + j * 13) % 64;
        }
        for (int j = 0; j < QK_K/2; j++) {
            weights[i].qs[j] = (i * 23 + j * 17) % 256;
        }
        // FP16
        weights[i].d = 0x3F00 | ((i % 10) & 0xF);
        weights[i].dmin = 0x0000;
    }
}

void generate_f32_input(float* input, int k) {
    for (int i = 0; i < k; i++) {
        input[i] = (i % 10) * 0.1f;
    }
}

int main() {
    int m = 4096;
    int k = 4096;
    int iterations = 10000;

    printf("llama.cpp Metal Q4_K MV - Instruments Test\n");
    printf("M=%d, K=%d, iterations=%d\n", m, k, iterations);
    printf("\nStarting in 3 seconds...\n");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    printf("Starting...\n\n");

    int nb = k / QK_K;

    // Allocate and generate data
    std::vector<block_q4_K> weights(m * nb);
    std::vector<float> input(k);
    std::vector<float> output(m);

    generate_q4_k_weights(weights.data(), m, k);
    generate_f32_input(input.data(), k);

    // Use llama.cpp Metal API directly
    // This requires linking against ggml-metal library
    // For now, use a simple Metal compute shader directly

    // Create Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Failed to create Metal device\n");
        return 1;
    }

    // Create command queue
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Load precompiled Metal library
    NSString* shaderPath = @"../llama.cpp-0.5.0/build/ggml/src/ggml-metal/ggml-metal.metal";
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:nil];
    if (!shaderSource) {
        fprintf(stderr, "Failed to load Metal shaders\n");
        return 1;
    }

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:nil];
    if (!library) {
        fprintf(stderr, "Failed to compile Metal library\n");
        return 1;
    }

    // Get kernel function
    id<MTLFunction> kernel = [library newFunctionWithName:@"kernel_mul_mv_q4_K_f32"];
    if (!kernel) {
        fprintf(stderr, "Failed to get kernel function\n");
        return 1;
    }

    // Create pipeline
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:nil];
    if (!pipeline) {
        fprintf(stderr, "Failed to create pipeline\n");
        return 1;
    }

    // Create buffers
    id<MTLBuffer> weightsBuffer = [device newBufferWithBytes:weights.data()
                                                       length:m * nb * sizeof(block_q4_K)
                                                      options:MTLResourceStorageModeShared];

    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                     length:k * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    id<MTLBuffer> outputBuffer = [device newBufferWithLength:m * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

    // Warmup
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weightsBuffer offset:0 atIndex:1];
        [encoder setBuffer:inputBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        // Set args
        // [encoder setBytes:&args length:sizeof(args) atIndex:0];
        // [encoder dispatchThreadgroups:... threadsPerThreadgroup:...];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:weightsBuffer offset:0 atIndex:1];
            [encoder setBuffer:inputBuffer offset:0 atIndex:2];
            [encoder setBuffer:outputBuffer offset:0 atIndex:3];
            // [encoder setBytes:&args length:sizeof(args) atIndex:0];
            // [encoder dispatchThreadgroups:... threadsPerThreadgroup:...];
            [encoder endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = duration.count() / (double)iterations;

    printf("Completed %d iterations\n", iterations);
    printf("Total time: %.2f ms\n", duration.count() / 1000.0);
    printf("Avg time: %.2f us\n", avg_us);
    printf("GFLOPS: %.2f\n", (2.0 * m * k) / (avg_us * 1e-6) / 1e9);

    return 0;
}
