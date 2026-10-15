// Direct llama.cpp SSM kernel benchmark
// Compiles and runs llama.cpp's Metal kernels with same test data
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

// Metal headers
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// Test dimensions matching our Rust benchmark
const int D_CONV = 4;
const int D_INNER_CONV = 10240;
const int D_STATE = 16;
const int D_INNER_SCAN = 4096;
const int D_STATE_SSD = 64;
const int D_INNER_SSD = 64; // head_dim

// FLOP calculations
double compute_conv_flops(int d_inner, int n_tok, int d_conv) {
    return (double)d_inner * n_tok * (d_conv * 2);
}

double compute_scan_flops(int d_inner, int n_tok, int d_state) {
    return (double)d_inner * n_tok * (10.0 + d_state * 5.0);
}

class LlamaCppSSMBench {
private:
    MTL::Device* device;
    MTL::CommandQueue* queue;
    MTL::Library* library;

    // Kernels
    MTL::ComputePipelineState* conv_kernel;
    MTL::ComputePipelineState* scan_kernel;
    MTL::ComputePipelineState* ssd_kernel;

public:
    bool init() {
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            printf("Failed to create Metal device\n");
            return false;
        }

        queue = device->newCommandQueue();
        if (!queue) {
            printf("Failed to create command queue\n");
            return false;
        }

        // Load Metal library from llama.cpp
        // Try multiple paths
        const char* paths[] = {
            "../llama.cpp-0.6.0/ggml/src/ggml-metal/ggml-metal.metal",
            "llama.cpp-0.6.0/ggml/src/ggml-metal/ggml-metal.metal",
            "/Users/jia/Desktop/infer_train/llama.cpp-0.6.0/ggml/src/ggml-metal/ggml-metal.metal"
        };

        NSString* source = nil;
        for (int i = 0; i < 3; i++) {
            @autoreleasepool {
                NSString* path = [NSString stringWithUTF8String:paths[i]];
                if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                    NSError* error = nil;
                    source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
                    if (source) {
                        printf("Loaded Metal source from: %s\n", paths[i]);
                        break;
                    }
                }
            }
        }

        if (!source) {
            printf("Failed to load Metal source\n");
            return false;
        }

        @autoreleasepool {
            NSError* error = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
            [options setLanguageVersion:MTLLanguageVersion3_0];

            library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) {
                printf("Failed to compile Metal library: %s\n", [[error localizedDescription] UTF8String]);
                return false;
            }
        }

        // Get kernel functions
        @autoreleasepool {
            // Conv1D kernel (nc=4)
            NSString* conv_name = @"kernel_ssm_conv_f32_f32_4";
            MTLFunction* conv_func = [library newFunctionWithName:conv_name];
            if (!conv_func) {
                printf("Failed to find kernel: %s\n", [conv_name UTF8String]);
                return false;
            }
            conv_kernel = [device newComputePipelineStateWithFunction:conv_func error:&error];
            if (!conv_kernel) {
                printf("Failed to create pipeline: %s\n", [[error localizedDescription] UTF8String]);
                return false;
            }

            // Scan kernel
            NSString* scan_name = @"kernel_ssm_scan_f32";
            MTLFunction* scan_func = [library newFunctionWithName:scan_name];
            if (!scan_func) {
                printf("Failed to find kernel: %s\n", [scan_name UTF8String]);
                return false;
            }
            scan_kernel = [device newComputePipelineStateWithFunction:scan_func error:&error];
            if (!scan_kernel) {
                printf("Failed to create pipeline: %s\n", [[error localizedDescription] UTF8String]);
                return false;
            }

            // SSD kernel
            NSString* ssd_name = @"kernel_ssm_scan_ssd_mma_f32";
            MTLFunction* ssd_func = [library newFunctionWithName:ssd_name];
            if (ssd_func) {
                ssd_kernel = [device newComputePipelineStateWithFunction:ssd_func error:&error];
                if (!ssd_kernel) {
                    printf("Warning: SSD kernel not found, skipping SSD benchmark\n");
                    ssd_kernel = nullptr;
                }
            } else {
                printf("Warning: SSD kernel not found, skipping SSD benchmark\n");
                ssd_kernel = nullptr;
            }
        }

        printf("llama.cpp SSM kernels loaded successfully\n");
        return true;
    }

    // Benchmark Conv1D
    void bench_conv1d(int* seq_lengths, int n_lengths) {
        printf("\n=== llama.cpp SSM Conv1D (d_conv=%d, d_inner=%d) ===\n",
               D_CONV, D_INNER_CONV);
        printf("seq_len | time (us) | GFLOPS | GB/s\n");
        printf("--------|-----------|--------|------\n");

        for (int i = 0; i < n_lengths; i++) {
            int n_tok = seq_lengths[i];
            int x_size = D_INNER_CONV * (n_tok + D_CONV - 1);
            int w_size = D_INNER_CONV * D_CONV;

            // Create test data
            std::vector<float> x(x_size, 0.1f);
            std::vector<float> w(w_size, 0.05f);
            std::vector<float> bias(D_INNER_CONV, 0.01f);
            std::vector<float> y(D_INNER_CONV * n_tok, 0.0f);

            // Create buffers
            @autoreleasepool {
                MTLBuffer* x_buf = [device newBufferWithBytes:x.data()
                                                       length:x_size * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
                MTLBuffer* w_buf = [device newBufferWithBytes:w.data()
                                                       length:w_size * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
                MTLBuffer* y_buf = [device newBufferWithBytes:y.data()
                                                       length:y.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

                // Warm up
                for (int j = 0; j < 10; j++) {
                    run_conv1d_kernel(x_buf, w_buf, y_buf, n_tok);
                }

                // Measure
                int iterations = 100;
                double total_ns = 0;
                for (int j = 0; j < iterations; j++) {
                    uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
                    run_conv1d_kernel(x_buf, w_buf, y_buf, n_tok);
                    uint64_t end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
                    total_ns += (end - start);
                }

                double mean_ns = total_ns / iterations;
                double mean_us = mean_ns / 1000.0;
                double gflops = compute_conv_flops(D_INNER_CONV, n_tok, D_CONV) / mean_ns * 1000.0;

                // Bandwidth
                int bytes_read = (x_size + w_size + D_INNER_CONV) * 4;
                int bytes_write = D_INNER_CONV * n_tok * 4;
                double gbps = (bytes_read + bytes_write) / mean_ns * 1000.0;

                printf("%7d | %7.2f | %6.2f | %5.1f\n", n_tok, mean_us, gflops, gbps);
            }
        }
    }

private:
    void run_conv1d_kernel(MTLBuffer* x_buf, MTLBuffer* w_buf, MTLBuffer* y_buf, int n_tok) {
        @autoreleasepool {
            MTLCommandBuffer* cmd_buf = queue->commandBuffer();
            MTLComputeCommandEncoder* encoder = cmd_buf->computeCommandEncoder();

            encoder->setComputePipelineState(conv_kernel);
            encoder->setBuffer(x_buf, 0, 0);
            encoder->setBuffer(w_buf, 0, 1);
            encoder->setBuffer(y_buf, 0, 2);

            // Grid: (d_inner, n_tok, 1)
            MTLSize grid = MTLSizeMake(D_INNER_CONV, n_tok, 1);
            MTLSize tg = MTLSizeMake(1, 1, 1);

            encoder->dispatchThreads(grid, tg);
            encoder->endEncoding();

            cmd_buf->commit();
            cmd_buf->waitUntilCompleted();
        }
    }
};

int main() {
    @autoreleasepool {
        LlamaCppSSMBench bench;
        if (!bench.init()) {
            printf("Failed to initialize llama.cpp benchmark\n");
            return 1;
        }

        printf("\n========================================\n");
        printf("llama.cpp SSM Kernel Performance\n");
        printf("========================================\n");

        int conv_lengths[] = {1, 16, 64, 128, 256, 512, 1024, 2048};
        bench.bench_conv1d(conv_lengths, 8);
    }

    return 0;
}
