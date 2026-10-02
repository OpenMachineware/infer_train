// llama.cpp Metal Q4_K MV - fair comparison test
// Same conditions as Rust test: single kernel per iteration, no special optimizations

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <thread>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#define QK_K 256

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

    printf("llama.cpp Metal Q4_K MV - Fair Comparison\n");
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

    // Initialize Metal backend
    // This is the same as ggml_backend_metal_init()
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Failed to create Metal device\n");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Load Metal library from source (same as Rust)
        NSString* shaderPath = @"../../shaders/mul_mv_template.metal";
        NSString* source = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:nil];
        if (!source) {
            // Try alternative path
            shaderPath = @"../../shaders/mul_mv_template.metal";
            source = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:nil];
        }

        if (!source) {
            fprintf(stderr, "Failed to load Metal shader from %s\n", [shaderPath UTF8String]);
            fprintf(stderr, "Trying absolute path...\n");
            shaderPath = @"/Users/jia/Desktop/infer_train/shaders/mul_mv_template.metal";
            source = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:nil];
        }

        if (!source) {
            fprintf(stderr, "Failed to load Metal shader\n");
            return 1;
        }

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "Failed to create Metal library\n");
            return 1;
        }

        // Create function with function constants (same as Rust)
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        short nsg = 2, ne12 = 1, r2 = 1, r3 = 1;
        [fc setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
        [fc setConstantValue:&ne12 type:MTLDataTypeShort atIndex:602];
        [fc setConstantValue:&r2 type:MTLDataTypeShort atIndex:603];
        [fc setConstantValue:&r3 type:MTLDataTypeShort atIndex:604];

        id<MTLFunction> kernel = [library newFunctionWithName:@"kernel_mul_mv_q4_K_f32"
                                               constantValues:fc
                                                        error:nil];
        if (!kernel) {
            fprintf(stderr, "Failed to create kernel function\n");
            return 1;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:nil];
        if (!pipeline) {
            fprintf(stderr, "Failed to create pipeline\n");
            return 1;
        }

        // Create buffers (same as Rust: with data for weights/input, empty for output)
        id<MTLBuffer> weightsBuffer = [device newBufferWithBytes:weights.data()
                                                           length:m * nb * sizeof(block_q4_K)
                                                          options:MTLResourceStorageModeShared];

        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                         length:k * sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        id<MTLBuffer> outputBuffer = [device newBufferWithLength:m * sizeof(float)
                                                          options:MTLResourceStorageModeShared];

        // Create args struct (same layout as Rust)
        struct __attribute__((packed)) MvArgs {
            int32_t ne00, ne01, ne02;
            uint64_t nb00, nb01, nb02, nb03;
            int32_t ne10, ne11, ne12;
            uint64_t nb10, nb11, nb12, nb13;
            int32_t ne0, ne1, nr0;
            int16_t r2, r3;
        };

        struct MvArgs args = {
            .ne00 = k, .ne01 = m, .ne02 = 1,
            .nb00 = 1, .nb01 = nb * sizeof(block_q4_K), .nb02 = 0, .nb03 = 0,
            .ne10 = k, .ne11 = 1, .ne12 = 1,
            .nb10 = 4, .nb11 = k * 4ULL, .nb12 = 0, .nb13 = 0,
            .ne0 = m, .ne1 = 1, .nr0 = 2,
            .r2 = 1, .r3 = 1
        };

        // Warmup
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&args length:sizeof(args) atIndex:0];
            [encoder setBuffer:weightsBuffer offset:0 atIndex:1];
            [encoder setBuffer:inputBuffer offset:0 atIndex:2];
            [encoder setBuffer:outputBuffer offset:0 atIndex:3];

            NSUInteger nsg_val = 2, nr0_val = 2;
            NSUInteger rows_per_tg = nsg_val * nr0_val;
            MTLSize tg_count = {(m + rows_per_tg - 1) / rows_per_tg, 1, 1};
            MTLSize tg_size = {32, nsg_val, 1};
            [encoder dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];

            [encoder endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Benchmark - SAME LOOP STRUCTURE AS RUST
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&args length:sizeof(args) atIndex:0];
            [encoder setBuffer:weightsBuffer offset:0 atIndex:1];
            [encoder setBuffer:inputBuffer offset:0 atIndex:2];
            [encoder setBuffer:outputBuffer offset:0 atIndex:3];

            NSUInteger nsg_val = 2, nr0_val = 2;
            NSUInteger rows_per_tg = nsg_val * nr0_val;
            MTLSize tg_count = {(m + rows_per_tg - 1) / rows_per_tg, 1, 1};
            MTLSize tg_size = {32, nsg_val, 1};
            [encoder dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];

            [encoder endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_us = duration.count() / (double)iterations;

        printf("Completed %d iterations\n", iterations);
        printf("Total time: %.2f ms\n", duration.count() / 1000.0);
        printf("Avg time: %.2f us\n", avg_us);
        printf("GFLOPS: %.2f\n", (2.0 * m * k) / (avg_us * 1e-6) / 1e9);
    }

    return 0;
}
