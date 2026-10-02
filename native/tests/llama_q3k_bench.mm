// Standalone benchmark for Q3_K MV kernel - llama.cpp version
// Compile: clang++ -fobjc-arc -framework Metal -framework Foundation -O3 -o llama_q3k_bench llama_q3k_bench.mm
// Run: ./llama_q3k_bench

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>

// Q3_K block structure (110 bytes)
struct BlockQ3K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};

// Kernel args
struct MvArgs {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
};

int main() {
    @autoreleasepool {
        printf("=== llama.cpp Q3_K MV Benchmark ===\n\n");

        // Create device and queue
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Load precompiled kernels from llama.cpp build
        NSString* shaderPath = @"llama.cpp-0.5.0/build/bin/kernels/mul_mv.metal";

        NSError* error = nil;
        NSString* mulmvSrc = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
        if (!mulmvSrc) {
            printf("Failed to load shader: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        [options setFastMathEnabled:YES];

        id<MTLLibrary> library = [device newLibraryWithSource:mulmvSrc options:options error:&error];
        if (!library) {
            printf("Shader compilation failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        // Create pipeline with function constants
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        int16_t nsg = 2, ne12 = 1, r2 = 1, r3 = 1;
        [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
        [constants setConstantValue:&ne12 type:MTLDataTypeShort atIndex:602];
        [constants setConstantValue:&r2 type:MTLDataTypeShort atIndex:603];
        [constants setConstantValue:&r3 type:MTLDataTypeShort atIndex:604];

        id<MTLFunction> kernel = [library newFunctionWithName:@"kernel_mul_mv_q3_K_f32"
                                                constantValues:constants error:&error];
        if (!kernel) {
            printf("Failed to get kernel: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
        if (!pipeline) {
            printf("Failed to create pipeline: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        printf("Pipeline created. threadExecutionWidth: %lu\n", pipeline.threadExecutionWidth);

        // Test dimensions
        const int M = 4096, K = 4096;
        const int QK = 256;
        const int nb = K / QK;
        const int block_size = 110;

        // Generate weights (deterministic)
        std::vector<BlockQ3K> weights(M * nb);
        for (int i = 0; i < M * nb; i++) {
            for (int j = 0; j < 32; j++) weights[i].hmask[j] = (uint8_t)((i * 17 + j * 23) % 256);
            for (int j = 0; j < 64; j++) weights[i].qs[j] = (uint8_t)((i * 13 + j * 7) % 256);
            for (int j = 0; j < 12; j++) weights[i].scales[j] = (uint8_t)((i * 31 + j * 11) % 128);
            weights[i].d = (uint16_t)(0.5f + (i % 10) * 0.1f);
        }

        // Generate input (deterministic)
        std::vector<float> input(K);
        for (int i = 0; i < K; i++) input[i] = (i % 100) * 0.01f;

        // Create buffers
        id<MTLBuffer> weightsBuf = [device newBufferWithBytes:weights.data()
                                                       length:M * nb * block_size
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> inputBuf = [device newBufferWithBytes:input.data()
                                                      length:K * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [device newBufferWithLength:M * sizeof(float)
                                                       options:MTLResourceStorageModeShared];

        // Args
        MvArgs args;
        args.ne00 = K; args.ne01 = M; args.ne02 = 1;
        args.nb00 = block_size; args.nb01 = nb * block_size; args.nb02 = 0; args.nb03 = 0;
        args.ne10 = K; args.ne11 = 1; args.ne12 = 1;
        args.nb10 = 4; args.nb11 = K * 4; args.nb12 = 0; args.nb13 = 0;
        args.ne0 = M; args.ne1 = 1; args.nr0 = 2;
        args.r2 = 1; args.r3 = 1;

        id<MTLBuffer> argsBuf = [device newBufferWithBytes:&args
                                                     length:sizeof(MvArgs)
                                                    options:MTLResourceStorageModeShared];

        // Thread configuration
        const int nsg_val = 2;
        const int nr0_val = 2;
        MTLSize threadgroups = MTLSizeMake((M + nsg_val * nr0_val - 1) / (nsg_val * nr0_val), 1, 1);
        MTLSize threadsPerTG = MTLSizeMake(32, nsg_val, 1);

        // Warmup
        printf("Warming up...\n");
        for (int i = 0; i < 100; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:argsBuf offset:0 atIndex:0];
            [enc setBuffer:weightsBuf offset:0 atIndex:1];
            [enc setBuffer:inputBuf offset:0 atIndex:2];
            [enc setBuffer:outputBuf offset:0 atIndex:3];
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Benchmark
        const int iterations = 10000;
        printf("Running benchmark (%d iterations)...\n", iterations);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:argsBuf offset:0 atIndex:0];
            [enc setBuffer:weightsBuf offset:0 atIndex:1];
            [enc setBuffer:inputBuf offset:0 atIndex:2];
            [enc setBuffer:outputBuf offset:0 atIndex:3];
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        auto end = std::chrono::high_resolution_clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        double avg_us = total_us / iterations;
        double gflops = 2.0 * M * K / (avg_us * 1e-6) / 1e9;

        printf("\nResults:\n");
        printf("  Total time: %.2f ms\n", total_us / 1000.0);
        printf("  Avg time:   %.2f us\n", avg_us);
        printf("  GFLOPS:     %.1f\n", gflops);

        // Print first few output values for verification
        float* output = (float*)outputBuf.contents;
        printf("\nFirst 5 outputs:\n");
        for (int i = 0; i < 5; i++) {
            printf("  output[%d] = %.6f\n", i, output[i]);
        }

        return 0;
    }
}
