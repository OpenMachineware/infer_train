// llama.cpp Q4_K GPU test
// Measures CPU time and GPU time separately
// Build: clang++ -framework Metal -framework Foundation llama_q4_k_fair.cpp -o llama_q4_k_fair

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <iostream>
#include <vector>

struct BlockQ4K {
    uint16_t d;      // half
    uint16_t dmin;   // half
    uint8_t scales[12];
    uint8_t qs[128];
};

int main() {
    @autoreleasepool {
        std::cout << "=== llama.cpp Q4_K GPU Test ===" << std::endl;

        const int M = 1024;
        const int K = 4096;
        const int NB = K / 256;
        const int BLOCK_SIZE = 144;
        const int ITERATIONS = 10000;

        // STEP 1: Generate data
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> weights(M * NB * BLOCK_SIZE);
        std::vector<float> input(K);

        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = (i * 17 + 13) % 256;
        }
        for (int i = 0; i < K; i++) {
            input[i] = (i % 100) * 0.01f;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Step 1 - Generate data: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
                  << " µs" << std::endl;

        // STEP 2: Create GPU buffers
        auto t3 = std::chrono::high_resolution_clock::now();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLBuffer> weightsBuf = [device newBufferWithBytes:weights.data()
                                                        length:weights.size()
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> inputBuf = [device newBufferWithBytes:input.data()
                                                      length:K * 4
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [device newBufferWithLength:M * 4
                                                       options:MTLResourceStorageModeShared];
        auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << "Step 2 - Create buffers: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()
                  << " µs" << std::endl;

        // STEP 3: Load precompiled kernel
        auto t5 = std::chrono::high_resolution_clock::now();
        NSString* shaderPath = @"/tmp/q4_k_standalone.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                           encoding:NSUTF8StringEncoding
                                                              error:nil];

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        [options setFastMathEnabled:YES];

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                       options:options
                                                         error:&error];
        if (!library) {
            std::cerr << "Shader compilation failed: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        // Set function constants
        MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
        short nsg = 2, ne12 = 1, r2 = 1, r3 = 1;
        [constantValues setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
        [constantValues setConstantValue:&ne12 type:MTLDataTypeShort atIndex:602];
        [constantValues setConstantValue:&r2 type:MTLDataTypeShort atIndex:603];
        [constantValues setConstantValue:&r3 type:MTLDataTypeShort atIndex:604];

        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"kernel_mul_mv_q4_K_f32"
                                                constantValues:constantValues
                                                         error:&error];
        if (!kernelFunc) {
            std::cerr << "Function creation failed: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunc
                                                                                    error:&error];
        auto t6 = std::chrono::high_resolution_clock::now();
        std::cout << "Step 3 - Compile kernel: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count()
                  << " µs" << std::endl;

        // Args
        struct {
            int32_t ne00, ne01, ne02;
            uint64_t nb00, nb01, nb02, nb03;
            int32_t ne10, ne11, ne12;
            uint64_t nb10, nb11, nb12, nb13;
            int32_t ne0, ne1, nr0;
            int16_t r2, r3;
        } args = {K, M, 1, 1, NB * BLOCK_SIZE, 0, 0, K, 1, 1, 4, K * 4, 0, 0, M, 1, 2, 1, 1};

        id<MTLBuffer> argsBuf = [device newBufferWithBytes:&args
                                                     length:sizeof(args)
                                                    options:MTLResourceStorageModeShared];

        // Warmup
        for (int i = 0; i < 100; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:argsBuf offset:0 atIndex:0];
            [enc setBuffer:weightsBuf offset:0 atIndex:1];
            [enc setBuffer:inputBuf offset:0 atIndex:2];
            [enc setBuffer:outputBuf offset:0 atIndex:3];
            MTLSize grid = MTLSizeMake(M / 4, 1, 1);
            MTLSize tg = MTLSizeMake(64, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Benchmark
        std::cout << "\nBenchmarking " << ITERATIONS << " iterations..." << std::endl;
        auto bench_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < ITERATIONS; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:argsBuf offset:0 atIndex:0];
            [enc setBuffer:weightsBuf offset:0 atIndex:1];
            [enc setBuffer:inputBuf offset:0 atIndex:2];
            [enc setBuffer:outputBuf offset:0 atIndex:3];
            MTLSize grid = MTLSizeMake(M / 4, 1, 1);
            MTLSize tg = MTLSizeMake(64, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        auto bench_end = std::chrono::high_resolution_clock::now();
        double total_us = std::chrono::duration_cast<std::chrono::microseconds>(bench_end - bench_start).count();
        double avg_us = total_us / ITERATIONS;

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Total time: " << total_us << " µs" << std::endl;
        std::cout << "Avg time: " << avg_us << " µs" << std::endl;
        std::cout << "GFLOPS: " << (2.0 * M * K) / avg_us / 1000.0 << std::endl;

        // Verify output
        float* result = (float*)[outputBuf contents];
        std::cout << "\nFirst 5 results: ";
        for (int i = 0; i < 5; i++) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
