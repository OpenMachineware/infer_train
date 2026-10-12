// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// llama.cpp Q4_K MV test - calls real ggml-metal API
// Compile: clang++ -O3 -fobjc-arc -framework Metal -framework Foundation -o llama_q4_k_test llama_q4_k_test.mm

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K/2];
} block_q4_K;

void generate_weights(block_q4_K* w, int m, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < m * nb; i++) {
        for (int j = 0; j < 12; j++) w[i].scales[j] = (i * 17 + j * 13) % 64;
        for (int j = 0; j < QK_K/2; j++) w[i].qs[j] = (i * 23 + j * 17) % 256;
        w[i].d = 0x3F00 | ((i % 10) & 0xF);
        w[i].dmin = 0;
    }
}

void generate_input(float* x, int k) {
    for (int i = 0; i < k; i++) x[i] = (i % 10) * 0.1f;
}

int main(int argc, char** argv) {
    int m = 4096, k = 4096, iter = 10000;
    if (argc >= 3) { m = atoi(argv[1]); k = atoi(argv[2]); }
    if (argc >= 4) iter = atoi(argv[3]);

    printf("llama.cpp Q4_K MV: M=%d, K=%d, iter=%d\n", m, k, iter);

    int nb = k / QK_K;
    std::vector<block_q4_K> weights(m * nb);
    std::vector<float> input(k), output(m);
    generate_weights(weights.data(), m, k);
    generate_input(input.data(), k);

    // Use ggml-metal API directly
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Compile shader
        NSString* src = [NSString stringWithContentsOfFile:@"../shaders/mul_mv_template.metal"
                                                  encoding:NSUTF8StringEncoding error:nil];
        if (!src) src = [NSString stringWithContentsOfFile:@"../../shaders/mul_mv_template.metal"
                                                  encoding:NSUTF8StringEncoding error:nil];
        if (!src) src = [NSString stringWithContentsOfFile:@"/Users/jia/Desktop/infer_train/shaders/mul_mv_template.metal"
                                                  encoding:NSUTF8StringEncoding error:nil];

        id<MTLLibrary> lib = [device newLibraryWithSource:src options:[[MTLCompileOptions alloc] init] error:nil];

        // Create function with constants
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        short nsg = 2, ne12 = 1, r2 = 1, r3 = 1;
        [fc setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
        [fc setConstantValue:&ne12 type:MTLDataTypeShort atIndex:602];
        [fc setConstantValue:&r2 type:MTLDataTypeShort atIndex:603];
        [fc setConstantValue:&r3 type:MTLDataTypeShort atIndex:604];

        id<MTLFunction> kernel = [lib newFunctionWithName:@"kernel_mul_mv_q4_K_f32" constantValues:fc error:nil];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:nil];

        // Create buffers
        id<MTLBuffer> wbuf = [device newBufferWithBytes:weights.data() length:m*nb*sizeof(block_q4_K) options:MTLResourceStorageModeShared];
        id<MTLBuffer> xbuf = [device newBufferWithBytes:input.data() length:k*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ybuf = [device newBufferWithLength:m*sizeof(float) options:MTLResourceStorageModeShared];

        // Args
        struct { int32_t ne00,ne01,ne02; uint64_t nb00,nb01,nb02,nb03; int32_t ne10,ne11,ne12;
                 uint64_t nb10,nb11,nb12,nb13; int32_t ne0,ne1,nr0; int16_t r2,r3; } args =
            {k,m,1, 1,nb*sizeof(block_q4_K),0,0, k,1,1, 4,k*4ULL,0,0, m,1,2, 1,1};

        NSUInteger rows = 4;
        MTLSize tgcount = {(m + rows - 1) / rows, 1, 1}, tgsize = {32, 2, 1};

        // Warmup
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:wbuf offset:0 atIndex:1];
            [enc setBuffer:xbuf offset:0 atIndex:2];
            [enc setBuffer:ybuf offset:0 atIndex:3];
            [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgsize];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Benchmark
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iter; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:wbuf offset:0 atIndex:1];
                [enc setBuffer:xbuf offset:0 atIndex:2];
                [enc setBuffer:ybuf offset:0 atIndex:3];
                [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgsize];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / (double)iter;
        printf("Avg: %.2f us, GFLOPS: %.2f\n", us, (2.0*m*k)/(us*1e-6)/1e9);
    }
    return 0;
}
