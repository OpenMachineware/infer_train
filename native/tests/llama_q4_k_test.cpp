// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Single kernel test: llama.cpp Q4_K GPU
// Measures each step separately
// Build: clang++ -framework Metal -framework Foundation -I../.. llama_q4_k_test.cpp -o llama_q4_k_test

#include <metal/metal.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

using namespace std::chrono;

struct BlockQ4K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct MulMVArgs {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int32_t nr0;
    int16_t r2, r3;
};

int main() {
    std::cout << "=== llama.cpp Q4_K GPU Test ===" << std::endl;

    // Dimensions
    const int M = 1024;
    const int K = 4096;
    const int N = 1;
    const int NB = K / 256;
    const int BLOCK_SIZE = 144;  // Q4_K block size
    const int ITERATIONS = 10000;

    // ===== STEP 1: Generate random data =====
    auto t1_start = high_resolution_clock::now();

    std::vector<BlockQ4K> weights(M * NB);
    std::vector<float> input(K);
    std::vector<float> output(M);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint8_t> dis(0, 255);

    for (int i = 0; i < M * NB; i++) {
        weights[i].d = 0.5f;
        weights[i].dmin = 0.1f;
        for (int j = 0; j < 12; j++) weights[i].scales[j] = dis(gen);
        for (int j = 0; j < 128; j++) weights[i].qs[j] = dis(gen);
    }

    for (int i = 0; i < K; i++) {
        input[i] = (i % 100) * 0.01f;
    }

    auto t1_end = high_resolution_clock::now();
    double t1_us = duration_cast<microseconds>(t1_end - t1_start).count();
    std::cout << "Step 1: Generate data - " << t1_us << " µs" << std::endl;

    // ===== STEP 2: Create Metal device and buffers =====
    auto t2_start = high_resolution_clock::now();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    id<MTLBuffer> weightsBuf = [device newBufferWithBytes:weights.data()
                                            length:M * NB * BLOCK_SIZE
                                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> inputBuf = [device newBufferWithBytes:input.data()
                                         length:K * sizeof(float)
                                        options:MTLResourceStorageModeShared];

    id<MTLBuffer> outputBuf = [device newBufferWithLength:M * sizeof(float)
                                                options:MTLResourceStorageModeShared];

    auto t2_end = high_resolution_clock::now();
    double t2_us = duration_cast<microseconds>(t2_end - t2_start).count();
    std::cout << "Step 2: Create GPU buffers - " << t2_us << " µs" << std::endl;

    // ===== STEP 3: Compile kernel =====
    auto t3_start = high_resolution_clock::now();

    NSString* shaderSource = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define QK_K 256\n"
    "#define FC_MUL_MV 600\n"
    "constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];\n"
    "constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];\n"
    "constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];\n"
    "constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];\n"
    "typedef struct { int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02, nb03; "
    "int32_t ne10, ne11, ne12; uint64_t nb10, nb11, nb12, nb13; int32_t ne0, ne1, nr0; int16_t r2, r3; } ggml_metal_kargs_mul_mv;\n"
    "typedef struct { half d, dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;\n"
    // ... [Include full kernel code here]
    "kernel void kernel_mul_mv_q4_K_f32("
    "constant ggml_metal_kargs_mul_mv& args [[buffer(0)]],"
    "device const char* src0 [[buffer(1)]],"
    "device const char* src1 [[buffer(2)]],"
    "device char* dst [[buffer(3)]],"
    "uint3 tgpig [[threadgroup_position_in_grid]],"
    "ushort tiisg [[thread_index_in_simdgroup]],"
    "ushort sgitg [[simdgroup_index_in_threadgroup]]);";

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                options:nil
                                                  error:&error];

    auto t3_end = high_resolution_clock::now();
    double t3_us = duration_cast<microseconds>(t3_end - t3_start).count();
    std::cout << "Step 3: Compile kernel - " << t3_us << " µs" << std::endl;

    // ===== STEP 4: Create pipeline =====
    auto t4_start = high_resolution_clock::now();

    id<MTLFunction> kernelFunc = [library newFunctionWithName:@"kernel_mul_mv_q4_K_f32"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunc
                                                                                 error:&error];

    auto t4_end = high_resolution_clock::now();
    double t4_us = duration_cast<microseconds>(t4_end - t4_start).count();
    std::cout << "Step 4: Create pipeline - " << t4_us << " µs" << std::endl;

    // ===== STEP 5: Warmup =====
    for (int i = 0; i < 10; i++) {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weightsBuf offset:0 atIndex:1];
        [encoder setBuffer:inputBuf offset:0 atIndex:2];
        [encoder setBuffer:outputBuf offset:0 atIndex:3];
        MTLSize gridSize = MTLSizeMake(M, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(64, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    // ===== STEP 6: Benchmark =====
    std::vector<double> gpu_times;
    std::vector<double> cpu_times;

    for (int i = 0; i < ITERATIONS; i++) {
        auto cpu_start = high_resolution_clock::now();

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weightsBuf offset:0 atIndex:1];
        [encoder setBuffer:inputBuf offset:0 atIndex:2];
        [encoder setBuffer:outputBuf offset:0 atIndex:3];
        MTLSize gridSize = MTLSizeMake(M, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(64, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        auto cpu_end = high_resolution_clock::now();
        double cpu_us = duration_cast<microseconds>(cpu_end - cpu_start).count();
        cpu_times.push_back(cpu_us);

        // GPU time from command buffer
        double gpu_us = [cmdBuf kernelStartTime] * 1000000.0; // Convert to µs
        gpu_times.push_back(gpu_us);
    }

    // ===== Results =====
    double total_cpu = 0, total_gpu = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        total_cpu += cpu_times[i];
        total_gpu += gpu_times[i];
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Iterations: " << ITERATIONS << std::endl;
    std::cout << "Avg CPU time: " << total_cpu / ITERATIONS << " µs" << std::endl;
    std::cout << "Avg GPU time: " << total_gpu / ITERATIONS << " µs" << std::endl;
    std::cout << "GFLOPS: " << (2.0 * M * K) / (total_gpu / ITERATIONS) / 1000.0 << std::endl;

    return 0;
}
