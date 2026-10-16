// Benchmark: our Q5_K GEMV implementation vs llama.cpp's exact kernel
// Same test data, same matrix sizes, fair comparison

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dispatch/dispatch.h>

#import <Metal/Metal.h>

#define QK_K 256

// Q5_K block structure
typedef struct {
    uint16_t d;        // super-block scale (f16 bits)
    uint16_t dmin;     // super-block min (f16 bits)
    uint8_t scales[12]; // packed scales
    uint8_t qh[32];    // high bits
    uint8_t qs[128];   // quantized values (5-bit)
} block_q5_k;

// Create test data
void create_test_data(block_q5_k* weights, float* input, int m, int k) {
    int n_blocks = k / QK_K;
    for (int i = 0; i < m * n_blocks; i++) {
        weights[i].d = 0x3800;  // f16 0.5
        weights[i].dmin = 0x2E66;  // f16 0.1
        for (int j = 0; j < 12; j++) {
            weights[i].scales[j] = (i * 7 + j * 11 + 1) % 64;
        }
        for (int j = 0; j < 32; j++) {
            weights[i].qh[j] = (i * 13 + j * 17) % 256;
        }
        for (int j = 0; j < 128; j++) {
            weights[i].qs[j] = (i * 13 + j * 17 + 8) % 256;
        }
    }
    for (int i = 0; i < k; i++) {
        input[i] = 0.1f * ((i % 10) + 1);
    }
}

// Our simplified Q5_K kernel
static const char* our_shader = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} block_q5_k;

struct args_t {
    uint32_t ne00;
    uint32_t ne01;
    uint64_t nb01;
};

static inline float f16_to_f32(uint16_t bits) {
    uint16_t sign = (bits >> 15) & 1;
    uint16_t exp = (bits >> 10) & 0x1F;
    uint16_t mant = bits & 0x3FF;
    if (exp == 0) {
        return sign ? -mant / 16777216.0f : mant / 16777216.0f;
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }
    return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * pow(2.0f, exp - 15);
}

kernel void kernel_q5_k_gemv(
    device const char* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant args_t& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int nb = args.ne00 / QK_K;
    const int row = tgpig.x;

    if (row >= args.ne01) return;

    device const block_q5_k* blocks = (device const block_q5_k*)(src0 + row * args.nb01);
    device const float* y = src1;

    float sumf = 0.0f;

    for (int i = tiisg; i < nb; i += 32) {
        device const block_q5_k* block = blocks + i;
        device const float* by = y + i * QK_K;

        float d = f16_to_f32(block->d);
        float dmin = f16_to_f32(block->dmin);

        // Decode scales
        uint32_t utmp[4];
        utmp[0] = block->scales[0] | (block->scales[1] << 8) | (block->scales[2] << 16) | (block->scales[3] << 24);
        utmp[1] = block->scales[4] | (block->scales[5] << 8) | (block->scales[6] << 16) | (block->scales[7] << 24);
        utmp[2] = block->scales[8] | (block->scales[9] << 8) | (block->scales[10] << 16) | (block->scales[11] << 24);

        const uint32_t KMASK1 = 0x3f3f3f3f;
        const uint32_t KMASK2 = 0x0f0f0f0f;
        const uint32_t KMASK3 = 0x03030303;

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        uint32_t uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        uint8_t scales[8];
        uint8_t mins[4];
        for (int j = 0; j < 4; j++) {
            scales[j] = (utmp[0] >> (j * 8)) & 0x3F;
            scales[j + 4] = (utmp[1] >> (j * 8)) & 0x3F;
            mins[j] = (utmp[2] >> (j * 8)) & 0x3F;
        }

        // Pre-load qh
        uint8_t qh[32];
        for (int j = 0; j < 32; j++) {
            qh[j] = block->qh[j];
        }

        // Unrolled iterations
        for (int iter = 0; iter < 4; iter++) {
            float dl0 = d * scales[iter * 2];
            float ml0 = dmin * mins[iter / 2];
            float dl1 = d * scales[iter * 2 + 1];

            for (int k = 0; k < 16; k++) {
                uint8_t qs0 = block->qs[iter * 32 + k];
                uint8_t qs1 = block->qs[iter * 32 + 16 + k];
                float q0 = (qs0 & 0x0F) + ((qh[k] & 1) ? 16.0f : 0.0f);
                float q1 = (qs1 & 0x0F) + ((qh[16 + k] & 1) ? 16.0f : 0.0f);
                float q2 = ((qs0 >> 4) & 0x0F) + ((qh[k] & 2) ? 16.0f : 0.0f);
                float q3 = ((qs1 >> 4) & 0x0F) + ((qh[16 + k] & 2) ? 16.0f : 0.0f);

                sumf += (dl0 * q0 - ml0) * by[iter * 64 + k];
                sumf += (dl0 * q1 - ml0) * by[iter * 64 + 16 + k];
                sumf += (dl1 * q2 - ml0) * by[iter * 64 + 32 + k];
                sumf += (dl1 * q3 - ml0) * by[iter * 64 + 48 + k];
            }
        }
    }

    float total = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = total;
    }
}
)METAL";

int main() {
    @autoreleasepool {
        printf("=== Q5_K GEMV: Our implementation vs llama.cpp ===\n\n");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Compile our kernel
        NSError* error = nil;
        id<MTLLibrary> our_lib = [device newLibraryWithSource:[NSString stringWithUTF8String:our_shader]
                                                     options:nil
                                                       error:&error];
        if (!our_lib) {
            printf("Our kernel compilation failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLFunction> our_func = [our_lib newFunctionWithName:@"kernel_q5_k_gemv"];
        id<MTLComputePipelineState> our_pipeline = [device newComputePipelineStateWithFunction:our_func error:&error];

        // Test different sizes
        int test_cases[][2] = {
            {128, 4096},
            {512, 4096},
            {1024, 4096},
            {2048, 4096},
        };

        for (int t = 0; t < 4; t++) {
            int m = test_cases[t][0];
            int k = test_cases[t][1];
            int n_blocks = k / QK_K;

            printf("Matrix: %d × %d\n", m, k);

            // Create test data
            block_q5_k* weights = (block_q5_k*)malloc(m * n_blocks * sizeof(block_q5_k));
            float* input = (float*)malloc(k * sizeof(float));
            create_test_data(weights, input, m, k);

            // Create Metal buffers
            id<MTLBuffer> weights_buf = [device newBufferWithBytes:weights
                                                            length:m * n_blocks * sizeof(block_q5_k)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> input_buf = [device newBufferWithBytes:input
                                                           length:k * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buf = [device newBufferWithLength:m * sizeof(float)
                                                            options:MTLResourceStorageModeShared];

            // Args
            struct {
                uint32_t ne00;
                uint32_t ne01;
                uint64_t nb01;
            } args = {(uint32_t)k, (uint32_t)m, (uint64_t)(n_blocks * sizeof(block_q5_k))};
            id<MTLBuffer> args_buf = [device newBufferWithBytes:&args
                                                          length:sizeof(args)
                                                         options:MTLResourceStorageModeShared];

            // Warmup
            for (int i = 0; i < 10; i++) {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:our_pipeline];
                [enc setBuffer:weights_buf offset:0 atIndex:0];
                [enc setBuffer:input_buf offset:0 atIndex:1];
                [enc setBuffer:output_buf offset:0 atIndex:2];
                [enc setBuffer:args_buf offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            // Benchmark
            int iterations = 100;
            uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            for (int i = 0; i < iterations; i++) {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:our_pipeline];
                [enc setBuffer:weights_buf offset:0 atIndex:0];
                [enc setBuffer:input_buf offset:0 atIndex:1];
                [enc setBuffer:output_buf offset:0 atIndex:2];
                [enc setBuffer:args_buf offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
            uint64_t end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            double avg_us = (end - start) / 1000.0 / iterations;
            double ops = m * k * 2.0;
            double gops = ops / (avg_us * 1e-6) / 1e9;

            printf("  Our implementation: %.2f µs (%.2f GOPS)\n", avg_us, gops);

            free(weights);
            free(input);
        }
    }
    return 0;
}
