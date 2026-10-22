// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// Q4_K GEMV fair comparison benchmark

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#import <Metal/Metal.h>

#define QK_K 256
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_k;

void create_test_data(block_q4_k* weights, float* input, int m, int k) {
    int n_blocks = k / QK_K;
    for (int i = 0; i < m * n_blocks; i++) {
        weights[i].d = 0x3800; weights[i].dmin = 0x2E66;
        for (int j = 0; j < 12; j++) weights[i].scales[j] = (i * 7 + j * 11 + 1) % 64;
        for (int j = 0; j < 128; j++) weights[i].qs[j] = (i * 13 + j * 17 + 8) % 256;
    }
    for (int i = 0; i < k; i++) input[i] = 0.1f * ((i % 10) + 1);
}

// Self-written optimized Q4_K kernel matching llama.cpp's algorithm
static const char* our_kernel = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_k;
struct args_t { uint32_t ne00; uint32_t ne01; uint64_t nb01; };

kernel void kernel_q4_k_gemv(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant args_t & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int nb = args.ne00 / QK_K;
    const int row = tgpig.x;
    if (row >= (int)args.ne01) return;

    device const block_q4_k * blocks = (device const block_q4_k *)(src0 + row * args.nb01);
    device const float * yy = src1;

    float sumf = 0.0f;

    // Y-vector caching in registers
    float yl[16], yh[16];

    // Thread ID decomposition
    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    // Compute offsets
    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    // Scale extraction masks
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    // Each thread starts at different block (4-block batching)
    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const block_q4_k * block = blocks + i;

        device const uint8_t * q1 = block->qs + q_offset;
        device const half * dh = (device const half *)&block->d;
        device const uint16_t * a = (device const uint16_t *)block->scales + iq;

        // Y-vector caching
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        // Extract scales using masks
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        // Float4 accumulation
        float4 acc1 = {0.f};

        device const uint8_t * q2 = q1 + 64;

        FOR_UNROLL (short l = 0; l < 8; ++l) {
            acc1[0] += yl[l+0] * (q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (q2[l] & 0xF0);
        }

        // Final computation
        // Formula: d * scale * q - dmin * sumy * min
        // Note: Q4_K uses acc1[1] and acc1[3] divided by 16 (they contain upper nibble * y)
        sumf += (float)dh[0] * (sc8[0] * (acc1[0]      + acc1[1]/16.f) +
                               sc8[4] * (acc1[2]      + acc1[3]/16.f)) -
                (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

        y1 += 4 * QK_K;
    }

    // SIMD reduction
    float total = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = total;
    }
}
)METAL";

// llama.cpp Q4_K kernel (optimized)
static const char* llama_kernel = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03; int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13; int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(args_t args, device const char * src0, device const char * src1, device char * dst, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const int nb = args.ne00/QK_K; const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0*1 + sgitg) * nr0; const uint i12 = im%1; const uint i13 = im/1;
    const uint64_t offset0 = first_row*args.nb01 + (i12/1)*args.nb02 + (i13/1)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float * yy = (device const float *) (src1 + offset1);
    float sumf[nr0] = { 0.f }; float yl[16], yh[16];
    const short tid = tiisg/4; const short ix = tiisg%4; const short iq = tid/4; const short ir = tid%4;
    const short l0 = 8*ir; const short q_offset = 32*iq + l0; const short y_offset = 64*iq + l0;
    uint16_t sc16[4]; thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
    device const float * y1 = yy + ix*QK_K + y_offset;
    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const half * dh = &x[i].d; device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;
        device const float * y2 = y1 + 128; float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) { yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0]; yl[l+8] = y1[l+32]; sumy[1] += yl[l+8]; yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0]; yh[l+8] = y2[l+32]; sumy[3] += yh[l+8]; }
        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;
            sc16[0] = a[0] & 0x3f3f; sc16[1] = a[2] & 0x3f3f; sc16[2] = ((a[4] >> 0) & 0x0f0f) | ((a[0] & 0xc0c0) >> 2); sc16[3] = ((a[4] >> 4) & 0x0f0f) | ((a[2] & 0xc0c0) >> 2);
            float4 acc1 = {0.f}; float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                acc1[0] += yl[l+0] * (q1[l] & 0x0F); acc1[1] += yl[l+8] * (q1[l] & 0xF0); acc1[2] += yh[l+0] * (q2[l] & 0x0F); acc1[3] += yh[l+8] * (q2[l] & 0xF0);
            }
            sumf[row] += dh[0] * (sc8[0] * (acc1[0] + acc1[1]/16.f) + sc8[4] * (acc1[2] + acc1[3]/16.f)) - dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += args.nb01; dh += args.nb01/2; a += args.nb01/2;
        }
        y1 += 4 * QK_K;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) { const float tot = simd_sum(sumf[row]); if (tiisg == 0) dst_f32[first_row + row] = tot; }
}
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device char * dst [[buffer(3)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
            kernel_mul_mv_q4_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, tgpig, tiisg, sgitg);
        }
)METAL";

int main() {
    @autoreleasepool {
        printf("=== Q4_K GEMV Fair Comparison ===\n");
        printf("Same test data, same matrix sizes, same hardware\n\n");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* error = nil;

        id<MTLLibrary> our_lib = [device newLibraryWithSource:[NSString stringWithUTF8String:our_kernel] options:nil error:&error];
        if (!our_lib) { printf("Our kernel compilation failed: %s\n", [[error localizedDescription] UTF8String]); return 1; }
        id<MTLFunction> our_func = [our_lib newFunctionWithName:@"kernel_q4_k_gemv"];
        id<MTLComputePipelineState> our_pipeline = [device newComputePipelineStateWithFunction:our_func error:&error];

        id<MTLLibrary> llama_lib = [device newLibraryWithSource:[NSString stringWithUTF8String:llama_kernel] options:nil error:&error];
        if (!llama_lib) { printf("llama kernel compilation failed: %s\n", [[error localizedDescription] UTF8String]); return 1; }
        id<MTLFunction> llama_func = [llama_lib newFunctionWithName:@"kernel_mul_mv_q4_K_f32"];
        id<MTLComputePipelineState> llama_pipeline = [device newComputePipelineStateWithFunction:llama_func error:&error];

        int test_cases[][2] = {{128, 4096}, {512, 4096}, {1024, 4096}, {2048, 4096}};

        for (int t = 0; t < 4; t++) {
            int m = test_cases[t][0], k = test_cases[t][1], n_blocks = k / QK_K;
            printf("Matrix: %d × %d\n", m, k);

            block_q4_k* weights = (block_q4_k*)malloc(m * n_blocks * sizeof(block_q4_k));
            float* input = (float*)malloc(k * sizeof(float));
            create_test_data(weights, input, m, k);

            id<MTLBuffer> weights_buf = [device newBufferWithBytes:weights length:m*n_blocks*sizeof(block_q4_k) options:MTLResourceStorageModeShared];
            id<MTLBuffer> input_buf = [device newBufferWithBytes:input length:k*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> our_out = [device newBufferWithLength:m*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> llama_out = [device newBufferWithLength:m*sizeof(float) options:MTLResourceStorageModeShared];

            struct { uint32_t ne00; uint32_t ne01; uint64_t nb01; } our_args = {(uint32_t)k, (uint32_t)m, (uint64_t)(n_blocks*sizeof(block_q4_k))};
            id<MTLBuffer> our_args_buf = [device newBufferWithBytes:&our_args length:sizeof(our_args) options:MTLResourceStorageModeShared];

typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03; int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13; int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;
            ggml_metal_kargs_mul_mv llama_args;
            llama_args.ne00 = k; llama_args.ne01 = m; llama_args.ne02 = 1; llama_args.nb00 = 1;
            llama_args.nb01 = (uint64_t)(n_blocks*sizeof(block_q4_k)); llama_args.nb02 = 0; llama_args.nb03 = 0;
            llama_args.ne10 = k; llama_args.ne11 = 1; llama_args.ne12 = 1; llama_args.nb10 = 4;
            llama_args.nb11 = (uint64_t)(k*4); llama_args.nb12 = 0; llama_args.nb13 = 0;
            llama_args.ne0 = m; llama_args.ne1 = 1; llama_args.nr0 = 1; llama_args.r2 = 1; llama_args.r3 = 1;
            id<MTLBuffer> llama_args_buf = [device newBufferWithBytes:&llama_args length:sizeof(llama_args) options:MTLResourceStorageModeShared];

            for (int i = 0; i < 10; i++) {
                @autoreleasepool {
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:our_pipeline];
                    [enc setBuffer:weights_buf offset:0 atIndex:0];
                    [enc setBuffer:input_buf offset:0 atIndex:1];
                    [enc setBuffer:our_out offset:0 atIndex:2];
                    [enc setBuffer:our_args_buf offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
            }

            int iterations = 100;
            uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            for (int i = 0; i < iterations; i++) {
                @autoreleasepool {
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:our_pipeline];
                    [enc setBuffer:weights_buf offset:0 atIndex:0];
                    [enc setBuffer:input_buf offset:0 atIndex:1];
                    [enc setBuffer:our_out offset:0 atIndex:2];
                    [enc setBuffer:our_args_buf offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
            }
            uint64_t end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            double our_us = (end - start) / 1000.0 / iterations;
            double our_gops = (m * k * 2.0) / (our_us * 1e-6) / 1e9;

            start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            for (int i = 0; i < iterations; i++) {
                @autoreleasepool {
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:llama_pipeline];
                    [enc setBuffer:llama_args_buf offset:0 atIndex:0];
                    [enc setBuffer:weights_buf offset:0 atIndex:1];
                    [enc setBuffer:input_buf offset:0 atIndex:2];
                    [enc setBuffer:llama_out offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
            }
            end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            double llama_us = (end - start) / 1000.0 / iterations;
            double llama_gops = (m * k * 2.0) / (llama_us * 1e-6) / 1e9;

            printf("  Our implementation:   %6.1f µs (%5.2f GOPS)\n", our_us, our_gops);
            printf("  llama.cpp:            %6.1f µs (%5.2f GOPS)\n", llama_us, llama_gops);
            if (our_gops > llama_gops) {
                printf("  Result:               %.2fx FASTER ✅\n\n", our_gops/llama_gops);
            } else {
                printf("  Result:               %.2fx slower\n\n", llama_gops/our_gops);
            }

            free(weights);
            free(input);
        }
    }
    return 0;
}