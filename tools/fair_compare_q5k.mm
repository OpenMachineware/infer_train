// FAIR COMPARISON: Our optimized Q5_K vs llama.cpp's exact kernel
// Same data, same hardware, same conditions

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#import <Metal/Metal.h>

#define QK_K 256

typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_k;

void create_test_data(block_q5_k* weights, float* input, int m, int k, uint64_t seed) {
    srand((unsigned)seed);
    int n_blocks = k / QK_K;
    for (int i = 0; i < m * n_blocks; i++) {
        weights[i].d = 0x3800 + (rand() % 256);
        weights[i].dmin = 0x2E66 + (rand() % 256);
        for (int j = 0; j < 12; j++) weights[i].scales[j] = rand() % 64;
        for (int j = 0; j < 32; j++) weights[i].qh[j] = rand() % 256;
        for (int j = 0; j < 128; j++) weights[i].qs[j] = rand() % 256;
    }
    for (int i = 0; i < k; i++) input[i] = 0.1f * (1 + rand() % 10);
}

// Our optimized kernel (from src/ops/metal/gemv/kquant_mv.metal)
static const char* our_kernel_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define QK_K 256

typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_k;
struct args_t { uint32_t ne00; uint32_t ne01; uint64_t nb01; };

static inline float f16_to_f32(uint16_t bits) {
    uint16_t sign = (bits >> 15) & 1, exp = (bits >> 10) & 0x1F, mant = bits & 0x3FF;
    if (exp == 0) return sign ? -mant/16777216.0f : mant/16777216.0f;
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    return (sign?-1.0f:1.0f) * (1.0f + mant/1024.0f) * pow(2.0f, exp-15);
}

kernel void kernel_q5_k_gemv(
    device const char* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant args_t& args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]) {

    const int nb = args.ne00 / QK_K, row = tgpig.x;
    if (row >= (int)args.ne01) return;

    device const block_q5_k* blocks = (device const block_q5_k*)(src0 + row * args.nb01);
    device const float* y = src1;
    float sumf = 0.0f;

    for (int i = tiisg; i < nb; i += 32) {
        device const block_q5_k* block = blocks + i;
        device const float* by = y + i * QK_K;
        float d = f16_to_f32(block->d), dmin = f16_to_f32(block->dmin);

        uint32_t utmp[4];
        utmp[0] = block->scales[0]|(block->scales[1]<<8)|(block->scales[2]<<16)|(block->scales[3]<<24);
        utmp[1] = block->scales[4]|(block->scales[5]<<8)|(block->scales[6]<<16)|(block->scales[7]<<24);
        utmp[2] = block->scales[8]|(block->scales[9]<<8)|(block->scales[10]<<16)|(block->scales[11]<<24);
        const uint32_t KMASK1 = 0x3f3f3f3f, KMASK2 = 0x0f0f0f0f, KMASK3 = 0x03030303;
        utmp[3] = ((utmp[2]>>4)&KMASK2) | (((utmp[1]>>6)&KMASK3)<<4);
        uint32_t uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2]&KMASK2) | (((utmp[0]>>6)&KMASK3)<<4);
        utmp[2] = uaux; utmp[0] &= KMASK1;

        uint8_t scales[8], mins[4];
        for (int j = 0; j < 4; j++) {
            scales[j] = (utmp[0]>>(j*8))&0x3F;
            scales[j+4] = (utmp[1]>>(j*8))&0x3F;
            mins[j] = (utmp[2]>>(j*8))&0x3F;
        }
        uint8_t qh[32];
        for (int j = 0; j < 32; j++) qh[j] = block->qh[j];

        for (int iter = 0; iter < 4; iter++) {
            float dl0 = d*scales[iter*2], ml0 = dmin*mins[iter/2], dl1 = d*scales[iter*2+1];
            for (int k = 0; k < 16; k++) {
                uint8_t qs0 = block->qs[iter*32+k], qs1 = block->qs[iter*32+16+k];
                float q0 = (qs0&0x0F)+((qh[k]&1)?16.0f:0.0f), q1 = (qs1&0x0F)+((qh[16+k]&1)?16.0f:0.0f);
                float q2 = ((qs0>>4)&0x0F)+((qh[k]&2)?16.0f:0.0f), q3 = ((qs1>>4)&0x0F)+((qh[16+k]&2)?16.0f:0.0f);
                sumf += (dl0*q0-ml0)*by[iter*64+k]; sumf += (dl0*q1-ml0)*by[iter*64+16+k];
                sumf += (dl1*q2-ml0)*by[iter*64+32+k]; sumf += (dl1*q3-ml0)*by[iter*64+48+k];
            }
        }
    }
    if (tiisg == 0) dst[row] = simd_sum(sumf);
}
)METAL";

// llama.cpp kernel (exact from mul_mv.metal:1632-1747)
static const char* llama_kernel_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03; int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13; int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

template<int nr0>
void kernel_mul_mv_q5_K_f32_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x, r1 = tgpig.y, im = tgpig.z;
    const int first_row = (r0*1 + sgitg) * nr0;
    const uint i12 = im%1, i13 = im/1;
    const uint64_t offset0 = first_row*args.nb01 + (i12/1)*args.nb02 + (i13/1)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float * yy = (device const float *) (src1 + offset1);

    float sumf[nr0] = {0.f};
    float yl[16], yh[16];
    constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const short tid = tiisg/4, ix = tiisg%4, iq = tid/4, ir = tid%4;
    const short l0 = 8*ir, q_offset = 32*iq + l0, y_offset = 64*iq + l0;
    const uint8_t hm1 = 1u << (2*iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }
        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f}, acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            sumf[row] += dh[0] * (sc8[0] * (acc1[0] + 16.f*acc2[0]) + sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) + sc8[4] * (acc1[2] + 16.f*acc2[2]) + sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) - dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += args.nb01; qh += args.nb01; dh += args.nb01/2; a += args.nb01/2;
        }
        y1 += 4 * QK_K;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        if (tiisg == 0) dst_f32[first_row + row] = simd_sum(sumf[row]);
    }
}

kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device       char * dst [[buffer(3)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_impl<1>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}
)METAL";

int main() {
    @autoreleasepool {
        printf("=== FAIR COMPARISON: Q5_K GEMV ===\n");
        printf("Our optimized implementation vs llama.cpp's exact kernel\n");
        printf("Same test data, same hardware, same conditions\n\n");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* error = nil;

        id<MTLLibrary> our_lib = [device newLibraryWithSource:[NSString stringWithUTF8String:our_kernel_source] options:nil error:&error];
        if (!our_lib) { printf("Our kernel error: %s\n", [[error localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> our_pipeline = [device newComputePipelineStateWithFunction:[our_lib newFunctionWithName:@"kernel_q5_k_gemv"] error:&error];

        id<MTLLibrary> llama_lib = [device newLibraryWithSource:[NSString stringWithUTF8String:llama_kernel_source] options:nil error:&error];
        if (!llama_lib) { printf("llama.cpp kernel error: %s\n", [[error localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> llama_pipeline = [device newComputePipelineStateWithFunction:[llama_lib newFunctionWithName:@"kernel_mul_mv_q5_K_f32"] error:&error];

        int sizes[][2] = {{128, 4096}, {512, 4096}, {1024, 4096}, {2048, 4096}};

        for (int t = 0; t < 4; t++) {
            int m = sizes[t][0], k = sizes[t][1], n_blocks = k / QK_K;
            printf("Matrix: %d × %d (%d blocks)\n", m, k, n_blocks);

            // SAME test data for both
            block_q5_k* weights = (block_q5_k*)malloc(m * n_blocks * sizeof(block_q5_k));
            float* input = (float*)malloc(k * sizeof(float));
            create_test_data(weights, input, m, k, 42); // Fixed seed for reproducibility

            id<MTLBuffer> weights_buf = [device newBufferWithBytes:weights length:m*n_blocks*sizeof(block_q5_k) options:MTLResourceStorageModeShared];
            id<MTLBuffer> input_buf = [device newBufferWithBytes:input length:k*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> our_out = [device newBufferWithLength:m*sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> llama_out = [device newBufferWithLength:m*sizeof(float) options:MTLResourceStorageModeShared];

            // Our args
            struct { uint32_t ne00; uint32_t ne01; uint64_t nb01; } our_args = {(uint32_t)k, (uint32_t)m, (uint64_t)(n_blocks*sizeof(block_q5_k))};
            id<MTLBuffer> our_args_buf = [device newBufferWithBytes:&our_args length:sizeof(our_args) options:MTLResourceStorageModeShared];

            // llama.cpp args
            struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03; int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13; int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } llama_args = {k, m, 1, 1, (uint64_t)(n_blocks*sizeof(block_q5_k)), 0, 0, k, 1, 1, 4, (uint64_t)(k*4), 0, 0, m, 1, 1, 1, 1};
            id<MTLBuffer> llama_args_buf = [device newBufferWithBytes:&llama_args length:sizeof(llama_args) options:MTLResourceStorageModeShared];

            // Warmup
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
                    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                }
                @autoreleasepool {
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:llama_pipeline];
                    [enc setBuffer:llama_args_buf offset:0 atIndex:0];
                    [enc setBuffer:weights_buf offset:0 atIndex:1];
                    [enc setBuffer:input_buf offset:0 atIndex:2];
                    [enc setBuffer:llama_out offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                }
            }

            // Measure our performance
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
                    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                }
            }
            uint64_t end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            double our_us = (end - start) / 1000.0 / iterations;
            double our_gops = (m * k * 2.0) / (our_us * 1e-6) / 1e9;

            // Measure llama.cpp performance
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
                    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                }
            }
            end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            double llama_us = (end - start) / 1000.0 / iterations;
            double llama_gops = (m * k * 2.0) / (llama_us * 1e-6) / 1e9;

            printf("  Our implementation:   %6.1f µs (%5.2f GOPS)\n", our_us, our_gops);
            printf("  llama.cpp:            %6.1f µs (%5.2f GOPS)\n", llama_us, llama_gops);

            if (our_gops > llama_gops * 1.01) { // >1% faster
                printf("  ✓ %.2fx FASTER than llama.cpp\n\n", our_gops/llama_gops);
            } else if (llama_gops > our_gops * 1.01) { // >1% slower
                printf("  ✗ %.2fx slower than llama.cpp\n\n", llama_gops/our_gops);
            } else {
                printf("  ≈ Performance match (within 1%%)\n\n");
            }

            free(weights); free(input);
        }
    }
    return 0;
}
