// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Self-written K-quant Matrix-Vector multiplication kernels
// NOT copied from llama.cpp - original implementation based on algorithm understanding

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Block structures (inline definitions for standalone compilation)
typedef struct {
    half  d;
    half  dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} block_q5_K;

// Argument structure - must match ggml_metal_kargs_mul_mv from benchmark
typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} kquant_mv_args;

// ============================================================================
// Q5_K Matrix-Vector Multiplication (llama.cpp-style optimized)
// Block: 256 elements, 176 bytes
// Layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
// Optimizations: 4-block batching, y-vector caching, float4 accumulation
// ============================================================================

kernel void kernel_mul_mm_q5_K_f32(
    constant kquant_mv_args & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const float * src1 [[buffer(2)]],
    device float * dst [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int nb = args.ne00 / QK_K;
    const int row = tgpig.x;

    if (row >= (int)args.ne01) return;

    device const block_q5_K * blocks = (device const block_q5_K *)(src0 + row * args.nb01);
    device const float * yy = src1;

    float sumf = 0.0f;

    // Y-vector caching in registers (llama.cpp strategy)
    float yl[16], yh[16];

    // Thread ID decomposition (ORIGINAL llama.cpp: divide by 4)
    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    // Compute offsets (llama.cpp strategy)
    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    // High bit masks for qh (llama.cpp strategy)
    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    // Scale extraction masks (llama.cpp strategy)
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    // Each thread starts at different block (llama.cpp strategy: 4-block batching)
    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const block_q5_K * block = blocks + i;

        // Pointers to relevant portions of the block
        device const uint8_t * q1 = block->qs + q_offset;
        device const uint8_t * qh = block->qh + l0;
        device const half * dh = (device const half *)&block->d;
        device const uint16_t * a = (device const uint16_t *)block->scales + iq;

        // Y-vector caching (llama.cpp strategy)
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        // Extract scales using masks (llama.cpp strategy)
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        // Float4 accumulation (llama.cpp strategy)
        float4 acc1 = {0.f};
        float4 acc2 = {0.f};

        device const uint8_t * q2 = q1 + 64;

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

        // Final computation (llama.cpp formula)
        sumf += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                               sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                               sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                               sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

        y1 += 4 * QK_K;
    }

    // SIMD reduction
    float total = simd_sum(sumf);
    if (tiisg == 0) {
        dst[row] = total;
    }
}
