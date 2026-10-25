// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// F16 GEMV kernels

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH 32

// Function constants for runtime kernel specialization
// Base index 600 matches llama.cpp convention
constant short FC_mul_mv_nsg [[function_constant(600)]];

// Parameter struct for matrix-vector multiplication
typedef struct {
    int32_t  ne00;  // K
    int32_t  ne01;  // M (rows)
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;  // row stride
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
    int16_t  nsg;   // Number of simdgroups (passed as parameter)
    int16_t  ne12_batch;  // batch dimension
} mul_mv_args;

// ============================================================================
// Helper: reduce and write
// ============================================================================
template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

// ============================================================================
// 1. Short version: K < 32 (single-batch optimized)
// Minimal overhead: direct offsets, no batch/multimodal calculations
// ============================================================================
kernel void kernel_mul_mv_f16_f32_short(
        constant mul_mv_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int r0 = tgpig.x * 32 + tiisg;
    const int r1 = tgpig.y;
    if (r0 >= args.ne01) return;

    device const half * x = (device const half *) (src0 + r0 * args.nb01);
    device const float * y = (device const float *) (src1 + r1 * args.nb11);
    device float * dst_f32 = (device float *) dst + (uint64_t)r1 * args.ne0;

    float res = 0.0f;
    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * y[i];
    }
    dst_f32[r0] = res;
}

// ============================================================================
// 2. Basic version: optimized with float4 and strided processing
// ============================================================================
kernel void kernel_mul_mv_f16_f32(
        constant mul_mv_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NW = N_SIMDWIDTH;  // 32

    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;

    float sumf[NR0] = { 0.f, 0.f };

    // Cast to float4 pointers for vectorized access
    device const float4 * y4 = (device const float4 *) (src1 + r1 * args.nb11);
    device const half4 * ax4[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *) (src0 + (r0 + row) * args.nb01);
    }

    const int n4 = args.ne00 / 4;  // Number of float4 vectors

    // Process float4 vectors in strided fashion for complete coverage
    for (int i4 = tiisg; i4 < n4; i4 += NW) {
        float4 yv = y4[i4];
        for (short row = 0; row < NR0; ++row) {
            float4 wv = float4(ax4[row][i4]);
            float4 prod = wv * yv;
            sumf[row] += prod[0] + prod[1] + prod[2] + prod[3];
        }
    }

    // Tail: remaining elements (K % 4)
    if (n4 * 4 < args.ne00) {
        device const half * ax[NR0];
        device const float * y = (device const float *) (src1 + r1 * args.nb11);
        for (short row = 0; row < NR0; ++row) {
            ax[row] = (device const half *) (src0 + (r0 + row) * args.nb01);
        }

        for (int i = n4 * 4 + tiisg; i < args.ne00; i += NW) {
            for (short row = 0; row < NR0; ++row) {
                sumf[row] += (float)ax[row][i] * y[i];
            }
        }
    }

    // Reduce within simdgroup
    for (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    // Write result
    device float * dst_f32 = (device float *) dst + (uint64_t)r1 * args.ne0;

    if (tiisg == 0) {
        for (short row = 0; row < NR0; ++row) {
            if (r0 + row < args.ne01) {
                dst_f32[r0 + row] = sumf[row];
            }
        }
    }
}

// ============================================================================
// 3. Optimized version: float4 with multi-simdgroup cooperation (K % 4 == 0)
// Exact match of llama.cpp implementation
// ============================================================================

kernel void kernel_mul_mv_f16_f32_4(
        constant mul_mv_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;  // Use function constant instead of template

    constexpr short NR0 = 2;
    constexpr short NW = N_SIMDWIDTH;  // 32
    constexpr short NB  = 32;  // Block size in elements
    constexpr short NF  = 16;  // Floats per thread per iteration
    constexpr short NF4 = NF/4;  // Float4 vectors per thread

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;

    const int nb = args.ne00/NB;  // Number of 32-element blocks

    // Base pointers
    device const float  * y  = (device const float *) (src1 + r1*args.nb11);
    device const float4 * y4 = (device const float4 *) (src1 + r1*args.nb11);

    device const half  * ax [NR0];
    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax [row] = (device const half *) (src0 + (r0 + row)*args.nb01);
        ax4[row] = (device const half4 *) (src0 + (r0 + row)*args.nb01);
    }

    float sumf[NR0] = { 0.f };

    // Work distribution exactly matching llama.cpp
    const short ix = tiisg/(NW/NF);  // tiisg/2, range 0-15
    const short il = tiisg%(NW/NF);  // tiisg%2, range 0-1

    const int ib0 = sgitg*NF + ix;  // Starting block

    float4 yl4[NF4];  // Local cache for y values

    // Initial y pointer - MATCHES LLAMA.CPP EXACTLY
    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

    // Main loop - stride by NSG*NF blocks
    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        // Load y values into local cache
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        // Compute dot products
        for (short row = 0; row < NR0; ++row) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        // Advance y pointer - CRITICAL DIFFERENCE
        yb4 += NSG*NF*NW/4;
    }

    // Handle tail elements
    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; ++row) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    // Reduce and write
    device float * dst_f32 = (device float *) dst + (uint64_t)r1*args.ne0;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// ============================================================================
// F16×F16 GEMV for output projection with F16 weights and F16 input
// ============================================================================
kernel void kernel_mul_mv_f16_f16(
        constant mul_mv_args & args,
        device const char * src0,  // F16 weights [M, K]
        device const char * src1,  // F16 input [K]
        device       char * dst,   // F32 output [M]
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        threadgroup  char * shmem [[threadgroup(0)]]) {
    const int r0 = tgpig.x;
    if (r0 >= args.ne01) return;

    device const half * ax = (device const half *) (src0 + r0*args.nb01);
    device const half * y  = (device const half *) src1;

    float sum = 0.0f;

    // Process elements in chunks
    for (int i = tiisg; i < args.ne00; i += N_SIMDWIDTH) {
        sum += float(ax[i]) * float(y[i]);
    }

    // Reduce across simdgroup
    sum = simd_sum(sum);

    // Write result
    if (tiisg == 0) {
        device float * dst_f32 = (device float *) dst;
        dst_f32[r0] = sum;
    }
}
