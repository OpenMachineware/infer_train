// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// F16 GEMV kernels

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH 32

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
// 2. Basic version: simple strided loop for all K >= 32
// Each thread processes blocks with stride=32 to ensure complete coverage
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
    constexpr short NB = 32;

    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;

    device const half * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax[row] = (device const half *) (src0 + (r0 + row) * args.nb01);
    }

    device const float * y = (device const float *) (src1 + r1 * args.nb11);

    float sumf[NR0] = { 0.f };

    // Main loop: process 32 elements per iteration
    // Each thread starts at its index, stride by 32 to ensure all elements covered
    for (int i = tiisg; i < args.ne00; i += NW) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            sumf[row] += (float)ax[row][i] * y[i];
        }
    }

    // Reduce within simdgroup
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    // Write result
    device float * dst_f32 = (device float *) dst + (uint64_t)r1 * args.ne0;

    if (tiisg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (r0 + row < args.ne01) {
                dst_f32[r0 + row] = sumf[row];
            }
        }
    }
}

// ============================================================================
// 3. Optimized version: float4 vectorization
// Use vec4 for better throughput, simple strided loop pattern
// ============================================================================
kernel void kernel_mul_mv_f16_f32_4(
        constant mul_mv_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr short NR0 = 2;
    constexpr short NW = N_SIMDWIDTH;  // 32
    constexpr short VEC_WIDTH = 4;     // Process 4 elements at a time

    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;

    // Cast to vec4 pointers for vectorized access
    device const float4 * y4 = (device const float4 *) (src1 + r1 * args.nb11);
    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *) (src0 + (r0 + row) * args.nb01);
    }

    float sumf[NR0] = { 0.f };

    const int num_vec4 = args.ne00 / VEC_WIDTH;  // Number of float4 elements

    // Vectorized loop: each thread processes vec4 elements with stride 32
    for (int i = tiisg; i < num_vec4; i += NW) {
        float4 y4_val = y4[i];

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            float4 w4_val = float4(ax4[row][i]);
            sumf[row] += dot(w4_val, y4_val);
        }
    }

    // Tail: handle remaining elements (K % 4 != 0)
    device const float * y = (device const float *) (src1 + r1 * args.nb11);
    device const half * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax[row] = (device const half *) (src0 + (r0 + row) * args.nb01);
    }

    for (int i = num_vec4 * VEC_WIDTH + tiisg; i < args.ne00; i += NW) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            sumf[row] += (float)ax[row][i] * y[i];
        }
    }

    // Reduce within simdgroup
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    // Write result
    device float * dst_f32 = (device float *) dst + (uint64_t)r1 * args.ne0;

    if (tiisg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (r0 + row < args.ne01) {
                dst_f32[r0 + row] = sumf[row];
            }
        }
    }
}
