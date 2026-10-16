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
// 2. Basic version: NF=8
// ============================================================================
kernel void kernel_mul_mv_f16_f32(
        constant mul_mv_args & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = args.nsg;  // Get NSG from parameters

    constexpr short NR0 = 2;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00 / NB;

    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12_batch;
    const uint i13 = im / args.ne12_batch;

    const uint64_t offset1 = r1 * args.nb11 + (i12) * args.nb12 + (i13) * args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const half * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
        ax[row] = (device const half *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg / (NW / NF);
    const short il = tiisg % (NW / NF);

    const int ib0 = sgitg * NF + ix;

    float yl[NF];

    device const float * yb = y + (ib0 * NB + il * NF);

    for (int ib = ib0; ib < nb; ib += NSG * NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const half * xb = ax[row] + (ib * NB + il * NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG * NF * NW;
    }

    for (int i = nb * NB + sgitg * NW + tiisg; i < args.ne00; i += NW * NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// ============================================================================
// 3. Optimized version: float4, simple strided loop
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
    constexpr short NR0 = 2;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;

    const int nb = args.ne00 / NB;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;

    device const float4 * y4 = (device const float4 *) (src1 + r1 * args.nb11);

    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *) (src0 + (r0 + row) * args.nb01);
    }

    float sumf[NR0] = { 0.f };

    // Simple strided loop: each thread processes blocks tiisg, tiisg+32, tiisg+64, etc.
    for (int ib = tiisg; ib < nb; ib += NW) {
        float4 y4b[4] = { y4[ib + 0], y4[ib + 1], y4[ib + 2], y4[ib + 3] };

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            float4 acc = 0.0f;
            for (int j = 0; j < 4; ++j) {
                acc += float4(ax4[row][ib * 4 + j]) * y4b[j];
            }
            sumf[row] += acc[0] + acc[1] + acc[2] + acc[3];
        }
    }

    // Tail handling for K % 32 != 0
    device const half * ax[NR0];
    device const float * y = (device const float *) (src1 + r1 * args.nb11);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax[row] = (device const half *) (src0 + (r0 + row) * args.nb01);
    }

    for (int i = nb * NB + tiisg; i < args.ne00; i += NW) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    // Reduce and write
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)r1 * args.ne0;

    if (tiisg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            if (r0 + row < args.ne01) {
                dst_f32[r0 + row] = sumf[row];
            }
        }
    }
}
