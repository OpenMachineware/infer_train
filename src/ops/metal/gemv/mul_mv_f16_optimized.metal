// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Optimized F16 GEMV kernel with nxpsg support
// Matches llama.cpp kernel_mul_mv_ext_q4_f32_impl structure

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Function constants
constant short FC_mul_mv_nsg   [[function_constant(600)]];
constant short FC_mul_mv_nxpsg [[function_constant(601)]];
constant short FC_mul_mv_ne12  [[function_constant(602)]];
constant short FC_mul_mv_r2    [[function_constant(603)]];
constant short FC_mul_mv_r3    [[function_constant(604)]];

// Argument structure matching llama.cpp
typedef struct {
    int32_t  ne00;  // K
    int32_t  ne01;  // M (rows)
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;  // batch size
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
} mul_mv_ext_args;

// Helper macro to write kernels for different r1ptg values
#define KERNEL_MUL_MV_EXT_F16_F32(r1ptg, name) \
kernel void name( \
        constant mul_mv_ext_args & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3   tgpig[[threadgroup_position_in_grid]], \
        ushort  tiisg[[thread_index_in_simdgroup]], \
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) { \
    \
    const short NSG   = FC_mul_mv_nsg; \
    const short nxpsg = FC_mul_mv_nxpsg; \
    \
    const short chpt = 4; \
    const short nypsg = 32 / nxpsg; \
    \
    const short tx = tiisg % nxpsg; \
    const short ty = tiisg / nxpsg; \
    \
    const int i01 = tgpig.x * (nypsg * NSG) + nypsg * sgitg + ty; \
    const int i11 = tgpig.y * r1ptg; \
    const int i1m = tgpig.z; \
    \
    const int i12 = i1m % FC_mul_mv_ne12; \
    const int i13 = i1m / FC_mul_mv_ne12; \
    \
    const uint64_t offset0 = i01 * args.nb01 + (i12 / FC_mul_mv_r2) * args.nb02 + (i13 / FC_mul_mv_r3) * args.nb03; \
    const uint64_t offset1 = i11 * args.nb11 + i12 * args.nb12 + i13 * args.nb13; \
    \
    device const half4 * xq = (i01 < args.ne01) \
        ? (device const half4 *) (src0 + offset0) + tx \
        : (device const half4 *) src0; \
    \
    device const float4 * y4[r1ptg]; \
    \
    FOR_UNROLL(int ir1 = 0; ir1 < r1ptg; ++ir1) { \
        y4[ir1] = (i11 + ir1 < args.ne11) \
            ? (device const float4 *) (src1 + offset1 + ir1 * args.nb11) + tx \
            : (device const float4 *) src1; \
    } \
    \
    float sumf[r1ptg] = { 0.0f }; \
    \
    for (int ich = tx; 4 * ich < args.ne00; ich += chpt * nxpsg) { \
        float4 lx[chpt]; \
        \
        FOR_UNROLL(short ch = 0; ch < chpt; ++ch) { \
            lx[ch] = float4(xq[ch * nxpsg]); \
        } \
        \
        FOR_UNROLL(short ch = 0; ch < chpt; ++ch) { \
            FOR_UNROLL(short ir1 = 0; ir1 < r1ptg; ++ir1) { \
                sumf[ir1] += dot(lx[ch], y4[ir1][ch * nxpsg]); \
            } \
        } \
        \
        xq += chpt * nxpsg; \
        FOR_UNROLL(short ir1 = 0; ir1 < r1ptg; ++ir1) { \
            y4[ir1] += chpt * nxpsg; \
        } \
    } \
    \
    FOR_UNROLL(short ir1 = 0; ir1 < r1ptg; ++ir1) { \
        if (nxpsg >= 32) { \
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16); \
        } \
        if (nxpsg >= 16) { \
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 8); \
        } \
        if (nxpsg >= 8) { \
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 4); \
        } \
        if (nxpsg >= 4) { \
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 2); \
        } \
        if (nxpsg >= 2) { \
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 1); \
        } \
    } \
    \
    if (tx == 0) { \
        FOR_UNROLL(short ir1 = 0; ir1 < r1ptg; ++ir1) { \
            if (i11 + ir1 < args.ne11) { \
                device float * dst_f32 = (device float *) dst + (uint64_t)i1m * args.ne0 * args.ne1 + (uint64_t)(i11 + ir1) * args.ne0; \
                \
                if (i01 < args.ne01) { \
                    dst_f32[i01] = sumf[ir1]; \
                } \
            } \
        } \
    } \
}

// Generate kernels for r1ptg = 1, 2, 3, 4, 5
KERNEL_MUL_MV_EXT_F16_F32(1, kernel_mul_mv_ext_f16_f32_r1_1)
KERNEL_MUL_MV_EXT_F16_F32(2, kernel_mul_mv_ext_f16_f32_r1_2)
KERNEL_MUL_MV_EXT_F16_F32(3, kernel_mul_mv_ext_f16_f32_r1_3)
KERNEL_MUL_MV_EXT_F16_F32(4, kernel_mul_mv_ext_f16_f32_r1_4)
KERNEL_MUL_MV_EXT_F16_F32(5, kernel_mul_mv_ext_f16_f32_r1_5)
