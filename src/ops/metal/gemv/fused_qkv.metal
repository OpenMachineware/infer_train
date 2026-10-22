// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Fused QKV GEMM - combines Q, K, V projections into one kernel
// Instead of 3 separate GEMVs with small M (256 each), we do one GEMM with M=Q+K+V

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32

// Include block structure definitions
#define GGML_COMMON_DECL_METAL
#include "../include/blocks.h"

// Fused QKV kernel for Q5_K format
// Processes Q, K, V projections in one kernel launch
// Grid: [M_total / rows_per_tg, batch_size]
// Each threadgroup handles multiple rows to increase parallelism

kernel void kernel_fused_qkv_q5_k(
    device const char * src0_q [[buffer(0)]],  // Q weights
    device const char * src0_k [[buffer(1)]],  // K weights
    device const char * src0_v [[buffer(2)]],  // V weights
    device const float * src1 [[buffer(3)]],   // Input vector
    device float * dst_q [[buffer(4)]],        // Q output
    device float * dst_k [[buffer(5)]],        // K output
    device float * dst_v [[buffer(6)]],        // V output
    constant uint32_t & m_q [[buffer(7)]],     // Q output size
    constant uint32_t & m_k [[buffer(8)]],     // K output size
    constant uint32_t & m_v [[buffer(9)]],     // V output size
    constant uint32_t & k_dim [[buffer(10)]],  // Hidden size
    constant uint64_t & nb01_q [[buffer(11)]], // Q row stride
    constant uint64_t & nb01_k [[buffer(12)]], // K row stride
    constant uint64_t & nb01_v [[buffer(13)]], // V row stride
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int m_total = m_q + m_k + m_v;
    const int nb = k_dim / QK_K;

    // Determine which projection this threadgroup handles
    const int row_global = tgpig.x;

    device const char * weight_ptr;
    device float * dst_ptr;
    uint64_t nb01;
    int row_local;

    if (row_global < (int)m_q) {
        // Q projection
        weight_ptr = src0_q;
        dst_ptr = dst_q;
        nb01 = nb01_q;
        row_local = row_global;
    } else if (row_global < (int)(m_q + m_k)) {
        // K projection
        weight_ptr = src0_k;
        dst_ptr = dst_k;
        nb01 = nb01_k;
        row_local = row_global - m_q;
    } else {
        // V projection
        weight_ptr = src0_v;
        dst_ptr = dst_v;
        nb01 = nb01_v;
        row_local = row_global - m_q - m_k;
    }

    device const block_q5_K * blocks = (device const block_q5_K *)(weight_ptr + row_local * nb01);
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
        device const block_q5_K * block = blocks + i;

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

        // Process 4 values at a time
        FOR_UNROLL(short l = 0; l < 8; ++l) {
            const uint8_t aux = q1[l];
            const float d = dh[0];
            const float m = dh[1];

            float sumq = d * sc8[l];
            acc1[0] += sumq * (aux & 0x0F);
            acc1[1] += sumq * (aux >> 4);
        }

        FOR_UNROLL(short l = 0; l < 8; ++l) {
            const uint8_t aux = q2[l];
            const float d = dh[0];
            const float m = dh[1];

            float sumq = d * sc8[8+l];
            acc1[2] += sumq * (aux & 0x0F);
            acc1[3] += sumq * (aux >> 4);
        }

        sumf += (acc1[0] + acc1[1]) * (sumy[0] + sumy[1]) +
                (acc1[2] + acc1[3]) * (sumy[2] + sumy[3]);

        y1 += 4*QK_K;
    }

    // Simdgroup reduction
    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        dst_ptr[row_local] = sumf;
    }
}
