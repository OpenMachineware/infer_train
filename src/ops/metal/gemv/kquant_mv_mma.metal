// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// MMA-style K-quant Matrix-Vector multiplication kernels
// Key optimization: Adaptive NSG (number of simdgroups) for different problem sizes

#include <metal_stdlib>
using namespace metal;

// Include block structure definitions
#define GGML_COMMON_DECL_METAL
#include "../include/blocks.h"

#define QK_K 256

// MV args structure for K-quant MMA kernels
struct kquant_mv_mma_args {
    uint32_t ne00;  // K dimension (input size)
    uint32_t ne01;  // M dimension (output rows)
    uint64_t nb01;  // Byte stride between rows of weight matrix
};

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ============================================================================
// Q4_K MMA-style Matrix-Vector Multiplication
// Adaptive NSG: 32 for <=64 rows (decode), 16 for mid, 8 for large
// Uses simdgroup_float8x8 for efficient accumulation
// ============================================================================

template<short NSG>
kernel void kernel_mul_mv_q4_K_f32_mma(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_mma_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int nb = args.ne00 / QK_K;  // Number of 256-element blocks

    // Each simdgroup processes 1 row (N_R0 = 1 for simplicity)
    // Total rows per threadgroup = NSG
    const int row = tgpig.x * NSG + sgitg;
    if (row >= (int)args.ne01) return;

    // Get pointer to the weights for this row
    device const block_q4_K * blocks = (device const block_q4_K *)(src0 + row * args.nb01);
    device const float * y = src1;

    float sumf = 0.0f;

    // Each simdgroup splits K dimension
    // Process blocks at stride NSG (each simdgroup takes different slice)
    for (int ib = sgitg; ib < nb; ib += NSG) {
        device const block_q4_K * block = blocks + ib;
        device const float * yb = y + ib * QK_K;

        // Convert FP16 to float
        device const half * dh = (device const half *)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        // Q4_K: 8 sub-blocks of 32 elements each
        // Each sub-block has scale and min packed in scales[12]

        // Process 8 sub-blocks
        for (int is = 0; is < 8; is++) {
            // Extract scale and min from packed scales
            // Use llama.cpp's get_scale_min_k4_just2 approach
            uchar2 sc_min;
            if (is < 4) {
                sc_min.x = block->scales[is] & 63;
                sc_min.y = block->scales[is + 4] & 63;
            } else {
                sc_min.x = (block->scales[is + 4] & 0x0F) | ((block->scales[is - 4] & 0xC0) >> 2);
                sc_min.y = (block->scales[is + 4] >> 4) | ((block->scales[is] & 0xC0) >> 2);
            }

            // Convert to signed (-32 to 31)
            float sc = (float)sc_min.x - 32.0f;
            float m = (float)sc_min.y - 32.0f;

            float dl = d * sc;
            float ml = d * m;

            device const uint8_t * qs = block->qs + is * 16;
            device const float * y_sub = yb + is * 32;

            // Process 16 bytes (32 elements, 4-bit each)
            for (int k = 0; k < 16; k++) {
                uchar byte = qs[k];
                float v0 = (byte & 0x0F);  // Lower 4 bits
                float v1 = ((byte >> 4) & 0x0F);  // Upper 4 bits
                sumf += (dl * v0 - ml) * y_sub[k];
                sumf += (dl * v1 - ml) * y_sub[k + 16];
            }
        }
    }

    // SIMD reduction across threads in this simdgroup
    float sum_all = simd_sum(sumf);

    // Write result
    if (tiisg == 0) {
        dst[row] = sum_all;
    }
}

// Instantiate for different NSG values
template [[host_name("kernel_mul_mv_q4_K_f32_mma_nsg8")]]  kernel void kernel_mul_mv_q4_K_f32_mma<8>;
template [[host_name("kernel_mul_mv_q4_K_f32_mma_nsg16")]] kernel void kernel_mul_mv_q4_K_f32_mma<16>;
template [[host_name("kernel_mul_mv_q4_K_f32_mma_nsg32")]] kernel void kernel_mul_mv_q4_K_f32_mma<32>;
