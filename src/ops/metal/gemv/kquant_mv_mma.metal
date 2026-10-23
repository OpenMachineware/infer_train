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
// Correct architecture: All simdgroups collaborate on K dimension for same row
// Threads within a simdgroup split the work within each block
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
    const int row = tgpig.x;  // Each threadgroup processes ONE row

    if (row >= (int)args.ne01) return;

    // Get pointer to the weights for this row
    device const block_q4_K * blocks = (device const block_q4_K *)(src0 + row * args.nb01);
    device const float * y = src1;

    float sumf = 0.0f;

    // Each simdgroup processes a different K-slice
    // Simdgroup 0: blocks [0, nb/NSG)
    // Simdgroup 1: blocks [nb/NSG, 2*nb/NSG)
    // ...
    for (int ib = sgitg; ib < nb; ib += NSG) {
        device const block_q4_K * block = blocks + ib;
        device const float * yb = y + ib * QK_K;

        // Convert FP16 to float
        device const half * dh = (device const half *)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        // Q4_K: 8 sub-blocks of 32 elements each
        // Each thread processes a subset of the sub-blocks
        // 32 threads total, 8 sub-blocks -> 4 threads per sub-block
        // Each thread processes 4 bytes (8 quantized values)
        const int sub_blocks_per_thread = (8 * 16 + 31) / 32; // How many bytes each thread processes

        for (int idx = tiisg; idx < 8 * 16; idx += 32) {
            const int is = idx / 16; // Sub-block index (0-7)
            const int k = idx % 16;  // Byte index within sub-block (0-15)

            // Extract scale and min from packed scales
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

            uchar byte = qs[k];
            float v0 = (byte & 0x0F);  // Lower 4 bits
            float v1 = ((byte >> 4) & 0x0F);  // Upper 4 bits
            sumf += (dl * v0 - ml) * y_sub[k];
            sumf += (dl * v1 - ml) * y_sub[k + 16];
        }
    }

    // Simdgroup-level reduction
    float sum_sg = simd_sum(sumf);

    // Threadgroup memory for cross-simdgroup reduction
    threadgroup float partial_sums[32];  // Max NSG = 32
    if (tiisg == 0) {
        partial_sums[sgitg] = sum_sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of simdgroup 0 sums all partial results
    if (sgitg == 0 && tiisg == 0) {
        float total = 0.0f;
        for (short i = 0; i < NSG; i++) {
            total += partial_sums[i];
        }
        dst[row] = total;
    }
}

// Debug version that writes partial sums for inspection
template<short NSG>
kernel void kernel_mul_mv_q4_K_f32_mma_debug(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_mma_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int nb = args.ne00 / QK_K;
    const int row = tgpig.x;

    if (row >= (int)args.ne01) return;

    device const block_q4_K * blocks = (device const block_q4_K *)(src0 + row * args.nb01);
    device const float * y = src1;

    float sumf = 0.0f;

    for (int ib = sgitg; ib < nb; ib += NSG) {
        device const block_q4_K * block = blocks + ib;
        device const float * yb = y + ib * QK_K;

        device const half * dh = (device const half *)&block->d;
        float d = dh[0];

        // Each thread processes a unique subset of the block
        for (int idx = tiisg; idx < 8 * 16; idx += 32) {
            const int is = idx / 16;
            const int k = idx % 16;

            uchar2 sc_min;
            if (is < 4) {
                sc_min.x = block->scales[is] & 63;
                sc_min.y = block->scales[is + 4] & 63;
            } else {
                sc_min.x = (block->scales[is + 4] & 0x0F) | ((block->scales[is - 4] & 0xC0) >> 2);
                sc_min.y = (block->scales[is + 4] >> 4) | ((block->scales[is] & 0xC0) >> 2);
            }

            float sc = (float)sc_min.x - 32.0f;
            float m = (float)sc_min.y - 32.0f;
            float dl = d * sc;
            float ml = d * m;

            device const uint8_t * qs = block->qs + is * 16;
            device const float * y_sub = yb + is * 32;

            uchar byte = qs[k];
            float v0 = (byte & 0x0F);
            float v1 = ((byte >> 4) & 0x0F);
            sumf += (dl * v0 - ml) * y_sub[k];
            sumf += (dl * v1 - ml) * y_sub[k + 16];
        }
    }

    float sum_sg = simd_sum(sumf);

    // Write debug output
    if (tiisg == 0) {
        dst[row * (NSG + 1) + sgitg] = sum_sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float total = 0.0f;
        for (short i = 0; i < NSG; i++) {
            total += dst[row * (NSG + 1) + i];
        }
        dst[row * (NSG + 1) + NSG] = total;
    }
}

// Instantiate for different NSG values using typedef pattern
typedef decltype(kernel_mul_mv_q4_K_f32_mma<32>) kquant_mma_t;

template [[host_name("kernel_mul_mv_q4_K_f32_mma_nsg8")]]  kernel kquant_mma_t kernel_mul_mv_q4_K_f32_mma<8>;
template [[host_name("kernel_mul_mv_q4_K_f32_mma_nsg16")]] kernel kquant_mma_t kernel_mul_mv_q4_K_f32_mma<16>;
template [[host_name("kernel_mul_mv_q4_K_f32_mma_nsg32")]] kernel kquant_mma_t kernel_mul_mv_q4_K_f32_mma<32>;

// ============================================================================
// Q4_K Multi-Row GEMV for Decode
// NT: Number of 8-row tiles per threadgroup (1-4)
// Each threadgroup processes NT*8 rows simultaneously
// Better memory coalescing and GPU utilization for decode scenario
// ============================================================================
template<short NT, short NSG>
kernel void kernel_mul_mv_q4_K_f32_decode(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant kquant_mv_mma_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int nb = args.ne00 / QK_K;  // Number of 256-element blocks
    const int row0 = tgpig.x * NT * 8;  // First row for this threadgroup

    if (row0 >= (int)args.ne01) return;

    // Each simdgroup processes a different K-slice
    float sumf[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // One sum per row in tile

    for (int ib = sgitg; ib < nb; ib += NSG) {
        device const float * yb = src1 + ib * QK_K;

        // Process all NT*8 rows for this K-slice
        for (short t = 0; t < NT; t++) {
            for (short r = 0; r < 8; r++) {
                const int row = row0 + t * 8 + r;
                if (row >= (int)args.ne01) break;

                device const block_q4_K * block =
                    (device const block_q4_K *)(src0 + row * args.nb01) + ib;

                device const half * dh = (device const half *)&block->d;
                float d = dh[0];

                // Each thread processes a subset of the block
                for (int idx = tiisg; idx < 8 * 16; idx += 32) {
                    const int is = idx / 16;
                    const int k = idx % 16;

                    uchar2 sc_min;
                    if (is < 4) {
                        sc_min.x = block->scales[is] & 63;
                        sc_min.y = block->scales[is + 4] & 63;
                    } else {
                        sc_min.x = (block->scales[is + 4] & 0x0F) | ((block->scales[is - 4] & 0xC0) >> 2);
                        sc_min.y = (block->scales[is + 4] >> 4) | ((block->scales[is] & 0xC0) >> 2);
                    }

                    float sc = (float)sc_min.x - 32.0f;
                    float m = (float)sc_min.y - 32.0f;
                    float dl = d * sc;
                    float ml = d * m;

                    device const uint8_t * qs = block->qs + is * 16;
                    device const float * y_sub = yb + is * 32;

                    uchar byte = qs[k];
                    float v0 = (byte & 0x0F);
                    float v1 = ((byte >> 4) & 0x0F);
                    sumf[t * 8 + r] += (dl * v0 - ml) * y_sub[k];
                    sumf[t * 8 + r] += (dl * v1 - ml) * y_sub[k + 16];
                }
            }
        }
    }

    // Simdgroup-level reduction for each row
    float sum_sg[8];
    for (short i = 0; i < 8; i++) {
        sum_sg[i] = simd_sum(sumf[i]);
    }

    // Threadgroup memory for cross-simdgroup reduction
    threadgroup float partial_sums[32 * 8];  // Max NSG * 8 rows
    if (tiisg == 0) {
        for (short i = 0; i < 8; i++) {
            partial_sums[sgitg * 8 + i] = sum_sg[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of simdgroup 0 writes final results
    if (sgitg == 0 && tiisg == 0) {
        for (short t = 0; t < NT; t++) {
            for (short r = 0; r < 8; r++) {
                const int row = row0 + t * 8 + r;
                if (row >= (int)args.ne01) break;

                float total = 0.0f;
                for (short sg = 0; sg < NSG; sg++) {
                    total += partial_sums[sg * 8 + t * 8 + r];
                }
                dst[row] = total;
            }
        }
    }
}

// Instantiate decode kernels for different NT/NSG combinations
typedef decltype(kernel_mul_mv_q4_K_f32_decode<4, 8>) kquant_decode_t;

template [[host_name("kernel_mul_mv_q4_K_f32_decode_nt1_nsg8")]]  kernel kquant_decode_t kernel_mul_mv_q4_K_f32_decode<1, 8>;
template [[host_name("kernel_mul_mv_q4_K_f32_decode_nt2_nsg8")]]  kernel kquant_decode_t kernel_mul_mv_q4_K_f32_decode<2, 8>;
template [[host_name("kernel_mul_mv_q4_K_f32_decode_nt4_nsg8")]]  kernel kquant_decode_t kernel_mul_mv_q4_K_f32_decode<4, 8>;
template [[host_name("kernel_mul_mv_q4_K_f32_decode_nt4_nsg16")]] kernel kquant_decode_t kernel_mul_mv_q4_K_f32_decode<4, 16>;
