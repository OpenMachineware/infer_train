// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// F16 GEMV with multi-simdgroup optimization

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Multi-simdgroup version for K >= 512
// Uses NSG = 4 simdgroups per threadgroup
kernel void kernel_mul_mv_f16_f32_nsg4(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NSG = 4;  // Number of simdgroups
    constexpr short NR0 = 2;  // Rows per threadgroup
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = k/NB;
    const int r0 = tgpig.x*NR0;
    if (r0 >= m) return;

    device const float4* y4 = (device const float4*)input;

    // Pointers to weight rows
    device const half4* ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4*)(weights + (r0 + row) * k);
    }

    float sumf[NR0] = { [ 0 ... NR0 - 1 ] = 0.0f };

    // Compute starting indices
    const short ix = tiisg/(NW/NF);  // 0 or 1
    const short il = tiisg%(NW/NF);  // 0..15
    const int ib0 = sgitg*NF + ix;   // Starting block for this simdgroup

    device const float4* yb4 = y4 + (ib0*NB + il*NF)/4;

    // Main loop with multi-simdgroup stride
    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const half4* xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.0f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    // Tail
    for (int i = nb*NB + sgitg*NW + tiisg; i < k; i += NW*NSG) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            sumf[row] += float(weights[(r0 + row)*k + i]) * input[i];
        }
    }

    // Reduce and write
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const int row_idx = r0 + row;
            if (row_idx < m) output[row_idx] = sumf[row];
        }
    }
}
