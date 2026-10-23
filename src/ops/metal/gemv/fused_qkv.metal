// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Fused Q/K/V GEMV kernel - single dispatch for three projections

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32

// Fused GEMV args for Q/K/V projections
struct fused_qkv_args {
    int32_t ne00;  // K dimension (hidden_size)
    int32_t ne01_q; // M dimension for Q
    int32_t ne01_k; // M dimension for K
    int32_t ne01_v; // M dimension for V
    uint64_t nb01_q; // Row stride for Q weight
    uint64_t nb01_k; // Row stride for K weight
    uint64_t nb01_v; // Row stride for V weight
    int32_t ne0_q;   // Output rows for Q
    int32_t ne0_k;   // Output rows for K
    int32_t ne0_v;   // Output rows for V
};

// Fused Q/K/V GEMV kernel for Q4_K quantization
// Reads input once, computes all three projections
template<short nr0>
void kernel_fused_qkv_q4_k_f32_impl(
    constant fused_qkv_args & args,
    device const char * src_q,  // Q weights
    device const char * src_k,  // K weights
    device const char * src_v,  // V weights
    device const float * src_x, // Input vector
    device float * dst_q,       // Q output
    device float * dst_k,       // K output
    device float * dst_v,       // V output
    uint3 tgpig,
    ushort tiisg,
    ushort sgitg)
{
    const short NSG = 2; // Number of simdgroups

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    // Y-vector is shared across Q/K/V
    device const float * y = src_x;
    float yl[16];
    float yh[16];

    // Process each projection separately to handle different sizes
    for (int proj = 0; proj < 3; proj++) {
        // Determine which weight and output to use
        device const block_q4_K * x;
        device float * dst;
        int ne01, out_size;
        uint64_t nb01;

        switch (proj) {
            case 0: // Q projection
                x = (device const block_q4_K *) (src_q + first_row * args.nb01_q);
                dst = dst_q + first_row;
                ne01 = args.ne01_q;
                out_size = args.ne0_q;
                nb01 = args.nb01_q;
                break;
            case 1: // K projection
                x = (device const block_q4_K *) (src_k + first_row * args.nb01_k);
                dst = dst_k + first_row;
                ne01 = args.ne01_k;
                out_size = args.ne0_k;
                nb01 = args.nb01_k;
                break;
            case 2: // V projection
                x = (device const block_q4_K *) (src_v + first_row * args.nb01_v);
                dst = dst_v + first_row;
                ne01 = args.ne01_v;
                out_size = args.ne0_v;
                nb01 = args.nb01_v;
                break;
        }

        float sumf[nr0] = {0.f};

        device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

        for (int ib = ix; ib < nb; ib += 4) {
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            // Load Y vector (shared input)
            for (short i = 0; i < 8; ++i) {
                yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
                yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
                yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
                yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
            }

            device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t * q2 = q1 + 32;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                #pragma unroll
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
                }

                sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                      (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                      (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                      (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += nb01/2;
                sc += nb01/2;
                dh += nb01/2;
            }

            y4 += 4 * QK_K;
        }

        // Write results
        for (int row = 0; row < nr0 && first_row + row < ne01; ++row) {
            float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst[first_row + row] = sum_all;
            }
        }
    }
}

[[host_name("kernel_fused_qkv_q4_k_f32")]]
kernel void kernel_fused_qkv_q4_k_f32(
    device const char * src_q [[buffer(0)]],
    device const char * src_k [[buffer(1)]],
    device const char * src_v [[buffer(2)]],
    device const float * src_x [[buffer(3)]],
    device float * dst_q [[buffer(4)]],
    device float * dst_k [[buffer(5)]],
    device float * dst_v [[buffer(6)]],
    constant fused_qkv_args & args [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    kernel_fused_qkv_q4_k_f32_impl<2>(args, src_q, src_k, src_v, src_x, dst_q, dst_k, dst_v, tgpig, tiisg, sgitg);
}
