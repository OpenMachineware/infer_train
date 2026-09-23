#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define QK4_NL 32
#define N_SIMDWIDTH 32

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// IQ4_NL lookup table
constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// Block structures
struct block_iq4_nl {
    half d;
    uint8_t qs[QK4_NL/2];
};

struct block_tq2_0 {
    uint8_t qs[QK4_NL/4];
    half d;
};

struct mv_args {
    uint32_t ne00;
    uint32_t ne01;
    uint64_t nb01;
};

// ===== IQ4_NL kernel =====
#define N_R0_IQ4_NL 4
#define N_SG_IQ4_NL 2

kernel void kernel_mul_mv_iq4_nl_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_IQ4_NL;
    const short NR0 = N_R0_IQ4_NL;

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + first_row * args.nb01);

    const int nb = args.ne00/QK4_NL;
    const int ns01 = args.nb01/2;  // Byte stride / sizeof(block)

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = src1 + ix*QK4_NL + it*8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            qf2 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            qf2 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 16 * QK4_NL;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}

// ===== TQ2_0 kernel =====
// Based on llama.cpp mul_mv.metal kernel_mul_mv_tq2_0_f32_impl
#define N_R0_TQ2_0 2
#define N_SG_TQ2_0 2

kernel void kernel_mul_mv_tq2_0_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = N_R0_TQ2_0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint64_t offset1 = r1 * 4; // nb11 = 4 (sizeof(float))

    device const float * y = (device const float *)(src1 + offset1);

    device const block_tq2_0 * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_tq2_0 *)(src0 + (first_row + row) * args.nb01);
    }

    float sumf[NR0] = {0.f};

    // 8 threads per block, NBLOCK blocks per pass, 2 halves per block per pass
    constexpr short NBLOCK = 4;
    constexpr short NB = N_SIMDWIDTH/NBLOCK; // threads per block = 8

    const short blk = tiisg / NB;    // 0..NBLOCK-1
    const short htg = tiisg % NB;    // 0..NB-1

    // byte and y base offsets within the block (32 elements per thread, 4 per byte)
    device const float4 * yb4 = (device const float4 *)(y + 4*htg + blk*QK_K);

    // hoisted per-byte coefficients (from y) and total y-sum, shared across rows
    float4 coef[4];

    for (int ib = blk; ib < nb; ib += NBLOCK) {
        FOR_UNROLL (short h0 = 0; h0 < 2; ++h0) {
            const float4 y0 = yb4[ 0 + 32*h0];
            const float4 y1 = yb4[ 8 + 32*h0];
            const float4 y2 = yb4[16 + 32*h0];
            const float4 y3 = yb4[24 + 32*h0];

            float sumy = 0.f;
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                coef[j] = float4(
                        y0[j],
                        y1[j] - 4.0f*y0[j],
                        y2[j] - 4.0f*y1[j],
                        y3[j] - 4.0f*y2[j]);

                sumy += (y0[j] + y1[j]) + (y2[j] + y3[j]);
            }

            FOR_UNROLL (short row = 0; row < NR0; ++row) {
                device const block_tq2_0 & xb = ax[row][ib];
                device const uchar * qs = xb.qs + 4*htg + 32*h0;

                float sum = -sumy;
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    // express the 2-bit field shifts (v>>2, v>>4, v>>6) as float floor ops
                    const float v = (float)qs[j];

                    const float f0 = v;
                    const float f1 = floor(v*0.25f);    // v>>2
                    const float f2 = floor(v*0.0625);   // v>>4
                    const float f3 = floor(v*0.015625); // v>>6

                    sum += coef[j][0]*f0 + coef[j][1]*f1 + coef[j][2]*f2 + coef[j][3]*f3;
                }

                sumf[row] += (float)xb.d * sum;
            }
        }

        yb4 += QK_K * NBLOCK / 4;
    }

    device float * dst_f32 = dst + r1*args.ne00 + first_row;

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}
