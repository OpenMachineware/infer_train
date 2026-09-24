#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q5_K_LARGE 2
#define N_R0_Q5_K_SMALL 8
#define N_SG_Q5_K 2
#define N_COLS_LARGE 4
#define N_COLS_SMALL 4

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct gemm_args {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint64_t nb01;
    uint64_t nb11;
};

kernel void kernel_gemm_q5_k_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q5_K;
    const int nb = args.ne00/QK_K;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    if (nb > 4) {
        // ===== Large K path =====
        const short NR0 = N_R0_Q5_K_LARGE;
        const short NC = N_COLS_LARGE;
        const short ix = tiisg/8;
        const short it = tiisg%8;
        const short iq = it/4;
        const short ir = it%4;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        device const block_q5_K * x = (device const block_q5_K *) (src0 + first_row * args.nb01);

        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

        float sumf[NR0][NC] = {{0.f}};
        float yl[NC][16];
        float yh[NC][16];
        float4 sumy[NC];

        for (int ib = ix; ib < nb; ib += 4) {
            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                device const float * y = src1 + (first_col + c) * args.ne00 + ib * QK_K;
                device const float * y4 = y + 64 * iq + 8 * ir;

                sumy[c] = {0.f, 0.f, 0.f, 0.f};

                for (short i = 0; i < 8; ++i) {
                    yl[c][i+0] = y4[i+  0]; sumy[c][0] += yl[c][i+0];
                    yl[c][i+8] = y4[i+ 32]; sumy[c][1] += yl[c][i+8];
                    yh[c][i+0] = y4[i+128]; sumy[c][2] += yh[c][i+0];
                    yh[c][i+8] = y4[i+160]; sumy[c][3] += yh[c][i+8];
                }
            }

            FOR_UNROLL (short row = 0; row < NR0; row++) {
                device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
                device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
                device const uint16_t * qh = (device const uint16_t *)x[ib].qh + 4 * ir;
                device const half * dh = &x[ib].d;

                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t * q2 = q1 + 32;

                for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        uint16_t qh_i = qh[i];

                        uint16_t q1_i = q1[i];
                        acc1[0] += yl[c][2*i + 0] * ((q1_i & 0x000F) + ((qh_i & 1) << 4));
                        acc1[1] += yl[c][2*i + 1] * ((q1_i & 0x0F00) >> 4) + ((qh_i & 0x10) << 4);
                        acc1[2] += yl[c][2*i + 8] * ((q1_i & 0x00F0) >> 0) + ((qh_i & 0x100) >> 4);
                        acc1[3] += yl[c][2*i + 9] * ((q1_i & 0xF000) >> 8) + ((qh_i & 0x1000) >> 8);

                        uint16_t q2_i = q2[i];
                        acc2[0] += yh[c][2*i + 0] * ((q2_i & 0x000F) + ((qh_i & 0x10000) >> 16));
                        acc2[1] += yh[c][2*i + 1] * ((q2_i & 0x0F00) >> 4) + ((qh_i & 0x100000) >> 20);
                        acc2[2] += yh[c][2*i + 8] * ((q2_i & 0x00F0) >> 0) + ((qh_i & 0x1000000) >> 24);
                        acc2[3] += yh[c][2*i + 9] * ((q2_i & 0xF000) >> 8) + ((qh_i & 0x10000000) >> 28);
                    }

                    sumf[row][c] += (float)dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                                    (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                                    (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                                    (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                                   (float)dh[1] * (sumy[c][0] * sc8[2] + sumy[c][1] * sc8[3] +
                                                   sumy[c][2] * sc8[6] + sumy[c][3] * sc8[7]);
                }

                sc += args.nb01/2;
                q1 += args.nb01/2;
                qh += args.nb01/2;
                dh += args.nb01/2;
            }
        }

        for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float sum_all = simd_sum(sumf[row][c]);
                if (tiisg == 0) {
                    dst[(first_row + row) * args.ne02 + (first_col + c)] = sum_all;
                }
            }
        }
    } else {
        // ===== Small K path =====
        const short NR0 = N_R0_Q5_K_SMALL;
        const short NC = N_COLS_SMALL;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        const int local_row = tiisg / NC;
        const int local_col = tiisg % NC;

        const int row = first_row + local_row;
        const int col = first_col + local_col;

        if (row >= args.ne01 || col >= args.ne02) return;

        device const block_q5_K * x = (device const block_q5_K *) (src0 + row * args.nb01);
        device const float * y = src1 + col * args.ne00;

        float sum = 0.0f;

        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

        for (int ib = 0; ib < nb; ib++) {
            for (int it = 0; it < 8; it++) {
                const int iq = it / 4;
                const int ir = it % 4;

                float yl[16];
                float yh[16];
                float4 sumy = {0.f, 0.f, 0.f, 0.f};

                device const float * y4 = y + ib * QK_K + 64 * iq + 8 * ir;
                for (short i = 0; i < 8; ++i) {
                    yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
                    yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
                    yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
                    yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
                }

                device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
                device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
                device const uint16_t * qh = (device const uint16_t *)x[ib].qh + 4 * ir;
                device const half * dh = &x[ib].d;

                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t * q2 = q1 + 32;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    uint16_t qh_i = qh[i];

                    uint16_t q1_i = q1[i];
                    acc1[0] += yl[2*i + 0] * ((q1_i & 0x000F) + ((qh_i & 1) << 4));
                    acc1[1] += yl[2*i + 1] * ((q1_i & 0x0F00) >> 4) + ((qh_i & 0x10) << 4);
                    acc1[2] += yl[2*i + 8] * ((q1_i & 0x00F0) >> 0) + ((qh_i & 0x100) >> 4);
                    acc1[3] += yl[2*i + 9] * ((q1_i & 0xF000) >> 8) + ((qh_i & 0x1000) >> 8);

                    uint16_t q2_i = q2[i];
                    acc2[0] += yh[2*i + 0] * ((q2_i & 0x000F) + ((qh_i & 0x10000) >> 16));
                    acc2[1] += yh[2*i + 1] * ((q2_i & 0x0F00) >> 4) + ((qh_i & 0x100000) >> 20);
                    acc2[2] += yh[2*i + 8] * ((q2_i & 0x00F0) >> 0) + ((qh_i & 0x1000000) >> 24);
                    acc2[3] += yh[2*i + 9] * ((q2_i & 0xF000) >> 8) + ((qh_i & 0x10000000) >> 28);
                }

                sum += (float)dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                       (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                       (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                       (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                       (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                       sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
        }

        dst[row * args.ne02 + col] = sum;
    }
}
