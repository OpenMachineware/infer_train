#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q3_K_LARGE 2
#define N_R0_Q3_K_SMALL 8
#define N_SG_Q3_K 2
#define N_COLS_LARGE 4
#define N_COLS_SMALL 4

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q3_K {
    half d;
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
};

struct gemm_args {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint64_t nb01;
    uint64_t nb11;
};

kernel void kernel_gemm_q3_k_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q3_K;
    const int nb = args.ne00/QK_K;

    constexpr uint16_t kmask1 = 0x0f0f;
    constexpr uint16_t kmask2 = 0x3030;

    if (nb > 4) {
        // ===== Large K path =====
        const short NR0 = N_R0_Q3_K_LARGE;
        const short NC = N_COLS_LARGE;
        const short ix = tiisg/8;
        const short it = tiisg%8;
        const short iq = it/4;
        const short ir = it%4;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        device const block_q3_K * x = (device const block_q3_K *) (src0 + first_row * args.nb01);

        float sumf[NR0][NC] = {{0.f}};
        float yl[NC][16];
        float yh[NC][16];

        for (int ib = ix; ib < nb; ib += 4) {
            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                device const float * y = src1 + (first_col + c) * args.ne00 + ib * QK_K;
                device const float * y4 = y + 8 * ir;

                for (short i = 0; i < 8; ++i) {
                    yl[c][i+0] = y4[i+  0];
                    yl[c][i+8] = y4[i+ 32];
                    yh[c][i+0] = y4[i+128];
                    yh[c][i+8] = y4[i+160];
                }
            }

            FOR_UNROLL (short row = 0; row < NR0; row++) {
                device const uint8_t * q = x[ib].qs + 8 * ir;
                device const uint8_t * hm = x[ib].hmask + 8 * ir;
                device const uint16_t * a = (device const uint16_t *)x[ib].scales + iq;
                device const half * dh = &x[ib].d;

                uint16_t aux16[2];
                aux16[0] = a[0] & kmask1;
                aux16[1] = a[2] & kmask1;
                thread const int8_t * sc = (thread const int8_t *)aux16;

                uint16_t tmp = a[4] >> 4;
                int8_t m0 = (tmp & kmask2) | ((a[0] & kmask2) >> 2);
                int8_t m1 = (tmp >> 8) | ((a[2] & kmask2) >> 2);

                for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                    float acc = 0.f;

                    FOR_UNROLL (short i = 0; i < 8; ++i) {
                        uint8_t q_i = q[i];
                        int8_t h_i = hm[i];

                        uint8_t q0 = (q_i & 0x03) | ((h_i & 1) << 2);
                        uint8_t q1 = ((q_i >> 2) & 0x03) | ((h_i >> 1) & 0x04);
                        uint8_t q2 = ((q_i >> 4) & 0x03) | ((h_i >> 2) & 0x04);
                        uint8_t q3 = ((q_i >> 6) & 0x03) | ((h_i >> 3) & 0x04);

                        int8_t v0 = (int8_t)q0 - 4;
                        int8_t v1 = (int8_t)q1 - 4;
                        int8_t v2 = (int8_t)q2 - 4;
                        int8_t v3 = (int8_t)q3 - 4;

                        acc += yl[c][i] * (v0 * sc[0] + v1 * sc[1]);
                        acc += yl[c][i+8] * (v2 * sc[0] + v3 * sc[1]);
                    }

                    FOR_UNROLL (short i = 0; i < 8; ++i) {
                        uint8_t q_i = q[i+32];
                        int8_t h_i = hm[i+32];

                        uint8_t q0 = (q_i & 0x03) | ((h_i & 1) << 2);
                        uint8_t q1 = ((q_i >> 2) & 0x03) | ((h_i >> 1) & 0x04);
                        uint8_t q2 = ((q_i >> 4) & 0x03) | ((h_i >> 2) & 0x04);
                        uint8_t q3 = ((q_i >> 6) & 0x03) | ((h_i >> 3) & 0x04);

                        int8_t v0 = (int8_t)q0 - 4;
                        int8_t v1 = (int8_t)q1 - 4;
                        int8_t v2 = (int8_t)q2 - 4;
                        int8_t v3 = (int8_t)q3 - 4;

                        acc += yh[c][i] * (v0 * sc[4] + v1 * sc[5]);
                        acc += yh[c][i+8] * (v2 * sc[4] + v3 * sc[5]);
                    }

                    sumf[row][c] += (float)dh[0] * acc;
                }

                q += args.nb01;
                hm += args.nb01;
                a += args.nb01/2;
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
        const short NR0 = N_R0_Q3_K_SMALL;
        const short NC = N_COLS_SMALL;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        const int local_row = tiisg / NC;
        const int local_col = tiisg % NC;

        const int row = first_row + local_row;
        const int col = first_col + local_col;

        if (row >= args.ne01 || col >= args.ne02) return;

        device const block_q3_K * x = (device const block_q3_K *) (src0 + row * args.nb01);
        device const float * y = src1 + col * args.ne00;

        float sum = 0.0f;

        for (int ib = 0; ib < nb; ib++) {
            for (int ir = 0; ir < 4; ir++) {
                float yl[16];
                float yh[16];

                device const float * y4 = y + ib * QK_K + 8 * ir;
                for (short i = 0; i < 8; ++i) {
                    yl[i+0] = y4[i+  0];
                    yl[i+8] = y4[i+ 32];
                    yh[i+0] = y4[i+128];
                    yh[i+8] = y4[i+160];
                }

                device const uint8_t * q = x[ib].qs + 8 * ir;
                device const uint8_t * hm = x[ib].hmask + 8 * ir;
                device const half * dh = &x[ib].d;

                float acc = 0.f;

                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    uint8_t q_i = q[i];
                    int8_t h_i = hm[i];

                    uint8_t q0 = (q_i & 0x03) | ((h_i & 1) << 2);
                    uint8_t q1 = ((q_i >> 2) & 0x03) | ((h_i >> 1) & 0x04);
                    uint8_t q2 = ((q_i >> 4) & 0x03) | ((h_i >> 2) & 0x04);
                    uint8_t q3 = ((q_i >> 6) & 0x03) | ((h_i >> 3) & 0x04);

                    int8_t v0 = (int8_t)q0 - 4;
                    int8_t v1 = (int8_t)q1 - 4;
                    int8_t v2 = (int8_t)q2 - 4;
                    int8_t v3 = (int8_t)q3 - 4;

                    uint8_t sc = x[ib].scales[ir];

                    acc += yl[i] * v0 * ((sc >> 0) & 0x0F);
                    acc += yl[i] * v1 * ((sc >> 4) & 0x0F);
                    acc += yl[i+8] * v2 * ((sc >> 0) & 0x0F);
                    acc += yl[i+8] * v3 * ((sc >> 4) & 0x0F);
                }

                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    uint8_t q_i = q[i+32];
                    int8_t h_i = hm[i+32];

                    uint8_t q0 = (q_i & 0x03) | ((h_i & 1) << 2);
                    uint8_t q1 = ((q_i >> 2) & 0x03) | ((h_i >> 1) & 0x04);
                    uint8_t q2 = ((q_i >> 4) & 0x03) | ((h_i >> 2) & 0x04);
                    uint8_t q3 = ((q_i >> 6) & 0x03) | ((h_i >> 3) & 0x04);

                    int8_t v0 = (int8_t)q0 - 4;
                    int8_t v1 = (int8_t)q1 - 4;
                    int8_t v2 = (int8_t)q2 - 4;
                    int8_t v3 = (int8_t)q3 - 4;

                    uint8_t sc = x[ib].scales[ir+4];

                    acc += yh[i] * v0 * ((sc >> 0) & 0x0F);
                    acc += yh[i] * v1 * ((sc >> 4) & 0x0F);
                    acc += yh[i+8] * v2 * ((sc >> 0) & 0x0F);
                    acc += yh[i+8] * v3 * ((sc >> 4) & 0x0F);
                }

                sum += (float)dh[0] * acc;
            }
        }

        dst[row * args.ne02 + col] = sum;
    }
}
