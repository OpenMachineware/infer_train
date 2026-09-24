#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q2_K_LARGE 2
#define N_R0_Q2_K_SMALL 8
#define N_SG_Q2_K 2
#define N_COLS_LARGE 4
#define N_COLS_SMALL 4

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q2_K {
    half d;
    half dmin;
    uint8_t scales[16];
    uint8_t qs[64];
};

struct gemm_args {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint64_t nb01;
    uint64_t nb11;
};

kernel void kernel_gemm_q2_k_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q2_K;
    const int nb = args.ne00/QK_K;

    if (nb > 4) {
        // ===== Large K path: parallelize across blocks =====
        const short NR0 = N_R0_Q2_K_LARGE;
        const short NC = N_COLS_LARGE;
        const short ix = tiisg/8;
        const short it = tiisg%8;
        const short ir = it%4;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        device const block_q2_K * x = (device const block_q2_K *) (src0 + first_row * args.nb01);

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
                device const uint8_t * sc = x[ib].scales;
                device const half * dh = &x[ib].d;

                for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 8; ++i) {
                        uint8_t q_i = q[i];
                        float qv[4];
                        qv[0] = (q_i & 0x03);
                        qv[1] = (q_i & 0x0C) >> 2;
                        qv[2] = (q_i & 0x30) >> 4;
                        qv[3] = (q_i & 0xC0) >> 6;

                        acc1[0] += yl[c][i] * qv[0];
                        acc1[1] += yl[c][i] * qv[1];
                        acc1[2] += yl[c][i] * qv[2];
                        acc1[3] += yl[c][i] * qv[3];
                    }

                    FOR_UNROLL (short i = 0; i < 8; ++i) {
                        uint8_t q_i = q[i+32];
                        float qv[4];
                        qv[0] = (q_i & 0x03);
                        qv[1] = (q_i & 0x0C) >> 2;
                        qv[2] = (q_i & 0x30) >> 4;
                        qv[3] = (q_i & 0xC0) >> 6;

                        acc2[0] += yh[c][i] * qv[0];
                        acc2[1] += yh[c][i] * qv[1];
                        acc2[2] += yh[c][i] * qv[2];
                        acc2[3] += yh[c][i] * qv[3];
                    }

                    float sum = (acc1[0] + acc2[0]) * (sc[ir] & 0x0F) +
                                (acc1[1] + acc2[1]) * (sc[ir+4] & 0x0F) +
                                (acc1[2] + acc2[2]) * (sc[ir+8] & 0x0F) +
                                (acc1[3] + acc2[3]) * (sc[ir+12] & 0x0F);

                    sumf[row][c] += (float)dh[0] * sum;
                }

                q += args.nb01;
                sc += args.nb01;
                dh += args.nb01;
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
        // ===== Small K path: each thread processes unique (row, col) output element =====
        const short NR0 = N_R0_Q2_K_SMALL;
        const short NC = N_COLS_SMALL;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        const int local_row = tiisg / NC;
        const int local_col = tiisg % NC;

        const int row = first_row + local_row;
        const int col = first_col + local_col;

        if (row >= args.ne01 || col >= args.ne02) return;

        device const block_q2_K * x = (device const block_q2_K *) (src0 + row * args.nb01);
        device const float * y = src1 + col * args.ne00;

        float sum = 0.0f;

        for (int ib = 0; ib < nb; ib++) {
            // Process all 4 sub-blocks (ir=0..3)
            for (int ir = 0; ir < 4; ir++) {
                // Load 32 input values
                float yl[16];
                float yh[16];

                device const float * y4 = y + ib * QK_K + 8 * ir;
                for (short i = 0; i < 8; ++i) {
                    yl[i+0] = y4[i+  0];
                    yl[i+8] = y4[i+ 32];
                    yh[i+0] = y4[i+128];
                    yh[i+8] = y4[i+160];
                }

                // Process quantized values
                device const uint8_t * q = x[ib].qs + 8 * ir;
                device const uint8_t * sc = x[ib].scales;
                device const half * dh = &x[ib].d;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    uint8_t q_i = q[i];
                    float qv[4];
                    qv[0] = (q_i & 0x03);
                    qv[1] = (q_i & 0x0C) >> 2;
                    qv[2] = (q_i & 0x30) >> 4;
                    qv[3] = (q_i & 0xC0) >> 6;

                    acc1[0] += yl[i] * qv[0];
                    acc1[1] += yl[i] * qv[1];
                    acc1[2] += yl[i] * qv[2];
                    acc1[3] += yl[i] * qv[3];
                }

                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    uint8_t q_i = q[i+32];
                    float qv[4];
                    qv[0] = (q_i & 0x03);
                    qv[1] = (q_i & 0x0C) >> 2;
                    qv[2] = (q_i & 0x30) >> 4;
                    qv[3] = (q_i & 0xC0) >> 6;

                    acc2[0] += yh[i] * qv[0];
                    acc2[1] += yh[i] * qv[1];
                    acc2[2] += yh[i] * qv[2];
                    acc2[3] += yh[i] * qv[3];
                }

                float partial = (acc1[0] + acc2[0]) * (sc[ir] & 0x0F) +
                                (acc1[1] + acc2[1]) * (sc[ir+4] & 0x0F) +
                                (acc1[2] + acc2[2]) * (sc[ir+8] & 0x0F) +
                                (acc1[3] + acc2[3]) * (sc[ir+12] & 0x0F);

                sum += (float)dh[0] * partial;
            }
        }

        dst[row * args.ne02 + col] = sum;
    }
}
