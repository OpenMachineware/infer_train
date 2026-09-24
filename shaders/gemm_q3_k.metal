#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q3_K 2
#define N_SG_Q3_K 2
#define N_COLS 4

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
    const short NR0 = N_R0_Q3_K;
    const short NC = N_COLS;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

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
            device const uint8_t * sc = (device const uint8_t *)x[ib].scales;
            device const half * d = &x[ib].d;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    uint8_t q_i = q[i];
                    uint8_t hm_i = hm[i];

                    float qv[4];
                    qv[0] = (q_i & 0x03) - (hm_i & 0x04 ? 4 : 0);
                    qv[1] = ((q_i >> 2) & 0x03) - (hm_i & 0x08 ? 4 : 0);
                    qv[2] = ((q_i >> 4) & 0x03) - (hm_i & 0x10 ? 4 : 0);
                    qv[3] = ((q_i >> 6) & 0x03) - (hm_i & 0x20 ? 4 : 0);

                    acc[0] += yl[c][i] * qv[0];
                    acc[1] += yl[c][i] * qv[1];
                    acc[2] += yl[c][i] * qv[2];
                    acc[3] += yl[c][i] * qv[3];
                }

                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    uint8_t q_i = q[i+32];
                    uint8_t hm_i = hm[i];

                    float qv[4];
                    qv[0] = (q_i & 0x03) - (hm_i & 0x01 ? 4 : 0);
                    qv[1] = ((q_i >> 2) & 0x03) - (hm_i & 0x02 ? 4 : 0);
                    qv[2] = ((q_i >> 4) & 0x03) - (hm_i & 0x40 ? 4 : 0);
                    qv[3] = ((q_i >> 6) & 0x03) - (hm_i & 0x80 ? 4 : 0);

                    acc[0] += yh[c][i] * qv[0];
                    acc[1] += yh[c][i] * qv[1];
                    acc[2] += yh[c][i] * qv[2];
                    acc[3] += yh[c][i] * qv[3];
                }

                float scale = (sc[ir] & 0x0F) * 0.5f + 0.5f;
                sumf[row][c] += d[0] * scale * (acc[0] + acc[1] + acc[2] + acc[3]);
            }

            q += args.nb01;
            hm += args.nb01;
            sc += args.nb01;
            d += args.nb01;
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
}
