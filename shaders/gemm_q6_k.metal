#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q6_K 2
#define N_SG_Q6_K 2
#define N_COLS 4

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

struct gemm_args {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint64_t nb01;
    uint64_t nb11;
};

kernel void kernel_gemm_q6_k_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q6_K;
    const short NR0 = N_R0_Q6_K;
    const short NC = N_COLS;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + first_row * args.nb01);

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
            device const uint8_t * ql = x[ib].ql + 16 * ir;
            device const uint8_t * qh = x[ib].qh + 8 * ir;
            device const int8_t * sc = x[ib].scales;
            device const half * d = &x[ib].d;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 8; ++i) {
                    uint8_t qh_i = qh[i];

                    float qv[4];
                    qv[0] = (ql[i] & 0x0F) + ((qh_i & 0x01) << 4) - 32;
                    qv[1] = (ql[i] >> 4) + ((qh_i & 0x02) << 3) - 32;
                    qv[2] = (ql[i+8] & 0x0F) + ((qh_i & 0x04) << 2) - 32;
                    qv[3] = (ql[i+8] >> 4) + ((qh_i & 0x08) << 1) - 32;

                    acc[0] += yl[c][i] * qv[0];
                    acc[1] += yl[c][i] * qv[1];
                    acc[2] += yh[c][i] * qv[2];
                    acc[3] += yh[c][i] * qv[3];
                }

                sumf[row][c] += d[0] * sc[ir] * (acc[0] + acc[1] + acc[2] + acc[3]);
            }

            ql += args.nb01;
            qh += args.nb01;
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
