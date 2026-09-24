#include <metal_stdlib>
using namespace metal;

#include "iq_grid_tables.h"

#define QK_K 256
#define QK4_NL 32
#define N_SIMDWIDTH 32
#define N_COLS 4

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

struct gemm_args {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint64_t nb01;
    uint64_t nb11;
};

// ===== IQ4_NL GEMM =====
#define N_R0_IQ4_NL 2
#define N_SG_IQ4_NL 2

struct block_iq4_nl {
    half d;
    uint8_t qs[QK4_NL/2];
};

kernel void kernel_gemm_iq4_nl_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_IQ4_NL;
    const short NR0 = N_R0_IQ4_NL;
    const short NC = N_COLS;

    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + first_row * args.nb01);
    const int nb = args.ne00/QK4_NL;
    const int ns01 = args.nb01/2;

    const short ix = tiisg/2;
    const short it = tiisg%2;

    threadgroup float * shmem_f32 = (threadgroup float *)shmem;
    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf[NR0][NC] = {{0.f}};
    float4 yl[NC][4];

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;
    float4 qf1, qf2;

    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        // Load input for all columns
        for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
            device const float * yb = src1 + (first_col + c) * args.ne00 + ib * QK4_NL + it * 8;
            device const float4 * y4 = (device const float4 *)yb;
            yl[c][0] = y4[0];
            yl[c][1] = y4[4];
            yl[c][2] = y4[1];
            yl[c][3] = y4[5];
        }

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float4 acc1 = {0.f}, acc2 = {0.f};

                aux32[0] = q4[0] | (q4[1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                acc1 += yl[c][0] * qf1;
                acc2 += yl[c][1] * qf2;

                aux32[0] = q4[2] | (q4[3] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                acc1 += yl[c][2] * qf1;
                acc2 += yl[c][3] * qf2;

                acc1 += acc2;
                sumf[row][c] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
            }
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

// ===== IQ4_XS GEMM =====
#define N_R0_IQ4_XS 2

struct block_iq4_xs {
    half d;
    uint16_t scales_h;
    uint8_t qs[128];
    uint8_t scales_l[4];
};

kernel void kernel_gemm_iq4_xs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ4_XS;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + first_row * args.nb01);

    float sumf[NR0][NC] = {{0.f}};
    float yl[NC][16];

    for (int ib = ix; ib < nb; ib += 4) {
        for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
            device const float * y = src1 + (first_col + c) * args.ne00 + ib * QK_K + 64 * iq + 8 * ir;
            for (short i = 0; i < 8; ++i) {
                yl[c][i] = y[i];
                yl[c][i+8] = y[i+32];
            }
        }

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_xs & xb = x[row * nb + ib];
            device const uint8_t * q4 = xb.qs + 32 * iq + 8 * ir;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 8; ++i) {
                    uint8_t q = q4[i];
                    float v0 = kvalues_iq4nl_f[q & 0x0F];
                    float v1 = kvalues_iq4nl_f[(q >> 4) & 0x0F];
                    acc += yl[c][i] * v0 + yl[c][i+8] * v1;
                }

                uint32_t sc = ((xb.scales_l[iq/2] >> 6*(iq%2)) & 0x3F) | (((xb.scales_h >> 12*(iq/2)) & 0x0F) << 6);
                sumf[row][c] += (float)xb.d * acc * sc;
            }
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

// ===== IQ1_S GEMM =====
#define N_R0_IQ1_S 4

struct block_iq1_s {
    half d;
    uint8_t qs[32];
    uint16_t qh;
};

kernel void kernel_gemm_iq1_s_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ1_S;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + first_row * args.nb01);

    // Load grid table to threadgroup memory
    threadgroup uint32_t * shmem_grid = (threadgroup uint32_t *)shmem;
    for (int i = tiisg; i < 512; i += N_SIMDWIDTH) {
        shmem_grid[i] = iq1s_grid[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf[NR0][NC] = {{0.f}};
    float yl[NC][8];

    for (int ib = ix; ib < nb; ib += 4) {
        for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
            device const float * y = src1 + (first_col + c) * args.ne00 + ib * QK_K + 8 * ir;
            for (short i = 0; i < 8; ++i) {
                yl[c][i] = y[i];
            }
        }

        for (short row = 0; row < NR0; row++) {
            device const block_iq1_s & xb = x[row * nb + ib];
            device const uint8_t * qs8 = xb.qs + 4 * ir;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint8_t q = qs8[i];
                    uint32_t grid_idx = q & 0x7F;
                    uint32_t signs = q >> 7;

                    uint32_t grid_val = shmem_grid[grid_idx];
                    float values[2];
                    values[0] = float((grid_val >> 0) & 0xF) * (signs & 1 ? -1.0f : 1.0f);
                    values[1] = float((grid_val >> 4) & 0xF) * (signs & 2 ? -1.0f : 1.0f);

                    acc += yl[c][i] * values[0];
                    acc += yl[c][i+4] * values[1];
                }

                sumf[row][c] += (float)xb.d * acc;
            }
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

// ===== IQ1_M GEMM =====
#define N_R0_IQ1_M 4

struct block_iq1_m {
    uint8_t qs[32];
    uint8_t ds[2];
};

kernel void kernel_gemm_iq1_m_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ1_M;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + first_row * args.nb01);

    threadgroup uint32_t * shmem_grid = (threadgroup uint32_t *)shmem;
    for (int i = tiisg; i < 512; i += N_SIMDWIDTH) {
        shmem_grid[i] = iq1s_grid[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf[NR0][NC] = {{0.f}};
    float yl[NC][8];

    for (int ib = ix; ib < nb; ib += 4) {
        for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
            device const float * y = src1 + (first_col + c) * args.ne00 + ib * QK_K + 8 * ir;
            for (short i = 0; i < 8; ++i) {
                yl[c][i] = y[i];
            }
        }

        for (short row = 0; row < NR0; row++) {
            device const block_iq1_m & xb = x[row * nb + ib];
            device const uint8_t * qs8 = xb.qs + 4 * ir;

            uint16_t ds = *(device const uint16_t *)xb.ds;
            // Decode merged scale (same as IQ1_S but merged into block)
            float d = float(ds & 0x7FFF) / 32768.0f;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint8_t q = qs8[i];
                    uint32_t grid_idx = q & 0x7F;
                    uint32_t signs = q >> 7;

                    uint32_t grid_val = shmem_grid[grid_idx];
                    float values[2];
                    values[0] = float((grid_val >> 0) & 0xF) * (signs & 1 ? -1.0f : 1.0f);
                    values[1] = float((grid_val >> 4) & 0xF) * (signs & 2 ? -1.0f : 1.0f);

                    acc += yl[c][i] * values[0];
                    acc += yl[c][i+4] * values[1];
                }

                sumf[row][c] += (float)d * acc;
            }
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

// ===== TQ2_0 GEMM =====
#define N_R0_TQ2_0 2

struct block_tq2_0 {
    uint8_t qs[8];
    half d;
};

kernel void kernel_gemm_tq2_0_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_TQ2_0;
    const short NC = N_COLS;
    const short NSG = 2;

    const int nb = args.ne00/QK4_NL;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_tq2_0 * x = (device const block_tq2_0 *) (src0 + first_row * args.nb01);

    float sumf[NR0][NC] = {{0.f}};
    float yl[NC][16];

    const short ix = tiisg/2;
    const short it = tiisg%2;

    for (int ib = ix; ib < nb; ib += 16) {
        for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
            device const float * y = src1 + (first_col + c) * args.ne00 + ib * QK4_NL + 8 * it;
            for (short i = 0; i < 16; ++i) {
                yl[c][i] = y[i];
            }
        }

        for (short row = 0; row < NR0; row++) {
            device const block_tq2_0 & xb = x[row * nb + ib];
            device const uint8_t * qs8 = xb.qs + 4 * it;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint8_t q = qs8[i];
                    float v0 = float(q & 0x03) - 1.5f;
                    float v1 = float((q >> 2) & 0x03) - 1.5f;
                    float v2 = float((q >> 4) & 0x03) - 1.5f;
                    float v3 = float((q >> 6) & 0x03) - 1.5f;

                    acc += yl[c][i] * v0 + yl[c][i+4] * v1 + yl[c][i+8] * v2 + yl[c][i+12] * v3;
                }

                sumf[row][c] += (float)xb.d * acc;
            }
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

// Additional IQ/TQ kernels would follow similar patterns
// For brevity, including core kernels above
