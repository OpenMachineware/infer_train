#include <metal_stdlib>
using namespace metal;

#include "iq_grid_tables.h"

#define QK_K 256
#define QK4_NL 32
#define N_SIMDWIDTH 32
#define N_COLS 4

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct gemm_args {
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint64_t nb01;
    uint64_t nb11;
};

// ===== IQ2_XXS GEMM =====
#define N_R0_IQ2_XXS 4

struct block_iq2_xxs {
    half d;
    uint16_t qs[32];
};

kernel void kernel_gemm_iq2_xxs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ2_XXS;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + first_row * args.nb01);

    // Load grid and signs tables
    threadgroup uint32_t * shmem_grid = (threadgroup uint32_t *)shmem;
    for (int i = tiisg; i < 512; i += N_SIMDWIDTH) {
        shmem_grid[i] = iq2xxs_grid[i];
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
            device const block_iq2_xxs & xb = x[row * nb + ib];
            device const uint16_t * q2 = xb.qs + 4 * ir;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint16_t q = q2[i];
                    uint32_t grid_idx = q & 0x1FF;
                    uint8_t signs = (q >> 9) & 0x7F;

                    uint32_t grid_val = shmem_grid[grid_idx];
                    uint8_t values[4];
                    values[0] = (grid_val >> 0) & 0x07;
                    values[1] = (grid_val >> 3) & 0x07;
                    values[2] = (grid_val >> 6) & 0x07;
                    values[3] = (grid_val >> 9) & 0x07;

                    for (short j = 0; j < 4; ++j) {
                        float v = float(values[j]) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                        acc += yl[c][i*2 + j%2] * v;
                    }
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

// ===== IQ2_XS GEMM =====
#define N_R0_IQ2_XS 4

struct block_iq2_xs {
    half d;
    uint16_t qs[32];
};

kernel void kernel_gemm_iq2_xs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ2_XS;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + first_row * args.nb01);

    threadgroup uint32_t * shmem_grid = (threadgroup uint32_t *)shmem;
    for (int i = tiisg; i < 512; i += N_SIMDWIDTH) {
        shmem_grid[i] = iq2xs_grid[i];
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
            device const block_iq2_xs & xb = x[row * nb + ib];
            device const uint16_t * q2 = xb.qs + 4 * ir;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint16_t q = q2[i];
                    uint32_t grid_idx = q & 0x1FF;
                    uint8_t signs = (q >> 9) & 0x1F;

                    uint32_t grid_val = shmem_grid[grid_idx];

                    for (short j = 0; j < 4; ++j) {
                        float v = float((grid_val >> (j*2)) & 0x03) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                        acc += yl[c][i] * v;
                    }
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

// ===== IQ2_S GEMM =====
#define N_R0_IQ2_S 4

struct block_iq2_s {
    half d;
    uint8_t qs[32];
    uint8_t scales[8];
};

kernel void kernel_gemm_iq2_s_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ2_S;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + first_row * args.nb01);

    threadgroup uint32_t * shmem_grid = (threadgroup uint32_t *)shmem;
    for (int i = tiisg; i < 512; i += N_SIMDWIDTH) {
        shmem_grid[i] = iq2xs_grid[i];
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
            device const block_iq2_s & xb = x[row * nb + ib];
            device const uint8_t * qs8 = xb.qs + 8 * ir;
            device const uint8_t * sc = xb.scales + 2 * ir;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 8; ++i) {
                    uint8_t q = qs8[i];
                    float scale = float((sc[i/4] >> (2*(i%4))) & 0x03) - 1.0f;
                    float v0 = float(q & 0x03) + scale;
                    float v1 = float((q >> 2) & 0x03) + scale;
                    acc += yl[c][i] * (v0 + v1);
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

// ===== IQ3_XXS GEMM =====
#define N_R0_IQ3_XXS 4

struct block_iq3_xxs {
    half d;
    uint8_t qs[32];
};

kernel void kernel_gemm_iq3_xxs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ3_XXS;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + first_row * args.nb01);

    threadgroup uint32_t * shmem_grid = (threadgroup uint32_t *)shmem;
    for (int i = tiisg; i < 256; i += N_SIMDWIDTH) {
        shmem_grid[i] = iq3xxs_grid[i];
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
            device const block_iq3_xxs & xb = x[row * nb + ib];
            device const uint8_t * qs8 = xb.qs + 4 * ir;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint8_t q = qs8[i];
                    uint32_t grid_idx = q & 0x3F;
                    uint8_t signs = (q >> 6) & 0x03;

                    uint32_t grid_val = shmem_grid[grid_idx];

                    for (short j = 0; j < 4; ++j) {
                        float v = float((grid_val >> (j*2)) & 0x03) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                        acc += yl[c][i] * v;
                    }
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

// ===== IQ3_S GEMM =====
#define N_R0_IQ3_S 4

struct block_iq3_s {
    half d;
    uint8_t qs[32];
    uint8_t scales[8];
};

kernel void kernel_gemm_iq3_s_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_IQ3_S;
    const short NC = N_COLS;
    const short NSG = 2;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + first_row * args.nb01);

    threadgroup uint32_t * shmem_grid = (threadgroup uint32_t *)shmem;
    for (int i = tiisg; i < 256; i += N_SIMDWIDTH) {
        shmem_grid[i] = iq3s_grid[i];
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
            device const block_iq3_s & xb = x[row * nb + ib];
            device const uint8_t * qs8 = xb.qs + 4 * ir;
            device const uint8_t * sc = xb.scales + ir;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint8_t q = qs8[i];
                    uint32_t grid_idx = q & 0x3F;
                    uint8_t signs = (q >> 6) & 0x03;
                    uint8_t scale = (sc[i] & 0x0F);

                    uint32_t grid_val = shmem_grid[grid_idx];

                    for (short j = 0; j < 4; ++j) {
                        float v = float(((grid_val >> (j*2)) & 0x03) | (scale << 2)) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                        acc += yl[c][i] * v;
                    }
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

// ===== TQ1_0 GEMM =====
#define N_R0_TQ1_0 2

struct block_tq1_0 {
    uint8_t qs[8];
    half d;
};

kernel void kernel_gemm_tq1_0_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NR0 = N_R0_TQ1_0;
    const short NC = N_COLS;
    const short NSG = 2;

    const int nb = args.ne00/QK4_NL;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_tq1_0 * x = (device const block_tq1_0 *) (src0 + first_row * args.nb01);

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
            device const block_tq1_0 & xb = x[row * nb + ib];
            device const uint8_t * qs8 = xb.qs + 4 * it;

            for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
                float acc = 0.f;

                for (short i = 0; i < 4; ++i) {
                    uint8_t q = qs8[i];
                    float v0 = float(q & 0x03);
                    float v1 = float((q >> 2) & 0x03);
                    float v2 = float((q >> 4) & 0x03);
                    float v3 = float((q >> 6) & 0x03);

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
