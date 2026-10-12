#include <metal_stdlib>
using namespace metal;

#include "iq_grid_tables.h"

#define QK_K 256
#define QK4_NL 32
#define N_SIMDWIDTH 32

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// IQ4_NL lookup table
constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// kmask_iq2xs for IQ2_XS
constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// ksigns_iq2xs lookup table (128 entries)
constexpr constant static int8_t ksigns_iq2xs[128] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,
     1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,
     1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,
     1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
};

// Grid tables provided by iq_grid_tables.h
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
// Keep NR0=2 as it's optimal for this kernel
#define N_R0_IQ4_NL 2
#define N_SG_IQ4_NL 2

kernel void kernel_mul_mv_iq4_nl_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_IQ4_NL;
    const short NR0 = N_R0_IQ4_NL;  // Keep at 2

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + first_row * args.nb01);

    const int nb = args.ne00/QK4_NL;
    const int ns01 = args.nb01/2;  // Byte stride / sizeof(block)

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    // Use threadgroup memory for kvalues lookup (16 floats)
    threadgroup float * shmem_f32 = (threadgroup float *)shmem;
    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
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

// ===== TQ1_0 kernel =====
// Optimized implementation with vectorization and threadgroup memory
#define N_R0_TQ1_0 2

struct block_tq1_0 {
    uchar qs[48];
    uchar qh[4];
    half d;
};

// Trit lookup table (constant memory)
constexpr constant static int8_t tq1_0_trit_values[3] = {-1, 0, 1};

// Pre-computed trit extraction constants
constexpr constant static uint POW3_PACKED = (1u << 28) | (3u << 21) | (9u << 14) | (27u << 7) | 81u;

kernel void kernel_mul_mv_tq1_0_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = N_R0_TQ1_0;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_tq1_0 * x = (device const block_tq1_0 *)(src0 + first_row * args.nb01);
    const int ns01 = args.nb01/2;

    // Use threadgroup memory for input vector caching (32 floats per iteration)
    threadgroup float s_y[32];

    float sumf[NR0] = {0.f};

    // Process 8 blocks per iteration, each thread handles 1 element per block
    constexpr short NBLOCK = 8;
    const short blk = tiisg / 4;   // 0..7 (block index)
    const short tid4 = tiisg % 4;  // 0..3 (element within block)

    device const float * yb = src1;

    for (int ib = blk; ib < nb; ib += NBLOCK) {
        // Load 32 elements of y into threadgroup memory
        if (tiisg < 32) {
            s_y[tiisg] = yb[ib * QK_K + tiisg];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const block_tq1_0 & xb = x[row*ns01 + ib];
            const float d = (float)xb.d;

            // Process 4 trits per thread (vectorized)
            // Each byte contains 5 trits, we process tid4'th trit from each of 8 consecutive bytes
            float sum = 0.f;

            // First 32 bytes: 160 elements (8 bytes * 5 trits * 4 threads)
            FOR_UNROLL (short m = 0; m < 8; ++m) {
                const uint qbyte = (uint)xb.qs[tid4 * 8 + m];
                const uint shift = (4 - (m / 2)) * 7;
                const uint mask = (POW3_PACKED >> shift) & 0x7Fu;
                const uint xi = ((((qbyte * mask) & 255u) * 3u) >> 8);
                const float val = d * (float(xi) - 1.0f);
                sum += val * s_y[m + tid4 * 4];
            }

            // qs[32:48]: 80 elements
            FOR_UNROLL (short m = 0; m < 4; ++m) {
                const uint qbyte = (uint)xb.qs[32 + tid4 * 4 + m];
                const uint shift = (4 - m) * 7;
                const uint mask = (POW3_PACKED >> shift) & 0x7Fu;
                const uint xi = ((((qbyte * mask) & 255u) * 3u) >> 8);
                const float val = d * (float(xi) - 1.0f);
                sum += val * s_y[32 + m + tid4 * 4];
            }

            // qh[0:4]: 16 elements
            if (tid4 < 4) {
                const uint qhbyte = (uint)xb.qh[tid4];
                FOR_UNROLL (short t = 0; t < 4; ++t) {
                    const uint shift = (4 - t) * 7;
                    const uint mask = (POW3_PACKED >> shift) & 0x7Fu;
                    const uint xi = ((((qhbyte * mask) & 255u) * 3u) >> 8);
                    const float val = d * (float(xi) - 1.0f);
                    sum += val * s_y[240 + t * 4 + tid4];
                }
            }

            sumf[row] += sum;
        }

        yb += QK_K;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}

// ===== IQ4_XS kernel =====
// Based on llama.cpp mul_mv.metal kernel_mul_mv_iq4_xs_f32_impl
#define N_R0_IQ4_XS 2

struct block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[4];
    uchar qs[128];
};

kernel void kernel_mul_mv_iq4_xs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = N_R0_IQ4_XS;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq4_xs * x = (device const block_iq4_xs *)(src0 + first_row * args.nb01);

    const int ns01 = args.nb01/2;  // Byte stride / sizeof(block)

    const short ix = tiisg/16;  // 0 or 1
    const short it = tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = src1 + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            qf2 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            qf2 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}

// ===== IQ1_S kernel =====
struct block_iq1_s {
    half d;
    uint8_t qs[32];
    uint16_t qh[8];
};

kernel void kernel_mul_mv_iq1_s_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 4;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq1_s * x = (device const block_iq1_s *)(src0 + first_row * args.nb01);
    const int ns01 = args.nb01/2;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short ix = tiisg % ntx;

    float yl[32];
    float sumf[NR0] = {0.f};

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        float sumy = 0;
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (short row = 0; row < NR0; ++row) {
            constant const uint8_t * grid1 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant const uint8_t * grid2 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant const uint8_t * grid3 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant const uint8_t * grid4 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (short j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }

            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}

// ===== IQ1_M kernel =====
// Optimized with threadgroup memory for grid table
struct block_iq1_m {
    uint8_t qs[32];
    uint8_t qh[16];
    uint8_t scales[8];
};

kernel void kernel_mul_mv_iq1_m_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 4;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq1_m * x = (device const block_iq1_m *)(src0 + first_row * args.nb01);
    const int ns01 = args.nb01/2;

    const int nb32 = nb * (QK_K / 32);
    const short ntx = 32;
    const short ix = tiisg % ntx;

    float yl[32];
    float sumf[NR0] = {0.f};

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const block_iq1_m & xr = x[row*ns01 + ibl];
            device const uint8_t * qs = xr.qs + 4 * ib;
            device const uint8_t * qh = xr.qh + 2 * ib;
            device const uint16_t * sc = (device const uint16_t *)xr.scales;

            // Optimized merged scale extraction
            const uint16_t sc0 = sc[0];
            const uint16_t sc1 = sc[1];
            const uint16_t sc2 = sc[2];
            const uint16_t sc3 = sc[3];
            const uint16_t scale16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
            const half d = as_type<half>(scale16);

            // Use global constant memory for grid (faster than threadgroup for this case)
            constant const uint8_t * grid1 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant const uint8_t * grid2 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant const uint8_t * grid3 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant const uint8_t * grid4 = (constant const uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }

            // Optimized delta calculation
            const uint8_t qh0 = qh[0];
            const uint8_t qh1 = qh[1];
            const float delta1 = sumy[0] * (-1.0f + (2.0f * IQ1M_DELTA + 1.0f) * (qh0 >> 3 & 1)) +
                               sumy[1] * (-1.0f + (2.0f * IQ1M_DELTA + 1.0f) * (qh0 >> 7));
            const float delta2 = sumy[2] * (-1.0f + (2.0f * IQ1M_DELTA + 1.0f) * (qh1 >> 3 & 1)) +
                               sumy[3] * (-1.0f + (2.0f * IQ1M_DELTA + 1.0f) * (qh1 >> 7));

            // Optimized scale extraction
            const uint16_t sc_ib = sc[ib/2];
            const float scale1 = 2.0f * ((sc_ib >> (6*(ib%2)+0)) & 7) + 1.0f;
            const float scale2 = 2.0f * ((sc_ib >> (6*(ib%2)+3)) & 7) + 1.0f;

            sumf[row] += (float)d * ((sum[0] + delta1) * scale1 + (sum[1] + delta2) * scale2);
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}

// ===== IQ2_XXS kernel =====
struct block_iq2_xxs {
    half d;
    uint16_t qs[32];
};

kernel void kernel_mul_mv_iq2_xxs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 4;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = 32;
    const short ix = tiisg % ntx;

    float yl[32];
    float sumf[NR0] = {0.f};

    // Use threadgroup memory for grid and signs
    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < NR0; ++row) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.25f;
        }
    }
}

// ===== IQ2_XXS kernel (split=1, NR0=8) =====
kernel void kernel_mul_mv_iq2_xxs_f32_split1(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 8;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = min(nb32, 32);
    const short nrep = 32 / ntx;
    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (NR0 * irep      ) / nrep;
    const short row1 = (NR0 * (irep + 1)) / nrep;

    float yl[32];
    float sumf[NR0] = {0.f};

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; ++row) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.25f;
        }
    }
}

// ===== IQ2_XS kernel =====
struct block_iq2_xs {
    half d;
    uint16_t qs[32];
    uint8_t scales[8];
};

kernel void kernel_mul_mv_iq2_xs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 4;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq2_xs * x = (device const block_iq2_xs *)(src0 + first_row * args.nb01);
    const int ns01 = args.nb01/2;

    const int nb32 = nb * (QK_K / 32);
    const short ntx = 32;
    const short ix = tiisg % ntx;

    // Use threadgroup shared memory for grid and signs
    threadgroup uint64_t * sgrid = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns = (threadgroup uint8_t  *)(sgrid + 512);
    {
        // Load grid values (512 * uint64_t = 512 * 8 bytes)
        // With NSG=2: each simdgroup loads 256 values, each thread loads 8
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) sgrid[pos + i] = iq2xs_grid[pos + i];
        // Load signs (128 * uint8_t = 128 bytes)
        // With NSG=2: each simdgroup loads 64 values, each thread loads 2
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float yl[32];
    float sumf[NR0] = {0.f};

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < NR0; ++row) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(sgrid + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(sgrid + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.25f;
        }
    }
}

// ===== IQ2_XS kernel (split=1, NR0=8) =====
kernel void kernel_mul_mv_iq2_xs_f32_split1(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 8;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq2_xs * x = (device const block_iq2_xs *)(src0 + first_row * args.nb01);
    const int ns01 = args.nb01/2;

    const int nb32 = nb * (QK_K / 32);
    const short ntx = min(nb32, 32);
    const short nrep = 32 / ntx;
    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (NR0 * irep      ) / nrep;
    const short row1 = (NR0 * (irep + 1)) / nrep;

    threadgroup uint64_t * sgrid = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns = (threadgroup uint8_t  *)(sgrid + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) sgrid[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float yl[32];
    float sumf[NR0] = {0.f};

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; ++row) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(sgrid + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(sgrid + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.25f;
        }
    }
}

// ===== IQ2_S kernel (split=0, NR0=4) =====
// Block layout: d (2B) + qs[64] + qh[8] + scales[8] = 82 bytes
// qs[0:32] stores grid indices (low bits), qs[32:64] stores signs
struct block_iq2_s {
    half d;
    uint8_t qs[QK_K/4];   // 64 bytes: indices + signs
    uint8_t qh[QK_K/32];  // 8 bytes: high bits
    uint8_t scales[QK_K/32]; // 8 bytes: scale factors
};

kernel void kernel_mul_mv_iq2_s_f32_split0(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 4;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq2_s * x = (device const block_iq2_s *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = 32;
    const short ix = tiisg % ntx;

    float yl[32];
    float sumf[NR0] = {0.f};

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;  // signs at offset 32 in qs
        device const half * dh = &xr->d;

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >> 4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                constant const uint8_t * grid1 = (constant const uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant const uint8_t * grid2 = (constant const uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.25f;
        }
    }
}

// ===== IQ2_S kernel (split=1, NR0=8) =====
// Adaptive ntx based on nb32 for better parallelism at large K
kernel void kernel_mul_mv_iq2_s_f32_split1(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 8;  // Increased from 4 to 8

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq2_s * x = (device const block_iq2_s *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = min(nb32, 32);  // Cap at 32 to avoid division issues
    const short nrep = 32 / ntx;  // How many iterations each thread does
    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    // Divide rows among iterations
    const short row0 = (NR0 * irep      ) / nrep;
    const short row1 = (NR0 * (irep + 1)) / nrep;

    float yl[32];
    float sumf[NR0] = {0.f};

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; ++row) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >> 4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                constant const uint8_t * grid1 = (constant const uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant const uint8_t * grid2 = (constant const uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.25f;
        }
    }
}

// ===== IQ3_XXS kernel =====
struct block_iq3_xxs {
    half d;
    uint8_t qs[96];
};

kernel void kernel_mul_mv_iq3_xxs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 4;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = 32;
    const short ix = tiisg % ntx;

    float yl[32];
    float sumf[NR0] = {0.f};

    // Threadgroup memory for grid and signs
    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < NR0; ++row) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.5f;
        }
    }
}

// ===== IQ3_XXS kernel (split=1, NR0=8) =====
kernel void kernel_mul_mv_iq3_xxs_f32_split1(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 8;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = min(nb32, 32);
    const short nrep = 32 / ntx;
    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (NR0 * irep      ) / nrep;
    const short row1 = (NR0 * (irep + 1)) / nrep;

    float yl[32];
    float sumf[NR0] = {0.f};

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; ++row) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all * 0.5f;
        }
    }
}

// ===== IQ3_S kernel =====
struct block_iq3_s {
    half d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
};

kernel void kernel_mul_mv_iq3_s_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 4;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq3_s * x = (device const block_iq3_s *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = 32;
    const short ix = tiisg % ntx;

    float yl[32];
    float sumf[NR0] = {0.f};

    // Threadgroup memory for grid
    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < NR0; ++row) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}

// ===== IQ3_S kernel (split=1, NR0=8) =====
kernel void kernel_mul_mv_iq3_s_f32_split1(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_TQ2_0;
    const short NR0 = 8;

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq3_s * x = (device const block_iq3_s *)(src0 + first_row * args.nb01);

    const int nb32 = nb * (QK_K / 32);
    const short ntx = min(nb32, 32);
    const short nrep = 32 / ntx;
    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (NR0 * irep      ) / nrep;
    const short row1 = (NR0 * (irep + 1)) / nrep;

    float yl[32];
    float sumf[NR0] = {0.f};

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * y4 = src1 + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; ++row) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}
