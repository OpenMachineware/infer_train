#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q4_K_LARGE 2   // Rows per SIMD group for large K (>=4 blocks)
#define N_R0_Q4_K_SMALL 8   // Rows per SIMD group for small K (<4 blocks) - more work per threadgroup
#define N_SG_Q4_K 2         // SIMD groups per threadgroup
#define N_COLS_LARGE 4      // Columns per threadgroup for large K
#define N_COLS_SMALL 4      // Columns per threadgroup for small K

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Q4_K block structure matching Rust BlockQ4K
struct block_q4_K {
    half d;            // Super-block scale
    half dmin;         // Super-block scale for mins
    uint8_t scales[12]; // Scales and mins
    uint8_t qs[QK_K/2]; // 4-bit quants
};

// GEMM args
struct gemm_args {
    uint32_t ne00;  // K dimension
    uint32_t ne01;  // M dimension (rows in weights)
    uint32_t ne02;  // N dimension (batch size / number of input vectors)
    uint64_t nb01;  // Byte stride for rows in weights
    uint64_t nb11;  // Byte stride for rows in input (each row is an input vector)
};

// Q4_K × F32 GEMM kernel
// Uses different NR0 based on K size to maximize GPU utilization
kernel void kernel_gemm_q4_k_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q4_K;
    const int nb = args.ne00/QK_K;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    if (nb > 4) {
        // ===== Large K path: parallelize across blocks =====
        const short NR0 = N_R0_Q4_K_LARGE;
        const short NC = N_COLS_LARGE;
        const short ix = tiisg/8;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        device const block_q4_K * x = (device const block_q4_K *) (src0 + first_row * args.nb01);

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
                        acc1[0] += yl[c][2*i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[c][2*i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[c][2*i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[c][2*i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[c][2*i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[c][2*i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[c][2*i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[c][2*i + 9] * (q2[i] & 0xF000);
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
        // ===== Small K path: each thread processes unique (row, col) output element =====
        // For nb=1-3, process more output elements in parallel
        // Each thread computes a complete dot product for its assigned output element
        const short NR0 = N_R0_Q4_K_SMALL;
        const short NC = N_COLS_SMALL;

        const int first_row = (tgpig.x * NSG + sgitg) * NR0;
        const int first_col = tgpig.y * NC;

        if (first_row >= args.ne01) return;

        // Map thread to output element within threadgroup's tile
        // tiisg=0..31 maps to (row, col) pairs in 8×4 tile
        const int local_row = tiisg / NC;   // 0..7
        const int local_col = tiisg % NC;   // 0..3

        const int row = first_row + local_row;
        const int col = first_col + local_col;

        if (row >= args.ne01 || col >= args.ne02) return;

        device const block_q4_K * x = (device const block_q4_K *) (src0 + row * args.nb01);
        device const float * y = src1 + col * args.ne00;

        float sum = 0.0f;

        // Process all K blocks
        for (int ib = 0; ib < nb; ib++) {
            // Load input for all 256 elements of this block
            // We'll process in 8 iterations, each handling 32 elements
            // Use same logic as large K path but without thread partitioning

            uint16_t sc16[4];
            thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

            // Process 8 sub-blocks (each covers 32 elements)
            // This matches the it=0..7 logic from large K path
            for (int it = 0; it < 8; it++) {
                const int iq = it / 4;
                const int ir = it % 4;

                // Load 32 input values
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

                // Load scales
                device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
                device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
                device const half * dh = &x[ib].d;

                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t * q2 = q1 + 32;

                // Accumulate for this sub-block
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
                }

                sum += (float)dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                       (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                       (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                       (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                       (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                       sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
        }

        // Store result directly
        dst[row * args.ne02 + col] = sum;
    }
}
