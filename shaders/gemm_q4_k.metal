#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q4_K 2   // Rows per SIMD group (matching MV kernel)
#define N_SG_Q4_K 2   // SIMD groups per threadgroup
#define N_COLS 4      // Columns (input vectors) per threadgroup

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
// Processes NR0 rows × N_COLS columns per threadgroup
// Each output element (row, col) is dot product of weight row with input vector
kernel void kernel_gemm_q4_k_f32(
    device const char * src0 [[buffer(0)]],    // Weights: M × K (BlockQ4K)
    device const float * src1 [[buffer(1)]],   // Input: N × K (FP32, N input vectors)
    device float * dst [[buffer(2)]],          // Output: M × N
    constant gemm_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q4_K;
    const short NR0 = N_R0_Q4_K;
    const short NC = N_COLS;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;

    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int first_col = tgpig.y * NC;

    if (first_row >= args.ne01) return;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + first_row * args.nb01);

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    // Sum for each row × column
    float sumf[NR0][NC] = {{0.f}};

    // Load input data for each column (input vector)
    // yl[nc][i] stores 16 floats for column nc
    float yl[NC][16];
    float yh[NC][16];
    float4 sumy[NC];

    // Process all K blocks
    for (int ib = ix; ib < nb; ib += 4) {
        // Load input data for all columns
        for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
            // Input is N × K, so input vector c starts at src1 + c * K
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

        // Process each row
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half * dh = &x[ib].d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            // For each column
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

                sumf[row][c] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                         (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                         (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                         (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                               dh[1] * (sumy[c][0] * sc8[2] + sumy[c][1] * sc8[3] +
                                        sumy[c][2] * sc8[6] + sumy[c][3] * sc8[7]);
            }

            // Move to next row (advance pointers)
            sc += args.nb01/2;
            q1 += args.nb01/2;
            dh += args.nb01/2;
        }
    }

    // Store results
    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        for (int c = 0; c < NC && first_col + c < args.ne02; ++c) {
            float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst[(first_row + row) * args.ne02 + (first_col + c)] = sum_all;
            }
        }
    }
}
