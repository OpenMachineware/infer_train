#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32

// Loop unroll pragma matching llama.cpp
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Q2_K block structure
struct block_q2_K {
    half d;            // Super-block scale
    half dmin;         // Super-block min scale
    uint8_t scales[16]; // Scales
    uint8_t qs[QK_K/4]; // 2-bit quants
};

// Q3_K block structure
struct block_q3_K {
    half d;            // Super-block scale
    uint8_t hmask[QK_K/8]; // High bit mask
    uint8_t qs[QK_K/4];    // 2-bit quants
    uint8_t scales[12];    // Scales
};

// Q6_K block structure
struct block_q6_K {
    uint8_t ql[128];   // Lower 4 bits
    uint8_t qh[64];    // Upper 2 bits
    int8_t scales[16]; // Scales
    half d;            // Scale
};

// Args structure
struct mv_args {
    uint32_t ne00;  // K dimension
    uint32_t ne01;  // M dimension (rows)
    uint64_t nb01;  // Byte stride for rows in src0
};

// ===== Q2_K kernel =====
#define N_R0_Q2_K 2
#define N_SG_Q2_K 2

kernel void kernel_mul_mv_q2_K_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q2_K;
    const short NR0 = N_R0_Q2_K;

    const int nb = args.ne00/QK_K;

    const int first_row = (tgpig.x * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + first_row * args.nb01);

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;          // 0 or 1
    const short il  = 2*((tid%4)/2);  // 0 or 2
    const short ir  = tid%2;
    const short l0  = 8*ir;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = src1 + ix*QK_K + y_offset;

    float sumf[NR0]={0.f};

    for (int i = ix; i < nb; i += 4) {
        device const float * y2 = y1 + 128;

        float sumy = 0;
        for (short l = 0; l < 8; ++l) {
            sumy += y1[l] + y2[l];
        }

        device const uint8_t * sc = (device const uint8_t *)x[i].scales;
        device const uint8_t * q = x[i].qs + q_offset;
        device const half * dh = &x[i].d;

        for (short row = 0; row < NR0; ++row) {
            const float d = (float)dh[0];
            const float dmin = (float)dh[1];

            float acc = 0;
            for (short l = 0; l < 8; ++l) {
                acc += y1[l] * (q[l] & 0x03) + y2[l] * (q[l] & 0x0C);
            }

            sumf[row] += d * acc * sc[ip] - dmin * sumy;

            dh += args.nb01/2;
            sc += args.nb01;
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}
