#include <metal_stdlib>
using namespace metal;

#define QK4_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q4_0 4   // Rows per SIMD group (llama.cpp: N_R0_Q4_0)
#define N_SG_Q4_0 2   // SIMD groups per threadgroup (llama.cpp: N_SG_Q4_0)

// Loop unroll pragma matching llama.cpp
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};

// Helper function matching llama.cpp exactly
// Calculate dot product between half a Q4_0 block and 16 floats
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// Q4_0 x F32 MV kernel matching llama.cpp exactly
// Args structure to match llama.cpp
struct mv_args {
    uint32_t ne00;  // K dimension
    uint32_t ne01;  // M dimension (rows)
    uint64_t nb01;  // Byte stride for rows in src0
};

kernel void kernel_mul_mv_q4_0_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q4_0;
    const short NR0 = N_R0_Q4_0;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;

    const int nb = args.ne00 / QK4_0;  // Number of blocks per row
    const int r0 = (tgpig.x * NSG + sgitg) * NR0;

    if (r0 >= args.ne01) return;

    // Load pointers to NR0 rows
    device const block_q4_0 * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01;
        ax[row] = (device const block_q4_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    const short ix = (tiisg / (NW / NQ));
    const short il = (tiisg % (NW / NQ)) * 8;

    const int ib0 = ix;

    float yl[16]; // src1 vector cache
    device const float * yb = src1 + ib0 * QK4_0 + il;

    // Each thread in a SIMD group deals with half a block
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1] / 256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.f;
            yl[i + 9] = yb[i + 17] / 4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }

    device float * dst_f32 = dst + r0;

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}
