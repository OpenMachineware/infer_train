#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define N_R0_Q5_K 2   // Rows per SIMD group (llama.cpp: N_R0_Q5_K)
#define N_SG_Q5_K 2   // SIMD groups per threadgroup (llama.cpp: N_SG_Q5_K)

// Loop unroll pragma matching llama.cpp
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Q5_K block structure matching Rust BlockQ5K
struct block_q5_K {
    half d;            // Super-block scale for quantized scales
    half dmin;         // Super-block scale for quantized mins
    uint8_t scales[12]; // Scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8]; // High quants
    uint8_t qs[QK_K/2]; // 4-bit quants
};

// Args structure
struct mv_args {
    uint32_t ne00;  // K dimension
    uint32_t ne01;  // M dimension (rows)
    uint64_t nb01;  // Byte stride for rows in src0
};

kernel void kernel_mul_mv_q5_K_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = N_SG_Q5_K;
    const short NR0 = N_R0_Q5_K;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    const int nb = args.ne00/QK_K;

    const int first_row = (tgpig.x * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + first_row * args.nb01);

    float sumf[NR0]={0.f};

    float yl[16], yh[16];

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = src1 + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+  0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+ 32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+  0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+ 32]; sumy[3] += yh[l+8];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};

            for (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * ((q1[l] & 0x0F) + ((h & hm1) ? 16 : 0));
                acc1[1] += yl[l+8] * ((q1[l] & 0xF0) + ((h & hm2) ? 256 : 0));
                acc1[2] += yh[l+0] * ((q2[l] & 0x0F) + ((h & hm3) ? 16 : 0));
                acc1[3] += yh[l+8] * ((q2[l] & 0xF0) + ((h & hm4) ? 256 : 0));
            }

            sumf[row] += dh[0] * (acc1[0] * sc8[0] +
                                  acc1[1] * sc8[1] * 1.f/16.f +
                                  acc1[2] * sc8[4] +
                                  acc1[3] * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            dh += args.nb01/2;
            a  += args.nb01;
            qh += args.nb01;
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
