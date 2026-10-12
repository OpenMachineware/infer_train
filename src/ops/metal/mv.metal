#include <metal_stdlib>
using namespace metal;

#define QK 32

// IQ4_NL lookup table
constant float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

struct BlockIQ4NL {
    half d;
    uint8_t qs[16];
};

struct BlockQ8_0 {
    half d;
    int8_t qs[32];
};

// Matrix-Vector: M rows × K columns
// Each thread processes one row
kernel void mv_iq4_nl_q8_0(
    device const BlockIQ4NL* weights [[buffer(0)]],  // M × (K/QK) blocks
    device const BlockQ8_0* input [[buffer(1)]],      // (K/QK) blocks
    device float* output [[buffer(2)]],                // M outputs
    constant uint& m [[buffer(3)]],                     // number of rows
    constant uint& nb [[buffer(4)]],                    // number of blocks per row (K/QK)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= m) return;

    float sumf = 0.0f;
    uint row_offset = tid * nb;

    for (uint i = 0; i < nb; i++) {
        const device BlockIQ4NL& wb = weights[row_offset + i];
        const device BlockQ8_0& ib = input[i];

        float sumi = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t q = wb.qs[j];
            float v0 = kvalues_iq4nl_f[q & 0xf];
            float v1 = kvalues_iq4nl_f[(q >> 4) & 0xf];
            float q8_0 = float(ib.qs[j]);
            float q8_1 = float(ib.qs[j + 16]);
            sumi += v0 * q8_0 + v1 * q8_1;
        }

        sumf += float(wb.d) * float(ib.d) * sumi;
    }

    output[tid] = sumf;
}
