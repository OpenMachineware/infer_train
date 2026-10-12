#include <metal_stdlib>
using namespace metal;

#define QK 32  // Q8_0 block size

// IQ4_NL lookup table (non-linear quantization - only 16 values for 4-bit indices)
constant float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

struct BlockIQ4NL {
    half d;          // scale
    uint8_t qs[16];  // quantized values (32 values in 16 bytes)
};

struct BlockQ8_0 {
    half d;          // scale
    int8_t qs[32];   // quantized values
};

// IQ4_NL x Q8_0 vector dot product
kernel void vec_dot_iq4_nl_q8_0(
    device const BlockIQ4NL* x [[buffer(0)]],
    device const BlockQ8_0* y [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& nb [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]])
{
    float sumf = 0.0f;

    for (uint i = 0; i < nb; i++) {
        const device BlockIQ4NL& xb = x[i];
        const device BlockQ8_0& yb = y[i];

        float sumi = 0.0f;

        for (int j = 0; j < 16; j++) {
            uint8_t q = xb.qs[j];

            // Extract two 4-bit indices
            float v0 = kvalues_iq4nl_f[q & 0xf];
            float v1 = kvalues_iq4nl_f[(q >> 4) & 0xf];

            // Corresponding Q8 values (low 4-bit maps to qs[j], high 4-bit maps to qs[j+16])
            float q8_0 = float(yb.qs[j]);
            float q8_1 = float(yb.qs[j + 16]);

            sumi += v0 * q8_0 + v1 * q8_1;
        }

        float d = float(xb.d) * float(yb.d);
        sumf += d * sumi;
    }

    *result = sumf;
}
