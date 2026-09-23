#include <metal_stdlib>
using namespace metal;

#define QK 32
#define N_SIMDWIDTH 32
#define NR0 4   // Rows per SIMD group
#define NSG 2   // SIMD groups per threadgroup

struct BlockQ4_0 {
    half d;
    uint8_t qs[QK/2];
};

// Helper function matching llama.cpp
// Calculate dot product between half a Q4_0 block and 16 floats
inline float block_q4_0_dot_y(device const BlockQ4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = float(qb_curr->d);

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // Load qs as uint16_t for efficient bit extraction
    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// Q4_0 x F32: matching llama.cpp kernel exactly
// Each SIMD group processes NR0 rows
// Each threadgroup has NSG SIMD groups
// Total rows per threadgroup = NSG * NR0 = 8
kernel void mv_q4_0_f32_simd(
    device const BlockQ4_0* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& nb [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int r0 = (tgpig.x * NSG + sgitg) * NR0;
    if (r0 >= m) return;

    // Which blocks this thread processes
    const short NQ = 16;
    const short ix = tiisg / 2;      // Block index (0-15)
    const short il = (tiisg % 2) * 8; // Offset within block (0 or 8)

    // Load pointers to NR0 rows
    device const BlockQ4_0* ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        if (r0 + row < m) {
            ax[row] = weights + (r0 + row) * nb;
        }
    }

    float sumf[NR0] = {0.f};

    device const float * yb = input + ix * QK + il;

    // Load y values and apply scaling for bit extraction
    float yl[16];
    float sumy[2] = { 0.f, 0.f };

    // Process blocks with stride NQ=16
    for (int ib = ix; ib < nb; ib += NQ) {
        // Load and scale y values (matching llama.cpp pattern)
        for (int i = 0; i < 8; i += 2) {
            sumy[0] += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1] / 256.f;

            sumy[1] += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.f;
            yl[i + 9] = yb[i + 17] / 4096.f;
        }

        // Compute for all NR0 rows
        for (int row = 0; row < NR0; ++row) {
            if (r0 + row >= m) break;
            sumf[row] += block_q4_0_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK * NQ;
    }

    // SIMD reduction and write results
    device float* dst = output + r0;
    for (int row = 0; row < NR0; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < m) {
            dst[row] = tot;
        }
    }
}
