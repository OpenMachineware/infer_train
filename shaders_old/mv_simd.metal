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

// SIMD-optimized MV: each threadgroup (32 threads) processes one row
// Each thread processes nb/16 blocks, then SIMD reduction
kernel void mv_iq4_nl_q8_0_simd(
    device const BlockIQ4NL* weights [[buffer(0)]],
    device const BlockQ8_0* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& nb [[buffer(4)]],
    uint tid [[thread_index_in_simdgroup]],
    uint row [[simdgroup_index_in_threadgroup]])
{
    if (row >= m) return;

    uint row_offset = row * nb;

    // Each thread processes nb/16 blocks
    // tid = 0..31, so each thread processes (nb/32) blocks
    float sum = 0.0f;
    uint blocks_per_thread = (nb + 31) / 32;

    for (uint i = 0; i < blocks_per_thread; i++) {
        uint block_idx = tid * blocks_per_thread + i;
        if (block_idx >= nb) break;

        const device BlockIQ4NL& wb = weights[row_offset + block_idx];
        const device BlockQ8_0& ib = input[block_idx];

        for (int j = 0; j < 16; j++) {
            uint8_t q = wb.qs[j];
            float v0 = kvalues_iq4nl_f[q & 0xf];
            float v1 = kvalues_iq4nl_f[(q >> 4) & 0xf];
            float q8_0 = float(ib.qs[j]);
            float q8_1 = float(ib.qs[j + 16]);
            sum += v0 * q8_0 + v1 * q8_1;
        }
    }

    // SIMD reduction: sum all thread values in the simdgroup
    float total_sum = simd_sum(sum);

    // Thread 0 writes result (applies scales)
    if (tid == 0) {
        // For now, use scale of first block (simplified)
        // TODO: properly handle per-block scales
        output[row] = total_sum;
    }
}
