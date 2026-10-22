// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// FusedGateUp: Compute gate and up projections in a single kernel
// Instead of two separate GEMV calls, this kernel computes:
//   gate = W_gate @ input
//   up = W_up @ input
// in one dispatch, saving one kernel launch overhead.

#include <metal_stdlib>
using namespace metal;

// Include block structure definitions for quantized formats
#define GGML_COMMON_DECL_METAL
#include "../include/blocks.h"

#define QK_K 256

// Helper function for Q4_K scale extraction (from kquant_mv.metal)
static inline uchar2 get_scale_min_k4(int j, int k, device const uchar * scales) {
    if (j < 4) {
        return uchar2(uchar(scales[j+0+k] & 63), uchar(scales[j+4+k] & 63));
    } else {
        return uchar2(
            uchar((scales[j+4+k] & 0x0F) | ((scales[j-4+k] & 0xC0) >> 2)),
            uchar((scales[j+4+k] >> 4) | ((scales[j-0+k] & 0xC0) >> 2))
        );
    }
}

// ============================================================================
// F16 FusedGateUp kernel
// ============================================================================

kernel void kernel_fused_gate_up_f16(
    device const float* input [[buffer(0)]],           // [K] input vector
    device const half* gate_weights [[buffer(1)]],     // [M, K] gate projection weights
    device const half* up_weights [[buffer(2)]],       // [M, K] up projection weights
    device float* gate_output [[buffer(3)]],           // [M] gate output
    device float* up_output [[buffer(4)]],             // [M] up output
    constant uint& m [[buffer(5)]],                    // M (output size)
    constant uint& k [[buffer(6)]],                    // K (input size)
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    // Compute dot product for gate projection
    device const half* gate_row = gate_weights + row * k;
    float gate_sum = 0.0f;
    for (uint i = lane_id; i < k; i += group_size) {
        gate_sum += float(gate_row[i]) * input[i];
    }
    gate_sum = simd_sum(gate_sum);

    // Compute dot product for up projection
    device const half* up_row = up_weights + row * k;
    float up_sum = 0.0f;
    for (uint i = lane_id; i < k; i += group_size) {
        up_sum += float(up_row[i]) * input[i];
    }
    up_sum = simd_sum(up_sum);

    // Store results (only first lane in simdgroup writes)
    if (group_size > 32) {
        threadgroup float gate_buf[32];
        threadgroup float up_buf[32];

        if (lane_in_simd == 0) {
            gate_buf[simd_id] = gate_sum;
            up_buf[simd_id] = up_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float gate_total = 0.0f;
            float up_total = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                gate_total += gate_buf[i];
                up_total += up_buf[i];
            }
            gate_output[row] = gate_total;
            up_output[row] = up_total;
        }
    } else {
        if (lane_in_simd == 0) {
            gate_output[row] = gate_sum;
            up_output[row] = up_sum;
        }
    }
}

// ============================================================================
// Q4_K FusedGateUp kernel
// ============================================================================

kernel void kernel_fused_gate_up_q4_k(
    device const float* input [[buffer(0)]],
    device const char* gate_weights [[buffer(1)]],     // Q4_K quantized
    device const char* up_weights [[buffer(2)]],       // Q4_K quantized
    device float* gate_output [[buffer(3)]],
    device float* up_output [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& gate_nb01 [[buffer(7)]],           // Byte stride between gate weight rows
    constant uint& up_nb01 [[buffer(8)]],             // Byte stride between up weight rows
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    const int nb = k / QK_K;

    // Gate projection
    device const char* gate_row_ptr = gate_weights + row * gate_nb01;
    device const block_q4_K* gate_blocks = (device const block_q4_K*)gate_row_ptr;

    float gate_sumf = 0.0f;
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q4_K* block = gate_blocks + ib;
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        for (int is = 0; is < 8; is++) {
            uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
            float sc = sc_min.x;
            float m = sc_min.y;
            float dl = d * sc;
            float ml = dmin * m;

            device const uchar* qs = block->qs + is * 16;
            for (int kk = 0; kk < 16; kk++) {
                uchar byte = qs[kk];
                float v0 = (byte & 0x0F);
                float v1 = ((byte >> 4) & 0x0F);
                float y0 = input[ib * QK_K + is * 32 + kk];
                float y1 = input[ib * QK_K + is * 32 + 16 + kk];
                gate_sumf += (dl * v0 - ml) * y0;
                gate_sumf += (dl * v1 - ml) * y1;
            }
        }
    }
    gate_sumf = simd_sum(gate_sumf);

    // Up projection
    device const char* up_row_ptr = up_weights + row * up_nb01;
    device const block_q4_K* up_blocks = (device const block_q4_K*)up_row_ptr;

    float up_sumf = 0.0f;
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q4_K* block = up_blocks + ib;
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        for (int is = 0; is < 8; is++) {
            uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
            float sc = sc_min.x;
            float m = sc_min.y;
            float dl = d * sc;
            float ml = dmin * m;

            device const uchar* qs = block->qs + is * 16;
            for (int kk = 0; kk < 16; kk++) {
                uchar byte = qs[kk];
                float v0 = (byte & 0x0F);
                float v1 = ((byte >> 4) & 0x0F);
                float y0 = input[ib * QK_K + is * 32 + kk];
                float y1 = input[ib * QK_K + is * 32 + 16 + kk];
                up_sumf += (dl * v0 - ml) * y0;
                up_sumf += (dl * v1 - ml) * y1;
            }
        }
    }
    up_sumf = simd_sum(up_sumf);

    // Store results
    if (group_size > 32) {
        threadgroup float gate_buf[32];
        threadgroup float up_buf[32];

        if (lane_in_simd == 0) {
            gate_buf[simd_id] = gate_sumf;
            up_buf[simd_id] = up_sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float gate_total = 0.0f;
            float up_total = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                gate_total += gate_buf[i];
                up_total += up_buf[i];
            }
            gate_output[row] = gate_total;
            up_output[row] = up_total;
        }
    } else {
        if (lane_in_simd == 0) {
            gate_output[row] = gate_sumf;
            up_output[row] = up_sumf;
        }
    }
}

// ============================================================================
// Q5_K FusedGateUp kernel
// ============================================================================

kernel void kernel_fused_gate_up_q5_k(
    device const float* input [[buffer(0)]],
    device const char* gate_weights [[buffer(1)]],
    device const char* up_weights [[buffer(2)]],
    device float* gate_output [[buffer(3)]],
    device float* up_output [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& gate_nb01 [[buffer(7)]],
    constant uint& up_nb01 [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    const int nb = k / QK_K;

    // Gate projection
    device const char* gate_row_ptr = gate_weights + row * gate_nb01;
    device const block_q5_K* gate_blocks = (device const block_q5_K*)gate_row_ptr;

    float gate_sumf = 0.0f;
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q5_K* block = gate_blocks + ib;
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        for (int is = 0; is < 8; is++) {
            uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
            float sc = sc_min.x;
            float m = sc_min.y;
            float dl = d * sc;
            float ml = dmin * m;

            device const uchar* qs = block->qs + is * 16;
            device const uchar* qh = block->qh + is * 4;

            for (int kk = 0; kk < 16; kk++) {
                uchar byte = qs[kk];
                uchar high_bits = qh[kk / 4];

                float v0 = (byte & 0x0F) + ((high_bits & 0x01) ? 16.0f : 0.0f);
                float v1 = ((byte >> 4) & 0x0F) + ((high_bits & 0x02) ? 16.0f : 0.0f);
                float y0 = input[ib * QK_K + is * 32 + kk];
                float y1 = input[ib * QK_K + is * 32 + 16 + kk];
                gate_sumf += (dl * v0 - ml) * y0;
                gate_sumf += (dl * v1 - ml) * y1;

                high_bits >>= 2;
            }
        }
    }
    gate_sumf = simd_sum(gate_sumf);

    // Up projection
    device const char* up_row_ptr = up_weights + row * up_nb01;
    device const block_q5_K* up_blocks = (device const block_q5_K*)up_row_ptr;

    float up_sumf = 0.0f;
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q5_K* block = up_blocks + ib;
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];
        float dmin = dh[1];

        for (int is = 0; is < 8; is++) {
            uchar2 sc_min = get_scale_min_k4(is, 0, block->scales);
            float sc = sc_min.x;
            float m = sc_min.y;
            float dl = d * sc;
            float ml = dmin * m;

            device const uchar* qs = block->qs + is * 16;
            device const uchar* qh = block->qh + is * 4;

            for (int kk = 0; kk < 16; kk++) {
                uchar byte = qs[kk];
                uchar high_bits = qh[kk / 4];

                float v0 = (byte & 0x0F) + ((high_bits & 0x01) ? 16.0f : 0.0f);
                float v1 = ((byte >> 4) & 0x0F) + ((high_bits & 0x02) ? 16.0f : 0.0f);
                float y0 = input[ib * QK_K + is * 32 + kk];
                float y1 = input[ib * QK_K + is * 32 + 16 + kk];
                up_sumf += (dl * v0 - ml) * y0;
                up_sumf += (dl * v1 - ml) * y1;

                high_bits >>= 2;
            }
        }
    }
    up_sumf = simd_sum(up_sumf);

    // Store results
    if (group_size > 32) {
        threadgroup float gate_buf[32];
        threadgroup float up_buf[32];

        if (lane_in_simd == 0) {
            gate_buf[simd_id] = gate_sumf;
            up_buf[simd_id] = up_sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float gate_total = 0.0f;
            float up_total = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                gate_total += gate_buf[i];
                up_total += up_buf[i];
            }
            gate_output[row] = gate_total;
            up_output[row] = up_total;
        }
    } else {
        if (lane_in_simd == 0) {
            gate_output[row] = gate_sumf;
            up_output[row] = up_sumf;
        }
    }
}

// ============================================================================
// Q6_K FusedGateUp kernel
// ============================================================================

kernel void kernel_fused_gate_up_q6_k(
    device const float* input [[buffer(0)]],
    device const char* gate_weights [[buffer(1)]],
    device const char* up_weights [[buffer(2)]],
    device float* gate_output [[buffer(3)]],
    device float* up_output [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& gate_nb01 [[buffer(7)]],
    constant uint& up_nb01 [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane_id [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (row >= m) return;

    const int nb = k / QK_K;

    // Gate projection
    device const char* gate_row_ptr = gate_weights + row * gate_nb01;
    device const block_q6_K* gate_blocks = (device const block_q6_K*)gate_row_ptr;

    float gate_sumf = 0.0f;
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q6_K* block = gate_blocks + ib;
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];

        for (int is = 0; is < 16; is++) {
            float sc = block->scales[is];
            device const uchar* ql = block->ql + is * 8;
            device const uchar* qh = block->qh + is * 4;

            for (int kk = 0; kk < 8; kk++) {
                uchar l = ql[kk];
                uchar h = qh[kk];

                float v0 = (l & 0x0F) + ((h & 0x03) << 4);
                float v1 = ((l >> 4) & 0x0F) + ((h >> 2) & 0x03) * 16.0f;
                float y0 = input[ib * QK_K + is * 16 + kk];
                float y1 = input[ib * QK_K + is * 16 + 8 + kk];
                gate_sumf += d * sc * v0 * y0;
                gate_sumf += d * sc * v1 * y1;
            }
        }
    }
    gate_sumf = simd_sum(gate_sumf);

    // Up projection
    device const char* up_row_ptr = up_weights + row * up_nb01;
    device const block_q6_K* up_blocks = (device const block_q6_K*)up_row_ptr;

    float up_sumf = 0.0f;
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q6_K* block = up_blocks + ib;
        device const half* dh = (device const half*)&block->d;
        float d = dh[0];

        for (int is = 0; is < 16; is++) {
            float sc = block->scales[is];
            device const uchar* ql = block->ql + is * 8;
            device const uchar* qh = block->qh + is * 4;

            for (int kk = 0; kk < 8; kk++) {
                uchar l = ql[kk];
                uchar h = qh[kk];

                float v0 = (l & 0x0F) + ((h & 0x03) << 4);
                float v1 = ((l >> 4) & 0x0F) + ((h >> 2) & 0x03) * 16.0f;
                float y0 = input[ib * QK_K + is * 16 + kk];
                float y1 = input[ib * QK_K + is * 16 + 8 + kk];
                up_sumf += d * sc * v0 * y0;
                up_sumf += d * sc * v1 * y1;
            }
        }
    }
    up_sumf = simd_sum(up_sumf);

    // Store results
    if (group_size > 32) {
        threadgroup float gate_buf[32];
        threadgroup float up_buf[32];

        if (lane_in_simd == 0) {
            gate_buf[simd_id] = gate_sumf;
            up_buf[simd_id] = up_sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_sgs = group_size / 32;
        if (simd_id == 0) {
            float gate_total = 0.0f;
            float up_total = 0.0f;
            for (uint i = 0; i < n_sgs; i++) {
                gate_total += gate_buf[i];
                up_total += up_buf[i];
            }
            gate_output[row] = gate_total;
            up_output[row] = up_total;
        }
    } else {
        if (lane_in_simd == 0) {
            gate_output[row] = gate_sumf;
            up_output[row] = up_sumf;
        }
    }
}
