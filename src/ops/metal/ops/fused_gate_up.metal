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

    // Combined projection: read input once, use for both gate and up
    device const char* gate_row_ptr = gate_weights + row * gate_nb01;
    device const char* up_row_ptr = up_weights + row * up_nb01;
    device const block_q4_K* gate_blocks = (device const block_q4_K*)gate_row_ptr;
    device const block_q4_K* up_blocks = (device const block_q4_K*)up_row_ptr;

    float gate_sumf = 0.0f;
    float up_sumf = 0.0f;

    // Process both projections together, caching input reads
    for (int ib = lane_id; ib < nb; ib += group_size) {
        // Gate block
        device const block_q4_K* gate_block = gate_blocks + ib;
        device const half* gate_dh = (device const half*)&gate_block->d;
        float gate_d = gate_dh[0];
        float gate_dmin = gate_dh[1];

        // Up block
        device const block_q4_K* up_block = up_blocks + ib;
        device const half* up_dh = (device const half*)&up_block->d;
        float up_d = up_dh[0];
        float up_dmin = up_dh[1];

        // Process all 8 sub-blocks
        for (int is = 0; is < 8; is++) {
            // Gate scales
            uchar2 gate_sc_min = get_scale_min_k4(is, 0, gate_block->scales);
            float gate_sc = gate_sc_min.x;
            float gate_m = gate_sc_min.y;
            float gate_dl = gate_d * gate_sc;
            float gate_ml = gate_dmin * gate_m;

            // Up scales
            uchar2 up_sc_min = get_scale_min_k4(is, 0, up_block->scales);
            float up_sc = up_sc_min.x;
            float up_m = up_sc_min.y;
            float up_dl = up_d * up_sc;
            float up_ml = up_dmin * up_m;

            device const uchar* gate_qs = gate_block->qs + is * 16;
            device const uchar* up_qs = up_block->qs + is * 16;

            // Process 16 bytes = 32 elements per sub-block
            // Cache input values and use for both projections
            for (int kk = 0; kk < 16; kk++) {
                // Read input ONCE
                float y0 = input[ib * QK_K + is * 32 + kk];
                float y1 = input[ib * QK_K + is * 32 + 16 + kk];

                // Gate projection
                uchar gate_byte = gate_qs[kk];
                float gate_v0 = (gate_byte & 0x0F);
                float gate_v1 = ((gate_byte >> 4) & 0x0F);
                gate_sumf += (gate_dl * gate_v0 - gate_ml) * y0;
                gate_sumf += (gate_dl * gate_v1 - gate_ml) * y1;

                // Up projection
                uchar up_byte = up_qs[kk];
                float up_v0 = (up_byte & 0x0F);
                float up_v1 = ((up_byte >> 4) & 0x0F);
                up_sumf += (up_dl * up_v0 - up_ml) * y0;
                up_sumf += (up_dl * up_v1 - up_ml) * y1;
            }
        }
    }
    gate_sumf = simd_sum(gate_sumf);
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

    // Combined projection: read input once, use for both gate and up
    device const char* gate_row_ptr = gate_weights + row * gate_nb01;
    device const char* up_row_ptr = up_weights + row * up_nb01;
    device const block_q5_K* gate_blocks = (device const block_q5_K*)gate_row_ptr;
    device const block_q5_K* up_blocks = (device const block_q5_K*)up_row_ptr;

    float gate_sumf = 0.0f;
    float up_sumf = 0.0f;

    // Process both projections together, caching input reads
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q5_K* gate_block = gate_blocks + ib;
        device const half* gate_dh = (device const half*)&gate_block->d;
        float gate_d = gate_dh[0];
        float gate_dmin = gate_dh[1];

        device const block_q5_K* up_block = up_blocks + ib;
        device const half* up_dh = (device const half*)&up_block->d;
        float up_d = up_dh[0];
        float up_dmin = up_dh[1];

        for (int is = 0; is < 8; is++) {
            // Gate scales
            uchar2 gate_sc_min = get_scale_min_k4(is, 0, gate_block->scales);
            float gate_sc = gate_sc_min.x;
            float gate_m = gate_sc_min.y;
            float gate_dl = gate_d * gate_sc;
            float gate_ml = gate_dmin * gate_m;

            // Up scales
            uchar2 up_sc_min = get_scale_min_k4(is, 0, up_block->scales);
            float up_sc = up_sc_min.x;
            float up_m = up_sc_min.y;
            float up_dl = up_d * up_sc;
            float up_ml = up_dmin * up_m;

            device const uchar* gate_qs = gate_block->qs + is * 16;
            device const uchar* gate_qh = gate_block->qh + is * 4;
            device const uchar* up_qs = up_block->qs + is * 16;
            device const uchar* up_qh = up_block->qh + is * 4;

            for (int kk = 0; kk < 16; kk++) {
                // Read input ONCE
                float y0 = input[ib * QK_K + is * 32 + kk];
                float y1 = input[ib * QK_K + is * 32 + 16 + kk];

                // Gate projection
                uchar gate_byte = gate_qs[kk];
                uchar gate_high_bits = gate_qh[kk / 4];
                float gate_v0 = (gate_byte & 0x0F) + ((gate_high_bits & 0x01) ? 16.0f : 0.0f);
                float gate_v1 = ((gate_byte >> 4) & 0x0F) + ((gate_high_bits & 0x02) ? 16.0f : 0.0f);
                gate_sumf += (gate_dl * gate_v0 - gate_ml) * y0;
                gate_sumf += (gate_dl * gate_v1 - gate_ml) * y1;

                // Up projection
                uchar up_byte = up_qs[kk];
                uchar up_high_bits = up_qh[kk / 4];
                float up_v0 = (up_byte & 0x0F) + ((up_high_bits & 0x01) ? 16.0f : 0.0f);
                float up_v1 = ((up_byte >> 4) & 0x0F) + ((up_high_bits & 0x02) ? 16.0f : 0.0f);
                up_sumf += (up_dl * up_v0 - up_ml) * y0;
                up_sumf += (up_dl * up_v1 - up_ml) * y1;
            }
        }
    }
    gate_sumf = simd_sum(gate_sumf);
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

    // Combined projection: read input once, use for both gate and up
    device const char* gate_row_ptr = gate_weights + row * gate_nb01;
    device const char* up_row_ptr = up_weights + row * up_nb01;
    device const block_q6_K* gate_blocks = (device const block_q6_K*)gate_row_ptr;
    device const block_q6_K* up_blocks = (device const block_q6_K*)up_row_ptr;

    float gate_sumf = 0.0f;
    float up_sumf = 0.0f;

    // Process both projections together, caching input reads
    for (int ib = lane_id; ib < nb; ib += group_size) {
        device const block_q6_K* gate_block = gate_blocks + ib;
        device const half* gate_dh = (device const half*)&gate_block->d;
        float gate_d = gate_dh[0];

        device const block_q6_K* up_block = up_blocks + ib;
        device const half* up_dh = (device const half*)&up_block->d;
        float up_d = up_dh[0];

        for (int is = 0; is < 16; is++) {
            float gate_sc = gate_block->scales[is];
            device const uchar* gate_ql = gate_block->ql + is * 8;
            device const uchar* gate_qh = gate_block->qh + is * 4;

            float up_sc = up_block->scales[is];
            device const uchar* up_ql = up_block->ql + is * 8;
            device const uchar* up_qh = up_block->qh + is * 4;

            for (int kk = 0; kk < 8; kk++) {
                // Read input ONCE
                float y0 = input[ib * QK_K + is * 16 + kk];
                float y1 = input[ib * QK_K + is * 16 + 8 + kk];

                // Gate projection
                uchar gate_l = gate_ql[kk];
                uchar gate_h = gate_qh[kk];
                float gate_v0 = (gate_l & 0x0F) + ((gate_h & 0x03) << 4);
                float gate_v1 = ((gate_l >> 4) & 0x0F) + ((gate_h >> 2) & 0x03) * 16.0f;
                gate_sumf += gate_d * gate_sc * gate_v0 * y0;
                gate_sumf += gate_d * gate_sc * gate_v1 * y1;

                // Up projection
                uchar up_l = up_ql[kk];
                uchar up_h = up_qh[kk];
                float up_v0 = (up_l & 0x0F) + ((up_h & 0x03) << 4);
                float up_v1 = ((up_l >> 4) & 0x0F) + ((up_h >> 2) & 0x03) * 16.0f;
                up_sumf += up_d * up_sc * up_v0 * y0;
                up_sumf += up_d * up_sc * up_v1 * y1;
            }
        }
    }
    gate_sumf = simd_sum(gate_sumf);
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
