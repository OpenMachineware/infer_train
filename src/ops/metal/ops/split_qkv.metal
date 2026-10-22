// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Strided split kernel: split a fused QKV buffer into separate Q, K, V buffers
// This is a utility kernel for fused QKV projection

#include <metal_stdlib>
using namespace metal;

kernel void kernel_split_qkv(
    device const float * fused_qkv [[buffer(0)]],  // Input: fused [q_size + k_size + v_size]
    device float * q_out [[buffer(1)]],            // Output: Q [q_size]
    device float * k_out [[buffer(2)]],            // Output: K [k_size]
    device float * v_out [[buffer(3)]],            // Output: V [v_size]
    constant uint32_t & q_size [[buffer(4)]],
    constant uint32_t & k_size [[buffer(5)]],
    constant uint32_t & v_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint32_t k_offset = q_size;
    const uint32_t v_offset = q_size + k_size;
    const uint32_t total_size = q_size + k_size + v_size;

    // Each thread handles one element
    if (tid < total_size) {
        if (tid < q_size) {
            // Q part
            q_out[tid] = fused_qkv[tid];
        } else if (tid < v_offset) {
            // K part
            k_out[tid - k_offset] = fused_qkv[tid];
        } else {
            // V part
            v_out[tid - v_offset] = fused_qkv[tid];
        }
    }
}