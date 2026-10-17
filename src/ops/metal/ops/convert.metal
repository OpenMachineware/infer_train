// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// F32 to F16 conversion kernel for KV cache updates

#include <metal_stdlib>
using namespace metal;

// Convert F32 to F16
// Each thread processes one element
// Grid: [n_heads, head_dim] for one position
kernel void kernel_convert_f32_to_f16(
        device const float* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& n_elements [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid < n_elements) {
        dst[tid] = half(src[tid]);
    }
}

// Convert and copy F32 to F16 with strided layout for KV cache
// Layout: [n_kv_heads, max_seq, head_dim]
// Each invocation handles one head's data at a specific position
kernel void kernel_convert_f32_to_f16_kv(
        device const float* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& n_heads [[buffer(2)]],
        constant uint& head_dim [[buffer(3)]],
        constant uint& pos [[buffer(4)]],     // current position in KV cache
        constant uint& max_seq [[buffer(5)]], // max_seq for stride calculation
        uint tid [[thread_position_in_grid]]) {

    // Total elements: n_heads * head_dim
    uint total_elements = n_heads * head_dim;
    if (tid >= total_elements) return;

    // Calculate head and offset within head
    uint head_idx = tid / head_dim;
    uint offset_in_head = tid % head_dim;

    // Source layout: [n_heads, head_dim] (contiguous)
    uint src_idx = head_idx * head_dim + offset_in_head;

    // Destination layout: [n_heads, max_seq, head_dim] (strided)
    uint dst_idx = head_idx * max_seq * head_dim + pos * head_dim + offset_in_head;

    dst[dst_idx] = half(src[src_idx]);
}
