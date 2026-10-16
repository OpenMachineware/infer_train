// KV Cache Update Kernel: F32 -> F16 with strided write
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <metal_stdlib>
using namespace metal;

// Update K cache: Convert F32 to F16 and write to strided position
// src: F32 buffer [n_kv_heads, head_dim]
// dst: F16 KV cache [n_kv_heads, max_seq, head_dim]
// pos: current sequence position
// max_seq: maximum sequence length
// n_kv_heads: number of KV heads
// head_dim: head dimension
kernel void kv_update_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& pos [[buffer(2)]],
    constant uint& max_seq [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 thread_id [[thread_position_in_grid]])
{
    // Each thread processes one element
    // Linear index: thread_id.x = element index across all heads
    uint idx = thread_id.x;
    uint total_elements = n_kv_heads * head_dim;

    if (idx >= total_elements) {
        return;
    }

    // Compute head and offset within head
    uint h = idx / head_dim;
    uint d = idx % head_dim;

    // Read from source: [n_kv_heads, head_dim]
    float val = src[idx];

    // Write to destination: [n_kv_heads, max_seq, head_dim]
    // Position: h * max_seq * head_dim + pos * head_dim + d
    uint dst_idx = h * max_seq * head_dim + pos * head_dim + d;

    dst[dst_idx] = half(val);
}

// Update V cache (same logic)
kernel void kv_update_v_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& pos [[buffer(2)]],
    constant uint& max_seq [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 thread_id [[thread_position_in_grid]])
{
    uint idx = thread_id.x;
    uint total_elements = n_kv_heads * head_dim;

    if (idx >= total_elements) {
        return;
    }

    uint h = idx / head_dim;
    uint d = idx % head_dim;

    float val = src[idx];
    uint dst_idx = h * max_seq * head_dim + pos * head_dim + d;

    dst[dst_idx] = half(val);
}
