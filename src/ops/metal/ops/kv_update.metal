// KV Cache Update Kernel: F32 -> F16 with strided write
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#include <metal_stdlib>
using namespace metal;

// Update K cache: Copy F16 to strided position (no conversion)
// src: F16 buffer [n_kv_heads, head_dim]
// dst: F16 KV cache [n_kv_heads, max_seq, head_dim]
// Used with RoPE F16 output to avoid redundant F32->F16 conversion
kernel void kv_update_f16_to_f16(
    device const half* src [[buffer(0)]],
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

    // Read F16 directly
    half val = src[idx];

    // Write to strided position
    uint dst_idx = h * max_seq * head_dim + pos * head_dim + d;
    dst[dst_idx] = val;
}

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

// Batched KV cache update for prefill
// Update multiple positions at once from batched input
// src: [batch, n_kv_heads, head_dim] F16
// dst: [n_kv_heads, max_seq, head_dim] F16 KV cache
// start_pos: starting position in KV cache
// batch: number of positions to update
kernel void kv_update_batched_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& start_pos [[buffer(2)]],
    constant uint& max_seq [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    uint3 thread_id [[thread_position_in_grid]])
{
    // thread_id.x = element index within a head
    // thread_id.y = head index
    // thread_id.z = batch position
    uint d = thread_id.x;
    uint h = thread_id.y;
    uint b = thread_id.z;

    if (d >= head_dim || h >= n_kv_heads || b >= batch) {
        return;
    }

    // Source index: [batch, n_kv_heads, head_dim]
    uint src_idx = b * n_kv_heads * head_dim + h * head_dim + d;

    // Destination index: [n_kv_heads, max_seq, head_dim]
    uint pos = start_pos + b;
    uint dst_idx = h * max_seq * head_dim + pos * head_dim + d;

    dst[dst_idx] = src[src_idx];
}

// Batched K cache update with strided input
// src: [batch, n_kv_heads, head_dim] with src_stride elements per row
// Used when input is not contiguous (e.g., from GEMM output)
kernel void kv_update_batched_f16_strided(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& start_pos [[buffer(2)]],
    constant uint& max_seq [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& src_stride [[buffer(7)]],  // stride in elements
    uint3 thread_id [[thread_position_in_grid]])
{
    uint d = thread_id.x;
    uint h = thread_id.y;
    uint b = thread_id.z;

    if (d >= head_dim || h >= n_kv_heads || b >= batch) {
        return;
    }

    // Source with stride
    uint src_idx = b * src_stride + h * head_dim + d;

    uint pos = start_pos + b;
    uint dst_idx = h * max_seq * head_dim + pos * head_dim + d;

    dst[dst_idx] = src[src_idx];
}
