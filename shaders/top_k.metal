// TOP_K Metal kernel - Self-written implementation
// Algorithm: Histogram-based selection (radix selection)
// 1. Convert float to sortable uint
// 2. Find k-th largest value via histogram
// 3. Output indices of top-k elements
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Convert float to sortable unsigned integer
// Maps -inf to 0x00000000, +inf to 0xFFFFFFFF
// Preserves ordering: f1 < f2  =>  uint(f1) < uint(f2)
static uint float_to_sortable_key(float f) {
    uint bits = as_type<uint>(f);
    // If negative, flip all bits (makes larger magnitudes smaller)
    // If positive, set sign bit (makes all positives larger than negatives)
    if ((bits & 0x80000000u) != 0u) {
        bits ^= 0xFFFFFFFFu;
    } else {
        bits |= 0x80000000u;
    }
    return bits;
}

kernel void top_k_f32_i32(
    device const float* values [[buffer(0)]],
    device int32_t* indices [[buffer(1)]],
    constant int& num_elements [[buffer(2)]],
    constant int& k [[buffer(3)]],
    threadgroup atomic_uint* histogram [[threadgroup(0)]],
    threadgroup uint* selected_bucket [[threadgroup(1)]],
    threadgroup uint* count_above [[threadgroup(2)]],
    threadgroup atomic_uint* output_counter [[threadgroup(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Phase 1: Radix selection to find threshold
    uint threshold_key = 0;
    uint remaining = k;

    // Process 8 bits at a time (256 buckets)
    for (int shift = 24; shift >= 0; shift -= 8) {
        // Clear histogram
        for (uint i = tid; i < 256; i += group_size) {
            atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Build histogram for current byte
        const uint hi_mask = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << uint(shift + 8));
        const uint prefix_hi = threshold_key & hi_mask;

        for (uint i = tid; i < uint(num_elements); i += group_size) {
            const uint key = float_to_sortable_key(values[i]);
            if ((key & hi_mask) == prefix_hi) {
                const uint bucket = (key >> uint(shift)) & 0xFFu;
                atomic_fetch_add_explicit(&histogram[bucket], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find bucket containing k-th element (top-down scan for descending order)
        if (tid == 0) {
            uint accumulated = 0;
            uint bucket = 0;
            for (int b = 255; b >= 0; b--) {
                const uint cnt = atomic_load_explicit(&histogram[b], memory_order_relaxed);
                if (accumulated + cnt >= remaining) {
                    bucket = uint(b);
                    break;
                }
                accumulated += cnt;
            }
            *selected_bucket = bucket;
            *count_above = accumulated;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threshold_key |= (*selected_bucket) << uint(shift);
        remaining -= *count_above;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: Output indices
    if (tid == 0) {
        atomic_store_explicit(output_counter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First, output all elements strictly above threshold
    for (uint i = tid; i < uint(num_elements); i += group_size) {
        if (float_to_sortable_key(values[i]) > threshold_key) {
            const uint pos = atomic_fetch_add_explicit(output_counter, 1u, memory_order_relaxed);
            if (pos < uint(k)) {
                indices[pos] = int32_t(i);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Then, fill remaining slots from elements equal to threshold
    for (uint i = tid; i < uint(num_elements); i += group_size) {
        if (float_to_sortable_key(values[i]) == threshold_key) {
            const uint pos = atomic_fetch_add_explicit(output_counter, 1u, memory_order_relaxed);
            if (pos < uint(k)) {
                indices[pos] = int32_t(i);
            }
        }
    }
}