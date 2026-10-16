// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Algorithm: Histogram-based radix selection + atomic output
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

static uint ggml_top_k_f2ui(float x) {
    uint y = as_type<uint>(x);
    if ((y & 0x80000000u) != 0u) {
        y ^= 0xFFFFFFFFu; // negative: flip all bits
    } else {
        y |= 0x80000000u; // positive: set sign bit
    }
    return y;
}

struct ggml_metal_kargs_top_k {
    int32_t ne00;   // number of columns (elements per row)
    int32_t ne01;   // number of rows
    int32_t ne02;
    int32_t ne03;
    uint64_t nb01;  // byte stride in src0
    uint64_t nb02;
    uint64_t nb03;
    int32_t top_k;  // k
};

kernel void top_k_f32_i32(
    constant ggml_metal_kargs_top_k& args [[buffer(0)]],
    device const char* src0 [[buffer(1)]],
    device int32_t* dst [[buffer(2)]],
    threadgroup atomic_uint* histogram [[threadgroup(0)]],
    threadgroup uint* sh_bucket [[threadgroup(1)]],
    threadgroup uint* sh_above [[threadgroup(2)]],
    threadgroup atomic_uint* out_count [[threadgroup(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]]
) {
    const uint ncols = args.ne00;
    const uint top_k = args.top_k;
    const uint i01 = tgpig[0];
    const uint i02 = tgpig[1];
    const uint i03 = tgpig[2];

    device const float* src0_row = (device const float*)(src0 + args.nb01 * i01 + args.nb02 * i02 + args.nb03 * i03);
    device int32_t* dst_row = dst + top_k * (i01 + args.ne01 * i02 + args.ne01 * args.ne02 * i03);

    const uint tid = tpitg.x;
    const uint ntg_x = ntg.x;

    uint prefix = 0;
    uint desired = top_k;

    // Radix selection: find the k-th largest value
    for (int shift = 24; shift >= 0; shift -= 8) {
        // Clear histogram
        for (uint i = tid; i < 256; i += ntg_x) {
            atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hi_mask = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << uint(shift + 8));
        const uint prefix_hi = prefix & hi_mask;

        // Build histogram
        for (uint i = tid; i < ncols; i += ntg_x) {
            const uint key = ggml_top_k_f2ui(src0_row[i]);
            if ((key & hi_mask) == prefix_hi) {
                atomic_fetch_add_explicit(&histogram[(key >> uint(shift)) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find bucket containing k-th element (top-down for descending order)
        if (tid == 0) {
            uint acc = 0;
            uint b = 0;
            for (int bb = 255; bb >= 0; --bb) {
                const uint c = atomic_load_explicit(&histogram[bb], memory_order_relaxed);
                if (acc + c >= desired) {
                    b = uint(bb);
                    break;
                }
                acc += c;
            }
            *sh_bucket = b;
            *sh_above = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        prefix |= *sh_bucket << uint(shift);
        desired -= *sh_above;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Output indices
    if (tid == 0) {
        atomic_store_explicit(out_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint threshold = prefix;

    // Output elements above threshold
    for (uint i = tid; i < ncols; i += ntg_x) {
        if (ggml_top_k_f2ui(src0_row[i]) > threshold) {
            const uint pos = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            dst_row[pos] = (int32_t)i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fill remaining with elements at threshold (ties)
    for (uint i = tid; i < ncols; i += ntg_x) {
        if (ggml_top_k_f2ui(src0_row[i]) == threshold) {
            const uint pos = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            if (pos < top_k) {
                dst_row[pos] = (int32_t)i;
            }
        }
    }
}
