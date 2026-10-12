// ARGSORT Metal kernel - Self-written bitonic sort implementation
// Algorithm: Bitonic sort with template for sort order
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

enum SortOrder {
    ASC = 0,
    DESC = 1
};

// Self-designed struct layout (matches benchmark but with different field names for clarity)
struct ArgsortParams {
    int64_t row_len;      // ne00: elements per row
    int64_t n_rows;       // ne01: number of rows
    int64_t batch_y;      // ne02: batch dimension y
    int64_t batch_z;      // ne03: batch dimension z
    int64_t row_stride;   // nb01: byte stride per row
    int64_t batch_y_stride; // nb02
    int64_t batch_z_stride; // nb03
    int64_t out_len;      // ne0: output size
    int64_t top_k;        // k value
};

// Self-written bitonic sort - different padding handling approach
template<SortOrder order>
kernel void argsort_sort(
    constant ArgsortParams& p [[buffer(0)]],
    device const char* src [[buffer(1)]],
    device int32_t* out [[buffer(2)]],
    threadgroup int32_t* idx [[threadgroup(0)]],
    ushort3 gid [[threadgroup_position_in_grid]],
    ushort3 tid_vec [[thread_position_in_threadgroup]],
    ushort3 group_size_vec [[threads_per_threadgroup]]
) {
    const ushort tid = tid_vec.x;
    const ushort group_size = group_size_vec.x;

    // Calculate row pointer
    const int row = gid.x % p.n_rows;
    const int b1 = gid.y;
    const int b2 = gid.z;

    device const float* data = (device const float*)(src + p.row_stride * row + p.batch_y_stride * b1 + p.batch_z_stride * b2);

    // Initialize indices - use thread index directly
    // For elements beyond row_len, initialize to invalid value (row_len)
    idx[tid] = (tid < p.row_len) ? tid : int(p.row_len);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort with different comparison logic
    for (uint step = 2; step <= group_size; step <<= 1) {
        for (uint dist = step >> 1; dist > 0; dist >>= 1) {
            const uint partner = tid ^ dist;

            // Only one thread performs swap per pair
            if (partner > tid) {
                const int a = idx[tid];
                const int b = idx[partner];

                // Check if both indices are valid
                const bool a_valid = (a < p.row_len);
                const bool b_valid = (b < p.row_len);

                bool do_swap = false;

                // Determine swap direction based on sort order and phase
                const bool ascending_phase = ((tid & step) == 0);

                if (!a_valid && b_valid) {
                    // Invalid index should move to end
                    do_swap = ascending_phase;
                } else if (a_valid && !b_valid) {
                    do_swap = !ascending_phase;
                } else if (a_valid && b_valid) {
                    // Both valid - compare values
                    const float va = data[a];
                    const float vb = data[b];

                    if (order == DESC) {
                        // Descending: larger values first
                        do_swap = ascending_phase ? (va < vb) : (va > vb);
                    } else {
                        // Ascending: smaller values first
                        do_swap = ascending_phase ? (va > vb) : (va < vb);
                    }
                }

                if (do_swap) {
                    idx[tid] = b;
                    idx[partner] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write top-k results
    // Calculate correct output index (must match llama.cpp layout)
    const int ib = gid.x / p.n_rows;  // batch index
    const int i0 = ib * p.top_k;       // base offset for this batch

    if (tid < p.top_k) {
        // Output layout: [batch][row][batch_y][batch_z][k]
        const int out_offset = i0 + p.out_len * row + p.out_len * p.n_rows * b1 + p.out_len * p.n_rows * p.batch_y * b2;
        out[out_offset + tid] = idx[tid];
    }
}

// Template instantiations with llama-compatible names
template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel void argsort_sort<ASC>(
    constant ArgsortParams& p, device const char* src, device int32_t* out,
    threadgroup int32_t* idx, ushort3 gid, ushort3 tid_vec, ushort3 group_size_vec);

template [[host_name("kernel_argsort_f32_i32_desc")]] kernel void argsort_sort<DESC>(
    constant ArgsortParams& p, device const char* src, device int32_t* out,
    threadgroup int32_t* idx, ushort3 gid, ushort3 tid_vec, ushort3 group_size_vec);

// Backward compatible wrapper with simpler interface
kernel void argsort_f32_i32(
    device const float* values [[buffer(0)]],
    device int32_t* indices [[buffer(1)]],
    constant int& num_elements [[buffer(2)]],
    constant int& top_k [[buffer(3)]],
    constant int& sort_order [[buffer(4)]],
    threadgroup int32_t* shmem [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    ushort tid [[thread_position_in_threadgroup]],
    ushort group_size [[threads_per_threadgroup]]
) {
    device const float* row_data = values + row_idx * num_elements;

    // Initialize indices with sentinel for out-of-bounds
    shmem[tid] = (tid < num_elements) ? tid : num_elements;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort
    for (uint k = 2; k <= group_size; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;

            if (other > tid) {
                const int ai = shmem[tid];
                const int bi = shmem[other];

                const bool a_in_bounds = (ai < num_elements);
                const bool b_in_bounds = (bi < num_elements);

                bool swap_needed = false;
                const bool first_half = ((tid & k) == 0);

                if (!a_in_bounds && b_in_bounds) {
                    swap_needed = first_half;
                } else if (a_in_bounds && !b_in_bounds) {
                    swap_needed = !first_half;
                } else if (a_in_bounds && b_in_bounds) {
                    const float av = row_data[ai];
                    const float bv = row_data[bi];

                    if (sort_order == 1) { // DESC
                        swap_needed = first_half ? (av < bv) : (av > bv);
                    } else { // ASC
                        swap_needed = first_half ? (av > bv) : (av < bv);
                    }
                }

                if (swap_needed) {
                    shmem[tid] = bi;
                    shmem[other] = ai;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Output
    if (tid < top_k) {
        indices[row_idx * top_k + tid] = shmem[tid];
    }
}
