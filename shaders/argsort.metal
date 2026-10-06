// ARGSORT Metal kernel - Self-written implementation
// Algorithm: Bitonic sort with parallel execution
// Each threadgroup sorts one row of data
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

enum SortOrder : int {
    Ascending = 0,
    Descending = 1
};

// Bitonic sort - swaps elements to create sorted sequences
kernel void argsort_f32_i32(
    device const float* values [[buffer(0)]],
    device int32_t* indices [[buffer(1)]],
    constant int& num_elements [[buffer(2)]],
    constant int& top_k [[buffer(3)]],
    constant int& sort_order [[buffer(4)]],
    threadgroup int32_t* shmem_indices [[threadgroup(0)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one row
    device const float* row_values = values + row_idx * num_elements;

    // Initialize indices
    const int start = tid;
    if (start < num_elements) {
        shmem_indices[start] = start;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort: create bitonic sequences and then merge them
    for (int k = 2; k <= int(group_size); k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int idx_a = start;
            const int idx_b = start ^ j;

            if (idx_b > idx_a && idx_a < num_elements && idx_b < num_elements) {
                // Determine if we should swap based on sort order and phase
                const bool in_k_segment = (idx_a & k) == 0;

                const int32_t a_idx = shmem_indices[idx_a];
                const int32_t b_idx = shmem_indices[idx_b];

                const float val_a = row_values[a_idx];
                const float val_b = row_values[b_idx];

                bool should_swap;
                if (sort_order == SortOrder::Descending) {
                    // Descending: larger values first
                    if (in_k_segment) {
                        should_swap = (val_a < val_b);
                    } else {
                        should_swap = (val_a > val_b);
                    }
                } else {
                    // Ascending: smaller values first
                    if (in_k_segment) {
                        should_swap = (val_a > val_b);
                    } else {
                        should_swap = (val_a < val_b);
                    }
                }

                if (should_swap) {
                    shmem_indices[idx_a] = b_idx;
                    shmem_indices[idx_b] = a_idx;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Copy top-k results to output
    if (tid < uint(top_k)) {
        device int32_t* output_row = indices + row_idx * top_k;
        output_row[tid] = shmem_indices[tid];
    }
}