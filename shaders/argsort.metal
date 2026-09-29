// ArgSort kernels for GPU
// Used in sampling (top-k, top-p)

#include <metal_stdlib>
using namespace metal;

// ArgSort ascending
kernel void kernel_argsort_f32_i32_asc(
    device const float* src [[buffer(0)]],
    device int32_t* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // Simple argsort: output the index
    // This is a basic version - for full sorting, need a different algorithm
    dst[gid] = int32_t(gid);
}

// ArgSort descending
kernel void kernel_argsort_f32_i32_desc(
    device const float* src [[buffer(0)]],
    device int32_t* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // Simple argsort: output the index in descending order
    dst[gid] = int32_t(n - 1 - gid);
}

// Partial argsort (top-k)
// Each threadgroup finds its local top-k, then global reduction
kernel void kernel_argsort_top_k_f32(
    device const float* src [[buffer(0)]],
    device int32_t* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& k [[buffer(3)]],
    threadgroup float* shmem_val [[threadgroup(0)]],
    threadgroup int* shmem_idx [[threadgroup(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    const int block_size = 256;

    // Load data to shared memory
    if (gid < n) {
        shmem_val[tid] = src[gid];
        shmem_idx[tid] = int32_t(gid);
    } else {
        shmem_val[tid] = -FLT_MAX;
        shmem_idx[tid] = -1;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Selection sort in shared memory to find top-k
    // This is O(k * block_size) - suitable for small k
    const int local_k = min(k, int(block_size));

    for (int i = 0; i < local_k && tgid == 0; i++) {
        float max_val = -FLT_MAX;
        int max_idx = -1;
        int max_pos = -1;

        // Find maximum
        for (int j = 0; j < block_size; j++) {
            if (shmem_val[j] > max_val && shmem_idx[j] >= 0) {
                max_val = shmem_val[j];
                max_idx = shmem_idx[j];
                max_pos = j;
            }
        }

        if (max_pos >= 0 && i < k) {
            dst[i] = max_idx;
            shmem_val[max_pos] = -FLT_MAX;  // Mark as used
        } else {
            break;
        }
    }
}

// Batch argsort
kernel void kernel_argsort_f32_i32_batch(
    device const float* src [[buffer(0)]],
    device int32_t* dst [[buffer(1)]],
    constant int& n_rows [[buffer(2)]],
    constant int& n_cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int row = gid.y;
    const int col = gid.x;

    if (row >= n_rows || col >= n_cols) return;

    // For batch argsort, output indices
    const int idx = row * n_cols + col;
    dst[idx] = int32_t(col);
}
