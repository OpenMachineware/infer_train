// Reduce kernels for GPU
// Sum, Max, Min, etc.

#include <metal_stdlib>
using namespace metal;

// Reduce sum
kernel void kernel_reduce_sum_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]]
) {
    // Load data to shared memory
    float sum = 0.0f;
    for (int i = tid; i < n; i += ntg) {
        sum += src[i];
    }

    shmem[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce in shared memory
    for (uint stride = ntg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        dst[tgid] = shmem[0];
    }
}

// Reduce max
kernel void kernel_reduce_max_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& n [[buffer(2)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg [[threads_per_threadgroup]]
) {
    float max_val = -FLT_MAX;
    for (int i = tid; i < n; i += ntg) {
        max_val = max(max_val, src[i]);
    }

    shmem[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] = max(shmem[tid], shmem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        dst[tgid] = shmem[0];
    }
}

// Row-wise reduce sum
kernel void kernel_reduce_sum_f32_row(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& n_rows [[buffer(2)]],
    constant int& n_cols [[buffer(3)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]
) {
    const int row = gid.x;
    if (row >= n_rows) return;

    device const float* row_data = src + row * n_cols;

    float sum = 0.0f;
    for (int i = tid; i < n_cols; i += ntg) {
        sum += row_data[i];
    }

    shmem[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        dst[row] = shmem[0];
    }
}
