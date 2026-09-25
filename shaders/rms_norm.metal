// RMSNorm Metal kernel
// Direct copy from llama.cpp's kernel_rms_norm_fuse_impl

#include <metal_stdlib>
using namespace metal;

// Simple RMSNorm: y = x * scale * w, where scale = 1/sqrt(mean(x^2) + eps)
// Each threadgroup processes one row
kernel void kernel_rms_norm_f32(
        device const float * src0,
        device const float * src1,  // weight
        device       float * dst,
        constant int32_t & ne00,    // hidden dimension
        constant float   & eps,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {

    // Initialize shared memory for reduction
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Get row pointer
    device const float * x = (device const float *) (src0 + tgpig * ne00);
    device const float * w = src1;
    device       float * y = (device       float *) (dst  + tgpig * ne00);

    float sumf = 0.0f;

    // Parallel sum of squares - each thread processes multiple elements
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }

    // Reduce within SIMD group
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First thread of each SIMD group writes to shared memory
    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SIMD groups
    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    // Compute scale
    const float mean  = sumf / ne00;
    const float scale = 1.0f / sqrt(mean + eps);

    // Apply normalization and weight
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] * scale * w[i00];
    }
}
