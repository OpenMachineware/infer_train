// Binary Broadcast kernels for GPU
// Element-wise operations with broadcasting support

#include <metal_stdlib>
using namespace metal;

// Broadcast types
// 0: no broadcast
// 1: broadcast src1 across rows
// 2: broadcast src0 across rows

// Add with broadcasting
kernel void kernel_add_f32_bcast(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne01 [[buffer(4)]],
    constant int& ne02 [[buffer(5)]],
    constant int& ne10 [[buffer(6)]],
    constant int& ne11 [[buffer(7)]],
    constant int& bcast_type [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int i0 = gid.x;
    const int i1 = gid.y;
    const int i2 = gid.z;

    if (i0 >= ne00 || i1 >= ne01 || i2 >= ne02) return;

    const int idx0 = i2 * ne01 * ne00 + i1 * ne00 + i0;
    const float v0 = src0[idx0];

    float v1;
    if (bcast_type == 0) {
        // No broadcast
        const int idx1 = i2 * ne11 * ne10 + i1 * ne10 + i0;
        v1 = src1[idx1];
    } else if (bcast_type == 1) {
        // Broadcast src1 across rows (ne11 == 1)
        const int idx1 = i2 * ne10 + i0;
        v1 = src1[idx1];
    } else {
        v1 = src1[0];  // Scalar
    }

    dst[idx0] = v0 + v1;
}

// Sub with broadcasting
kernel void kernel_sub_f32_bcast(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne01 [[buffer(4)]],
    constant int& ne02 [[buffer(5)]],
    constant int& ne10 [[buffer(6)]],
    constant int& ne11 [[buffer(7)]],
    constant int& bcast_type [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int i0 = gid.x;
    const int i1 = gid.y;
    const int i2 = gid.z;

    if (i0 >= ne00 || i1 >= ne01 || i2 >= ne02) return;

    const int idx0 = i2 * ne01 * ne00 + i1 * ne00 + i0;
    const float v0 = src0[idx0];

    float v1;
    if (bcast_type == 0) {
        const int idx1 = i2 * ne11 * ne10 + i1 * ne10 + i0;
        v1 = src1[idx1];
    } else if (bcast_type == 1) {
        const int idx1 = i2 * ne10 + i0;
        v1 = src1[idx1];
    } else {
        v1 = src1[0];
    }

    dst[idx0] = v0 - v1;
}

// Mul with broadcasting
kernel void kernel_mul_f32_bcast(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne01 [[buffer(4)]],
    constant int& ne02 [[buffer(5)]],
    constant int& ne10 [[buffer(6)]],
    constant int& ne11 [[buffer(7)]],
    constant int& bcast_type [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int i0 = gid.x;
    const int i1 = gid.y;
    const int i2 = gid.z;

    if (i0 >= ne00 || i1 >= ne01 || i2 >= ne02) return;

    const int idx0 = i2 * ne01 * ne00 + i1 * ne00 + i0;
    const float v0 = src0[idx0];

    float v1;
    if (bcast_type == 0) {
        const int idx1 = i2 * ne11 * ne10 + i1 * ne10 + i0;
        v1 = src1[idx1];
    } else if (bcast_type == 1) {
        const int idx1 = i2 * ne10 + i0;
        v1 = src1[idx1];
    } else {
        v1 = src1[0];
    }

    dst[idx0] = v0 * v1;
}

// Div with broadcasting
kernel void kernel_div_f32_bcast(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int& ne00 [[buffer(3)]],
    constant int& ne01 [[buffer(4)]],
    constant int& ne02 [[buffer(5)]],
    constant int& ne10 [[buffer(6)]],
    constant int& ne11 [[buffer(7)]],
    constant int& bcast_type [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int i0 = gid.x;
    const int i1 = gid.y;
    const int i2 = gid.z;

    if (i0 >= ne00 || i1 >= ne01 || i2 >= ne02) return;

    const int idx0 = i2 * ne01 * ne00 + i1 * ne00 + i0;
    const float v0 = src0[idx0];

    float v1;
    if (bcast_type == 0) {
        const int idx1 = i2 * ne11 * ne10 + i1 * ne10 + i0;
        v1 = src1[idx1];
    } else if (bcast_type == 1) {
        const int idx1 = i2 * ne10 + i0;
        v1 = src1[idx1];
    } else {
        v1 = src1[0];
    }

    dst[idx0] = v0 / v1;
}
