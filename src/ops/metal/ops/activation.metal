// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Activation functions and element-wise operations for GPU
#include <metal_stdlib>
using namespace metal;

constant float GELU_COEF_A     = 0.044715f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float GELU_QUICK_COEF = -1.702f;

// ========== Element-wise operations ==========

// Add two tensors: dst = src0 + src1
kernel void kernel_add_f32(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src0[gid] + src1[gid];
}

// Mul two tensors: dst = src0 * src1
kernel void kernel_mul_f32(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src0[gid] * src1[gid];
}

// ========== dispatch_threads kernels (for small sizes) ==========
// Uses thread_position_in_grid - simpler but slightly slower for large sizes

// SiLU (Swish) activation: silu(x) = x / (1 + exp(-x))
kernel void kernel_silu_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src[gid] / (1.0f + exp(-src[gid]));
}

// GELU activation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * x * (1 + 0.044715 * x^2)))
kernel void kernel_gelu_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = src[gid];
    dst[gid] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

// GELU-Quick: gelu_quick(x) = x * sigmoid(-1.702 * x)
kernel void kernel_gelu_quick_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = src[gid];
    dst[gid] = x / (1.0f + exp(GELU_QUICK_COEF * x));
}

// ========== float4 kernels with dispatch_threads ==========
// For hybrid approach: float4 processing with simpler dispatch

kernel void kernel_silu_f32_4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 x = src[gid];
    dst[gid] = x / (1.0f + exp(-x));
}

kernel void kernel_gelu_f32_4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 x = src[gid];
    dst[gid] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

kernel void kernel_gelu_quick_f32_4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 x = src[gid];
    dst[gid] = x / (1.0f + exp(GELU_QUICK_COEF * x));
}

// ========== dispatch_thread_groups kernels (for large sizes) ==========
// Uses threadgroup_position_in_grid + thread_position_in_threadgroup
// Faster for large sizes due to more efficient GPU scheduling

kernel void kernel_silu_f32_tg(
    constant int & ne00,
    device const float* src0,
    device float* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = x / (1.0f + exp(-x));
}

kernel void kernel_gelu_f32_tg(
    constant int & ne00,
    device const float* src0,
    device float* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

kernel void kernel_gelu_quick_f32_tg(
    constant int & ne00,
    device const float* src0,
    device float* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    const float x = src0[i0];
    dst[i0] = x / (1.0f + exp(GELU_QUICK_COEF * x));
}

// ========== float4 kernels (4 elements per thread) ==========
// Much faster for large sizes due to better memory throughput

kernel void kernel_silu_f32_4_tg(
    constant int & ne00,
    device const float4* src0,
    device float4* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 * 4 >= ne00) return;
    float4 x = src0[i0];
    dst[i0] = x / (1.0f + exp(-x));
}

kernel void kernel_gelu_f32_4_tg(
    constant int & ne00,
    device const float4* src0,
    device float4* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 * 4 >= ne00) return;
    float4 x = src0[i0];
    dst[i0] = 0.5f * x * (1.0f + precise::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

kernel void kernel_gelu_quick_f32_4_tg(
    constant int & ne00,
    device const float4* src0,
    device float4* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 * 4 >= ne00) return;
    float4 x = src0[i0];
    dst[i0] = x / (1.0f + exp(GELU_QUICK_COEF * x));
}

// ========== Fused kernels ==========

// Fused SwiGLU: output = SiLU(gate) * up
// Combines SiLU activation and element-wise multiplication
// Reduces dispatch overhead and intermediate memory writes
kernel void kernel_fused_swiglu_f32(
    device const float* gate [[buffer(0)]],   // Gate projection output
    device const float* up [[buffer(1)]],     // Up projection output (SiLU applied to this)
    device float* dst [[buffer(2)]],          // Output: gate * SiLU(up)
    constant int & ne00 [[buffer(3)]],        // Size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)ne00) return;
    float g = gate[gid];
    float u = up[gid];
    float silu_u = u / (1.0f + exp(-u));  // SiLU on up
    dst[gid] = g * silu_u;               // gate * silu(up)
}

// Fused SwiGLU with float4 for better memory throughput
kernel void kernel_fused_swiglu_f32_4(
    device const float4* gate [[buffer(0)]],
    device const float4* up [[buffer(1)]],
    device float4* dst [[buffer(2)]],
    constant int & ne00 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid * 4 >= (uint)ne00) return;
    float4 g = gate[gid];
    float4 u = up[gid];
    float4 silu_u = u / (1.0f + exp(-u));  // SiLU on up
    dst[gid] = g * silu_u;                // gate * silu(up)
}

// Fused SwiGLU with threadgroups for large sizes
kernel void kernel_fused_swiglu_f32_tg(
    constant int & ne00,
    device const float* gate,
    device const float* up,
    device float* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 >= ne00) return;
    float g = gate[i0];
    float u = up[i0];
    float silu_u = u / (1.0f + exp(-u));  // SiLU on up
    dst[i0] = g * silu_u;                // gate * silu(up)
}

// Fused SwiGLU with float4 and threadgroups
kernel void kernel_fused_swiglu_f32_4_tg(
    constant int & ne00,
    device const float4* gate,
    device const float4* up,
    device float4* dst,
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]])
{
    const int i0 = tgpig.x * ntg.x + tpitg.x;
    if (i0 * 4 >= ne00) return;
    float4 g = gate[i0];
    float4 u = up[i0];
    float4 silu_u = u / (1.0f + exp(-u));  // SiLU on up
    dst[i0] = g * silu_u;                // gate * silu(up)
}

// ========== Batched kernels for prefill ==========

// Batched Add: dst[i] = src0[i] + src1[i]
// Process [batch, size] tensor
kernel void kernel_add_batched_f32(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& total_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_size) return;
    dst[gid] = src0[gid] + src1[gid];
}

// Batched Mul: dst[i] = src0[i] * src1[i]
kernel void kernel_mul_batched_f32(
    device const float* src0 [[buffer(0)]],
    device const float* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint& total_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_size) return;
    dst[gid] = src0[gid] * src1[gid];
}

// F32 to F16 conversion
kernel void kernel_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& total_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_size) return;
    dst[gid] = half(src[gid]);
}

// F16 to F32 conversion
kernel void kernel_f16_to_f32(
    device const half* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& total_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_size) return;
    dst[gid] = float(src[gid]);
}

// Batched F32 to F16 (with stride support for non-contiguous input)
// src_stride: stride in elements for source rows
// dst_stride: stride in elements for destination rows
// rows: number of rows (batch size)
// cols: elements per row
kernel void kernel_f32_to_f16_strided(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& src_stride [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row >= rows || col >= cols) return;

    uint src_idx = row * src_stride + col;
    uint dst_idx = row * dst_stride + col;

    dst[dst_idx] = half(src[src_idx]);
}

// ========== Fused Operations ==========

// Fused SiLU and multiplication: output = SiLU(gate) * up
// Combines two operations into single kernel for better memory locality
kernel void kernel_fused_swi_glu(
    device const float * gate [[buffer(0)]],
    device const float * up [[buffer(1)]],
    device float * output [[buffer(2)]],
    constant uint32_t & n [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int idx = tgpig.x * 32 + tiisg;
    if (idx >= (int)n) return;

    // SiLU activation: x * sigmoid(x)
    float g = gate[idx];
    float silu = g / (1.0f + exp(-g));

    // Element-wise multiply
    output[idx] = silu * up[idx];
}
