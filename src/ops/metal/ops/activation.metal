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
    device const float* up [[buffer(1)]],     // Up projection output
    device float* dst [[buffer(2)]],          // Output: SiLU(gate) * up
    constant int & ne00 [[buffer(3)]],        // Size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)ne00) return;
    float g = gate[gid];
    float u = up[gid];
    float silu_g = g / (1.0f + exp(-g));  // SiLU activation
    dst[gid] = silu_g * u;               // Element-wise multiplication
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
    float4 silu_g = g / (1.0f + exp(-g));
    dst[gid] = silu_g * u;
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
    float silu_g = g / (1.0f + exp(-g));
    dst[i0] = silu_g * u;
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
    float4 silu_g = g / (1.0f + exp(-g));
    dst[i0] = silu_g * u;
}
