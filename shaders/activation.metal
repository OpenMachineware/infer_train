// Activation functions for GPU
#include <metal_stdlib>
using namespace metal;

constant float GELU_COEF_A     = 0.044715f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float GELU_QUICK_COEF = -1.702f;

// SiLU (Swish) activation: silu(x) = x / (1 + exp(-x))
kernel void kernel_silu_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    dst[gid] = src[gid] / (1.0f + exp(-src[gid]));
}

// SiLU with vec4 for better throughput
kernel void kernel_silu_f32_4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 x = src[gid];
    dst[gid] = x / (1.0f + exp(-x));
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

// GELU with vec4 for better throughput
kernel void kernel_gelu_f32_4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 x = src[gid];
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

// GELU-Quick with vec4 for better throughput
kernel void kernel_gelu_quick_f32_4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 x = src[gid];
    dst[gid] = x / (1.0f + exp(GELU_QUICK_COEF * x));
}
