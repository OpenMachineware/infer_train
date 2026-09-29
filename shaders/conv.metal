// Convolution kernels for GPU
// Used in vision encoders (CLIP, ViT, LLaVA)

#include <metal_stdlib>
using namespace metal;

// 2D Convolution (forward)
kernel void kernel_conv2d_f32(
    device const float* input [[buffer(0)]],    // [batch, in_channels, in_h, in_w]
    device const float* weight [[buffer(1)]],   // [out_channels, in_channels, kH, kW]
    device const float* bias [[buffer(2)]],     // [out_channels]
    device float* output [[buffer(3)]],         // [batch, out_channels, out_h, out_w]
    constant int& batch_size [[buffer(4)]],
    constant int& in_channels [[buffer(5)]],
    constant int& out_channels [[buffer(6)]],
    constant int& in_h [[buffer(7)]],
    constant int& in_w [[buffer(8)]],
    constant int& kH [[buffer(9)]],
    constant int& kW [[buffer(10)]],
    constant int& stride_h [[buffer(11)]],
    constant int& stride_w [[buffer(12)]],
    constant int& pad_h [[buffer(13)]],
    constant int& pad_w [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int out_h_size = (in_h + 2 * pad_h - kH) / stride_h + 1;
    const int out_w_size = (in_w + 2 * pad_w - kW) / stride_w + 1;

    const int batch = gid.z;
    const int out_ch = gid.y;
    const int out_idx = gid.x;

    if (batch >= batch_size || out_ch >= out_channels || out_idx >= out_h_size * out_w_size) return;

    const int out_h_idx = out_idx / out_w_size;
    const int out_w_idx = out_idx % out_w_size;

    float sum = bias[out_ch];

    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                const int in_h_idx = out_h_idx * stride_h - pad_h + kh;
                const int in_w_idx = out_w_idx * stride_w - pad_w + kw;

                if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                    const int input_idx = batch * in_channels * in_h * in_w +
                                         in_ch * in_h * in_w +
                                         in_h_idx * in_w + in_w_idx;

                    const int weight_idx = out_ch * in_channels * kH * kW +
                                          in_ch * kH * kW +
                                          kh * kW + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    const int output_idx = batch * out_channels * out_h_size * out_w_size +
                          out_ch * out_h_size * out_w_size +
                          out_h_idx * out_w_size + out_w_idx;
    output[output_idx] = sum;
}

// Depthwise Conv2D (for MobileNet-style models)
kernel void kernel_conv2d_dw_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& channels [[buffer(5)]],
    constant int& in_h [[buffer(6)]],
    constant int& in_w [[buffer(7)]],
    constant int& kH [[buffer(8)]],
    constant int& kW [[buffer(9)]],
    constant int& stride_h [[buffer(10)]],
    constant int& stride_w [[buffer(11)]],
    constant int& pad_h [[buffer(12)]],
    constant int& pad_w [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int out_h_size = (in_h + 2 * pad_h - kH) / stride_h + 1;
    const int out_w_size = (in_w + 2 * pad_w - kW) / stride_w + 1;

    const int batch = gid.z;
    const int ch = gid.y;
    const int out_idx = gid.x;

    if (batch >= batch_size || ch >= channels || out_idx >= out_h_size * out_w_size) return;

    const int out_h_idx = out_idx / out_w_size;
    const int out_w_idx = out_idx % out_w_size;

    float sum = bias[ch];

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            const int in_h_idx = out_h_idx * stride_h - pad_h + kh;
            const int in_w_idx = out_w_idx * stride_w - pad_w + kw;

            if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                const int input_idx = batch * channels * in_h * in_w +
                                     ch * in_h * in_w +
                                     in_h_idx * in_w + in_w_idx;

                const int weight_idx = ch * kH * kW + kh * kW + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    const int output_idx = batch * channels * out_h_size * out_w_size +
                          ch * out_h_size * out_w_size +
                          out_h_idx * out_w_size + out_w_idx;
    output[output_idx] = sum;
}
