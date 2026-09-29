// Pooling kernels for GPU
// Used in vision models for downsampling

#include <metal_stdlib>
using namespace metal;

// Max Pooling 2D
kernel void kernel_pool2d_max_f32(
    device const float* input [[buffer(0)]],    // [batch, channels, in_h, in_w]
    device float* output [[buffer(1)]],         // [batch, channels, out_h, out_w]
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_h [[buffer(4)]],
    constant int& in_w [[buffer(5)]],
    constant int& kH [[buffer(6)]],
    constant int& kW [[buffer(7)]],
    constant int& stride_h [[buffer(8)]],
    constant int& stride_w [[buffer(9)]],
    constant int& pad_h [[buffer(10)]],
    constant int& pad_w [[buffer(11)]],
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

    float max_val = -FLT_MAX;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            const int in_h_idx = out_h_idx * stride_h - pad_h + kh;
            const int in_w_idx = out_w_idx * stride_w - pad_w + kw;

            if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                const int input_idx = batch * channels * in_h * in_w +
                                     ch * in_h * in_w +
                                     in_h_idx * in_w + in_w_idx;
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    const int output_idx = batch * channels * out_h_size * out_w_size +
                          ch * out_h_size * out_w_size +
                          out_h_idx * out_w_size + out_w_idx;
    output[output_idx] = max_val;
}

// Average Pooling 2D
kernel void kernel_pool2d_avg_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_h [[buffer(4)]],
    constant int& in_w [[buffer(5)]],
    constant int& kH [[buffer(6)]],
    constant int& kW [[buffer(7)]],
    constant int& stride_h [[buffer(8)]],
    constant int& stride_w [[buffer(9)]],
    constant int& pad_h [[buffer(10)]],
    constant int& pad_w [[buffer(11)]],
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

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            const int in_h_idx = out_h_idx * stride_h - pad_h + kh;
            const int in_w_idx = out_w_idx * stride_w - pad_w + kw;

            if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
                const int input_idx = batch * channels * in_h * in_w +
                                     ch * in_h * in_w +
                                     in_h_idx * in_w + in_w_idx;
                sum += input[input_idx];
                count++;
            }
        }
    }

    const int output_idx = batch * channels * out_h_size * out_w_size +
                          ch * out_h_size * out_w_size +
                          out_h_idx * out_w_size + out_w_idx;
    output[output_idx] = count > 0 ? sum / count : 0.0f;
}

// Global Average Pooling
kernel void kernel_pool2d_global_avg_f32(
    device const float* input [[buffer(0)]],    // [batch, channels, H, W]
    device float* output [[buffer(1)]],         // [batch, channels]
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_h [[buffer(4)]],
    constant int& in_w [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int batch = gid.y;
    const int ch = gid.x;

    if (batch >= batch_size || ch >= channels) return;

    float sum = 0.0f;
    const int total = in_h * in_w;

    for (int h = 0; h < in_h; h++) {
        for (int w = 0; w < in_w; w++) {
            const int input_idx = batch * channels * in_h * in_w +
                                 ch * in_h * in_w +
                                 h * in_w + w;
            sum += input[input_idx];
        }
    }

    const int output_idx = batch * channels + ch;
    output[output_idx] = sum / total;
}
