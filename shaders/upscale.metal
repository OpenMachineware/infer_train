// Upscale kernels for GPU
// Used in super-resolution and image generation

#include <metal_stdlib>
using namespace metal;

// Nearest neighbor upscale
kernel void kernel_upscale_nearest_f32(
    device const float* input [[buffer(0)]],    // [batch, channels, in_h, in_w]
    device float* output [[buffer(1)]],         // [batch, channels, out_h, out_w]
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_h [[buffer(4)]],
    constant int& in_w [[buffer(5)]],
    constant int& scale_factor [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int out_h = in_h * scale_factor;
    const int out_w = in_w * scale_factor;

    const int batch = gid.z;
    const int ch = gid.y;
    const int out_idx = gid.x;

    if (batch >= batch_size || ch >= channels || out_idx >= out_h * out_w) return;

    const int out_h_idx = out_idx / out_w;
    const int out_w_idx = out_idx % out_w;

    const int in_h_idx = out_h_idx / scale_factor;
    const int in_w_idx = out_w_idx / scale_factor;

    const int input_idx = batch * channels * in_h * in_w +
                         ch * in_h * in_w +
                         in_h_idx * in_w + in_w_idx;

    const int output_idx = batch * channels * out_h * out_w +
                          ch * out_h * out_w +
                          out_h_idx * out_w + out_w_idx;

    output[output_idx] = input[input_idx];
}

// Bilinear upscale
kernel void kernel_upscale_bilinear_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_h [[buffer(4)]],
    constant int& in_w [[buffer(5)]],
    constant int& scale_factor [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int out_h = in_h * scale_factor;
    const int out_w = in_w * scale_factor;

    const int batch = gid.z;
    const int ch = gid.y;
    const int out_idx = gid.x;

    if (batch >= batch_size || ch >= channels || out_idx >= out_h * out_w) return;

    const int out_h_idx = out_idx / out_w;
    const int out_w_idx = out_idx % out_w;

    // Bilinear interpolation
    const float in_h_float = (float)out_h_idx / scale_factor;
    const float in_w_float = (float)out_w_idx / scale_factor;

    const int h0 = (int)floor(in_h_float);
    const int w0 = (int)floor(in_w_float);
    const int h1 = min(h0 + 1, in_h - 1);
    const int w1 = min(w0 + 1, in_w - 1);

    const float h_alpha = in_h_float - h0;
    const float w_alpha = in_w_float - w0;

    const int idx000 = batch * channels * in_h * in_w + ch * in_h * in_w + h0 * in_w + w0;
    const int idx010 = batch * channels * in_h * in_w + ch * in_h * in_w + h0 * in_w + w1;
    const int idx100 = batch * channels * in_h * in_w + ch * in_h * in_w + h1 * in_w + w0;
    const int idx110 = batch * channels * in_h * in_w + ch * in_h * in_w + h1 * in_w + w1;

    const float val = (1.0f - h_alpha) * (1.0f - w_alpha) * input[idx000] +
                     (1.0f - h_alpha) * w_alpha * input[idx010] +
                     h_alpha * (1.0f - w_alpha) * input[idx100] +
                     h_alpha * w_alpha * input[idx110];

    const int output_idx = batch * channels * out_h * out_w +
                          ch * out_h * out_w +
                          out_h_idx * out_w + out_w_idx;
    output[output_idx] = val;
}

// Bicubic upscale (simplified)
kernel void kernel_upscale_bicubic_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_h [[buffer(4)]],
    constant int& in_w [[buffer(5)]],
    constant int& scale_factor [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int out_h = in_h * scale_factor;
    const int out_w = in_w * scale_factor;

    const int batch = gid.z;
    const int ch = gid.y;
    const int out_idx = gid.x;

    if (batch >= batch_size || ch >= channels || out_idx >= out_h * out_w) return;

    const int out_h_idx = out_idx / out_w;
    const int out_w_idx = out_idx % out_w;

    const float in_h_float = (float)out_h_idx / scale_factor - 0.5f;
    const float in_w_float = (float)out_w_idx / scale_factor - 0.5f;

    const int h0 = max(0, (int)floor(in_h_float));
    const int w0 = max(0, (int)floor(in_w_float));
    const int h1 = min(h0 + 1, in_h - 1);
    const int w1 = min(w0 + 1, in_w - 1);

    // Simplified bicubic: just use bilinear for now
    const float h_alpha = in_h_float - h0 + 0.5f;
    const float w_alpha = in_w_float - w0 + 0.5f;

    const int idx000 = batch * channels * in_h * in_w + ch * in_h * in_w + h0 * in_w + w0;
    const int idx010 = batch * channels * in_h * in_w + ch * in_h * in_w + h0 * in_w + w1;
    const int idx100 = batch * channels * in_h * in_w + ch * in_h * in_w + h1 * in_w + w0;
    const int idx110 = batch * channels * in_h * in_w + ch * in_h * in_w + h1 * in_w + w1;

    const float val = (1.0f - h_alpha) * (1.0f - w_alpha) * input[idx000] +
                     (1.0f - h_alpha) * w_alpha * input[idx010] +
                     h_alpha * (1.0f - w_alpha) * input[idx100] +
                     h_alpha * w_alpha * input[idx110];

    const int output_idx = batch * channels * out_h * out_w +
                          ch * out_h * out_w +
                          out_h_idx * out_w + out_w_idx;
    output[output_idx] = val;
}
