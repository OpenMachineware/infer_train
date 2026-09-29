// WKV (RWKV) kernels for GPU
// Based on RWKV architecture: Receptance Weighted Key Value

#include <metal_stdlib>
using namespace metal;

// WKV time mixing: core operation of RWKV
// Formula: wkv_t = (a * wkv_{t-1} + b * k_t * v_t) / (a * wkv_{t-1} + b * k_t)
// Where: a = exp(-exp(w)), b = exp(u)

kernel void kernel_wkv_f32(
    device const float* k [[buffer(0)]],      // key: [seq_len, d_model]
    device const float* v [[buffer(1)]],      // value: [seq_len, d_model]
    device const float* r [[buffer(2)]],      // receptance: [seq_len, d_model]
    device const float* w [[buffer(3)]],      // time decay: [d_model]
    device const float* u [[buffer(4)]],      // time bonus: [d_model]
    device float* y [[buffer(5)]],             // output: [seq_len, d_model]
    constant int& seq_len [[buffer(6)]],
    constant int& d_model [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d_model) return;

    // Initialize state
    float wkv_n = 0.0f;  // numerator
    float wkv_d = 1.0f;  // denominator

    const float w_val = exp(-exp(w[gid]));
    const float u_val = exp(u[gid]);

    // Process sequence
    for (int t = 0; t < seq_len; t++) {
        const float k_t = k[t * d_model + gid];
        const float v_t = v[t * d_model + gid];
        const float r_t = r[t * d_model + gid];

        // Compute WKV
        const float wkv = (wkv_n + u_val * k_t * v_t) / (wkv_d + u_val * k_t);
        y[t * d_model + gid] = r_t * wkv;

        // Update state
        wkv_n = w_val * wkv_n + w_val * k_t * v_t;
        wkv_d = w_val * wkv_d + w_val * k_t;
    }
}

// Batched WKV for multiple sequences
kernel void kernel_wkv_f32_batch(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* r [[buffer(2)]],
    device const float* w [[buffer(3)]],
    device const float* u [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant int& seq_len [[buffer(7)]],
    constant int& d_model [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int batch = gid.y;
    const int channel = gid.x;

    if (batch >= batch_size || channel >= d_model) return;

    const int offset = batch * seq_len * d_model + channel;

    float wkv_n = 0.0f;
    float wkv_d = 1.0f;

    const float w_val = exp(-exp(w[channel]));
    const float u_val = exp(u[channel]);

    for (int t = 0; t < seq_len; t++) {
        const float k_t = k[offset + t * d_model];
        const float v_t = v[offset + t * d_model];
        const float r_t = r[offset + t * d_model];

        const float wkv = (wkv_n + u_val * k_t * v_t) / (wkv_d + u_val * k_t);
        y[offset + t * d_model] = r_t * wkv;

        wkv_n = w_val * wkv_n + w_val * k_t * v_t;
        wkv_d = w_val * wkv_d + w_val * k_t;
    }
}

// WKV backward pass (for training)
kernel void kernel_wkv_f32_backward(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* r [[buffer(2)]],
    device const float* w [[buffer(3)]],
    device const float* u [[buffer(4)]],
    device const float* grad_y [[buffer(5)]],
    device float* grad_k [[buffer(6)]],
    device float* grad_v [[buffer(7)]],
    device float* grad_r [[buffer(8)]],
    constant int& seq_len [[buffer(9)]],
    constant int& d_model [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d_model) return;

    // Backward pass requires reversing the sequence
    // Simplified: just propagate gradients
    // Full implementation needs careful state management

    float wkv_n = 0.0f;
    float wkv_d = 1.0f;

    const float w_val = exp(-exp(w[gid]));
    const float u_val = exp(u[gid]);

    for (int t = seq_len - 1; t >= 0; t--) {
        const float k_t = k[t * d_model + gid];
        const float v_t = v[t * d_model + gid];
        const float r_t = r[t * d_model + gid];
        const float gy = grad_y[t * d_model + gid];

        // Gradient w.r.t r
        const float wkv = (wkv_n + u_val * k_t * v_t) / (wkv_d + u_val * k_t);
        grad_r[t * d_model + gid] = gy * wkv;

        // Gradient w.r.t k and v (simplified)
        grad_k[t * d_model + gid] = gy * r_t * v_t;
        grad_v[t * d_model + gid] = gy * r_t * k_t;

        // Update state
        wkv_n = w_val * wkv_n + w_val * k_t * v_t;
        wkv_d = w_val * wkv_d + w_val * k_t;
    }
}
