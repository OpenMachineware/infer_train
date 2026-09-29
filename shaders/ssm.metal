// State Space Model (SSM) kernels for GPU
// Based on Mamba architecture: Selective State Space Models

#include <metal_stdlib>
using namespace metal;

// SSM parameters:
// - A: state transition matrix (d_state x d_state)
// - B: input matrix (d_state x d_model)
// - C: output matrix (d_model x d_state)
// - D: skip connection (d_model)
// - dt: delta time (d_model)

// Selective scan - core operation of Mamba
// Each thread processes one channel across the sequence
kernel void kernel_ssm_scan_f32(
    device const float* u [[buffer(0)]],      // input: [d_model, seq_len]
    device const float* delta [[buffer(1)]],  // delta: [d_model, seq_len]
    device const float* A [[buffer(2)]],      // A: [d_state]
    device const float* B [[buffer(3)]],      // B: [d_state, seq_len]
    device const float* C [[buffer(4)]],      // C: [d_state, seq_len]
    device const float* D [[buffer(5)]],      // D: [d_model]
    device float* y [[buffer(6)]],             // output: [d_model, seq_len]
    constant int& d_model [[buffer(7)]],
    constant int& d_state [[buffer(8)]],
    constant int& seq_len [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d_model) return;

    // Initialize state
    float h = 0.0f;

    // Get D for skip connection
    const float d_val = D[gid];

    // Process sequence
    for (int t = 0; t < seq_len; t++) {
        const float u_t = u[gid * seq_len + t];
        const float delta_t = delta[gid * seq_len + t];

        // Discretize A
        const float da = exp(-delta_t * A[0]);  // Simplified: assuming diagonal A

        // Update state: h = da * h + delta_t * B_t * u_t
        float b_t = 0.0f;
        for (int s = 0; s < d_state; s++) {
            b_t += B[s * seq_len + t];
        }
        h = da * h + delta_t * b_t * u_t;

        // Output: y = C_t * h + D * u_t
        float c_t = 0.0f;
        for (int s = 0; s < d_state; s++) {
            c_t += C[s * seq_len + t];
        }
        y[gid * seq_len + t] = c_t * h + d_val * u_t;
    }
}

// Batched SSM scan for multiple channels
kernel void kernel_ssm_scan_f32_batch(
    device const float* u [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* B [[buffer(3)]],
    device const float* C [[buffer(4)]],
    device const float* D [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant int& batch_size [[buffer(7)]],
    constant int& d_model [[buffer(8)]],
    constant int& d_state [[buffer(9)]],
    constant int& seq_len [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int batch = gid.y;
    const int channel = gid.x;

    if (batch >= batch_size || channel >= d_model) return;

    const int offset = batch * d_model * seq_len + channel * seq_len;

    float h = 0.0f;
    const float d_val = D[channel];

    for (int t = 0; t < seq_len; t++) {
        const float u_t = u[offset + t];
        const float delta_t = delta[offset + t];

        const float da = exp(-delta_t * A[0]);

        float b_t = 0.0f;
        for (int s = 0; s < d_state; s++) {
            b_t += B[(batch * d_state + s) * seq_len + t];
        }
        h = da * h + delta_t * b_t * u_t;

        float c_t = 0.0f;
        for (int s = 0; s < d_state; s++) {
            c_t += C[(batch * d_state + s) * seq_len + t];
        }
        y[offset + t] = c_t * h + d_val * u_t;
    }
}

// Conv1d SSM variant (used in Mamba)
kernel void kernel_ssm_conv1d_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant int& d_model [[buffer(4)]],
    constant int& kernel_size [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int channel = gid.x;
    const int pos = gid.y;

    if (channel >= d_model || pos >= seq_len) return;

    float sum = bias[channel];

    for (int k = 0; k < kernel_size; k++) {
        const int x_pos = pos - kernel_size / 2 + k;
        if (x_pos >= 0 && x_pos < seq_len) {
            sum += x[x_pos * d_model + channel] * weight[k * d_model + channel];
        }
    }

    y[pos * d_model + channel] = sum;
}
