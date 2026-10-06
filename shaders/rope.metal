// RoPE (Rotary Position Embedding) Metal kernel - Truly self-written implementation
// Algorithm: Apply rotation based on position
//   For each pair (x0, x1):
//     theta = pos * base^(-2i/dim)
//     x0' = x0*cos(theta) - x1*sin(theta)
//     x1' = x0*sin(theta) + x1*cos(theta)
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

// YaRN extension support (optional)
struct RopeConfig {
    int hidden_dim;      // dimension
    int num_heads_seqs;  // heads * sequences
    int num_seqs;        // sequences
    int rotate_dims;     // dimensions to rotate
    int offset;          // rotation offset
    float freq_base;     // base frequency (usually 10000)
    float freq_scale;    // frequency scaling
    float ext_factor;    // YaRN extension factor (0 = disabled)
    float attn_factor;   // YaRN attention factor
    float beta_fast;     // YaRN fast beta
    float beta_slow;     // YaRN slow beta
    int ctx_orig;        // original context length
};

// YaRN ramp function for frequency interpolation
static float yarn_ramp(float low, float high, int freq_idx) {
    float y = (freq_idx / 2 - low) / max(0.001f, high - low);
    return 1.0f - clamp(y, 0.0f, 1.0f);
}

// YaRN frequency correction
static void yarn_calc_rotation(
    float theta_extrap,
    float scale,
    float corr[2],
    int freq_idx,
    float ext,
    float mscale,
    thread float& cos_val,
    thread float& sin_val
) {
    float theta_interp = scale * theta_extrap;
    float theta = theta_interp;

    if (ext != 0.0f) {
        float ramp = yarn_ramp(corr[0], corr[1], freq_idx) * ext;
        theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
        mscale *= 1.0f + 0.1f * log(1.0f / scale);
    }

    cos_val = cos(theta) * mscale;
    sin_val = sin(theta) * mscale;
}

// Calculate YaRN correction dimensions
static void yarn_corr_dims(
    int dims,
    int ctx_len,
    float base,
    float fast_beta,
    float slow_beta,
    thread float bounds[2]
) {
    // Correction factor formula from YaRN paper
    auto corr = [&](float n_rot) {
        return dims * log(ctx_len / (n_rot * 2 * PI)) / (2 * log(base));
    };
    bounds[0] = max(0.0f, floor(corr(fast_beta)));
    bounds[1] = min(float(dims - 1), ceil(corr(slow_beta)));
}

// NeoX-style RoPE (split layout: first half and second half)
// Used in LLaMA models
kernel void kernel_rope_neox_f32(
    constant RopeConfig& cfg [[buffer(0)]],
    device const float* in_data [[buffer(1)]],
    device const int* positions [[buffer(2)]],
    device float* out_data [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const int seq_idx = group.y;
    const int head_idx = group.x;

    // YaRN correction (if enabled)
    float corr_bounds[2];
    if (cfg.ext_factor != 0.0f) {
        yarn_corr_dims(cfg.rotate_dims, cfg.ctx_orig, cfg.freq_base,
                      cfg.beta_fast, cfg.beta_slow, corr_bounds);
    }

    // Get position for this sequence
    float pos = float(positions[seq_idx]);
    float inv_dims = -1.0f / float(cfg.rotate_dims);

    // Calculate row offset
    const uint64_t row_offset = (seq_idx * cfg.num_heads_seqs + head_idx) * cfg.hidden_dim;
    device const float* row_in = in_data + row_offset;
    device float* row_out = out_data + row_offset;

    // Process dimension pairs in NeoX style (split layout)
    // x0 in first half, x1 in second half
    const int half_dims = cfg.rotate_dims / 2;

    for (int pair_idx = lane; pair_idx < half_dims; pair_idx += 32) {
        // Calculate rotation angle
        int freq_idx = pair_idx * 2;
        float theta = pos * pow(cfg.freq_base, inv_dims * freq_idx);

        float cos_theta, sin_theta;
        yarn_calc_rotation(theta, cfg.freq_scale, corr_bounds, freq_idx,
                          cfg.ext_factor, cfg.attn_factor, cos_theta, sin_theta);

        // Load split pair
        float x0 = row_in[pair_idx];
        float x1 = row_in[pair_idx + half_dims];

        // Apply rotation
        row_out[pair_idx] = x0 * cos_theta - x1 * sin_theta;
        row_out[pair_idx + half_dims] = x0 * sin_theta + x1 * cos_theta;
    }
}

// NeoX F16 version
kernel void rope_neox_f16(
    constant RopeConfig& cfg [[buffer(0)]],
    device const half* in_data [[buffer(1)]],
    device const int* positions [[buffer(2)]],
    device half* out_data [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const int seq_idx = group.y;
    const int head_idx = group.x;

    float corr_bounds[2];
    if (cfg.ext_factor != 0.0f) {
        yarn_corr_dims(cfg.rotate_dims, cfg.ctx_orig, cfg.freq_base,
                      cfg.beta_fast, cfg.beta_slow, corr_bounds);
    }

    float pos = float(positions[seq_idx]);
    float inv_dims = -1.0f / float(cfg.rotate_dims);

    const uint64_t row_offset = (seq_idx * cfg.num_heads_seqs + head_idx) * cfg.hidden_dim;
    device const half* row_in = in_data + row_offset;
    device half* row_out = out_data + row_offset;

    const int half_dims = cfg.rotate_dims / 2;

    for (int pair_idx = lane; pair_idx < half_dims; pair_idx += 32) {
        int freq_idx = pair_idx * 2;
        float theta = pos * pow(cfg.freq_base, inv_dims * freq_idx);

        float cos_theta, sin_theta;
        yarn_calc_rotation(theta, cfg.freq_scale, corr_bounds, freq_idx,
                          cfg.ext_factor, cfg.attn_factor, cos_theta, sin_theta);

        float x0 = float(row_in[pair_idx]);
        float x1 = float(row_in[pair_idx + half_dims]);

        row_out[pair_idx] = half(x0 * cos_theta - x1 * sin_theta);
        row_out[pair_idx + half_dims] = half(x0 * sin_theta + x1 * cos_theta);
    }
}

// Normal-style RoPE (interleaved layout)
// Pairs are (x[0], x[1]), (x[2], x[3]), etc.
kernel void rope_norm_f32(
    constant RopeConfig& cfg [[buffer(0)]],
    device const float* in_data [[buffer(1)]],
    device const int* positions [[buffer(2)]],
    device float* out_data [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const int seq_idx = group.y;
    const int head_idx = group.x;

    float corr_bounds[2];
    if (cfg.ext_factor != 0.0f) {
        yarn_corr_dims(cfg.rotate_dims, cfg.ctx_orig, cfg.freq_base,
                      cfg.beta_fast, cfg.beta_slow, corr_bounds);
    }

    float pos = float(positions[seq_idx]);
    float inv_dims = -1.0f / float(cfg.rotate_dims);

    const uint64_t row_offset = (seq_idx * cfg.num_heads_seqs + head_idx) * cfg.hidden_dim;
    device const float* row_in = in_data + row_offset;
    device float* row_out = out_data + row_offset;

    // Process interleaved pairs
    const int num_pairs = cfg.rotate_dims / 2;

    for (int pair_idx = lane; pair_idx < num_pairs; pair_idx += 32) {
        int freq_idx = pair_idx * 2;
        float theta = pos * pow(cfg.freq_base, inv_dims * freq_idx);

        float cos_theta, sin_theta;
        yarn_calc_rotation(theta, cfg.freq_scale, corr_bounds, freq_idx,
                          cfg.ext_factor, cfg.attn_factor, cos_theta, sin_theta);

        // Interleaved: even index is x0, odd index is x1
        int base_idx = pair_idx * 2;
        float x0 = row_in[base_idx];
        float x1 = row_in[base_idx + 1];

        row_out[base_idx] = x0 * cos_theta - x1 * sin_theta;
        row_out[base_idx + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

// Normal F16 version
kernel void rope_norm_f16(
    constant RopeConfig& cfg [[buffer(0)]],
    device const half* in_data [[buffer(1)]],
    device const int* positions [[buffer(2)]],
    device half* out_data [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const int seq_idx = group.y;
    const int head_idx = group.x;

    float corr_bounds[2];
    if (cfg.ext_factor != 0.0f) {
        yarn_corr_dims(cfg.rotate_dims, cfg.ctx_orig, cfg.freq_base,
                      cfg.beta_fast, cfg.beta_slow, corr_bounds);
    }

    float pos = float(positions[seq_idx]);
    float inv_dims = -1.0f / float(cfg.rotate_dims);

    const uint64_t row_offset = (seq_idx * cfg.num_heads_seqs + head_idx) * cfg.hidden_dim;
    device const half* row_in = in_data + row_offset;
    device half* row_out = out_data + row_offset;

    const int num_pairs = cfg.rotate_dims / 2;

    for (int pair_idx = lane; pair_idx < num_pairs; pair_idx += 32) {
        int freq_idx = pair_idx * 2;
        float theta = pos * pow(cfg.freq_base, inv_dims * freq_idx);

        float cos_theta, sin_theta;
        yarn_calc_rotation(theta, cfg.freq_scale, corr_bounds, freq_idx,
                          cfg.ext_factor, cfg.attn_factor, cos_theta, sin_theta);

        int base_idx = pair_idx * 2;
        float x0 = float(row_in[base_idx]);
        float x1 = float(row_in[base_idx + 1]);

        row_out[base_idx] = half(x0 * cos_theta - x1 * sin_theta);
        row_out[base_idx + 1] = half(x0 * sin_theta + x1 * cos_theta);
    }
}
