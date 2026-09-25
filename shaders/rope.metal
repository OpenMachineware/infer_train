// RoPE (Rotary Position Embedding) Metal kernel
// Based on llama.cpp's kernel_rope_neox

#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

struct RopeArgs {
    int32_t  ne00;       // hidden_dim
    int32_t  ne01;       // n_heads * n_seqs
    int32_t  ne02;       // n_seqs
    uint64_t nb00;       // stride for dim
    uint64_t nb01;       // stride for head
    uint64_t nb02;       // stride for seq
    int32_t  n_dims;     // dimensions to rotate
    int32_t  n_offs;     // offset for rotation
    float    freq_base;  // base frequency
    float    freq_scale; // frequency scaling
    float    ext_factor; // YaRN extension factor
    float    attn_factor;// YaRN attention factor
    float    beta_fast;  // YaRN fast beta
    float    beta_slow;  // YaRN slow beta
    int32_t  n_ctx_orig; // original context length
};

static float rope_yarn_ramp(float low, float high, int i0) {
    float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0,
    float ext_factor, float mscale,
    thread float& cos_theta, thread float& sin_theta
) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * PI)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
    thread float dims[2]
) {
    dims[0] = max(0.0f, floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

// NeoX style RoPE (used by LLaMA)
kernel void kernel_rope_neox_f32(
    constant RopeArgs& args [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const int32_t* pos [[buffer(2)]],
    device float* dst [[buffer(3)]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort3 tptg [[threads_per_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]]
) {
    const int i2 = tgpig[1];  // sequence index
    const int i1 = tgpig[0];  // head index

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base,
                        args.beta_fast, args.beta_slow, corr_dims);

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.0f / args.n_dims;

    // Base offset for this (seq, head)
    const int row_offset = (i2 * args.ne01 + i1) * args.ne00;

    // Each thread processes multiple dimension pairs
    for (int i0 = tiitg; i0 < args.n_dims / 2; i0 += tptg.x) {
        const int iw = i0 * 2;
        const float theta = theta_base * powr(args.freq_base, inv_ndims * iw);

        float cos_theta, sin_theta;
        rope_yarn(theta, args.freq_scale, corr_dims, iw, args.ext_factor,
                  args.attn_factor, cos_theta, sin_theta);

        // Load x0 and x1 (separated by n_dims/2)
        const float x0 = src[row_offset + i0];
        const float x1 = src[row_offset + i0 + args.n_dims / 2];

        // Apply rotation
        dst[row_offset + i0] = x0 * cos_theta - x1 * sin_theta;
        dst[row_offset + i0 + args.n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
    }
}
