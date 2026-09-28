// Minimal test to compare assembly with llama.cpp
// Compile with: clang -O3 -S -march=armv8.2-a+fp16 test_attention_only.c

#include <arm_neon.h>
#include <math.h>
#include <string.h>

typedef __fp16 float16_t;
typedef uint16_t ggml_fp16_t;

static inline float vec_dot_f16(const float16_t* a, const float16_t* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 8) {
        float16x8_t va = vld1q_f16(a + i);
        float16x8_t vb = vld1q_f16(b + i);
        sum = vaddq_f32(sum, vaddq_f32(vcvt_f32_f16(vget_low_f16(va)) * vcvt_f32_f16(vget_low_f16(vb)),
                                        vcvt_high_f32_f16(va) * vcvt_high_f32_f16(vb)));
    }
    return vaddvq_f32(sum);
}

static inline void vec_scale_f16(ggml_fp16_t* y, float v, int n) {
    for (int i = 0; i < n; i += 8) {
        float16x8_t vy = vld1q_f16((const float16_t*)y + i);
        vy = vmulq_n_f16(vy, v);
        vst1q_f16((float16_t*)y + i, vy);
    }
}

static inline void vec_mad_f16(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    for (int i = 0; i < n; i += 8) {
        float16x8_t vy = vld1q_f16((const float16_t*)y + i);
        float16x8_t vx = vld1q_f16((const float16_t*)x + i);
        vy = vfmaq_n_f16(vy, vx, v);
        vst1q_f16((float16_t*)y + i, vy);
    }
}

void test_attention(
    const float16_t* q,
    const float16_t* k,
    const float16_t* v,
    float* output,
    int num_heads,
    int head_dim,
    int seq_len,
    float scale,
    void* vkq16
) {
    for (int h = 0; h < num_heads; h++) {
        const float16_t* q_head = q + h * head_dim;
        float* output_head = output + h * head_dim;

        memset(vkq16, 0, head_dim * sizeof(ggml_fp16_t));

        float max_val = -INFINITY;
        float sum_exp = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v + (s * num_heads + h) * head_dim;

            float score = vec_dot_f16(q_head, k_row, head_dim) * scale;

            float old_max = max_val;
            if (score > max_val) {
                max_val = score;
            }

            float weight;
            if (score > old_max) {
                float correction = expf(old_max - max_val);
                sum_exp *= correction;
                vec_scale_f16((ggml_fp16_t*)vkq16, correction, head_dim);
                weight = 1.0f;
            } else {
                weight = expf(score - max_val);
            }

            sum_exp += weight;
            vec_mad_f16((ggml_fp16_t*)vkq16, (const ggml_fp16_t*)v_row, weight, head_dim);
        }

        float inv_sum = 1.0f / sum_exp;
        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t acc = vld1q_f16((const float16_t*)vkq16 + d);
            float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
            float32x4_t acc1 = vcvt_high_f32_f16(acc);
            acc0 = vmulq_n_f32(acc0, inv_sum);
            acc1 = vmulq_n_f32(acc1, inv_sum);
            vst1q_f32(output_head + d, acc0);
            vst1q_f32(output_head + d + 4, acc1);
        }
    }
}
