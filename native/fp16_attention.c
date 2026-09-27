// Complete FP16 attention kernel in C
// Uses FP16 accumulation for maximum performance
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// FP16 type matching llama.cpp's ggml_fp16_t
typedef uint16_t ggml_fp16_t;

// Native FP16 × FP16 dot product with 4-way unrolling
static inline float vec_dot_f16(const float16_t* q, const float16_t* k, int n) {
    float16x8_t sum0 = vdupq_n_f16(0.0f);
    float16x8_t sum1 = vdupq_n_f16(0.0f);
    float16x8_t sum2 = vdupq_n_f16(0.0f);
    float16x8_t sum3 = vdupq_n_f16(0.0f);
    int i = 0;

    // Process 32 elements at a time (4×8)
    for (; i + 32 <= n; i += 32) {
        sum0 = vfmaq_f16(sum0, vld1q_f16(q + i),      vld1q_f16(k + i));
        sum1 = vfmaq_f16(sum1, vld1q_f16(q + i + 8),  vld1q_f16(k + i + 8));
        sum2 = vfmaq_f16(sum2, vld1q_f16(q + i + 16), vld1q_f16(k + i + 16));
        sum3 = vfmaq_f16(sum3, vld1q_f16(q + i + 24), vld1q_f16(k + i + 24));
    }

    // Reduce 4 accumulators to 1
    sum0 = vaddq_f16(sum0, sum1);
    sum2 = vaddq_f16(sum2, sum3);
    sum0 = vaddq_f16(sum0, sum2);

    // Handle remaining chunks of 8
    for (; i + 8 <= n; i += 8) {
        sum0 = vfmaq_f16(sum0, vld1q_f16(q + i), vld1q_f16(k + i));
    }

    // Horizontal reduction: FP16 → FP32 → sum
    float32x4_t s = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum0)),
                               vcvt_high_f32_f16(sum0));
    float result = vaddvq_f32(s);

    // Handle remaining elements (scalar)
    for (; i < n; i++) {
        result += (float)q[i] * (float)k[i];
    }
    return result;
}

// FP16 vec_mad with FP16 accumulation
static inline void vec_mad_f16(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    const float16_t v_f16 = (float16_t)v;
    float16x8_t vx = vdupq_n_f16(v_f16);
    const float16_t* x_ptr = (const float16_t*)x;
    float16_t* y_ptr = (float16_t*)y;

    int i = 0;
    for (; i + 32 <= n; i += 32) {
        float16x8_t ax0 = vld1q_f16(x_ptr + i);
        float16x8_t ay0 = vld1q_f16(y_ptr + i);
        ay0 = vfmaq_f16(ay0, ax0, vx);
        vst1q_f16(y_ptr + i, ay0);

        float16x8_t ax1 = vld1q_f16(x_ptr + i + 8);
        float16x8_t ay1 = vld1q_f16(y_ptr + i + 8);
        ay1 = vfmaq_f16(ay1, ax1, vx);
        vst1q_f16(y_ptr + i + 8, ay1);

        float16x8_t ax2 = vld1q_f16(x_ptr + i + 16);
        float16x8_t ay2 = vld1q_f16(y_ptr + i + 16);
        ay2 = vfmaq_f16(ay2, ax2, vx);
        vst1q_f16(y_ptr + i + 16, ay2);

        float16x8_t ax3 = vld1q_f16(x_ptr + i + 24);
        float16x8_t ay3 = vld1q_f16(y_ptr + i + 24);
        ay3 = vfmaq_f16(ay3, ax3, vx);
        vst1q_f16(y_ptr + i + 24, ay3);
    }

    for (; i + 8 <= n; i += 8) {
        float16x8_t ax = vld1q_f16(x_ptr + i);
        float16x8_t ay = vld1q_f16(y_ptr + i);
        ay = vfmaq_f16(ay, ax, vx);
        vst1q_f16(y_ptr + i, ay);
    }

    for (; i < n; i++) {
        float val_y = (float)((float16_t*)y)[i];
        float val_x = (float)((float16_t*)x)[i];
        ((float16_t*)y)[i] = (float16_t)(val_y + val_x * v);
    }
}

// Scale FP16 vector
static inline void vec_scale_f16(ggml_fp16_t* y, float v, int n) {
    const float16_t v_f16 = (float16_t)v;
    float16x8_t vx = vdupq_n_f16(v_f16);
    float16_t* y_ptr = (float16_t*)y;

    for (int i = 0; i + 8 <= n; i += 8) {
        float16x8_t ay = vld1q_f16(y_ptr + i);
        ay = vmulq_f16(ay, vx);
        vst1q_f16(y_ptr + i, ay);
    }
}

// Complete attention kernel with FP16 accumulation
// vkq16: pre-allocated work buffer of size head_dim * sizeof(uint16_t)
void decode_attention_f16_native(
    const unsigned short* q,
    const unsigned short* k,
    const unsigned short* v,
    float* output,
    int num_heads,
    int head_dim,
    int seq_len,
    float scale,
    void* vkq16
) {
    const float16_t* q_ptr = (const float16_t*)q;
    const float16_t* k_ptr = (const float16_t*)k;
    const float16_t* v_ptr = (const float16_t*)v;

    for (int h = 0; h < num_heads; h++) {
        const float16_t* q_head = q_ptr + h * head_dim;
        float* output_head = output + h * head_dim;

        // Zero FP16 accumulator
        memset(vkq16, 0, head_dim * sizeof(ggml_fp16_t));

        float max_val = -INFINITY;
        float sum_exp = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;

            float score = vec_dot_f16(q_head, k_row, head_dim) * scale;

            float old_max = max_val;
            if (score > max_val) {
                max_val = score;
            }

            float weight;
            if (score > old_max) {
                float correction = expf(old_max - max_val);
                sum_exp *= correction;
                vec_scale_f16(vkq16, correction, head_dim);
                weight = 1.0f;
            } else {
                weight = expf(score - max_val);
            }

            sum_exp += weight;
            vec_mad_f16(vkq16, (const ggml_fp16_t*)v_row, weight, head_dim);
        }

        // Convert FP16 accumulator to FP32 and normalize
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
