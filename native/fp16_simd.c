// Native FP16 SIMD functions for ARM NEON
// Compile with: clang -O3 -c fp16_simd.c -o libfp16_simd.a

#include <arm_neon.h>

// Native FP16 × FP16 dot product
// Processes 8 FP16 values per instruction
float vec_dot_f16_native(const unsigned short* q, const unsigned short* k, int n) {
    float16x8_t sum0 = vdupq_n_f16(0.0f);
    float16x8_t sum1 = vdupq_n_f16(0.0f);
    float16x8_t sum2 = vdupq_n_f16(0.0f);
    float16x8_t sum3 = vdupq_n_f16(0.0f);

    const float16_t* q_ptr = (const float16_t*)q;
    const float16_t* k_ptr = (const float16_t*)k;
    int i = 0;

    // Process 32 elements at a time (4 × 8)
    for (; i + 32 <= n; i += 32) {
        float16x8_t q0 = vld1q_f16(q_ptr + i);
        float16x8_t k0 = vld1q_f16(k_ptr + i);
        float16x8_t q1 = vld1q_f16(q_ptr + i + 8);
        float16x8_t k1 = vld1q_f16(k_ptr + i + 8);
        float16x8_t q2 = vld1q_f16(q_ptr + i + 16);
        float16x8_t k2 = vld1q_f16(k_ptr + i + 16);
        float16x8_t q3 = vld1q_f16(q_ptr + i + 24);
        float16x8_t k3 = vld1q_f16(k_ptr + i + 24);

        sum0 = vfmaq_f16(sum0, q0, k0);
        sum1 = vfmaq_f16(sum1, q1, k1);
        sum2 = vfmaq_f16(sum2, q2, k2);
        sum3 = vfmaq_f16(sum3, q3, k3);
    }

    // Reduce accumulators
    sum0 = vaddq_f16(sum0, sum1);
    sum2 = vaddq_f16(sum2, sum3);
    sum0 = vaddq_f16(sum0, sum2);

    // Process remaining 8 elements
    for (; i + 8 <= n; i += 8) {
        float16x8_t q = vld1q_f16(q_ptr + i);
        float16x8_t k = vld1q_f16(k_ptr + i);
        sum0 = vfmaq_f16(sum0, q, k);
    }

    // Horizontal reduce
    float32x4_t low = vcvt_f32_f16(vget_low_f16(sum0));
    float32x4_t high = vcvt_high_f32_f16(sum0);
    float result = vaddvq_f32(vaddq_f32(low, high));

    // Handle remaining elements
    for (; i < n; i++) {
        result += (float)q_ptr[i] * (float)k_ptr[i];
    }

    return result;
}

// Native FP16 vec_mad: output += v * weight
// output is FP32, v is FP16, weight is FP32
void vec_mad_f16_f32_native(float* output, const unsigned short* v, float weight, int n) {
    float16x8_t w_vec = vdupq_n_f16((float16_t)weight);
    const float16_t* v_ptr = (const float16_t*)v;
    int i = 0;

    for (; i + 8 <= n; i += 8) {
        // Load 8 FP16 values
        float16x8_t v_vec = vld1q_f16(v_ptr + i);
        // Load 8 FP32 values
        float32x4_t out0 = vld1q_f32(output + i);
        float32x4_t out1 = vld1q_f32(output + i + 4);

        // Convert v to FP32 and do FMA
        float32x4_t v0 = vcvt_f32_f16(vget_low_f16(v_vec));
        float32x4_t v1 = vcvt_high_f32_f16(v_vec);
        float32x4_t w0 = vdupq_n_f32(weight);

        out0 = vfmaq_f32(out0, v0, w0);
        out1 = vfmaq_f32(out1, v1, w0);

        vst1q_f32(output + i, out0);
        vst1q_f32(output + i + 4, out1);
    }

    for (; i < n; i++) {
        output[i] += weight * (float)v_ptr[i];
    }
}
