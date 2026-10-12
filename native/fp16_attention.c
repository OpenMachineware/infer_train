// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Complete FP16 attention kernel in C
// Uses FP16 accumulation for maximum performance
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// FP16 type matching llama.cpp's ggml_fp16_t
typedef uint16_t ggml_fp16_t;

// FP32 Q × FP16 K dot product (matches llama.cpp approach)
// Uses FP32 accumulation for better precision
static inline float vec_dot_f32_f16(const float* q, const float16_t* k, int n) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int i = 0;

    // Process 16 elements at a time (4×4)
    for (; i + 16 <= n; i += 16) {
        // Load 4 FP32 Q values
        float32x4_t q0 = vld1q_f32(q + i);
        float32x4_t q1 = vld1q_f32(q + i + 4);
        float32x4_t q2 = vld1q_f32(q + i + 8);
        float32x4_t q3 = vld1q_f32(q + i + 12);

        // Load and convert 4 FP16 K values to FP32
        float32x4_t k0 = vcvt_f32_f16(vld1_f16(k + i));
        float32x4_t k1 = vcvt_f32_f16(vld1_f16(k + i + 4));
        float32x4_t k2 = vcvt_f32_f16(vld1_f16(k + i + 8));
        float32x4_t k3 = vcvt_f32_f16(vld1_f16(k + i + 12));

        // FP32 FMA
        sum0 = vfmaq_f32(sum0, q0, k0);
        sum1 = vfmaq_f32(sum1, q1, k1);
        sum2 = vfmaq_f32(sum2, q2, k2);
        sum3 = vfmaq_f32(sum3, q3, k3);
    }

    // Reduce 4 accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float result = vaddvq_f32(sum0);

    // Handle remaining chunks of 4
    for (; i + 4 <= n; i += 4) {
        float32x4_t qv = vld1q_f32(q + i);
        float32x4_t kv = vcvt_f32_f16(vld1_f16(k + i));
        result += vaddvq_f32(vmulq_f32(qv, kv));
    }

    // Handle remaining elements (scalar)
    for (; i < n; i++) {
        result += q[i] * (float)((float16_t*)k)[i];
    }
    return result;
}

// Native FP16 × FP16 dot product with llama.cpp-style loop
static inline float vec_dot_f16(const float16_t* q, const float16_t* k, int n) {
    const int np = (n & ~31);
    float16x8_t sum[4] = { vdupq_n_f16(0.0f), vdupq_n_f16(0.0f), vdupq_n_f16(0.0f), vdupq_n_f16(0.0f) };

    for (int i = 0; i < np; i += 32) {
        sum[0] = vfmaq_f16(sum[0], vld1q_f16(q + i),      vld1q_f16(k + i));
        sum[1] = vfmaq_f16(sum[1], vld1q_f16(q + i + 8),  vld1q_f16(k + i + 8));
        sum[2] = vfmaq_f16(sum[2], vld1q_f16(q + i + 16), vld1q_f16(k + i + 16));
        sum[3] = vfmaq_f16(sum[3], vld1q_f16(q + i + 24), vld1q_f16(k + i + 24));
    }

    // Reduce 4 accumulators to 1
    sum[0] = vaddq_f16(sum[0], sum[1]);
    sum[2] = vaddq_f16(sum[2], sum[3]);
    sum[0] = vaddq_f16(sum[0], sum[2]);

    // Horizontal reduction: FP16 → FP32 → sum
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16(sum[0]));
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(sum[0]));
    float result = vaddvq_f32(vaddq_f32(t0, t1));

    // Handle remaining elements (scalar)
    for (int i = np; i < n; i++) {
        result += (float)q[i] * (float)k[i];
    }
    return result;
}

// FP16 vec_mad with llama.cpp-style inner loop
static inline void vec_mad_f16(ggml_fp16_t* y, const ggml_fp16_t* x, float v, int n) {
    const int np = (n & ~31);
    const float16x8_t vx = vdupq_n_f16((float16_t)v);

    for (int i = 0; i < np; i += 32) {
        float16x8_t ax[4], ay[4];
        for (int j = 0; j < 4; j++) {
            ax[j] = vld1q_f16((const float16_t*)(x + i + j * 8));
            ay[j] = vld1q_f16((float16_t*)(y + i + j * 8));
            ay[j] = vfmaq_f16(ay[j], ax[j], vx);
            vst1q_f16((float16_t*)(y + i + j * 8), ay[j]);
        }
    }

    // Handle remaining elements
    for (int i = np; i < n; i++) {
        y[i] = (float)((float16_t*)y)[i] + (float)((float16_t*)x)[i] * v;
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

// Chunked attention kernel with FP16 accumulation
// Partials buffer: [num_heads][num_chunks][M, S, VKQ[head_dim]]
// Returns 1 if chunking was used, 0 otherwise
int decode_attention_f16_chunked(
    const unsigned short* q,
    const unsigned short* k,
    const unsigned short* v,
    float* output,
    int num_heads,
    int head_dim,
    int seq_len,
    float scale,
    void* vkq16,
    void* partials,      // Pre-allocated partials buffer
    int chunk_size       // 0 = auto (seq_len / num_chunks)
) {
    const float16_t* q_ptr = (const float16_t*)q;
    const float16_t* k_ptr = (const float16_t*)k;
    const float16_t* v_ptr = (const float16_t*)v;

    // Determine if chunking is beneficial
    // For single-thread, chunking adds overhead without benefit
    // Disable for now - needs multi-threading to be beneficial
    return 0;  // Disable chunking for single-thread

    const int min_seq_len_for_chunking = 512;
    if (seq_len < min_seq_len_for_chunking) {
        return 0;  // Don't use chunking
    }

    // Calculate number of chunks (target ~256 positions per chunk)
    int num_chunks;
    if (chunk_size > 0) {
        num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    } else {
        // Auto: aim for ~256 positions per chunk
        chunk_size = 256;
        num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    }

    // Partials layout: [num_heads][num_chunks][M, S, VKQ[head_dim]]
    const int partial_size = 2 + head_dim;
    float* partials_base = (float*)partials;

    // Phase 1: Process each chunk, store partials
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int ic_start = chunk_idx * chunk_size;
        int ic_end = (ic_start + chunk_size < seq_len) ? ic_start + chunk_size : seq_len;

        for (int h = 0; h < num_heads; h++) {
            const float16_t* q_head = q_ptr + h * head_dim;

            // Zero FP16 accumulator
            memset(vkq16, 0, head_dim * sizeof(ggml_fp16_t));

            float M = -INFINITY;
            float S = 0.0f;

            // Process this chunk
            for (int s = ic_start; s < ic_end; s++) {
                const float16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
                const float16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;

                float score = vec_dot_f16(q_head, k_row, head_dim) * scale;

                float old_max = M;
                if (score > M) {
                    M = score;
                }

                float weight;
                if (score > old_max) {
                    float correction = expf(old_max - M);
                    S *= correction;
                    vec_scale_f16(vkq16, correction, head_dim);
                    weight = 1.0f;
                } else {
                    weight = expf(score - M);
                }

                S += weight;
                vec_mad_f16(vkq16, (const ggml_fp16_t*)v_row, weight, head_dim);
            }

            // Store partial: [M, S, VKQ]
            float* partial = partials_base + (h * num_chunks + chunk_idx) * partial_size;
            partial[0] = M;
            partial[1] = S;

            // Convert FP16 accumulator to FP32 for storage
            for (int d = 0; d < head_dim; d += 8) {
                float16x8_t acc = vld1q_f16((const float16_t*)vkq16 + d);
                float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
                float32x4_t acc1 = vcvt_high_f32_f16(acc);
                vst1q_f32(partial + 2 + d, acc0);
                vst1q_f32(partial + 2 + d + 4, acc1);
            }
        }
    }

    // Phase 2: Reduce partials for each query head
    for (int h = 0; h < num_heads; h++) {
        float M_final = -INFINITY;
        float S_final = 0.0f;
        float* VKQ_final = (float*)vkq16;  // Reuse buffer for FP32 VKQ
        memset(VKQ_final, 0, head_dim * sizeof(float));

        // Combine partials from all chunks (online softmax reduction)
        for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            const float* partial = partials_base + (h * num_chunks + chunk_idx) * partial_size;
            float M_chunk = partial[0];
            float S_chunk = partial[1];
            const float* VKQ_chunk = partial + 2;

            if (S_chunk == 0.0f) continue;

            float M_new = fmaxf(M_final, M_chunk);
            float scale_old = expf(M_final - M_new);
            float scale_new = expf(M_chunk - M_new);

            for (int d = 0; d < head_dim; d++) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write output
        float* output_head = output + h * head_dim;
        if (S_final != 0.0f) {
            float S_inv = 1.0f / S_final;
            for (int d = 0; d < head_dim; d += 4) {
                float32x4_t v = vld1q_f32(VKQ_final + d);
                v = vmulq_n_f32(v, S_inv);
                vst1q_f32(output_head + d, v);
            }
        }
    }

    return 1;  // Chunking was used
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
    // Use ggml_fp16_t pointers for consistency with llama.cpp
    const ggml_fp16_t* q_ptr = (const ggml_fp16_t*)q;
    const ggml_fp16_t* k_ptr = (const ggml_fp16_t*)k;
    const ggml_fp16_t* v_ptr = (const ggml_fp16_t*)v;
    ggml_fp16_t* VKQ = (ggml_fp16_t*)vkq16;

    for (int h = 0; h < num_heads; h++) {
        const ggml_fp16_t* q_head = q_ptr + h * head_dim;
        float* out_head = output + h * head_dim;

        // Zero FP16 accumulator
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));

        float M = -INFINITY;
        float S = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const ggml_fp16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
            const ggml_fp16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;

            float score = vec_dot_f16((const float16_t*)q_head, (const float16_t*)k_row, head_dim) * scale;

            // llama.cpp style: simpler branch structure
            if (score > M) {
                float ms = expf(M - score);
                S *= ms;
                vec_scale_f16(VKQ, ms, head_dim);
                M = score;
            }

            float vs = expf(score - M);
            S += vs;
            vec_mad_f16(VKQ, v_row, vs, head_dim);
        }

        // Normalize and write output
        float inv_S = 1.0f / S;
        for (int d = 0; d < head_dim; d++) {
            out_head[d] = (float)((float16_t*)VKQ)[d] * inv_S;
        }
    }
}

// ============================================================================
// FP32 Q + FP16 K/V Attention (matches llama.cpp approach)
// Key insight: Convert Q to FP16 ONCE, then use FP16×FP16 dot product
// ============================================================================

// FP32 Q + FP16 K/V attention with KV chunking for seq_len >= 512
// vkq32: work buffer for FP32 VKQ accumulator (size: head_dim * sizeof(float))
// q_f16: work buffer for FP16 Q (size: head_dim * sizeof(uint16_t))
// partials: buffer for chunk partials [num_heads][num_chunks][M, S, VKQ]
// Returns 1 if chunking was used, 0 otherwise
int decode_attention_f32_q_f16_kv(
    const float* q,
    const unsigned short* k,
    const unsigned short* v,
    float* output,
    int num_heads,
    int head_dim,
    int seq_len,
    float scale,
    void* vkq32,       // FP32 work buffer
    void* q_f16,       // FP16 Q buffer
    void* partials,    // Partials buffer (NULL to disable chunking)
    int chunk_size     // 0 = auto
) {
    const float16_t* k_ptr = (const float16_t*)k;
    const float16_t* v_ptr = (const float16_t*)v;
    float* VKQ = (float*)vkq32;
    ggml_fp16_t* Q_f16 = (ggml_fp16_t*)q_f16;

    // Prefetch distance: how many positions ahead to prefetch
    // Only prefetch for large KV caches where cache thrashing occurs
    // DISABLED for testing - prefetching may cause slowdown
    const int prefetch_distance = 0; // seq_len > 512 ? 32 : 0;

    // Determine if chunking is beneficial
    const int min_seq_len_for_chunking = 999999;  // Disable chunking for testing

    if (seq_len >= min_seq_len_for_chunking && partials != NULL) {
        // Calculate number of chunks
        if (chunk_size <= 0) {
            chunk_size = 256;
        }
        int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        const int partial_size = 2 + head_dim;
        float* partials_base = (float*)partials;

        // Phase 1: Process each chunk, store partials
        for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            int ic_start = chunk_idx * chunk_size;
            int ic_end = (ic_start + chunk_size < seq_len) ? ic_start + chunk_size : seq_len;

            for (int h = 0; h < num_heads; h++) {
                const float* q_head = q + h * head_dim;

                // Convert Q to FP16 (once per head)
                for (int d = 0; d < head_dim; d += 8) {
                    float32x4_t q0 = vld1q_f32(q_head + d);
                    float32x4_t q1 = vld1q_f32(q_head + d + 4);
                    vst1q_f16((float16_t*)(Q_f16 + d), vcvt_high_f16_f32(vcvt_f16_f32(q0), q1));
                }

                // Zero FP16 accumulator for VKQ
                float16x8_t* VKQ16 = (float16x8_t*)VKQ;
                for (int d = 0; d < head_dim; d += 8) {
                    VKQ16[d/8] = vdupq_n_f16(0.0f);
                }

                float M = -INFINITY;
                float S = 0.0f;

                // Process this chunk using FP16×FP16 dot product
                for (int s = ic_start; s < ic_end; s++) {
                    const float16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
                    const float16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;

                    // FP16 × FP16 dot product
                    float score = vec_dot_f16((const float16_t*)Q_f16, k_row, head_dim) * scale;

                    float old_max = M;
                    if (score > M) {
                        M = score;
                    }

                    float weight;
                    if (score > old_max) {
                        float correction = expf(old_max - M);
                        S *= correction;
                        vec_scale_f16((ggml_fp16_t*)VKQ, correction, head_dim);
                        weight = 1.0f;
                    } else {
                        weight = expf(score - M);
                    }

                    S += weight;
                    vec_mad_f16((ggml_fp16_t*)VKQ, (const ggml_fp16_t*)v_row, weight, head_dim);
                }

                // Store partial: [M, S, VKQ]
                float* partial = partials_base + (h * num_chunks + chunk_idx) * partial_size;
                partial[0] = M;
                partial[1] = S;

                // Convert FP16 VKQ to FP32 for storage
                for (int d = 0; d < head_dim; d += 8) {
                    float16x8_t acc = vld1q_f16((const float16_t*)VKQ + d);
                    float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
                    float32x4_t acc1 = vcvt_high_f32_f16(acc);
                    vst1q_f32(partial + 2 + d, acc0);
                    vst1q_f32(partial + 2 + d + 4, acc1);
                }
            }
        }

        // Phase 2: Reduce partials for each query head
        for (int h = 0; h < num_heads; h++) {
            float M_final = -INFINITY;
            float S_final = 0.0f;
            memset(VKQ, 0, head_dim * sizeof(float));

            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                const float* partial = partials_base + (h * num_chunks + chunk_idx) * partial_size;
                float M_chunk = partial[0];
                float S_chunk = partial[1];
                const float* VKQ_chunk = partial + 2;

                if (S_chunk == 0.0f) continue;

                float M_new = fmaxf(M_final, M_chunk);
                float scale_old = expf(M_final - M_new);
                float scale_new = expf(M_chunk - M_new);

                for (int d = 0; d < head_dim; d++) {
                    VKQ[d] = VKQ[d] * scale_old + VKQ_chunk[d] * scale_new;
                }
                S_final = S_final * scale_old + S_chunk * scale_new;
                M_final = M_new;
            }

            // Normalize and write output
            float* output_head = output + h * head_dim;
            if (S_final != 0.0f) {
                float S_inv = 1.0f / S_final;
                for (int d = 0; d < head_dim; d += 4) {
                    float32x4_t v = vld1q_f32(VKQ + d);
                    v = vmulq_n_f32(v, S_inv);
                    vst1q_f32(output_head + d, v);
                }
            }
        }
        return 1;  // Chunking was used
    }

    // Non-chunked version
    for (int h = 0; h < num_heads; h++) {
        const float* q_head = q + h * head_dim;
        float* out_head = output + h * head_dim;

        // Convert Q to FP16 (once per head)
        for (int d = 0; d < head_dim; d += 8) {
            float32x4_t q0 = vld1q_f32(q_head + d);
            float32x4_t q1 = vld1q_f32(q_head + d + 4);
            vst1q_f16((float16_t*)(Q_f16 + d), vcvt_high_f16_f32(vcvt_f16_f32(q0), q1));
        }

        // Zero FP16 accumulator
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));

        float M = -INFINITY;
        float S = 0.0f;

        for (int s = 0; s < seq_len; s++) {
            const ggml_fp16_t* k_row = (const ggml_fp16_t*)k_ptr + (s * num_heads + h) * head_dim;
            const ggml_fp16_t* v_row = (const ggml_fp16_t*)v_ptr + (s * num_heads + h) * head_dim;

            float score = vec_dot_f16((const float16_t*)Q_f16, (const float16_t*)k_row, head_dim) * scale;

            // llama.cpp style softmax
            if (score > M) {
                float ms = expf(M - score);
                S *= ms;
                vec_scale_f16((ggml_fp16_t*)VKQ, ms, head_dim);
                M = score;
            }

            float vs = expf(score - M);
            S += vs;
            vec_mad_f16((ggml_fp16_t*)VKQ, v_row, vs, head_dim);
        }

        // Convert FP16 VKQ to FP32 and normalize
        float inv_S = 1.0f / S;
        for (int d = 0; d < head_dim; d++) {
            out_head[d] = (float)((float16_t*)VKQ)[d] * inv_S;
        }
    }
    return 0;  // Chunking not used
}
