// 两边算法插桩对比 - 修正版（确保逻辑一致）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
#include <inttypes.h>

typedef __fp16 float16_t;
typedef uint16_t ggml_fp16_t;

static inline uint64_t get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

typedef struct {
    uint64_t q_conv_ns;
    uint64_t clear_ns;
    uint64_t dot_ns;
    uint64_t softmax_ns;
    uint64_t scale_ns;
    uint64_t mad_ns;
    uint64_t output_ns;
    int scale_count;
} Stats;

// 我们的实现
void our_attention(const float* q_f32, const ggml_fp16_t* k, const ggml_fp16_t* v,
                   float* output, int num_heads, int head_dim, int seq_len, float scale,
                   void* work, ggml_fp16_t* q_f16_buf, Stats* stats) {
    const float16_t* k_ptr = (const float16_t*)k;
    const float16_t* v_ptr = (const float16_t*)v;
    float* VKQ = (float*)work;
    ggml_fp16_t* Q_f16 = q_f16_buf;
    
    memset(stats, 0, sizeof(Stats));
    
    for (int h = 0; h < num_heads; h++) {
        const float* q_head = q_f32 + h * head_dim;
        float* output_head = output + h * head_dim;
        
        uint64_t t0 = get_ns();
        for (int d = 0; d < head_dim; d += 8) {
            float32x4_t q0 = vld1q_f32(q_head + d);
            float32x4_t q1 = vld1q_f32(q_head + d + 4);
            vst1q_f16((float16_t*)(Q_f16 + d), vcvt_high_f16_f32(vcvt_f16_f32(q0), q1));
        }
        stats->q_conv_ns += get_ns() - t0;
        
        t0 = get_ns();
        memset(VKQ, 0, head_dim * sizeof(ggml_fp16_t));
        stats->clear_ns += get_ns() - t0;
        
        float M = -INFINITY;
        float S = 0.0f;
        
        for (int s = 0; s < seq_len; s++) {
            const float16_t* k_row = k_ptr + (s * num_heads + h) * head_dim;
            const float16_t* v_row = v_ptr + (s * num_heads + h) * head_dim;
            
            // vec_dot (inline)
            t0 = get_ns();
            float16x8_t sum0 = vdupq_n_f16(0.0f);
            float16x8_t sum1 = vdupq_n_f16(0.0f);
            float16x8_t sum2 = vdupq_n_f16(0.0f);
            float16x8_t sum3 = vdupq_n_f16(0.0f);
            for (int i = 0; i < head_dim; i += 32) {
                sum0 = vfmaq_f16(sum0, vld1q_f16(Q_f16 + i),      vld1q_f16(k_row + i));
                sum1 = vfmaq_f16(sum1, vld1q_f16(Q_f16 + i + 8),  vld1q_f16(k_row + i + 8));
                sum2 = vfmaq_f16(sum2, vld1q_f16(Q_f16 + i + 16), vld1q_f16(k_row + i + 16));
                sum3 = vfmaq_f16(sum3, vld1q_f16(Q_f16 + i + 24), vld1q_f16(k_row + i + 24));
            }
            sum0 = vaddq_f16(sum0, sum1);
            sum2 = vaddq_f16(sum2, sum3);
            sum0 = vaddq_f16(sum0, sum2);
            float32x4_t s = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum0)),
                                       vcvt_high_f32_f16(sum0));
            float score = vaddvq_f32(s) * scale;
            stats->dot_ns += get_ns() - t0;
            
            t0 = get_ns();
            float old_max = M;
            if (score > M) M = score;
            
            float weight;
            if (score > old_max) {
                float correction = expf(old_max - M);
                S *= correction;
                weight = 1.0f;
            } else {
                weight = expf(score - M);
            }
            stats->softmax_ns += get_ns() - t0;
            
            if (score > old_max) {
                t0 = get_ns();
                float16x8_t vc = vdupq_n_f16((float16_t)correction);
                for (int i = 0; i < head_dim; i += 8) {
                    float16x8_t vy = vld1q_f16((const float16_t*)VKQ + i);
                    vst1q_f16((float16_t*)VKQ + i, vmulq_f16(vy, vc));
                }
                stats->scale_ns += get_ns() - t0;
                stats->scale_count++;
            }
            
            S += weight;
            
            t0 = get_ns();
            float16x8_t vw = vdupq_n_f16((float16_t)weight);
            for (int i = 0; i < head_dim; i += 8) {
                float16x8_t vy = vld1q_f16((const float16_t*)VKQ + i);
                float16x8_t vx = vld1q_f16(v_row + i);
                vst1q_f16((float16_t*)VKQ + i, vfmaq_f16(vy, vx, vw));
            }
            stats->mad_ns += get_ns() - t0;
        }
        
        t0 = get_ns();
        float inv_sum = 1.0f / S;
        for (int d = 0; d < head_dim; d += 8) {
            float16x8_t acc = vld1q_f16((const float16_t*)VKQ + d);
            float32x4_t acc0 = vcvt_f32_f16(vget_low_f16(acc));
            float32x4_t acc1 = vcvt_high_f32_f16(acc);
            vst1q_f32(output_head + d, vmulq_n_f32(acc0, inv_sum));
            vst1q_f32(output_head + d + 4, vmulq_n_f32(acc1, inv_sum));
        }
        stats->output_ns += get_ns() - t0;
    }
}

int main(void) {
    int num_heads = 32, head_dim = 128;
    int seq_lens[] = {128, 512, 1024, 2048};
    
    printf("=== Single Implementation Instrumentation ===\n\n");
    
    for (int t = 0; t < 4; t++) {
        int seq_len = seq_lens[t];
        
        size_t kv_size = (size_t)seq_len * num_heads * head_dim * sizeof(ggml_fp16_t);
        float* q_f32 = aligned_alloc(64, num_heads * head_dim * sizeof(float));
        ggml_fp16_t* k = aligned_alloc(64, kv_size);
        ggml_fp16_t* v = aligned_alloc(64, kv_size);
        float* output = aligned_alloc(64, num_heads * head_dim * sizeof(float));
        void* work = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));
        ggml_fp16_t* q_f16_buf = aligned_alloc(64, head_dim * sizeof(ggml_fp16_t));
        
        for (int i = 0; i < num_heads * head_dim; i++) {
            q_f32[i] = (float)((rand() % 2000 - 1000) / 1000.0);
        }
        for (size_t i = 0; i < (size_t)seq_len * num_heads * head_dim; i++) {
            k[i] = (rand() % 1000) / 1000.0;
            v[i] = (rand() % 1000) / 1000.0;
        }
        
        float scale = 1.0f / sqrtf((float)head_dim);
        Stats stats;
        
        // Warmup
        our_attention(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf, &stats);
        
        // Measure
        our_attention(q_f32, k, v, output, num_heads, head_dim, seq_len, scale, work, q_f16_buf, &stats);
        
        printf("seq_len=%d:\n", seq_len);
        printf("  Q->FP16:    %10" PRIu64 " ns/head\n", stats.q_conv_ns/num_heads);
        printf("  Clear:      %10" PRIu64 " ns/head\n", stats.clear_ns/num_heads);
        printf("  vec_dot:    %10" PRIu64 " ns/call  (total: %" PRIu64 " us)\n", 
               stats.dot_ns/(num_heads*seq_len), stats.dot_ns/1000);
        printf("  softmax:    %10" PRIu64 " ns/call\n", stats.softmax_ns/(num_heads*seq_len));
        printf("  vec_scale:  %10" PRIu64 " ns/call  (count: %d)\n", 
               stats.scale_ns/(stats.scale_count > 0 ? stats.scale_count : 1), stats.scale_count);
        printf("  vec_mad:    %10" PRIu64 " ns/call  (total: %" PRIu64 " us)\n",
               stats.mad_ns/(num_heads*seq_len), stats.mad_ns/1000);
        printf("  Output:     %10" PRIu64 " ns/head\n", stats.output_ns/num_heads);
        printf("  TOTAL:      %10" PRIu64 " us\n\n", 
               (stats.q_conv_ns + stats.clear_ns + stats.dot_ns + 
                stats.softmax_ns + stats.scale_ns + stats.mad_ns + stats.output_ns)/1000);
        
        free(q_f32); free(k); free(v); free(output); free(work); free(q_f16_buf);
    }
    
    return 0;
}
