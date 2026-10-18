// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Self-written Flash Attention kernels from first principles
// Uses online softmax to avoid materializing full attention matrix
//
// SPDX-License-Identifier: AGPL-3.0
// SPDX-FileCopyrightText: Copyright (c) 2025 Jia Liu

#include <metal_stdlib>
using namespace metal;

struct FlashAttnArgs {
    uint32_t batch;
    uint32_t heads;
    uint32_t seq_q;
    uint32_t seq_k;
    uint32_t head_dim;
    float scale;
    uint32_t q_stride_b;
    uint32_t q_stride_h;
    uint32_t q_stride_s;
    uint32_t k_stride_b;
    uint32_t k_stride_h;
    uint32_t k_stride_s;
    uint32_t v_stride_b;
    uint32_t v_stride_h;
    uint32_t v_stride_s;
    uint32_t o_stride_b;
    uint32_t o_stride_h;
    uint32_t o_stride_s;
};

// ============================================================================
// Decode Flash Attention
// Uses 128 threads, each processes stride of KV positions
// Online softmax across all positions
// ============================================================================

kernel void flash_decode_attention_f32(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const float * k [[buffer(2)]],
    device const float * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h;
    device const float * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const float * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = ntg.x;

    // Per-thread online softmax
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] = 0.0f;
    }

    // Process all KV positions assigned to this thread
    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += num_threads) {
        // Q*K^T
        device const float * k_ptr = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;
        for (ushort d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        // Online softmax update
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim && d < 128; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        device const float * v_ptr = v_base + kv_idx * args.v_stride_s;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            local_out[d] += weight * v_ptr[d];
        }
    }

    // Reduce across threads in SIMD group
    float sg_max_val = simd_max(local_max);

    // Rescale each thread's values to SIMD max
    float scale_factor = exp(local_max - sg_max_val);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] *= scale_factor;
    }

    // Reduce within SIMD group
    float sg_sum_val = simd_sum(scaled_sum);
    float sg_out_val[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        sg_out_val[d] = simd_sum(local_out[d]);
    }

    // Reduce across SIMD groups
    threadgroup float tg_max[4];
    threadgroup float tg_sum[4];
    threadgroup float tg_out[128 * 4];

    // Initialize threadgroup memory (CRITICAL: Metal doesn't zero threadgroup memory)
    if (tiisg == 0) {
        tg_max[sgitg] = 0.0f;
        tg_sum[sgitg] = 0.0f;
        for (ushort d = 0; d < 128; d++) {
            tg_out[sgitg * 128 + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results
    if (tiisg == 0) {
        tg_max[sgitg] = sg_max_val;
        tg_sum[sgitg] = sg_sum_val;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            tg_out[sgitg * 128 + d] = sg_out_val[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine SIMD group results
    if (sgitg == 0 && tiisg == 0) {
        // Find global max
        float global_max = tg_max[0];
        for (ushort i = 1; i < 4; i++) {
            if (i * 32 < num_threads) {
                global_max = max(global_max, tg_max[i]);
            }
        }

        // Rescale and sum
        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort i = 0; i < 4; i++) {
            if (i * 32 < num_threads) {
                float scale_g = exp(tg_max[i] - global_max);
                global_sum += tg_sum[i] * scale_g;
                for (ushort d = 0; d < head_dim && d < 128; d++) {
                    global_out[d] += tg_out[i * 128 + d] * scale_g;
                }
            }
        }

        // Write output
        device float * o_ptr = output + b * args.o_stride_b + h * args.o_stride_h;
        float inv_sum = global_sum > 0.0f ? 1.0f / global_sum : 0.0f;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            o_ptr[d] = global_out[d] * inv_sum;
        }
    }
}

// ============================================================================
// Prefill Flash Attention
// Same algorithm, one threadgroup per query position
// ============================================================================

kernel void flash_prefill_attention_f32(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const float * k [[buffer(2)]],
    device const float * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const uint sq = tgpig.x;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    device const float * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h + sq * args.q_stride_s;
    device const float * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const float * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = ntg.x;

    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] = 0.0f;
    }

    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += num_threads) {
        device const float * k_ptr = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;
        for (ushort d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim && d < 128; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        device const float * v_ptr = v_base + kv_idx * args.v_stride_s;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            local_out[d] += weight * v_ptr[d];
        }
    }

    float sg_max_val = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max_val);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum_val = simd_sum(scaled_sum);
    float sg_out_val[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        sg_out_val[d] = simd_sum(local_out[d]);
    }

    threadgroup float tg_max[4];
    threadgroup float tg_sum[4];
    threadgroup float tg_out[128 * 4];

    // Initialize threadgroup memory (CRITICAL: Metal doesn't zero threadgroup memory)
    if (tiisg == 0) {
        tg_max[sgitg] = 0.0f;
        tg_sum[sgitg] = 0.0f;
        for (ushort d = 0; d < 128; d++) {
            tg_out[sgitg * 128 + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results
    if (tiisg == 0) {
        tg_max[sgitg] = sg_max_val;
        tg_sum[sgitg] = sg_sum_val;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            tg_out[sgitg * 128 + d] = sg_out_val[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float global_max = tg_max[0];
        for (ushort i = 1; i < 4; i++) {
            if (i * 32 < num_threads) {
                global_max = max(global_max, tg_max[i]);
            }
        }

        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort i = 0; i < 4; i++) {
            if (i * 32 < num_threads) {
                float scale_g = exp(tg_max[i] - global_max);
                global_sum += tg_sum[i] * scale_g;
                for (ushort d = 0; d < head_dim && d < 128; d++) {
                    global_out[d] += tg_out[i * 128 + d] * scale_g;
                }
            }
        }

        device float * o_ptr = output + b * args.o_stride_b + h * args.o_stride_h + sq * args.o_stride_s;
        float inv_sum = global_sum > 0.0f ? 1.0f / global_sum : 0.0f;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            o_ptr[d] = global_out[d] * inv_sum;
        }
    }
}

// ============================================================================
// F16 variants
// ============================================================================

kernel void flash_decode_attention_f16(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const half * q [[buffer(1)]],
    device const half * k [[buffer(2)]],
    device const half * v [[buffer(3)]],
    device half * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    device const half * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h;
    device const half * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const half * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = ntg.x;

    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] = 0.0f;
    }

    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += num_threads) {
        device const half * k_ptr = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;
        for (ushort d = 0; d < head_dim; d++) {
            score += float(q_ptr[d]) * float(k_ptr[d]);
        }
        score *= scale;

        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim && d < 128; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        device const half * v_ptr = v_base + kv_idx * args.v_stride_s;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            local_out[d] += weight * float(v_ptr[d]);
        }
    }

    float sg_max_val = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max_val);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum_val = simd_sum(scaled_sum);
    float sg_out_val[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        sg_out_val[d] = simd_sum(local_out[d]);
    }

    threadgroup float tg_max[4];
    threadgroup float tg_sum[4];
    threadgroup float tg_out[128 * 4];

    // Initialize threadgroup memory (CRITICAL: Metal doesn't zero threadgroup memory)
    if (tiisg == 0) {
        tg_max[sgitg] = 0.0f;
        tg_sum[sgitg] = 0.0f;
        for (ushort d = 0; d < 128; d++) {
            tg_out[sgitg * 128 + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results
    if (tiisg == 0) {
        tg_max[sgitg] = sg_max_val;
        tg_sum[sgitg] = sg_sum_val;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            tg_out[sgitg * 128 + d] = sg_out_val[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float global_max = tg_max[0];
        for (ushort i = 1; i < 4; i++) {
            if (i * 32 < num_threads) {
                global_max = max(global_max, tg_max[i]);
            }
        }

        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort i = 0; i < 4; i++) {
            if (i * 32 < num_threads) {
                float scale_g = exp(tg_max[i] - global_max);
                global_sum += tg_sum[i] * scale_g;
                for (ushort d = 0; d < head_dim && d < 128; d++) {
                    global_out[d] += tg_out[i * 128 + d] * scale_g;
                }
            }
        }

        device half * o_ptr = output + b * args.o_stride_b + h * args.o_stride_h;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            o_ptr[d] = half(global_out[d] / global_sum);
        }
    }
}

kernel void flash_prefill_attention_f16(
    constant FlashAttnArgs & args [[buffer(0)]],
    device const half * q [[buffer(1)]],
    device const half * k [[buffer(2)]],
    device const half * v [[buffer(3)]],
    device half * output [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint b = tgpig.z;
    const uint h = tgpig.y;
    const uint sq = tgpig.x;
    const ushort head_dim = args.head_dim;
    const float scale = args.scale;
    const uint seq_k = args.seq_k;

    device const half * q_ptr = q + b * args.q_stride_b + h * args.q_stride_h + sq * args.q_stride_s;
    device const half * k_base = k + b * args.k_stride_b + h * args.k_stride_h;
    device const half * v_base = v + b * args.v_stride_b + h * args.v_stride_h;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = ntg.x;

    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] = 0.0f;
    }

    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += num_threads) {
        device const half * k_ptr = k_base + kv_idx * args.k_stride_s;
        float score = 0.0f;
        for (ushort d = 0; d < head_dim; d++) {
            score += float(q_ptr[d]) * float(k_ptr[d]);
        }
        score *= scale;

        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < head_dim && d < 128; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        device const half * v_ptr = v_base + kv_idx * args.v_stride_s;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            local_out[d] += weight * float(v_ptr[d]);
        }
    }

    float sg_max_val = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max_val);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum_val = simd_sum(scaled_sum);
    float sg_out_val[128];
    for (ushort d = 0; d < head_dim && d < 128; d++) {
        sg_out_val[d] = simd_sum(local_out[d]);
    }

    threadgroup float tg_max[4];
    threadgroup float tg_sum[4];
    threadgroup float tg_out[128 * 4];

    // Initialize threadgroup memory (CRITICAL: Metal doesn't zero threadgroup memory)
    if (tiisg == 0) {
        tg_max[sgitg] = 0.0f;
        tg_sum[sgitg] = 0.0f;
        for (ushort d = 0; d < 128; d++) {
            tg_out[sgitg * 128 + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results
    if (tiisg == 0) {
        tg_max[sgitg] = sg_max_val;
        tg_sum[sgitg] = sg_sum_val;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            tg_out[sgitg * 128 + d] = sg_out_val[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float global_max = tg_max[0];
        for (ushort i = 1; i < 4; i++) {
            if (i * 32 < num_threads) {
                global_max = max(global_max, tg_max[i]);
            }
        }

        float global_sum = 0.0f;
        float global_out[128];
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            global_out[d] = 0.0f;
        }

        for (ushort i = 0; i < 4; i++) {
            if (i * 32 < num_threads) {
                float scale_g = exp(tg_max[i] - global_max);
                global_sum += tg_sum[i] * scale_g;
                for (ushort d = 0; d < head_dim && d < 128; d++) {
                    global_out[d] += tg_out[i * 128 + d] * scale_g;
                }
            }
        }

        device half * o_ptr = output + b * args.o_stride_b + h * args.o_stride_h + sq * args.o_stride_s;
        for (ushort d = 0; d < head_dim && d < 128; d++) {
            o_ptr[d] = half(global_out[d] / global_sum);
        }
    }
}
