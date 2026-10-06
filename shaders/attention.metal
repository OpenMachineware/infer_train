// Self-written Attention kernels from first principles
// NOT copied from llama.cpp - written from understanding of the algorithm
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// ATTENTION FROM FIRST PRINCIPLES
// ============================================================================
//
// Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V
//
// Where:
// - Q: query tensor [batch, heads, seq_q, head_dim]
// - K: key tensor   [batch, heads, seq_k, head_dim]
// - V: value tensor [batch, heads, seq_k, head_dim]
// - scale: 1/sqrt(head_dim) for numerical stability
//
// Flash Attention optimization:
// - Don't materialize the full (seq_q, seq_k) attention matrix
// - Compute online softmax incrementally
// - Track: max value (M), sum of exp (S), weighted output (O)
//
// ============================================================================

// Argument structure for attention kernels
struct AttentionArgs {
    uint32_t batch;       // batch size
    uint32_t heads;       // number of attention heads
    uint32_t seq_q;       // query sequence length
    uint32_t seq_k;       // key/value sequence length
    uint32_t head_dim;    // head dimension
    float scale;          // 1/sqrt(head_dim)
    uint32_t q_stride_b;  // query stride for batch
    uint32_t q_stride_h;  // query stride for head
    uint32_t q_stride_s;  // query stride for sequence
    uint32_t k_stride_b;  // key stride for batch
    uint32_t k_stride_h;  // key stride for head
    uint32_t k_stride_s;  // key stride for sequence
    uint32_t v_stride_b;  // value stride for batch
    uint32_t v_stride_h;  // value stride for head
    uint32_t v_stride_s;  // value stride for sequence
    uint32_t o_stride_b;  // output stride for batch
    uint32_t o_stride_h;  // output stride for head
    uint32_t o_stride_s;  // output stride for sequence
};

// ============================================================================
// DECODE ATTENTION (seq_q = 1, small batch)
// One query against all cached KV
// Uses online softmax algorithm - no need to store all scores
// ============================================================================

kernel void decode_attention_f32(
    constant AttentionArgs& args [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one (batch, head) pair
    const uint batch_idx = tgpig.x;
    const uint head_idx = tgpig.y;

    // Get pointers for this batch and head
    device const float* q_ptr = q + batch_idx * args.q_stride_b + head_idx * args.q_stride_h;
    device const float* k_ptr = k + batch_idx * args.k_stride_b + head_idx * args.k_stride_h;
    device const float* v_ptr = v + batch_idx * args.v_stride_b + head_idx * args.v_stride_h;
    device float* o_ptr = out + batch_idx * args.o_stride_b + head_idx * args.o_stride_h;

    const uint head_dim = args.head_dim;
    const uint seq_k = args.seq_k;
    const float scale = args.scale;

    const uint tid = tpitg.x;
    const uint block_size = ntg.x;

    // Shared memory for query only
    threadgroup float shared_q[128];

    // Load query into shared memory
    for (uint i = tid; i < head_dim; i += block_size) {
        shared_q[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax: track max, sum of exp, and weighted output
    // Each thread maintains its own state
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float local_output[128];
    for (uint i = 0; i < head_dim; i++) {
        local_output[i] = 0.0f;
    }

    // Count how many positions this thread processes
    uint count = 0;

    // Process KV positions assigned to this thread
    // Online softmax update for each position
    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += block_size) {
        // Compute Q @ K^T
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += shared_q[d] * k_ptr[kv_idx * args.k_stride_s + d];
        }
        score *= scale;

        // Online softmax update
        float new_max = max(max_score, score);
        float rescale_old = exp(max_score - new_max);
        float exp_new = exp(score - new_max);

        sum_exp = sum_exp * rescale_old + exp_new;
        for (uint d = 0; d < head_dim; d++) {
            local_output[d] = local_output[d] * rescale_old + exp_new * v_ptr[kv_idx * args.v_stride_s + d];
        }
        max_score = new_max;
        count++;
    }

    // If this thread processed no positions, don't contribute to reduction
    // Set sum_exp to 0 (it already should be, but be explicit)
    if (count == 0) {
        sum_exp = 0.0f;
        // max_score is still -FLT_MAX, but that's fine for simd_max
    }

    // First, find the global max across all threads
    float global_max = simd_max(max_score);

    // Rescale each thread's sum_exp and output
    // Threads with count == 0 will contribute 0 after rescale (exp(-FLT_MAX - global_max) ≈ 0)
    float rescale = exp(max_score - global_max);
    sum_exp *= rescale;
    for (uint d = 0; d < head_dim; d++) {
        local_output[d] *= rescale;
    }

    // Reduce sum_exp
    float global_sum = simd_sum(sum_exp);

    // Reduce output - each dimension separately
    for (uint d = 0; d < head_dim; d++) {
        local_output[d] = simd_sum(local_output[d]);
    }

    // Use threadgroup memory to combine results from multiple SIMD groups
    threadgroup float tg_sum[2];  // Max 2 SIMD groups for 64 threads
    threadgroup float tg_output[2 * 128];  // 2 SIMD groups * max head_dim

    // Each SIMD group's thread 0 writes its partial result
    if (tiisg == 0) {
        tg_sum[sgitg] = global_sum;
        for (uint i = 0; i < head_dim; i++) {
            tg_output[sgitg * 128 + i] = local_output[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Only first thread of first SIMD group computes final result and writes
    if (tiisg == 0 && sgitg == 0) {
        // Sum across SIMD groups
        float total_sum = 0.0f;
        float total_output[128];
        for (uint i = 0; i < head_dim; i++) {
            total_output[i] = 0.0f;
        }

        // Count active SIMD groups (those with valid data)
        uint n_sg = (block_size + 31) / 32;  // Ceiling(block_size / 32)
        for (uint sg = 0; sg < n_sg; sg++) {
            total_sum += tg_sum[sg];
            for (uint i = 0; i < head_dim; i++) {
                total_output[i] += tg_output[sg * 128 + i];
            }
        }

        // Write final result
        for (uint i = 0; i < head_dim; i++) {
            o_ptr[i] = total_output[i] / total_sum;
        }
    }
}

// ============================================================================
// PREFILL ATTENTION (seq_q > 1)
// Multiple queries against all KV
// ============================================================================

kernel void prefill_attention_f32(
    constant AttentionArgs& args [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one (batch, head, query_position) tuple
    const uint batch_idx = tgpig.x / args.heads;
    const uint head_idx = tgpig.x % args.heads;
    const uint q_pos = tgpig.y;

    if (q_pos >= args.seq_q) {
        return;
    }

    // Get pointers
    device const float* q_ptr = q + batch_idx * args.q_stride_b + head_idx * args.q_stride_h + q_pos * args.q_stride_s;
    device const float* k_ptr = k + batch_idx * args.k_stride_b + head_idx * args.k_stride_h;
    device const float* v_ptr = v + batch_idx * args.v_stride_b + head_idx * args.v_stride_h;
    device float* o_ptr = out + batch_idx * args.o_stride_b + head_idx * args.o_stride_h + q_pos * args.o_stride_s;

    const uint head_dim = args.head_dim;
    const float scale = args.scale;
    const uint tid = tpitg.x;
    const uint block_size = ntg.x;

    // Shared memory for query and scores
    threadgroup float shared_q[128];
    threadgroup float shared_scores[256];  // Max seq_k

    // Load query
    for (uint i = tid; i < head_dim; i += block_size) {
        shared_q[i] = q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Compute all attention scores
    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += shared_q[d] * k_ptr[kv_idx * args.k_stride_s + d];
        }
        shared_scores[kv_idx] = score * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Find max
    float local_max = -FLT_MAX;
    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        local_max = max(local_max, shared_scores[kv_idx]);
    }
    float global_max = simd_max(local_max);

    // Phase 3: Compute exp sum
    float local_sum = 0.0f;
    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        float exp_score = exp(shared_scores[kv_idx] - global_max);
        shared_scores[kv_idx] = exp_score;
        local_sum += exp_score;
    }

    // Combine sums from all SIMD groups
    float global_sum = simd_sum(local_sum);

    // Phase 4: Accumulate weighted values
    float local_output[128];
    for (uint i = 0; i < head_dim; i++) {
        local_output[i] = 0.0f;
    }

    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        float weight = shared_scores[kv_idx];
        for (uint d = 0; d < head_dim; d++) {
            local_output[d] += weight * v_ptr[kv_idx * args.v_stride_s + d];
        }
    }

    // Reduce output across threads within SIMD group
    for (uint d = 0; d < head_dim; d++) {
        local_output[d] = simd_sum(local_output[d]);
    }

    // Use threadgroup memory to combine results from multiple SIMD groups
    threadgroup float tg_sum[2];
    threadgroup float tg_output[2 * 128];

    if (tiisg == 0) {
        tg_sum[sgitg] = global_sum;
        for (uint i = 0; i < head_dim; i++) {
            tg_output[sgitg * 128 + i] = local_output[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Only first thread of first SIMD group computes final result
    if (tiisg == 0 && sgitg == 0) {
        float total_sum = 0.0f;
        float total_output[128];
        for (uint i = 0; i < head_dim; i++) {
            total_output[i] = 0.0f;
        }

        uint n_sg = (block_size + 31) / 32;
        for (uint sg = 0; sg < n_sg; sg++) {
            total_sum += tg_sum[sg];
            for (uint i = 0; i < head_dim; i++) {
                total_output[i] += tg_output[sg * 128 + i];
            }
        }

        // Normalize and write
        for (uint i = 0; i < head_dim; i++) {
            o_ptr[i] = total_output[i] / total_sum;
        }
    }
}

// ============================================================================
// F16 VARIANTS
// ============================================================================

kernel void decode_attention_f16(
    constant AttentionArgs& args [[buffer(0)]],
    device const half* q [[buffer(1)]],
    device const half* k [[buffer(2)]],
    device const half* v [[buffer(3)]],
    device half* out [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]]
) {
    const uint batch_idx = tgpig.x;
    const uint head_idx = tgpig.y;

    device const half* q_ptr = q + batch_idx * args.q_stride_b + head_idx * args.q_stride_h;
    device const half* k_ptr = k + batch_idx * args.k_stride_b + head_idx * args.k_stride_h;
    device const half* v_ptr = v + batch_idx * args.v_stride_b + head_idx * args.v_stride_h;
    device half* o_ptr = out + batch_idx * args.o_stride_b + head_idx * args.o_stride_h;

    const uint head_dim = args.head_dim;
    const uint seq_k = args.seq_k;
    const float scale = args.scale;
    const uint tid = tpitg.x;
    const uint block_size = ntg.x;

    threadgroup float shared_q[128];

    for (uint i = tid; i < head_dim; i += block_size) {
        shared_q[i] = (float)q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float local_output[128];
    for (uint i = 0; i < head_dim; i++) {
        local_output[i] = 0.0f;
    }

    for (uint kv_idx = tid; kv_idx < seq_k; kv_idx += block_size) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += shared_q[d] * (float)k_ptr[kv_idx * args.k_stride_s + d];
        }
        score *= scale;

        float new_max = max(max_score, score);
        float rescale_old = exp(max_score - new_max);
        float exp_new = exp(score - new_max);

        sum_exp = sum_exp * rescale_old + exp_new;
        for (uint d = 0; d < head_dim; d++) {
            local_output[d] = local_output[d] * rescale_old + exp_new * (float)v_ptr[kv_idx * args.v_stride_s + d];
        }
        max_score = new_max;
    }

    float global_max = simd_max(max_score);

    float rescale = exp(max_score - global_max);
    sum_exp *= rescale;
    for (uint d = 0; d < head_dim; d++) {
        local_output[d] *= rescale;
    }

    float global_sum = simd_sum(sum_exp);

    for (uint d = 0; d < head_dim; d++) {
        float reduced = simd_sum(local_output[d]);
        if (tid == 0) {
            o_ptr[d] = (half)(reduced / global_sum);
        }
    }
}

kernel void prefill_attention_f16(
    constant AttentionArgs& args [[buffer(0)]],
    device const half* q [[buffer(1)]],
    device const half* k [[buffer(2)]],
    device const half* v [[buffer(3)]],
    device half* out [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]]
) {
    const uint batch_idx = tgpig.x / args.heads;
    const uint head_idx = tgpig.x % args.heads;
    const uint q_pos = tgpig.y;

    if (q_pos >= args.seq_q) {
        return;
    }

    device const half* q_ptr = q + batch_idx * args.q_stride_b + head_idx * args.q_stride_h + q_pos * args.q_stride_s;
    device const half* k_ptr = k + batch_idx * args.k_stride_b + head_idx * args.k_stride_h;
    device const half* v_ptr = v + batch_idx * args.v_stride_b + head_idx * args.v_stride_h;
    device half* o_ptr = out + batch_idx * args.o_stride_b + head_idx * args.o_stride_h + q_pos * args.o_stride_s;

    const uint head_dim = args.head_dim;
    const float scale = args.scale;
    const uint tid = tpitg.x;
    const uint block_size = ntg.x;

    threadgroup float shared_q[128];
    threadgroup float shared_scores[256];

    for (uint i = tid; i < head_dim; i += block_size) {
        shared_q[i] = (float)q_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += shared_q[d] * (float)k_ptr[kv_idx * args.k_stride_s + d];
        }
        shared_scores[kv_idx] = score * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -FLT_MAX;
    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        local_max = max(local_max, shared_scores[kv_idx]);
    }
    float global_max = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        float exp_score = exp(shared_scores[kv_idx] - global_max);
        shared_scores[kv_idx] = exp_score;
        local_sum += exp_score;
    }
    float global_sum = simd_sum(local_sum);

    float local_output[128];
    for (uint i = 0; i < head_dim; i++) {
        local_output[i] = 0.0f;
    }

    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += block_size) {
        float weight = shared_scores[kv_idx] / global_sum;
        for (uint d = 0; d < head_dim; d++) {
            local_output[d] += weight * (float)v_ptr[kv_idx * args.v_stride_s + d];
        }
    }

    for (uint d = 0; d < head_dim; d++) {
        local_output[d] = simd_sum(local_output[d]);
    }

    if (tid == 0) {
        for (uint i = 0; i < head_dim; i++) {
            o_ptr[i] = (half)local_output[i];
        }
    }
}
