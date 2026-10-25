// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// =============================================================================
// STATUS: EXPERIMENTAL - Batched prefill for server mode optimization
// =============================================================================
// Batched Flash Attention for Prefill
// Architecture: matches llama.cpp's approach
// - One threadgroup per 8 queries
// - KV loop is SEQUENTIAL within threadgroup (avoids load imbalance)
// - All threads collaborate on same KV chunk

#include <metal_stdlib>
using namespace metal;

struct FlashPrefillArgs {
    uint32_t batch;
    uint32_t n_heads;
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
    uint32_t start_pos;
};

kernel void kernel_flash_prefill_batch(
    constant FlashPrefillArgs& args [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const half* k [[buffer(2)]],
    device const half* v [[buffer(3)]],
    device float* output [[buffer(4)]],
    threadgroup half* shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint batch_idx = tgpig.z;
    const uint head_idx = tgpig.y;
    const uint q_block = tgpig.x;

    const ushort head_dim = args.head_dim;
    const float scale = args.scale;

    const ushort q_start = q_block * 8;
    const ushort local_q = sgitg * 2;

    threadgroup half*  sq = shmem;
    threadgroup half*  so = shmem + 8 * head_dim;
    threadgroup float* ss = (threadgroup float*)(so + 8 * head_dim);

    device const float* q_base = q + batch_idx * args.q_stride_b + head_idx * args.q_stride_h;

    for (ushort q_idx = 0; q_idx < 8; ++q_idx) {
        const ushort global_q = q_start + q_idx;
        if (global_q >= args.seq_q) break;

        device const float* q_row = q_base + global_q * args.q_stride_s;

        for (ushort d = tiisg + sgitg * 32; d < head_dim; d += 128) {
            sq[q_idx * head_dim + d] = (half)q_row[d];
        }
    }

    for (ushort q_idx = 0; q_idx < 8; ++q_idx) {
        for (ushort d = tiisg + sgitg * 32; d < head_dim; d += 128) {
            so[q_idx * head_dim + d] = (half)0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M[2] = { -FLT_MAX/2, -FLT_MAX/2 };
    float S[2] = { 0.0f, 0.0f };

    device const half* k_base = k + batch_idx * args.k_stride_b + head_idx * args.k_stride_h;
    device const half* v_base = v + batch_idx * args.v_stride_b + head_idx * args.v_stride_h;

    const uint total_kv = args.start_pos + args.seq_q;

    for (uint kv_chunk = 0; kv_chunk < total_kv; kv_chunk += 64) {
        const uint kv_start = kv_chunk;
        const uint kv_end = min(kv_chunk + 64, total_kv);

        for (ushort q_idx = 0; q_idx < 2; ++q_idx) {
            for (ushort c = tiisg; c < 64; c += 32) {
                ss[(local_q + q_idx) * 64 + c] = -FLT_MAX/2;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort q_idx = 0; q_idx < 2; ++q_idx) {
            const ushort global_q = q_start + local_q + q_idx;
            if (global_q >= args.seq_q) continue;

            const uint max_kv_for_q = args.start_pos + global_q;

            for (uint kv_idx = kv_start + tiisg; kv_idx < kv_end; kv_idx += 32) {
                if (kv_idx > max_kv_for_q) {
                    ss[(local_q + q_idx) * 64 + (kv_idx - kv_start)] = -FLT_MAX/2;
                    continue;
                }

                device const half* k_row = k_base + kv_idx * args.k_stride_s;

                float score = 0.0f;
                for (ushort d = 0; d < head_dim; ++d) {
                    score += (float)sq[(local_q + q_idx) * head_dim + d] * (float)k_row[d];
                }
                score *= scale;

                ss[(local_q + q_idx) * 64 + (kv_idx - kv_start)] = score;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort q_idx = 0; q_idx < 2; ++q_idx) {
            const ushort global_q = q_start + local_q + q_idx;
            if (global_q >= args.seq_q) continue;

            float chunk_max = -FLT_MAX/2;
            for (ushort c = tiisg; c < 64; c += 32) {
                chunk_max = max(chunk_max, ss[(local_q + q_idx) * 64 + c]);
            }
            chunk_max = simd_max(chunk_max);

            float old_max = M[q_idx];
            M[q_idx] = max(old_max, chunk_max);

            float chunk_sum = 0.0f;
            for (ushort c = tiisg; c < 64; c += 32) {
                float s = ss[(local_q + q_idx) * 64 + c];
                if (s > -FLT_MAX/4) {
                    float exp_s = exp(s - M[q_idx]);
                    ss[(local_q + q_idx) * 64 + c] = exp_s;
                    chunk_sum += exp_s;
                }
            }
            chunk_sum = simd_sum(chunk_sum);

            float scale_old = exp(old_max - M[q_idx]);
            S[q_idx] = S[q_idx] * scale_old + chunk_sum;

            for (ushort d = tiisg; d < head_dim; d += 32) {
                so[(local_q + q_idx) * head_dim + d] *= scale_old;
            }

            for (ushort c = tiisg; c < 64; c += 32) {
                const uint kv_idx = kv_start + c;
                if (kv_idx > args.start_pos + global_q) continue;

                float weight = ss[(local_q + q_idx) * 64 + c];
                if (weight <= 0.0f) continue;

                device const half* v_row = v_base + kv_idx * args.v_stride_s;

                // Each thread handles a subset of head_dim
                for (ushort d = tiisg; d < head_dim; d += 32) {
                    so[(local_q + q_idx) * head_dim + d] += weight * (float)v_row[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (ushort q_idx = 0; q_idx < 2; ++q_idx) {
        const ushort global_q = q_start + local_q + q_idx;
        if (global_q >= args.seq_q) continue;

        device float* o_row = output + batch_idx * args.o_stride_b
                             + head_idx * args.o_stride_h
                             + global_q * args.o_stride_s;

        for (ushort d = tiisg; d < head_dim; d += 32) {
            o_row[d] = (float)so[(local_q + q_idx) * head_dim + d] / S[q_idx];
        }
    }
}
