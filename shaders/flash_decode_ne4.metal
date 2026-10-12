// Flash Decode Kernel - Correct NE=4 Implementation
//
// Key insight: NE=4 means each thread processes 4 cache items in parallel
// - 32 threads / NE=4 = 8 parallel cache items per SIMD operation
// - C=32 cache items processed per iteration via 4 NE-operations
// - Each thread maintains 4 parallel scores (one per cache item)
//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Block structures
struct Q4_0Block { half d; uint8_t qs[16]; };
struct Q4_1Block { half d; half m; uint8_t qs[16]; };
struct Q5_0Block { half d; uint8_t qh[4]; uint8_t qs[16]; };
struct Q5_1Block { half d; half m; uint8_t qh[4]; uint8_t qs[16]; };
struct Q8_0Block { half d; int8_t qs[32]; };
struct F16Block { half values[4]; };

// Dequantization - returns 4 values from a block at a given lane
template<typename T4>
inline void dequant_q4_0(device const Q4_0Block* block, int lane, thread T4& result) {
    const half d = block->d;
    const int group = lane % 8;
    const int is_high = group / 4;
    const int idx = group % 4;
    device const uint16_t* qs16 = (device const uint16_t*)block->qs;
    const half scale = is_high ? d / 16.0h : d;
    const half offset = -8.0h * d;
    const uint16_t mask = is_high ? 0x00F0 : 0x000F;
    for (int i = 0; i < 2; i++) {
        uint16_t val = qs16[2 * idx + i];
        result[2*i + 0] = scale * half(val & mask) + offset;
        result[2*i + 1] = (scale / 256.0h) * half((val >> 8) & mask) + offset;
    }
}

template<typename T4>
inline void dequant_q4_1(device const Q4_1Block* block, int lane, thread T4& result) {
    const half d = block->d;
    const half m = block->m;
    const int group = lane % 8;
    const int is_high = group / 4;
    const int idx = group % 4;
    device const uint16_t* qs16 = (device const uint16_t*)block->qs;
    const half scale = is_high ? d / 16.0h : d;
    const uint16_t mask = is_high ? 0x00F0 : 0x000F;
    for (int i = 0; i < 2; i++) {
        uint16_t val = qs16[2 * idx + i];
        result[2*i + 0] = scale * half(val & mask) + m;
        result[2*i + 1] = (scale / 256.0h) * half((val >> 8) & mask) + m;
    }
}

template<typename T4>
inline void dequant_q5_0(device const Q5_0Block* block, int lane, thread T4& result) {
    const half d = block->d;
    const half md = -16.0h * d;
    const int group = lane % 8;
    const int is_high = group / 4;
    const int idx = group % 4;
    device const uint16_t* qs16 = (device const uint16_t*)block->qs;
    const uint32_t qh = *((device const uint32_t*)block->qh);
    const int x_mv = is_high ? 4 : 0;
    const int gh_mv = is_high ? 12 : 0;
    const int gh_bk = is_high ? 0 : 4;
    for (int i = 0; i < 2; i++) {
        int j = 2 * idx + i;
        uint8_t xh_0 = ((qh >> (gh_mv + 2*j)) << gh_bk) & 0x10;
        uint8_t xh_1 = ((qh >> (gh_mv + 2*j+1)) << gh_bk) & 0x10;
        int32_t x0 = ((qs16[j] & 0xFF) & (is_high ? 0xF0 : 0x0F)) >> x_mv | xh_0;
        int32_t x1 = ((qs16[j] >> 8) & (is_high ? 0xF0 : 0x0F)) >> x_mv | xh_1;
        result[2*i + 0] = d * x0 + md;
        result[2*i + 1] = d * x1 + md;
    }
}

template<typename T4>
inline void dequant_q5_1(device const Q5_1Block* block, int lane, thread T4& result) {
    const half d = block->d;
    const half m = block->m;
    const int group = lane % 8;
    const int is_high = group / 4;
    const int idx = group % 4;
    device const uint16_t* qs16 = (device const uint16_t*)block->qs;
    const uint32_t qh = *((device const uint32_t*)block->qh);
    const int x_mv = is_high ? 4 : 0;
    const int gh_mv = is_high ? 12 : 0;
    const int gh_bk = is_high ? 0 : 4;
    for (int i = 0; i < 2; i++) {
        int j = 2 * idx + i;
        uint8_t xh_0 = ((qh >> (gh_mv + 2*j)) << gh_bk) & 0x10;
        uint8_t xh_1 = ((qh >> (gh_mv + 2*j+1)) << gh_bk) & 0x10;
        int32_t x0 = ((qs16[j] & 0xFF) & (is_high ? 0xF0 : 0x0F)) >> x_mv | xh_0;
        int32_t x1 = ((qs16[j] >> 8) & (is_high ? 0xF0 : 0x0F)) >> x_mv | xh_1;
        result[2*i + 0] = d * x0 + m;
        result[2*i + 1] = d * x1 + m;
    }
}

template<typename T4>
inline void dequant_q8_0(device const Q8_0Block* block, int lane, thread T4& result) {
    const half d = block->d;
    const int idx = lane % 8;
    for (int i = 0; i < 4; i++) {
        result[i] = d * (half)block->qs[4 * idx + i];
    }
}

template<typename T4>
inline void dequant_f16(device const F16Block* block, int lane, thread T4& result) {
    for (int i = 0; i < 4; i++) {
        result[i] = block->values[i];
    }
}

// Main kernel - NE=4 parallel processing
template<
    typename KBlock, typename VBlock, typename T4,
    int DEQUANT_LANES,
    void (*DEQUANT_K)(device const KBlock*, int, thread T4&),
    void (*DEQUANT_V)(device const VBlock*, int, thread T4&),
    int HEAD_DIM, int NE, int C, int ELEM_SIZE
>
kernel void flash_decode_ne4(
    constant uint& seq_k [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device const char* key_cache [[buffer(3)]],
    device const char* value_cache [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    constexpr int NL = 32 / NE;  // 8
    constexpr int NB = HEAD_DIM / (4 * NL);  // 4

    const int tx = tiisg % NL;  // 0-7: dimension threads
    const int ty = tiisg / NL;  // 0-3: which of NE groups

    const int head = gid.y;

    // Load query into threadgroup memory
    threadgroup float4 s_query[32];
    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)(query + head * HEAD_DIM);
        s_query[tiisg] = q_ptr[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state - one per NE cache item
    float M[NE];
    float S[NE];
    float4 O[NE][NB];

    #pragma unroll
    for (int ne = 0; ne < NE; ne++) {
        M[ne] = -INFINITY;
        S[ne] = 0.0f;
        for (int i = 0; i < NB; i++) {
            O[ne][i] = float4(0.0f);
        }
    }

    // KV cache pointers
    device const char* k_base = key_cache + head * seq_k * HEAD_DIM * ELEM_SIZE;
    device const char* v_base = value_cache + head * seq_k * HEAD_DIM * ELEM_SIZE;

    // Process C cache items in C/NE iterations
    // Each iteration processes NE cache items in parallel
    const int nb_iterations = (seq_k + C - 1) / C;

    for (int iter = 0; iter < nb_iterations; iter++) {
        const int ic = iter * C;  // Base cache index for this iteration

        // Process NE cache items in parallel (ne=0..NE-1)
        float mqk[NE];

        #pragma unroll
        for (int ne = 0; ne < NE; ne++) {
            const int cache_idx = ic + ty * NE + ne;
            mqk[ne] = 0.0f;

            if (cache_idx < seq_k) {
                // Dequantize K values for this cache item
                device const KBlock* k_block = (device const KBlock*)
                    (k_base + cache_idx * HEAD_DIM * ELEM_SIZE);

                for (int i = 0; i < NB; i++) {
                    T4 k_vals;
                    DEQUANT_K(k_block, i * NL + tx, k_vals);
                    mqk[ne] += dot((float4)k_vals, s_query[i * NL + tx]);
                }
            }
        }

        // Tree reduction across tx dimension
        #pragma unroll
        for (int ne = 0; ne < NE; ne++) {
            mqk[ne] += simd_shuffle_down(mqk[ne], 4);
            mqk[ne] += simd_shuffle_down(mqk[ne], 2);
            mqk[ne] += simd_shuffle_down(mqk[ne], 1);
            mqk[ne] = simd_shuffle(mqk[ne], NL * ty);  // Broadcast from thread 0 of this ty group
        }

        // Store scores for this ty group (ty threads, each with NE scores)
        threadgroup float s_scores[C];

        #pragma unroll
        for (int ne = 0; ne < NE; ne++) {
            const int score_idx = ty * NE + ne;
            const int cache_idx = ic + score_idx;
            if (cache_idx < seq_k) {
                s_scores[score_idx] = mqk[ne] * scale;
            } else {
                s_scores[score_idx] = -INFINITY;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update
        const float s = s_scores[tiisg];
        const float prev_m = M[tiisg % NE];
        const float m_new = simd_max(max(prev_m, s));
        M[tiisg % NE] = simd_broadcast(m_new, 0);

        const float m_diff = prev_m - M[tiisg % NE];
        const float s_exp = exp(s - M[tiisg % NE]);

        S[tiisg % NE] = S[tiisg % NE] * exp(m_diff) + simd_sum(s_exp);

        // Scale previous output
        const float scale_prev = exp(m_diff);
        #pragma unroll
        for (int i = 0; i < NB; i++) {
            O[tiisg % NE][i] *= scale_prev;
        }

        s_scores[tiisg] = s_exp;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation
        const int v_idx = ic + tiisg;
        if (v_idx < seq_k) {
            const float weight = s_scores[tiisg];
            device const VBlock* v_block = (device const VBlock*)
                (v_base + v_idx * HEAD_DIM * ELEM_SIZE);

            for (int i = 0; i < NB; i++) {
                T4 v_vals;
                DEQUANT_V(v_block, i * NL + tx, v_vals);
                O[tiisg % NE][i] += weight * (float4)v_vals;
            }
        }
    }

    // Final normalize and write output
    const float inv_s = 1.0f / S[tiisg % NE];

    #pragma unroll
    for (int i = 0; i < NB; i++) {
        O[tiisg % NE][i] *= inv_s;
    }

    if (tiisg < HEAD_DIM / 4) {
        device float4* out_ptr = (device float4*)(output + head * HEAD_DIM);
        out_ptr[tiisg] = O[tiisg % NE][tiisg / NL];
    }
}

// Template instantiations
template [[host_name("flash_decode_ne4_f16")]]
kernel void flash_decode_ne4<F16Block, F16Block, float4, 8, dequant_f16, dequant_f16, 128, 4, 32, 2>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort);

template [[host_name("flash_decode_ne4_q4_0")]]
kernel void flash_decode_ne4<Q4_0Block, Q4_0Block, float4, 8, dequant_q4_0, dequant_q4_0, 128, 4, 32, 1>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort);

template [[host_name("flash_decode_ne4_q4_1")]]
kernel void flash_decode_ne4<Q4_1Block, Q4_1Block, float4, 8, dequant_q4_1, dequant_q4_1, 128, 4, 32, 1>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort);

template [[host_name("flash_decode_ne4_q5_0")]]
kernel void flash_decode_ne4<Q5_0Block, Q5_0Block, float4, 8, dequant_q5_0, dequant_q5_0, 128, 4, 32, 1>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort);

template [[host_name("flash_decode_ne4_q5_1")]]
kernel void flash_decode_ne4<Q5_1Block, Q5_1Block, float4, 8, dequant_q5_1, dequant_q5_1, 128, 4, 32, 1>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort);

template [[host_name("flash_decode_ne4_q8_0")]]
kernel void flash_decode_ne4<Q8_0Block, Q8_0Block, float4, 8, dequant_q8_0, dequant_q8_0, 128, 4, 32, 1>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort);
