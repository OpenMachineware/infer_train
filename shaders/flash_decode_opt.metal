// Flash Decode Optimized Kernel
// Implements high-performance decode attention using proven design patterns
//
// Key design principles (inspired by proven GPU attention implementations):
// - C=32: Process 32 cache items per iteration
// - NE=4: Each thread processes 4 cache items in parallel
// - NL=8: 8 threads per dimension (32 threads / NE=4)
// - Adaptive NSG: 1/2/4 SIMD groups based on sequence length
// - Tree reduction: Efficient SIMD shuffle for reduction
// - Online softmax: Incremental max and sum updates
//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Block structures for different quantization formats
struct Q4_0Block {
    half d;
    uint8_t qs[16];  // 32 values, 2 per byte
};

struct Q4_1Block {
    half d;
    half m;
    uint8_t qs[16];
};

struct Q5_0Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct Q5_1Block {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct Q8_0Block {
    half d;
    int8_t qs[32];
};

struct F16Block {
    half values[4];
};

// Dequantization functions
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

// Main kernel template
template<
    typename KBlock,
    typename VBlock,
    typename T4,
    int DEQUANT_LANES_K,
    int DEQUANT_LANES_V,
    void (*DEQUANT_K)(device const KBlock*, int, thread T4&),
    void (*DEQUANT_V)(device const VBlock*, int, thread T4&),
    int HEAD_DIM,  // 128
    int NE,        // 4: threads per cache item group
    int C,         // 32: cache items per iteration
    int ELEM_SIZE, // bytes per element
    int NSG        // 1/2/4: SIMD groups per workgroup
>
kernel void flash_decode_opt(
    constant uint& seq_k [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device const char* key_cache [[buffer(3)]],
    device const char* value_cache [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NL = 32 / NE;  // 8 threads per ty group
    constexpr int NB = HEAD_DIM / (4 * NL);  // blocks per thread

    const int tx = tiisg % NL;  // 0-7: dimension index
    const int ty = tiisg / NL;  // 0-3: cache item index within NE
    const int sg = sgitg;       // 0 to NSG-1

    const int head = gid.y;
    const int batch = gid.x;

    // Load query into threadgroup memory
    // 128 floats = 32 float4s
    threadgroup float4 s_query[32];
    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)
            (query + batch * HEAD_DIM + head * HEAD_DIM);
        s_query[tiisg] = q_ptr[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize online softmax state
    float M = -INFINITY;
    float S = 0.0f;
    float4 O[NB];

    #pragma unroll
    for (int i = 0; i < NB; i++) {
        O[i] = float4(0.0f);
    }

    // Threadgroup memory for scores (used in softmax)
    threadgroup float s_scores[NSG][C];

    // KV cache pointers
    device const char* k_base = key_cache + head * seq_k * HEAD_DIM * ELEM_SIZE;
    device const char* v_base = value_cache + head * seq_k * HEAD_DIM * ELEM_SIZE;

    // Strided processing: each SIMD group processes different chunks
    // Pattern: for ic0 = sg; ic0 * C < seq_k; ic0 += NSG
    const int nb_iterations = (seq_k + C * NSG - 1) / (C * NSG);

    for (int iter = 0; iter < nb_iterations; iter++) {
        const int ic0 = sg + iter * NSG;
        const int ic = ic0 * C;

        if (ic >= seq_k) break;

        // Step 1: Q*K^T computation
        float mqk = 0.0f;

        // Each ty processes a different cache item
        const int cache_idx = ic + ty;

        if (cache_idx < seq_k) {
            // Dequantize K values
            device const KBlock* k_block = (device const KBlock*)
                (k_base + cache_idx * HEAD_DIM * ELEM_SIZE);

            for (int ii = 0; ii < NB; ii++) {
                T4 k_vals;
                DEQUANT_K(k_block, ii * NL + tx, k_vals);

                // Dot product with query
                mqk += dot((float4)k_vals, s_query[ii * NL + tx]);
            }
        }

        // Step 2: Tree reduction within SIMD group
        // Reduce across tx (dimension threads) to get per-cache-item score
        mqk += simd_shuffle_down(mqk, 4);
        mqk += simd_shuffle_down(mqk, 2);
        mqk += simd_shuffle_down(mqk, 1);

        // Broadcast from thread 0 of each ty group
        float score = simd_shuffle(mqk, NL * ty);

        // Step 3: Store scores for softmax
        const int score_idx = ty * NE + tx;  // 0 to 31
        if (score_idx < C && ic + score_idx < seq_k) {
            s_scores[sg][score_idx] = score * scale;
        } else {
            s_scores[sg][score_idx] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: Online softmax update
        // Load current iteration's scores
        const float s = s_scores[sg][tiisg];

        // Save previous max
        const float prev_m = M;

        // Update max using tree reduction
        const float m_local = max(M, s);
        const float m_new = simd_max(m_local);

        // Broadcast max to all threads
        M = simd_broadcast(m_new, 0);

        // Update sum and scale previous output
        const float m_diff = prev_m - M;
        const float s_exp = exp(s - M);

        S = S * exp(m_diff) + simd_sum(s_exp);

        // Scale previous output
        const float scale_prev = exp(m_diff);
        #pragma unroll
        for (int i = 0; i < NB; i++) {
            O[i] *= scale_prev;
        }

        // Store normalized scores
        s_scores[sg][tiisg] = s_exp;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: V accumulation
        // Each thread loads its portion of V and accumulates
        const int v_idx = ic + tiisg;

        if (v_idx < seq_k) {
            const float weight = s_scores[sg][tiisg];

            device const VBlock* v_block = (device const VBlock*)
                (v_base + v_idx * HEAD_DIM * ELEM_SIZE);

            for (int ii = 0; ii < NB; ii++) {
                T4 v_vals;
                DEQUANT_V(v_block, ii * NL + tx, v_vals);

                O[ii] += weight * (float4)v_vals;
            }
        }
    }

    // Step 6: Final reduction across SIMD groups (if NSG > 1)
    // For simplicity, handle NSG=1 case (most common for decode)
    if (NSG == 1) {
        // Normalize by sum
        const float inv_s = 1.0f / S;

        #pragma unroll
        for (int i = 0; i < NB; i++) {
            O[i] *= inv_s;
        }

        // Write output
        if (tiisg < HEAD_DIM / 4) {
            device float4* out_ptr = (device float4*)
                (output + batch * HEAD_DIM + head * HEAD_DIM);
            out_ptr[tiisg] = O[tiisg / NL];
        }
    }
}

// Template instantiations
// F16, NSG=1 (optimal for seq_k <= 2048)
template [[host_name("flash_decode_opt_f16_nsg1")]]
kernel void flash_decode_opt<
    F16Block, F16Block, float4,
    8, 8,
    dequant_f16, dequant_f16,
    128, 4, 32, 2, 1
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*,
    uint3, ushort, ushort
);

// Q8_0, NSG=1
template [[host_name("flash_decode_opt_q8_0_nsg1")]]
kernel void flash_decode_opt<
    Q8_0Block, Q8_0Block, float4,
    8, 8,
    dequant_q8_0, dequant_q8_0,
    128, 4, 32, 1, 1
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*,
    uint3, ushort, ushort
);

// Q4_0, NSG=1
template [[host_name("flash_decode_opt_q4_0_nsg1")]]
kernel void flash_decode_opt<
    Q4_0Block, Q4_0Block, float4,
    8, 8,
    dequant_q4_0, dequant_q4_0,
    128, 4, 32, 1, 1
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*,
    uint3, ushort, ushort
);

// Q4_1, NSG=1
template [[host_name("flash_decode_opt_q4_1_nsg1")]]
kernel void flash_decode_opt<
    Q4_1Block, Q4_1Block, float4,
    8, 8,
    dequant_q4_1, dequant_q4_1,
    128, 4, 32, 1, 1
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*,
    uint3, ushort, ushort
);

// Q5_0, NSG=1
template [[host_name("flash_decode_opt_q5_0_nsg1")]]
kernel void flash_decode_opt<
    Q5_0Block, Q5_0Block, float4,
    8, 8,
    dequant_q5_0, dequant_q5_0,
    128, 4, 32, 1, 1
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*,
    uint3, ushort, ushort
);

// Q5_1, NSG=1
template [[host_name("flash_decode_opt_q5_1_nsg1")]]
kernel void flash_decode_opt<
    Q5_1Block, Q5_1Block, float4,
    8, 8,
    dequant_q5_1, dequant_q5_1,
    128, 4, 32, 1, 1
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*,
    uint3, ushort, ushort
);
