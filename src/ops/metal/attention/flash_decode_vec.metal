// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Decode Vectorized Kernel
// Implements high-performance decode attention based on proven design principles
//
// Key design principles learned from llama.cpp:
// - NE=4: Each SIMD group processes 4 cache items in parallel per iteration
// - C=32: Total cache items per iteration across the SIMD group
// - 2D thread layout: tx for head_dim, ty for cache items
// - Tree reduction: Efficient SIMD shuffle for reduction
// - Loop unrolling: FOR_UNROLL for compile-time unrolling
//
// How it works:
// - 32 threads per SIMD group
// - tx = lane_id % 8 (0-7): Head dimension parallelism
// - ty = lane_id / 8 (0-3): Cache item parallelism (NE=4)
// - Each ty group processes a different cache item simultaneously
// - C/NE = 8 iterations, each processing NE=4 cache items
// - Total: 8 * 4 = 32 cache items per SIMD group iteration
//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Block structures
struct Q4_0Block {
    half d;  // Scale
    uint8_t qs[16];  // 32 quantized values, 2 per byte
};

struct F16Block {
    half values[4];
};

// Dequantization: extracts 4 values from a Q4_0 block
// lane: which 4-element group within the 32-element block (0-7)
template<typename T4>
inline void dequant_q4_0(device const Q4_0Block* block, int lane, thread T4& result) {
    const half d = block->d;
    const int group = lane % 8;
    const int is_high = group / 4;  // High nibble or low nibble
    const int idx = group % 4;

    device const uint16_t* qs16 = (device const uint16_t*)block->qs;

    const half scale = is_high ? d / 16.0h : d;
    const half offset = -8.0h * d;
    const uint16_t mask = is_high ? 0x00F0 : 0x000F;

    // Extract 4 values from the 16-bit values
    for (int i = 0; i < 2; i++) {
        uint16_t val = qs16[2 * idx + i];
        // Low nibble
        result[2*i + 0] = scale * half(val & mask) + offset;
        // High nibble (shifted)
        result[2*i + 1] = (scale / 256.0h) * half((val >> 8) & mask) + offset;
    }
}

// Dequantization for F16 (identity)
template<typename T4>
inline void dequant_f16(device const F16Block* block, int lane, thread T4& result) {
    for (int i = 0; i < 4; i++) {
        result[i] = block->values[i];
    }
}

// Tree reduction for NE=4 case
// Reduces within groups of 8 threads (same ty)
// Input: each thread has a partial sum
// Output: all threads in the group have the complete sum
inline float reduce_ne4(float value, int tx, int ty, int NL) {
    // NE=4 means 4 groups of 8 threads each
    // Reduce within each group
    if (true) {  // NE <= 4
        value += simd_shuffle_down(value, 4);
    }
    if (true) {  // NE <= 8 (always true for NE=4)
        value += simd_shuffle_down(value, 2);
    }
    if (true) {  // NE <= 16
        value += simd_shuffle_down(value, 1);
    }

    // Broadcast from thread NL*ty (thread 0, 8, 16, or 24)
    // Thread NL*ty is the "leader" of each group
    return simd_shuffle(value, NL * ty);
}

// Main kernel template
template<
    typename KBlock,                    // Key block type
    typename VBlock,                    // Value block type
    typename T4,                        // 4-element accumulator type (float4)
    typename K4,                        // 4-element key read type (half4 for F16, float4 for others)
    typename V4,                        // 4-element value read type (half4 for F16, float4 for others)
    int DEQUANT_LANES_K,                // 4-element groups per K block
    int DEQUANT_LANES_V,                // 4-element groups per V block
    void (*DEQUANT_K)(device const KBlock*, int, thread T4&),
    void (*DEQUANT_V)(device const VBlock*, int, thread T4&),
    int HEAD_DIM,                       // 128
    int NE,                             // Cache items per iteration per SIMD group (4)
    int C,                              // Total cache items per iteration (32)
    int ELEM_SIZE>                      // Element size in bytes (2 for half, 4 for float)

kernel void flash_decode_vec_kernel(
    constant uint& seq_k [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device const char* key_cache [[buffer(3)]],
    device const char* value_cache [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    // Thread layout (key insight from llama.cpp)
    // 32 threads total
    // tx: 0-7 for head_dim dimension (8 threads)
    // ty: 0-3 for cache item dimension (4 groups)
    // NL = 8 threads per ty group

    constexpr int NL = 32 / NE;  // 8 threads per cache item group
    const int tx = tiisg % NL;   // 0-7
    const int ty = tiisg / NL;   // 0-3

    const int batch = gid.z;
    const int head = gid.y;

    // Shared memory for query (128 floats = 32 float4s)
    // All 32 threads cooperate to load it
    threadgroup float4 s_query[32];

    // Load query into shared memory
    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)
            (query + batch * HEAD_DIM + head * HEAD_DIM);
        s_query[tiisg] = q_ptr[tiisg];
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator (each thread accumulates part of head_dim)
    // 128 / 4 = 32 float4s, but each thread only handles some
    // With NL=8, each thread handles 32/8 = 4 float4s
    float4 O[HEAD_DIM / (4 * NL)];

    #pragma unroll
    for (int i = 0; i < HEAD_DIM / (4 * NL); i++) {
        O[i] = float4(0.0f);
    }

    // Process KV cache in chunks of C items
    // C/NE iterations, each processing NE=4 cache items in parallel
    const int nb_iterations = (seq_k + C - 1) / C;

    // Compile-time check: is_same<K4, T4>::value determines if we can use direct read
    // For F16: K4 = half4, need conversion, use direct read with type conversion
    // For quantized: need to use dequantization function
    constexpr bool IS_DIRECT_K = (DEQUANT_LANES_K == 1);
    constexpr bool IS_DIRECT_V = (DEQUANT_LANES_V == 1);

    // Pointer to key cache for this head
    // Note: stride calculation uses ELEM_SIZE for correct byte offset
    device const char* k_head = key_cache + head * HEAD_DIM * seq_k * ELEM_SIZE;
    device const char* v_head = value_cache + head * HEAD_DIM * seq_k * ELEM_SIZE;

    // Shared memory for scores - declare once outside the loop
    threadgroup float s_scores[C];

    for (int iter = 0; iter < nb_iterations; iter++) {
        const int ic = iter * C;  // Start index for this iteration

        // Q*K^T scores for NE=4 cache items per thread
        float mqk[C / NE];

        #pragma unroll
        for (int cc = 0; cc < C / NE; cc++) {
            mqk[cc] = 0.0f;
        }

        // Key insight from llama.cpp: pre-compute base pointer for iteration start
        // Then use strided reads inside the loop
        if (IS_DIRECT_K) {
            // Pre-compute base pointer for iteration start
            // Layout: [heads][seq_k][head_dim]
            // For F16: HEAD_DIM/4 = 32 float4s per cache item
            device const K4* k_base = (device const K4*)(k_head + ic * HEAD_DIM * ELEM_SIZE);
            // Each thread starts at its cache item position within the C items
            // ty*HEAD_DIM/4 = ty*32 is the offset for the ty-th cache item
            // tx is the position within the head_dim for this thread
            k_base += ty * (HEAD_DIM / 4) + tx;

            #pragma unroll
            for (int cc = 0; cc < C / NE; cc++) {
                const int cache_idx = ic + NE * cc + ty;

                if (cache_idx < seq_k) {
                    #pragma unroll
                    for (int ii = 0; ii < HEAD_DIM / (4 * NL); ii++) {
                        // Stride: cc*NE*HEAD_DIM/4 + ii*NL = cc*128 + ii*8
                        K4 k_values = k_base[cc * NE * (HEAD_DIM / 4) + ii * NL];
                        mqk[cc] += dot((float4)k_values, (float4)s_query[ii * NL + tx]);
                    }
                } else {
                    mqk[cc] = -INFINITY;
                }
            }
        } else {
            // Quantized path: use dequantization function
            #pragma unroll
            for (int cc = 0; cc < C / NE; cc++) {
                const int cache_idx = ic + NE * cc + ty;

                if (cache_idx < seq_k) {
                    device const KBlock* k_block = (device const KBlock*)(k_head + cache_idx * HEAD_DIM * ELEM_SIZE);

                    #pragma unroll
                    for (int ii = 0; ii < HEAD_DIM / (4 * NL); ii++) {
                        const int lane = ii * NL + tx;
                        T4 k_values;
                        DEQUANT_K(k_block, lane, k_values);
                        mqk[cc] += dot((float4)k_values, (float4)s_query[lane]);
                    }
                } else {
                    mqk[cc] = -INFINITY;
                }
            }
        }

        // Reduce mqk within each ty group
        // Each ty group (8 threads) computed partial sums for one cache item
        #pragma unroll
        for (int cc = 0; cc < C / NE; cc++) {
            mqk[cc] = reduce_ne4(mqk[cc], tx, ty, NL);
        }

        // Store scores to shared memory
        // Key insight from llama.cpp: store mqk[tx] at position NE*tx + ty
        // This maps each thread's score to the correct cache item position
        const int score_idx = NE * tx + ty;
        const int cache_idx_for_score = ic + score_idx;

        if (cache_idx_for_score < seq_k) {
            s_scores[score_idx] = mqk[tx] * scale;
        } else {
            s_scores[score_idx] = -INFINITY;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: all threads process their score in parallel
        // Each thread (tiisg) reads one score
        const float s = s_scores[tiisg];

        // Update max (need global reduction)
        const float m = M;
        M = simd_max(max(M, s));

        // Compute scaling factors
        const float ms = exp(m - M);
        const float vs = exp(s - M);

        // Update sum (need global reduction)
        S = S * ms + simd_sum(vs);

        // Store weight back to shared memory
        s_scores[tiisg] = vs;

        // Scale previous output
        #pragma unroll
        for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
            O[j] *= ms;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // V Accumulation - llama.cpp style: pre-compute base pointer with stride
        // Key insight: instead of each thread computing its own cache_idx and pointer,
        // compute the base pointer for iteration start, then use strided reads
        const float weight = s_scores[tiisg];
        const int cache_idx = ic + tiisg;

        if (cache_idx < seq_k) {
            if (IS_DIRECT_V) {
                // Direct read path: pre-compute base pointer, then use strided reads
                // Layout: [heads][seq_k][head_dim]
                // For F16: stride = HEAD_DIM/4 = 32 float4s per cache item
                device const V4* v_base = (device const V4*)(v_head + ic * HEAD_DIM * ELEM_SIZE);
                // Each thread starts at its cache item position within the C items
                // ty*HEAD_DIM/4 = ty*32 is the offset for the ty-th cache item
                // tx is the position within the head_dim for this thread
                v_base += ty * (HEAD_DIM / 4) + tx;

                #pragma unroll
                for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
                    // Stride: j*NL = j*8 positions in head_dim
                    V4 v_values = v_base[j * NL];
                    O[j] += weight * (float4)v_values;
                }
            } else {
                // Quantized path: use dequantization function
                device const VBlock* v_block = (device const VBlock*)(v_head + cache_idx * HEAD_DIM * ELEM_SIZE);

                #pragma unroll
                for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
                    const int lane = j * NL + tx;
                    T4 v_values;
                    DEQUANT_V(v_block, lane, v_values);
                    O[j] += weight * (float4)v_values;
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output (normalized by S)
    if (tiisg < HEAD_DIM / 4) {
        const float norm = 1.0f / S;
        device float4* out_ptr = (device float4*)
            (output + batch * HEAD_DIM + head * HEAD_DIM);
        out_ptr[tiisg] = O[tiisg / NL] * norm;
    }
}

// Template instantiations

// F16 NE=1 variant (matches llama.cpp default for HEAD_DIM=128)
// For NE=1, each thread processes 1 cache item, NL=32
template [[host_name("flash_decode_vec_f16_ne1")]]
kernel void flash_decode_vec_kernel<
    F16Block, F16Block, float4, half4, half4,
    1, 1,
    dequant_f16, dequant_f16,
    128, 1, 32, 2
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort
);

// F16 NE=4 variant
template [[host_name("flash_decode_vec_f16")]]
kernel void flash_decode_vec_kernel<
    F16Block, F16Block, float4, half4, half4,
    1, 1,
    dequant_f16, dequant_f16,
    128, 4, 32, 2
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort
);

// Q4_0 (quantized, byte-based access)
template [[host_name("flash_decode_vec_q4_0")]]
kernel void flash_decode_vec_kernel<
    Q4_0Block, Q4_0Block, float4, float4, float4,
    8, 8,
    dequant_q4_0, dequant_q4_0,
    128, 4, 32, 1
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, uint3, ushort
);
