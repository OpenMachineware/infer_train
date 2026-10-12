// Flash Decode Multi-Workgroup Kernel
// Implements parallel KV cache processing across multiple workgroups
// Based on llama.cpp's strided processing design
//
// Key design principles:
// - NWG workgroups process different chunks in parallel
// - Each workgroup starts at different stride: ic0 = iwg
// - Stride is NWG to avoid overlaps
// - Partial M/S values stored for final reduction
//
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

// Block structures
struct Q4_0Block {
    half d;
    uint8_t qs[16];
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

    const uint16_t mask = is_high ? 0x00F0 : 0x000F;
    const int x_mv = is_high ? 4 : 0;
    const int gh_mv = is_high ? 12 : 0;
    const int gh_bk = is_high ? 0 : 4;

    for (int i = 0; i < 2; i++) {
        int j = 2 * idx + i;
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*j)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*j+1)) << gh_bk) & 0x10;

        const int32_t x0 = (((qs16[j] & mask) >> x_mv) | xh_0);
        const int32_t x1 = (((qs16[j] >> 8) & mask) >> x_mv) | xh_1;

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

    const uint16_t mask = is_high ? 0x00F0 : 0x000F;
    const int x_mv = is_high ? 4 : 0;
    const int gh_mv = is_high ? 12 : 0;
    const int gh_bk = is_high ? 0 : 4;

    for (int i = 0; i < 2; i++) {
        int j = 2 * idx + i;
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*j)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*j+1)) << gh_bk) & 0x10;

        const int32_t x0 = (((qs16[j] & mask) >> x_mv) | xh_0);
        const int32_t x1 = (((qs16[j] >> 8) & mask) >> x_mv) | xh_1;

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

// Tree reduction for NE=4
inline float reduce_ne4(float value, int tx, int ty, int NL) {
    value += simd_shuffle_down(value, 4);
    value += simd_shuffle_down(value, 2);
    value += simd_shuffle_down(value, 1);
    return simd_shuffle(value, NL * ty);
}

// Partial result structure for inter-workgroup reduction
struct PartialResult {
    float M;  // Max value
    float S;  // Sum value
};

// Main kernel template - partial kernel
// Each workgroup processes a strided subset of the KV cache
template<
    typename KBlock,
    typename VBlock,
    typename T4,
    typename K4,
    typename V4,
    int DEQUANT_LANES_K,
    int DEQUANT_LANES_V,
    void (*DEQUANT_K)(device const KBlock*, int, thread T4&),
    void (*DEQUANT_V)(device const VBlock*, int, thread T4&),
    int HEAD_DIM,
    int NE,
    int C,
    int ELEM_SIZE,
    int NWG>  // Number of workgroups

kernel void flash_decode_multi_wg_partial(
    constant uint& seq_k [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device const char* key_cache [[buffer(3)]],
    device const char* value_cache [[buffer(4)]],
    device float* output [[buffer(5)]],
    device PartialResult* partial_results [[buffer(6)]],  // [heads][NWG] partial M/S
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NL = 32 / NE;
    constexpr int NSG = 1;  // For simplicity, 1 SIMD group per workgroup

    const int tx = tiisg % NL;
    const int ty = tiisg / NL;

    const int batch = gid.z;
    const int head = gid.y;
    const int iwg = gid.x;  // Workgroup index (0 to NWG-1)

    // Shared memory for query
    threadgroup float4 s_query[32];

    if (tiisg < HEAD_DIM / 4) {
        device const float4* q_ptr = (device const float4*)
            (query + batch * HEAD_DIM + head * HEAD_DIM);
        s_query[tiisg] = q_ptr[tiisg];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state
    float M = -INFINITY;
    float S = 0.0f;

    // Output accumulator
    float4 O[HEAD_DIM / (4 * NL)];

    #pragma unroll
    for (int i = 0; i < HEAD_DIM / (4 * NL); i++) {
        O[i] = float4(0.0f);
    }

    // Strided loop over KV cache
    // Each workgroup starts at different position and strides by NWG
    // Similar to llama.cpp: for (int ic0 = iwg; ic0 < seq_k/C; ic0 += NWG)

    constexpr bool IS_DIRECT_K = (DEQUANT_LANES_K == 1);
    constexpr bool IS_DIRECT_V = (DEQUANT_LANES_V == 1);

    device const char* k_head = key_cache + head * HEAD_DIM * seq_k * ELEM_SIZE;
    device const char* v_head = value_cache + head * HEAD_DIM * seq_k * ELEM_SIZE;

    threadgroup float s_scores[C];

    // Strided processing: each workgroup processes every NWG-th chunk
    for (int ic0 = iwg; ic0 * C < seq_k; ic0 += NWG) {
        const int ic = ic0 * C;  // Start index for this chunk

        // Q*K^T scores
        float mqk[C / NE];

        #pragma unroll
        for (int cc = 0; cc < C / NE; cc++) {
            mqk[cc] = 0.0f;
        }

        if (IS_DIRECT_K) {
            device const K4* k_base = (device const K4*)(k_head + ic * HEAD_DIM * ELEM_SIZE);
            k_base += ty * (HEAD_DIM / 4) + tx;

            #pragma unroll
            for (int cc = 0; cc < C / NE; cc++) {
                const int cache_idx = ic + NE * cc + ty;

                if (cache_idx < seq_k) {
                    #pragma unroll
                    for (int ii = 0; ii < HEAD_DIM / (4 * NL); ii++) {
                        K4 k_values = k_base[cc * NE * (HEAD_DIM / 4) + ii * NL];
                        mqk[cc] += dot((float4)k_values, s_query[ii * NL + tx]);
                    }
                } else {
                    mqk[cc] = -INFINITY;
                }
            }
        } else {
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
                        mqk[cc] += dot((float4)k_values, s_query[lane]);
                    }
                } else {
                    mqk[cc] = -INFINITY;
                }
            }
        }

        // Reduce within each ty group
        #pragma unroll
        for (int cc = 0; cc < C / NE; cc++) {
            mqk[cc] = reduce_ne4(mqk[cc], tx, ty, NL);
        }

        // Store scores
        const int score_idx = NE * tx + ty;
        const int cache_idx_for_score = ic + score_idx;

        if (cache_idx_for_score < seq_k) {
            s_scores[score_idx] = mqk[tx] * scale;
        } else {
            s_scores[score_idx] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        const float s = s_scores[tiisg];

        const float m = M;
        M = simd_max(max(M, s));

        const float ms = exp(m - M);
        const float vs = exp(s - M);

        S = S * ms + simd_sum(vs);

        s_scores[tiisg] = vs;

        #pragma unroll
        for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
            O[j] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V Accumulation
        const float weight = s_scores[tiisg];
        const int cache_idx = ic + tiisg;

        if (cache_idx < seq_k) {
            if (IS_DIRECT_V) {
                device const V4* v_base = (device const V4*)(v_head + ic * HEAD_DIM * ELEM_SIZE);
                v_base += ty * (HEAD_DIM / 4) + tx;

                #pragma unroll
                for (int j = 0; j < HEAD_DIM / (4 * NL); j++) {
                    V4 v_values = v_base[j * NL];
                    O[j] += weight * (float4)v_values;
                }
            } else {
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

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store partial output to global memory (interleaved)
    // Each workgroup stores to its own interleaved position
    if (tiisg < HEAD_DIM / 4) {
        const float norm = 1.0f / S;
        device float4* out_ptr = (device float4*)
            (output + batch * HEAD_DIM * NWG + head * HEAD_DIM * NWG + iwg * HEAD_DIM);
        out_ptr[tiisg] = O[tiisg / NL] * norm;
    }

    // Store partial M and S for final reduction
    if (tiisg == 0) {
        const int result_idx = head * NWG + iwg;
        partial_results[result_idx].M = M;
        partial_results[result_idx].S = S;
    }
}

// Final reduction kernel
// Combines partial results from all workgroups
template<int HEAD_DIM, int NWG>
kernel void flash_decode_multi_wg_finalize(
    device const float* partial_outputs [[buffer(0)]],
    device const PartialResult* partial_results [[buffer(1)]],
    device float* final_output [[buffer(2)]],
    constant uint& seq_k [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int batch = gid.z;
    const int head = gid.y;

    // Load partial M and S values
    float M[NWG];
    float S[NWG];

    for (int i = 0; i < NWG; i++) {
        const int idx = head * NWG + i;
        M[i] = partial_results[idx].M;
        S[i] = partial_results[idx].S;
    }

    // Find global max
    float global_M = -INFINITY;
    for (int i = 0; i < NWG; i++) {
        global_M = max(global_M, M[i]);
    }

    // Compute combined S with rescaling
    float global_S = 0.0f;
    for (int i = 0; i < NWG; i++) {
        global_S += S[i] * exp(M[i] - global_M);
    }

    // Combine outputs with rescaling
    if (tiisg < HEAD_DIM / 4) {
        float4 result = float4(0.0f);

        for (int i = 0; i < NWG; i++) {
            device const float4* partial_ptr = (device const float4*)
                (partial_outputs + batch * HEAD_DIM * NWG + head * HEAD_DIM * NWG + i * HEAD_DIM);

            const float scale = (S[i] / global_S) * exp(M[i] - global_M);
            result += scale * partial_ptr[tiisg];
        }

        // Write final output
        device float4* final_ptr = (device float4*)
            (final_output + batch * HEAD_DIM + head * HEAD_DIM);
        final_ptr[tiisg] = result;
    }
}

// Template instantiations

// F16 NWG=8
template [[host_name("flash_decode_multi_wg_partial_f16_nwg8")]]
kernel void flash_decode_multi_wg_partial<
    F16Block, F16Block, float4, half4, half4,
    1, 1,
    dequant_f16, dequant_f16,
    128, 4, 32, 2, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_finalize_f16_nwg8")]]
kernel void flash_decode_multi_wg_finalize<128, 8>(
    device const float*, device const PartialResult*, device float*,
    constant uint&, uint3, ushort
);

// F16 NWG=16
template [[host_name("flash_decode_multi_wg_partial_f16_nwg16")]]
kernel void flash_decode_multi_wg_partial<
    F16Block, F16Block, float4, half4, half4,
    1, 1,
    dequant_f16, dequant_f16,
    128, 4, 32, 2, 16
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_finalize_f16_nwg16")]]
kernel void flash_decode_multi_wg_finalize<128, 16>(
    device const float*, device const PartialResult*, device float*,
    constant uint&, uint3, ushort
);

// F16 NWG=32
template [[host_name("flash_decode_multi_wg_partial_f16_nwg32")]]
kernel void flash_decode_multi_wg_partial<
    F16Block, F16Block, float4, half4, half4,
    1, 1,
    dequant_f16, dequant_f16,
    128, 4, 32, 2, 32
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_finalize_f16_nwg32")]]
kernel void flash_decode_multi_wg_finalize<128, 32>(
    device const float*, device const PartialResult*, device float*,
    constant uint&, uint3, ushort
);

// Q4_0
template [[host_name("flash_decode_multi_wg_partial_q4_0_nwg8")]]
kernel void flash_decode_multi_wg_partial<
    Q4_0Block, Q4_0Block, float4, float4, float4,
    8, 8,
    dequant_q4_0, dequant_q4_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q4_0_nwg16")]]
kernel void flash_decode_multi_wg_partial<
    Q4_0Block, Q4_0Block, float4, float4, float4,
    16, 16,
    dequant_q4_0, dequant_q4_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q4_0_nwg32")]]
kernel void flash_decode_multi_wg_partial<
    Q4_0Block, Q4_0Block, float4, float4, float4,
    32, 32,
    dequant_q4_0, dequant_q4_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

// Q4_1
template [[host_name("flash_decode_multi_wg_partial_q4_1_nwg8")]]
kernel void flash_decode_multi_wg_partial<
    Q4_1Block, Q4_1Block, float4, float4, float4,
    8, 8,
    dequant_q4_1, dequant_q4_1,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q4_1_nwg16")]]
kernel void flash_decode_multi_wg_partial<
    Q4_1Block, Q4_1Block, float4, float4, float4,
    16, 16,
    dequant_q4_1, dequant_q4_1,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q4_1_nwg32")]]
kernel void flash_decode_multi_wg_partial<
    Q4_1Block, Q4_1Block, float4, float4, float4,
    32, 32,
    dequant_q4_1, dequant_q4_1,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

// Q5_0
template [[host_name("flash_decode_multi_wg_partial_q5_0_nwg8")]]
kernel void flash_decode_multi_wg_partial<
    Q5_0Block, Q5_0Block, float4, float4, float4,
    8, 8,
    dequant_q5_0, dequant_q5_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q5_0_nwg16")]]
kernel void flash_decode_multi_wg_partial<
    Q5_0Block, Q5_0Block, float4, float4, float4,
    16, 16,
    dequant_q5_0, dequant_q5_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q5_0_nwg32")]]
kernel void flash_decode_multi_wg_partial<
    Q5_0Block, Q5_0Block, float4, float4, float4,
    32, 32,
    dequant_q5_0, dequant_q5_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

// Q5_1
template [[host_name("flash_decode_multi_wg_partial_q5_1_nwg8")]]
kernel void flash_decode_multi_wg_partial<
    Q5_1Block, Q5_1Block, float4, float4, float4,
    8, 8,
    dequant_q5_1, dequant_q5_1,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q5_1_nwg16")]]
kernel void flash_decode_multi_wg_partial<
    Q5_1Block, Q5_1Block, float4, float4, float4,
    16, 16,
    dequant_q5_1, dequant_q5_1,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q5_1_nwg32")]]
kernel void flash_decode_multi_wg_partial<
    Q5_1Block, Q5_1Block, float4, float4, float4,
    32, 32,
    dequant_q5_1, dequant_q5_1,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

// Q8_0
template [[host_name("flash_decode_multi_wg_partial_q8_0_nwg8")]]
kernel void flash_decode_multi_wg_partial<
    Q8_0Block, Q8_0Block, float4, float4, float4,
    8, 8,
    dequant_q8_0, dequant_q8_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q8_0_nwg16")]]
kernel void flash_decode_multi_wg_partial<
    Q8_0Block, Q8_0Block, float4, float4, float4,
    16, 16,
    dequant_q8_0, dequant_q8_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);

template [[host_name("flash_decode_multi_wg_partial_q8_0_nwg32")]]
kernel void flash_decode_multi_wg_partial<
    Q8_0Block, Q8_0Block, float4, float4, float4,
    32, 32,
    dequant_q8_0, dequant_q8_0,
    128, 4, 32, 1, 8
>(
    constant uint&, constant float&, device const float*, device const char*,
    device const char*, device float*, device PartialResult*,
    uint3, ushort, ushort
);
