// Flash Attention kernels for GPU
// Based on llama.cpp's template approach

#include <metal_stdlib>
using namespace metal;

// Based on llama.cpp's dequantize.h



#define QK 32

// Block structures
struct block_q4_0 {
    half d;
    uint8_t qs[QK / 2];
};

struct block_q4_1 {
    half d;
    half m;
    uint8_t qs[QK / 2];
};

struct block_q5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[QK / 2];
};

struct block_q5_1 {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[QK / 2];
};

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};

// Type aliases for 4x4 matrices
typedef float4x4 float4x4_t;
typedef half4x4 half4x4_t;

// F16 dequantize (identity)
template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

// Q4_0 dequantize
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

// Q4_1 dequantize
template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_1_t4(device const block_q4_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = (il/4) ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float m = xb->m;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

// Q5_0 dequantize
template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.0h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_0_t4(device const block_q5_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.0h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;
    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ? 0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

// Q5_1 dequantize
template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;
    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ? 0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

// Q8_0 dequantize
template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}


#define QK 32
#define N_SIMDWIDTH 32
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// FA_TYPES macro for main kernel (uses simdgroup matrices)
#define FA_TYPES_MAIN \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8,  \
    float,  float4,    simdgroup_float8x8

// FA_TYPES macro for vec kernel (7 types)
// q4_t, k4_t, v4_t, qk_t, s_t, s4_t, o4_t
#define FA_TYPES_VEC \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

// Argument structures
struct FlashAttnArgs {
    uint32_t ne01;    // Q rows
    uint32_t ne02;    // Q heads
    uint32_t ne03;    // Q batches
    uint32_t ne11;    // K rows (seq_len)
    uint32_t ne_12_2; // KV heads
    uint32_t ne_12_3; // KV batches
    uint32_t nb01;    // Q stride (bytes)
    uint32_t nb02;    // Q stride (bytes)
    uint32_t nb03;    // Q stride (bytes)
    uint32_t nb11;    // K stride (bytes)
    uint32_t nb12;    // K stride (bytes)
    uint32_t nb13;    // K stride (bytes)
    uint32_t nb21;    // V stride (bytes)
    uint32_t nb22;    // V stride (bytes)
    uint32_t nb23;    // V stride (bytes)
    uint32_t nb31;    // mask stride
    uint32_t nb32;    // mask stride
    uint32_t nb33;    // mask stride
    uint32_t ne32;    // mask rows
    uint32_t ne33;    // mask batches
    float scale;      // 1/sqrt(head_dim)
    float max_bias;   // ALiBi
    float logit_softcap;
    uint32_t n_head_log2;
    float m0;
    float m1;
    uint32_t nsg;     // simdgroups per threadgroup
    uint32_t nwg;     // workgroups (for parallelization across seq_len)
};

// Function constants for compile-time configuration
// Using llama.cpp's indices (FC_FLASH_ATTN_EXT_VEC base = 1000, but we use relative indices)
// Index 0-4: boolean flags
// Index 20-23: integer values
constant bool FC_has_mask [[function_constant(0)]];
constant bool FC_has_sinks [[function_constant(1)]];
constant bool FC_has_bias [[function_constant(2)]];
constant bool FC_has_scap [[function_constant(3)]];
constant bool FC_has_kvpad [[function_constant(4)]];
constant int32_t FC_ns10 [[function_constant(20)]];
constant int32_t FC_ns20 [[function_constant(21)]];
constant int32_t FC_nsg  [[function_constant(22)]];
constant int32_t FC_nwg  [[function_constant(23)]];

// ============================================================================
// Vec kernel for small batch (decode case)
// Each simdgroup handles Q queries and C cache items
// ============================================================================

template<
    typename q4_t,    // query type in shared memory (half4)
    typename k4_t,    // key type in shared memory
    typename v4_t,    // value type in shared memory
    typename qk_t,    // Q*K type (float)
    typename s_t,     // softmax type (float)
    typename s4_t,    // softmax type 4 (float4)
    typename o4_t,    // output type (float4)
    typename kd4_t,   // key type in device memory
    short nl_k,       // number of 4-element groups per block (QK/4 = 8 for Q4_0)
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,   // value type in device memory
    short nl_v,       // number of 4-element groups per block
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,         // K head dimension
    short DV,         // V head dimension
    short NE = 1,     // elements per thread (1 for optimal parallelism)
    short Q  = 1,     // queries per simdgroup
    short C  = 32>    // cache items per simdgroup (llama.cpp uses 32)

kernel void kernel_flash_attn_ext_vec(
        constant FlashAttnArgs & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        device const char * idx,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

    const short iwg = tgpig[2] % FC_nwg;
    const ushort iq3 = tgpig[2] / FC_nwg;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*Q*C;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    // Shared memory layout (matches llama.cpp):
    // - sq4: query data, starts at offset 0
    // - ss:  attention scores, starts after Q*NSG*PK
    // - sm:  mask data, starts after ss
    // - so4: output data, starts after all scores
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                          0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH         + Q*FC_nsg*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH         + Q*FC_nsg*PK); // alias
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*Q*C + Q*FC_nsg*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*Q*PV       + Q*FC_nsg*PK + FC_nsg*SH);

    so4 += tiisg;

    // Setup pointers
    q += iq1 * Q * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03;
    const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);
    k += ikv2 * args.nb12 + ikv3 * args.nb13;
    v += ikv2 * args.nb22 + ikv3 * args.nb23;

    // Load Q to shared memory
    for (short qq = 0; qq < Q; ++qq) {
        const int iq1_q = iq1 * Q + qq;
        device const float4 * q4 = (device const float4 *) ((device const char *) q + qq * args.nb01);
        if (iq1_q < args.ne01) {
            for (short i = tiisg; i < PK4; i += NW) {
                if (i < DK4) {
                    sq4[qq * PK4 + i] = (q4_t) q4[i];
                } else {
                    sq4[qq * PK4 + i] = (q4_t) 0.0f;
                }
            }
        } else {
            for (short i = tiisg; i < PK4; i += NW) {
                sq4[qq * PK4 + i] = (q4_t) 0.0f;
            }
        }
    }

    // Zero output
    for (short qq = 0; qq < Q; ++qq) {
        for (short i = 0; i < DV4/NL; ++i) {
            so4[qq * DV4 + i * NL] = (o4_t) 0.0f;
        }
    }

    // Zero shared memory
    for (short i = tiisg; i < SH/4; i += NW) {
        ((threadgroup s4_t *)ss)[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[Q];
    float M[Q];
    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
        S[qq] = 0.0f;
        M[qq] = -FLT_MAX/2;
    }

    const short tx = tiisg % NL;
    const short ty = tiisg / NL;

    device const half * pm_base = (device const half *) (mask + iq1 * Q * args.nb31 + (iq2 % args.ne32) * args.nb32 + (iq3 % args.ne33) * args.nb33);

    // Loop over KV cache (matching llama.cpp's parallelization pattern)
    // Each threadgroup handles a slice of the KV cache based on iwg (workgroup index)
    // Each simdgroup within a threadgroup handles a different slice based on sgitg
    for (int ic0 = iwg * args.nsg + sgitg; ; ic0 += args.nwg * args.nsg) {
        int ic = ic0 * C;
        if (ic >= args.ne11) {
            break;
        }

        device const half * pm[Q];
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            pm[qq] = pm_base + ((iq1 * Q + qq) < args.ne01 ? qq * (args.nb31/sizeof(half)) : -iq1 * Q * (args.nb31/sizeof(half)));
        }

        // Handle padding for last chunk
        if (FC_has_kvpad && ic + C > args.ne11) {
            k = pad;
            v = k + args.nb11 * C * args.ne_12_2 * args.ne_12_3;
            k += (ikv2 + ikv3 * args.ne_12_2) * args.nb11 * C;
            v += (ikv2 + ikv3 * args.ne_12_2) * args.nb21 * C;
            ic = 0;
        }

        // Read mask
        if (FC_has_mask) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                sm[qq * C + tiisg] = pm[qq][ic + tiisg];
            }
        } else {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                sm[qq * C + tiisg] = (ic + tiisg) < args.ne11 ? (half)0.0f : (half)(-MAXHALF);
            }
        }

        // Q*K^T
        s_t mqk[Q][C/NE];
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                mqk[qq][cc] = 0.0f;
            }
        }

        // Compute Q*K
        if (is_same<kd4_t, k4_t>::value) {
            // F16 path - direct read (matching llama.cpp pattern)
            // pk4 points to start of K chunk for this simdgroup
            device const k4_t * pk4 = (device const k4_t *) (k + ic * args.nb11);
            pk4 += sgitg * (8 * FC_ns10/4);  // simdgroup offset

            FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                    const k4_t k_elem = pk4[cc * NE * FC_ns10/4 + ii * NL];
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq * PK4 + ii * NL + tx]);
                    }
                }
            }
        } else {
            // Quantized path - dequantize on the fly
            FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE * cc + ty) * args.nb11));

                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                    const short i = ii * NL + tx;
                    k4_t mk;
                    deq_k_t4(pk + i/nl_k, i % nl_k, mk);

                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq * PK4 + i]);
                    }
                }
            }
        }

        // Reduce within simdgroup (matching llama.cpp exactly)
        // Each thread has C/NE accumulators, reduce each one
        FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                if (NE == 1) {
                    mqk[qq][cc] = simd_sum(mqk[qq][cc]);
                } else {
                    // simdgroup reduce (NE = 4)
                    // [ 0 ..  7] -> [ 0]
                    // [ 8 .. 15] -> [ 8]
                    // [16 .. 23] -> [16]
                    // [24 .. 31] -> [24]
                    if (NE <= 1) mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 16);
                    if (NE <= 2) mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 8);
                    if (NE <= 4) mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 4);
                    if (NE <= 8) mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 2);
                    if (NE <= 16) mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 1);

                    // broadcast
                    mqk[qq][cc] = simd_shuffle(mqk[qq][cc], NL * ty);
                }
            }
        }

        // Apply scale and mask
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            mqk[qq][tx] *= args.scale;
            if (FC_has_bias) {
                // ALiBi
            }
            mqk[qq][tx] += (float) sm[qq * C + NE * tx + ty];
            ss[qq * C + NE * tx + ty] = mqk[qq][tx];
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const float m = M[qq];
            const float s = ss[qq * C + tiisg];

            M[qq] = simd_max(max(M[qq], s));

            const float ms = exp(m - M[qq]);
            const float vs = exp(s - M[qq]);

            S[qq] = S[qq] * ms + simd_sum(vs);

            ss[qq * C + tiisg] = vs;

            if ((DV4/NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[qq * DV4 + ii * NL] *= ms;
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = O + (Q*K^T)*V
        o4_t lo[Q][DV4/NL];
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                lo[qq][ii] = 0.0f;
            }
        }

        if (is_same<vd4_t, v4_t>::value) {
            // F16 path
            device const v4_t * pv4 = (device const v4_t *) (v + ic * args.nb21);
            pv4 += ty * FC_ns20/4 + tx;

            FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    const v4_t v_elem = pv4[cc * NE * FC_ns20/4 + ii * NL];
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        lo[qq][ii] += o4_t(float4(v_elem) * float4(ss[qq * C + cc * NE + ty]));
                    }
                }
            }
        } else {
            // Quantized path
            FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE * cc + ty) * args.nb21));

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    const short i = ii * NL + tx;
                    v4_t mv;
                    deq_v_t4(pv4 + i/nl_v, i % nl_v, mv);

                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        lo[qq][ii] += o4_t(float4(mv) * float4(ss[qq * C + cc * NE + ty]));
                    }
                }
            }
        }

        // Reduce and accumulate
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                if (NE > 1) {
                    lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 16);
                    lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 16);
                    lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 16);
                    lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 16);
                }
                if (NE > 2) {
                    lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 8);
                    lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 8);
                    lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 8);
                    lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 8);
                }
                if (NE > 4) {
                    lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 4);
                    lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 4);
                    lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 4);
                    lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 4);
                }
                if (NE > 8) {
                    lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 2);
                    lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 2);
                    lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 2);
                    lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 2);
                }
                if (NE > 16) {
                    lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 1);
                    lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 1);
                    lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 1);
                    lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 1);
                }
            }
        }

        if ((DV4/NL % NW == 0) || ty == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[qq * DV4 + ii * NL] += lo[qq][ii];
                }
            }
        }
    }

    // Store S and M for reduction
    if (tiisg == 0) {
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            ss[2*qq + 0] = S[qq];
            ss[2*qq + 1] = M[qq];
        }
    }

    so4 -= tiisg;  // Reset pointer offset

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduce across SIMD groups
    for (short r = FC_nsg/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float S0 = ss[                2*qq + 0];
                const float S1 = ss[r*(SH/2) +      2*qq + 0];

                const float M0 = ss[                2*qq + 1];
                const float M1 = ss[r*(SH/2) +      2*qq + 1];

                const float Mx  = max(M0, M1);

                const float ms0 = exp(M0 - Mx);
                const float ms1 = exp(M1 - Mx);

                const float Sx  = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[2*qq + 0] = Sx;
                    ss[2*qq + 1] = Mx;
                }

                // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[qq*DV4 + i] = so4[qq*DV4 + i]*ms0 + so4[qq*DV4 + i + r*Q*PV4]*ms1;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const int64_t nrows = args.ne03 * args.ne02 * args.ne01;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*FC_nwg; // the S and M are stored after the results

        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1 * Q + qq;
            if (iq1_q >= args.ne01) {
                continue;
            }

            const int64_t rid = iq3 * args.ne02 * args.ne01 + iq2 + iq1_q * args.ne01;

            const float Sval = FC_nwg == 1 ? (ss[2*qq + 0] == 0.0f ? 0.0f : 1.0f/ss[2*qq + 0]) : 1.0f;

            // Interleave the workgroup data
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid * DV4 * FC_nwg + FC_nwg * i + iwg] = (float4) so4[qq * DV4 + i] * Sval;
            }

            // Store S and M
            if (FC_nwg > 1) {
                if (tiisg == 0) {
                    dst1[rid * (2 * FC_nwg) + 2 * iwg + 0] = ss[2*qq + 0];
                    dst1[rid * (2 * FC_nwg) + 2 * iwg + 1] = ss[2*qq + 1];
                }
            }
        }
    }
}

// Template instantiation typedef
// Note: NE=4 for head_dim=128, nl_k=8 for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 (8 groups of 4 elements per block)
typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES_VEC, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

// Instantiations for head_dim=128, DK=128, DV=128, NE=4
// F16 (nl_k=1 because no quantization blocks)
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128")]]
kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_VEC, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>;

// Q4_0 (nl_k=8: 32 elements / 4 = 8 groups per block)
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128")]]
kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_VEC, block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4, 128, 128, 4>;

// Q4_1
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128")]]
kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_VEC, block_q4_1, 8, dequantize_q4_1_t4, block_q4_1, 8, dequantize_q4_1_t4, 128, 128, 4>;

// Q5_0
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128")]]
kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_VEC, block_q5_0, 8, dequantize_q5_0_t4, block_q5_0, 8, dequantize_q5_0_t4, 128, 128, 4>;

// Q5_1
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128")]]
kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_VEC, block_q5_1, 8, dequantize_q5_1_t4, block_q5_1, 8, dequantize_q5_1_t4, 128, 128, 4>;

// Q8_0
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]]
kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_VEC, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 128, 128, 4>;

// ============================================================================
// Main kernel for large batch (prefill case)
// Uses simdgroup_matrix for efficient computation
// ============================================================================

// FA_TYPES_MAIN expands to 6 types for main kernel:
// q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t

template<
    typename q_t,     // query type in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key type in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value type in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K type
    typename qk8x8_t,
    typename s_t,     // softmax type
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // output type
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head dimension
    short DV,         // V head dimension
    short Q,          // queries per threadgroup
    short C,          // cache items per threadgroup
    short NSG>        // number of simdgroups
kernel void kernel_flash_attn_ext(
        constant FlashAttnArgs & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {

    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0] * Q;

    constexpr short KV   = 8;
    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;
    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;
    constexpr short NW   = N_SIMDWIDTH;
    constexpr short NQ   = Q/NSG;
    constexpr short SH   = 2*C;
    constexpr short TS   = 2*SH;
    constexpr short T    = DK + 2*PV;

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);
    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    // Setup pointers
    {
        q += iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03;
        const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
        const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);
        k += ikv2 * args.nb12 + ikv3 * args.nb13;
        v += ikv2 * args.nb22 + ikv3 * args.nb23;
    }

    // Load Q to shared memory
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *) ((device const char *) q + j * args.nb01);
        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j * DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j * DK4 + i] = 0;
            }
        }
    }

    // Zero output
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (short i = tiisg; i < DV4; i += NW) {
            so4[j * PV4 + i] = 0;
        }
        for (short i = tiisg; i < SH; i += NW) {
            ss[j * SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };
    float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

    // Loop over KV cache
    for (int ic0 = 0; ; ++ic0) {
        int ic = ic0 * C;
        if (ic >= args.ne11) {
            break;
        }

        // Read mask
        if (FC_has_mask) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj * NSG + sgitg;
                device const half2 * pm2 = (device const half2 *) (mask + (iq1 + j) * args.nb31 + (iq2 % args.ne32) * args.nb32 + (iq3 % args.ne33) * args.nb33);
                sm2[j * SH + tiisg] = pm2[ic + tiisg];
            }
        } else {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                sm2[jj * SH + tiisg] = half2(0.0h, 0.0h);
            }
        }

        // Q*K^T
        if (is_same<kd4x4_t, k4x4_t>::value) {
            // F16 path - direct simdgroup matrix multiply
            device const k_t * pk = (device const k_t *) (k + ic * args.nb11);
            threadgroup const q_t * pq = sq;
            threadgroup s_t * ps = ss;

            pk += sgitg * (8 * 8);  // NS10 = 8 for DK=128
            ps += sgitg * 8;

            constexpr short NC = (C/8) / NSG;

            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                if (DK % 16 != 0) {
                    // Non-divisible case
                    k8x8_t mk;
                    q8x8_t mq;

                    FOR_UNROLL (short i = 0; i < DK8; ++i) {
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_load(mk, pk + 8*i, 8, 0, true);
                        simdgroup_load(mq, pq + 8*i, DK);
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                    }
                } else {
                    // Divisible case - optimized
                    k8x8_t mk[2];
                    q8x8_t mq[2];

                    FOR_UNROLL (short i = 0; i < DK8/2; ++i) {
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                        simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);
                        simdgroup_load(mk[0], pk + 0*8 + 16*i, 8, 0, true);
                        simdgroup_load(mk[1], pk + 1*8 + 16*i, 8, 0, true);
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                        simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                    }
                }

                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8 * (NSG * 8);
                ps += 8 * NSG;
            }
        } else {
            // Quantized path - dequantize on the fly
            for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                const short cc = ccc * NSG + sgitg;
                const short tx = tiisg % 4;
                const short ty = tiisg / 4;

                qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                for (short ii = 0; ii < DK16; ii += 4) {
                    device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty) * args.nb11));

                    if (DK16 % 4 == 0) {
                        k4x4_t tmp;
                        deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx) % nl_k, tmp);
                        sk4x4[4*ty + tx] = tmp;

                        simdgroup_barrier(mem_flags::mem_threadgroup);

                        FOR_UNROLL (short k = 0; k < 4; ++k) {
                            k8x8_t mk;
                            q8x8_t mq;

                            simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                            simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                            simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                            simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    }
                }

                simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const float m = M[jj];

            float2 s2 = ss2[j * SH/2 + tiisg] * args.scale;
            s2 += s2_t(sm2[j * SH + tiisg]);

            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

            const float ms = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);

            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[j * SH/2 + tiisg] = vs2;

            if (DV4 % NW == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                    const short i = ii * NW + tiisg;
                    so4[j * PV4 + i] *= ms;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = O + (Q*K^T)*V
        if (is_same<vd4x4_t, v4x4_t>::value) {
            // F16 path
            device const v_t * pv = (device const v_t *) (v + ic * args.nb21);
            pv += 8 * sgitg;

            constexpr short NO = PV8 / NSG;
            o8x8_t lo[NO];

            {
                auto sot = so + 8 * sgitg;
                FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }

            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                s8x8_t vs;
                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                    v8x8_t mv[2];
                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, 8, 0, false);
                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, 8, 0, false);
                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                }
                pv += 8 * 8;
            }

            {
                auto sot = so + 8 * sgitg;
                FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        } else {
            // Quantized path
            const short tx = tiisg % 4;
            const short ty = tiisg / 4;

            for (short cc = 0; cc < C/8; ++cc) {
                s8x8_t vs;
                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                    device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty) * args.nb21));

                    if (DV16 % 4 == 0) {
                        v4x4_t tmp;
                        deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx) % nl_v, tmp);
                        sv4x4[4*ty + tx] = tmp;

                        simdgroup_barrier(mem_flags::mem_threadgroup);

                        FOR_UNROLL (short k = 0; k < 4; ++k) {
                            v8x8_t mv[2];
                            o8x8_t lo[2];

                            simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                            simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                            simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                            simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                            simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                            simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
                            simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                            simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    {
        device float4 * dst4 = (device float4 *) (dst + iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03);

        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            if (iq1 + j < args.ne01) {
                for (short i = tiisg; i < DV4; i += NW) {
                    dst4[j * DV4 + i] = so4[j * PV4 + i] / S[jj];
                }
            }
        }
    }
}

// Main kernel instantiations
typedef decltype(kernel_flash_attn_ext<FA_TYPES_MAIN, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 128, 128, 8, 256, 2>) flash_attn_ext_t;

// F16
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]
kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_MAIN, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 128, 128, 8, 256, 2>;

// Q4_0
template [[host_name("kernel_flash_attn_ext_q4_0_dk128_dv128")]]
kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_MAIN, block_q4_0, 4, dequantize_q4_0, block_q4_0, 4, dequantize_q4_0, 128, 128, 8, 256, 2>;

// Q4_1
template [[host_name("kernel_flash_attn_ext_q4_1_dk128_dv128")]]
kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_MAIN, block_q4_1, 4, dequantize_q4_1, block_q4_1, 4, dequantize_q4_1, 128, 128, 8, 256, 2>;

// Q5_0
template [[host_name("kernel_flash_attn_ext_q5_0_dk128_dv128")]]
kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_MAIN, block_q5_0, 4, dequantize_q5_0, block_q5_0, 4, dequantize_q5_0, 128, 128, 8, 256, 2>;

// Q5_1
template [[host_name("kernel_flash_attn_ext_q5_1_dk128_dv128")]]
kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_MAIN, block_q5_1, 4, dequantize_q5_1, block_q5_1, 4, dequantize_q5_1, 128, 128, 8, 256, 2>;

// Q8_0
template [[host_name("kernel_flash_attn_ext_q8_0_dk128_dv128")]]
kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_MAIN, block_q8_0, 4, dequantize_q8_0, block_q8_0, 4, dequantize_q8_0, 128, 128, 8, 256, 2>;
