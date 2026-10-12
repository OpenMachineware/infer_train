// Combined llama.cpp GEMM template with all dependencies inlined
// This file is self-contained for standalone compilation

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// From ggml-metal-impl.h
// ============================================================================

#define SZ_SIMDGROUP 16
#define N_MM_NK 2
#define N_MM_NK_TOTAL (SZ_SIMDGROUP * N_MM_NK)

#define N_MM_BLOCK_X 4
#define N_MM_BLOCK_Y 2
#define N_MM_SIMD_GROUP_X 2
#define N_MM_SIMD_GROUP_Y 2

#define FC_MUL_MM 100

// ============================================================================
// From common.h
// ============================================================================

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define QK 32
#define QK_K 256
#define QK_NL 16

// Type aliases
typedef float4x4 float4x4_t;
typedef half4x4 half4x4_t;

// ============================================================================
// Block structures from ggml-common.h
// ============================================================================

struct block_q4_0 {
    half d;
    uint8_t qs[QK/2];
};

struct block_q4_1 {
    half d;
    half m;
    uint8_t qs[QK/2];
};

struct block_q5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[QK/2];
};

struct block_q5_1 {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[QK/2];
};

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};

struct block_q2_K {
    half d;
    half dmin;
    uint8_t scales[16];
    uint8_t qs[64];
};

struct block_q3_K {
    half d;
    uint8_t hmask[32];
    uint8_t qs[64];
    int8_t scales[12];
};

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[64];
};

struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[16];
    uint8_t qs[64];
};

struct block_q6_K {
    half d;
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
};

struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};

struct block_iq4_nl {
    half d;
    uint8_t qs[16];
};

// ============================================================================
// Kernel arguments
// ============================================================================

struct ggml_metal_kargs_mul_mm {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    uint16_t r2;
    uint16_t r3;
};

// ============================================================================
// Function constants
// ============================================================================

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];

// ============================================================================
// Dequantize functions (from dequantize.h, simplified)
// ============================================================================

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    reg = (type4x4)reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }
    reg = (type4x4)reg_f;
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }
    reg = (type4x4)reg_f;
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)(xb + 2));
    const half d = xb->d;
    const uint32_t qh = *(device const uint32_t *)(xb->qh);
    for (int i = 0; i < 16; ++i) {
        const uint8_t qval = (qs[i] >> (il ? 4 : 0)) & 0x0F;
        const uint32_t hbit = (qh >> (i + il * 16)) & 1;
        reg[i/4][i%4] = d * (qval | (hbit << 4)) - 16.0;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)(xb + 2));
    const half d = xb->d;
    const half m = xb->m;
    const uint32_t qh = *(device const uint32_t *)(xb->qh);
    for (int i = 0; i < 16; ++i) {
        const uint8_t qval = (qs[i] >> (il ? 4 : 0)) & 0x0F;
        const uint32_t hbit = (qh >> (i + il * 16)) & 1;
        reg[i/4][i%4] = d * (qval | (hbit << 4)) + m;
    }
}

// MXFP4
constexpr constant static float kvalues_mxfp4_f[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static inline float decode_e8m0_scale(uint8_t e) {
    return exp2(float(e) - 127.f);
}

template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;
    const float d = decode_e8m0_scale(xb->e);
    const uint8_t shr = il >= 1 ? 4 : 0;

    for (int i = 0; i < 4; ++i) {
        reg[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }
}

// IQ4_NL
constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

// K-quant dequantize functions (simplified versions)
template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    dl = d * (sc & 0xF) * coef;
    ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    il = (il/2)%4;

    const int8_t sc = scales[il];
    half dl = d_all * (il == 0 ? 4.f : il == 1 ? 2.f : il == 2 ? 1.f : 0.5f);

    for (int i = 0; i < 16; ++i) {
        half qval = q[i] & 3;
        half hbit = h[i] & 1;
        reg[i/4][i%4] = dl * (qval - (hbit ? 4.f : 0.f)) * sc;
    }
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = uchar2{uchar(xb->scales[is+0] & 63), uchar(xb->scales[is+4] & 63)};
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs + (il/4) * 32 + 16 * (il&1);
    device const uchar * qh = xb->qh;
    const float d = xb->d;
    const float min = xb->dmin;
    const uchar sc = xb->scales[il/2];
    const float dl = d * (sc & 0xF);
    const float ml = min * (sc >> 4);

    for (int i = 0; i < 16; ++i) {
        uchar qval = q[i] & 0x0F;
        uchar hbit = (qh[i] >> (il & 1)) & 1;
        reg[i/4][i%4] = dl * (qval | (hbit << 4)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d = xb->d;
    device const uint8_t * ql = xb->ql + 64*(il/4) + 32*((il/2)&1) + 16*(il&1);
    device const uint8_t * qh = xb->qh + 32*(il/4) + 16*(il&1);
    const int8_t sc = xb->scales[il/2];

    for (int i = 0; i < 16; ++i) {
        const half q0 = ql[i] & 0xF;
        const half q1 = ql[i] >> 4;
        const half q2 = qh[i] & 1;
        const half q3 = qh[i] & 2;
        reg[i/4][i%4] = d * sc * (q0 + 16*q2 + 64*q1 + 256*q3 - 512);
    }
}

// ============================================================================
// GEMM kernel template
// ============================================================================

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y;
    const int r1 = tgpig.x;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;

    device const block_q * x = (device const block_q *)(srcA + offset0);
    device const char    * y = srcB + offset1;

    const int nb = args.ne00 / (nl * 16);

    float result[NR0] = {0.f};

    for (int ib = tiitg / NL1; ib < nb; ib += 32 / NL1) {
        threadgroup S0 * yf = sa + 64 * (tiitg % NL1);
        threadgroup S1 * xf = sb + 32 * (tiitg % NL1);

        const int i = tiitg % NL1;

        S0_4x4 tmp;
        dequantize_func(x + ib, i, tmp);
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                yf[4*j + k] = tmp[j][k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ir = sgitg * 16; ir < NR0; ir += 64) {
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    acc[j] += yf[4*k + j] * xf[4*(ir/4 + k) + j];
                }
            }
            for (int j = 0; j < 4; ++j) {
                result[ir + j] += acc[j];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float * dstf = (device float *)(dst + im*args.ne0*args.ne1*4 + r1*args.ne0*4);
    for (int ir = sgitg * 16; ir < NR0; ir += 64) {
        float sum = 0.f;
        for (int i = tiitg; i < 16; i += 32) {
            sum += result[ir + i];
        }
        sum = simd_sum(sum);
        if (tiitg == 0) {
            dstf[ir] = sum;
        }
    }
}

// ============================================================================
// Template instantiations
// ============================================================================

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_0, 2, dequantize_q4_0, float, float4x4, float, float2x4>) mul_mm_t;

// Standard Q
template [[host_name("kernel_mul_mm_q4_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_0, 2, dequantize_q4_0, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_1, 2, dequantize_q4_1, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_0, 2, dequantize_q5_0, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_1, 2, dequantize_q5_1, float, float4x4, float, float2x4>;

// Special formats
template [[host_name("kernel_mul_mm_mxfp4_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mxfp4, 2, dequantize_mxfp4, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_nl, 2, dequantize_iq4_nl, float, float4x4, float, float2x4>;

// K-quant
template [[host_name("kernel_mul_mm_q2_K_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q3_K, QK_NL, dequantize_q3_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_K, QK_NL, dequantize_q5_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K, QK_NL, dequantize_q6_K, float, float4x4, float, float2x4>;
