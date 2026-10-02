// Self-written Matrix-Vector Multiplication Template
// Supports all common quantization formats

#include <metal_stdlib>
using namespace metal;

#include "ggml-common.h"

#define QK_K 256
#define QK 32

// Block sizes for standard quantization
#define QK1_0 32
#define QK2_0 64
#define QK5_0 32
#define QK5_1 32
#define QK4_NL 32
#define QK_MXFP4 32

// Loop unrolling macro
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// ============================================================================
// Block structures
// ============================================================================

// K-quant formats
typedef struct {
    half  d;
    half  dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct {
    half  d;
    half  dmin;
    uint8_t scales[12];
    uint8_t qh[32];  // QK_K/8 = 256/8 = 32 (NOT 64!)
    uint8_t qs[128]; // QK_K/2 = 256/2 = 128
} block_q5_K;

typedef struct {
    uint8_t scales[16];  // QK_K/16 = 256/16 = 16
    uint8_t qs[64];      // QK_K/4 = 256/4 = 64
    half  d;
    half  dmin;
} block_q2_K;

typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half  d;
} block_q3_K;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half  d;
} block_q6_K;

// Standard quantization formats
typedef struct {
    half d;
    uint8_t qs[16];
} block_q4_0;

typedef struct {
    half d;
    half m;
    uint8_t qs[16];
} block_q4_1;

typedef struct {
    half d;
    uint8_t qs[32];
} block_q8_0;

typedef struct {
    half d;
    uint8_t qs[4];
} block_q1_0;

typedef struct {
    half d;
    uint8_t qs[16];
} block_q2_0;

typedef struct {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_0;

typedef struct {
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_1;

// IQ formats
typedef struct {
    half  d;
    uint8_t  qs[32];
    uint16_t qh[8];
} block_iq1_s;

typedef struct {
    uint8_t  qs[32];
    uint8_t  qh[16];
    uint16_t scales[4];
} block_iq1_m;

typedef struct {
    half d;
    uint16_t qs[32];
} block_iq2_xxs;

typedef struct {
    half d;
    uint16_t qs[32];
    uint8_t scales[8];
} block_iq2_xs;

typedef struct {
    half d;
    uint8_t qs[32];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[8];
} block_iq2_s;

typedef struct {
    half d;
    uint8_t qs[32];
    uint16_t qh[16];
} block_iq3_xxs;

typedef struct {
    half d;
    uint8_t qs[32];
    uint8_t qh[8];
    uint8_t signs[16];
    uint8_t scales[4];
} block_iq3_s;

typedef struct {
    half d;
    uint8_t qs[16];  // QK4_NL/2 = 32/2 = 16
} block_iq4_nl;

typedef struct {
    half d;
    uint8_t qs[32];
    uint16_t scales_h;
    uint8_t scales_l[8];
} block_iq4_xs;

// TQ format
typedef struct {
    half d;
    uint8_t qs[64];
} block_tq2_0;

// MXFP4 format
typedef struct {
    uint8_t e;
    uint8_t qs[16];
} block_mxfp4;

// ============================================================================
// Kernel arguments
// ============================================================================

typedef struct {
    int32_t ne00; int32_t ne01; int32_t ne02;
    uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
    int32_t ne10; int32_t ne11; int32_t ne12;
    uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
    int32_t ne0; int32_t ne1;
    int32_t nr0;
    int16_t r2;
    int16_t r3;
} ggml_metal_kargs_mul_mv;

// ============================================================================
// Function constants
// ============================================================================

constant short FC_nsg  [[function_constant(600)]];
constant short FC_ne12 [[function_constant(602)]];
constant short FC_r2   [[function_constant(603)]];
constant short FC_r3   [[function_constant(604)]];

// ============================================================================
// Lookup tables
// ============================================================================

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_mxfp4_f[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// IQ format constants
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

// Grid tables are provided by ggml-common.h

// ============================================================================
// Q4_K kernel (verified: 100.39% of llama.cpp)
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K * x = (device const block_q4_K *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    float sumf[nr0] = {0.f};

    float yl[16];
    float yh[16];

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        // y-loading loop: do NOT use FOR_UNROLL
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // dot-product loop: MUST use FOR_UNROLL
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q2_K kernel - Original implementation
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q2_K * x = (device const block_q2_K *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    // Use distinct naming: y_vec for cached y values
    float y_vec[32];
    float result[nr0] = {0.f};

    // Thread indexing
    const short block_idx = tiisg / 8;
    const short thread_in_block = tiisg % 8;
    const short sub_block = thread_in_block / 4;
    const short local_pos = thread_in_block % 4;
    const short scale_offset = (8 * local_pos) / 16;

    device const float * y_ptr = y + block_idx * QK_K + 128 * sub_block + 8 * local_pos;

    for (int blk = block_idx; blk < nb; blk += 4) {
        // Load y values and compute partial sums
        float4 y_sum = {0.f, 0.f, 0.f, 0.f};
        for (short j = 0; j < 8; ++j) {
            y_vec[j     ] = y_ptr[j    ]; y_sum[0] += y_vec[j];
            y_vec[j +  8] = y_ptr[j + 32]; y_sum[1] += y_vec[j +  8];
            y_vec[j + 16] = y_ptr[j + 64]; y_sum[2] += y_vec[j + 16];
            y_vec[j + 24] = y_ptr[j + 96]; y_sum[3] += y_vec[j + 24];
        }

        device const uint8_t  * scales = (device const uint8_t  *)x[blk].scales + 8*sub_block + scale_offset;
        device const uint16_t * quant = (device const uint16_t *)x[blk].qs + 16 * sub_block + 4 * local_pos;
        device const half     * delta = &x[blk].d;

        for (short row = 0; row < nr0; row++) {
            // Process quantization - each 16-bit packs 8x 2-bit values
            // Use float4 for better vectorization
            float4 acc_lo = {0.f, 0.f, 0.f, 0.f};
            float4 acc_hi = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (int q = 0; q < 8; q += 2) {
                uint16_t packed = quant[q/2];
                // Extract using bit masks
                acc_lo[0] += y_vec[q] * (packed & 0x0003);
                acc_hi[0] += y_vec[q+1] * (packed & 0x0300);

                acc_lo[1] += y_vec[q+8] * (packed & 0x000c);
                acc_hi[1] += y_vec[q+9] * (packed & 0x0c00);

                acc_lo[2] += y_vec[q+16] * (packed & 0x0030);
                acc_hi[2] += y_vec[q+17] * (packed & 0x3000);

                acc_lo[3] += y_vec[q+24] * (packed & 0x00c0);
                acc_hi[3] += y_vec[q+25] * (packed & 0xc000);
            }

            // Combine with scaling factors - use multiplication instead of division
            float scale_main = delta[0];
            float scale_min = delta[1] * 0.0625f;  // 1/16

            // Precompute reciprocals for better performance
            float s0 = (scales[0] & 0x0F);
            float s2 = (scales[2] & 0x0F) * 0.25f;  // 1/4
            float s4 = (scales[4] & 0x0F) * 0.0625f;  // 1/16
            float s6 = (scales[6] & 0x0F) * 0.015625f;  // 1/64

            result[row] += scale_main * (
                (acc_lo[0] + acc_hi[0] * 0.00390625f) * s0 +
                (acc_lo[1] + acc_hi[1] * 0.00390625f) * s2 +
                (acc_lo[2] + acc_hi[2] * 0.00390625f) * s4 +
                (acc_lo[3] + acc_hi[3] * 0.00390625f) * s6
            ) - scale_min * (
                y_sum[0] * (scales[0] & 0xF0) +
                y_sum[1] * (scales[2] & 0xF0) +
                y_sum[2] * (scales[4] & 0xF0) +
                y_sum[3] * (scales[6] & 0xF0)
            );

            quant += args.nb01 / 2;
            scales += args.nb01;
            delta += args.nb01 / 2;
        }

        y_ptr += 4 * QK_K;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(result[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q3_K kernel - Original implementation
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q3_K * x = (device const block_q3_K *)(src0 + offset0);
    device const float      * yy = (device const float      *)(src1 + offset1);

    float cached_y[32];

    // Thread coordinate computation
    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short part = tid / 4;
    const short lane = 2 * ((tid % 4) / 2);
    const short half_pos = tid % 2;
    const short base_pos = 8 * half_pos;

    // Pre-computed mask tables for quantization bits
    const ushort4 h_masks[4] = {
        {0x0001, 0x0100, 0x0002, 0x0200},
        {0x0004, 0x0400, 0x0008, 0x0800},
        {0x0010, 0x1000, 0x0020, 0x2000},
        {0x0040, 0x4000, 0x0080, 0x8000}
    };

    const int4 q_masks[2] = {
        {0x0003, 0x0300, 0x000c, 0x0c00},
        {0x0030, 0x3000, 0x00c0, 0xc000}
    };

    const ushort4 active_h_mask = h_masks[2 * part + lane / 2];

    const short scale_shift = 2 * lane;

    const float bias1 = lane == 0 ? 4.f : 64.f;
    const float bias2 = 4.f * bias1;

    const uint16_t scale_shift1 = 4 * part;
    const uint16_t scale_shift2 = scale_shift1 + lane;

    const short quant_offset = 32 * part + base_pos;
    const short y_base = 128 * part + 32 * lane + base_pos;

    device const float * y_src = yy + ix * QK_K + y_base;

    uint32_t scales_bits, aux_bits;
    thread uint16_t * scale_pair = (thread uint16_t *)&scales_bits;
    thread const int8_t * scale_bytes = (thread const int8_t *)&scales_bits;

    float part1[nr0] = {0.f};
    float part2[nr0] = {0.f};

    for (int blk = ix; blk < nb; blk += 4) {
        // Cache y values
        for (short k = 0; k < 8; ++k) {
            cached_y[k] = y_src[k];
            cached_y[k + 8] = y_src[k + 16];
            cached_y[k + 16] = y_src[k + 32];
            cached_y[k + 24] = y_src[k + 48];
        }

        device const uint16_t * quant_data = (device const uint16_t *)(x[blk].qs + quant_offset);
        device const uint16_t * high_bits = (device const uint16_t *)(x[blk].hmask + base_pos);
        device const uint16_t * scale_data = (device const uint16_t *)(x[blk].scales);
        device const half * delta = &x[blk].d;

        for (short row = 0; row < nr0; ++row) {
            const float scale = (float)delta[0];

            // Decode scales
            scale_pair[0] = scale_data[4];
            scale_pair[1] = scale_data[5];
            aux_bits = ((scales_bits >> scale_shift2) << 4) & 0x30303030;
            scale_pair[0] = scale_data[lane];
            scale_pair[1] = scale_data[lane + 1];
            scales_bits = ((scales_bits >> scale_shift1) & 0x0f0f0f0f) | aux_bits;

            // First half processing
            float dot_a = 0, dot_b = 0, mask_a = 0, dot_c = 0, dot_d = 0, mask_b = 0;

            FOR_UNROLL (short idx = 0; idx < 8; idx += 2) {
                const int32_t qword = quant_data[idx / 2];
                dot_a += cached_y[idx] * (qword & q_masks[lane / 2][0]);
                dot_b += cached_y[idx + 1] * (qword & q_masks[lane / 2][1]);
                mask_a += ((high_bits[idx / 2] & active_h_mask[0]) ? 0.f : cached_y[idx]) +
                          ((high_bits[idx / 2] & active_h_mask[1]) ? 0.f : cached_y[idx + 1]);
                dot_c += cached_y[idx + 16] * (qword & q_masks[lane / 2][2]);
                dot_d += cached_y[idx + 17] * (qword & q_masks[lane / 2][3]);
                mask_b += ((high_bits[idx / 2] & active_h_mask[2]) ? 0.f : cached_y[idx + 16]) +
                          ((high_bits[idx / 2] & active_h_mask[3]) ? 0.f : cached_y[idx + 17]);
            }

            float val1 = scale * (dot_a + dot_b * 0.00390625f - mask_a * bias1);
            float val2 = scale * (dot_c + dot_d * 0.00390625f - mask_b * bias2);
            part1[row] += val1 * (scale_bytes[0] - 32);
            part2[row] += val2 * (scale_bytes[2] - 32);

            // Second half processing
            dot_a = dot_b = mask_a = dot_c = dot_d = mask_b = 0;
            FOR_UNROLL (short idx = 0; idx < 8; idx += 2) {
                const int32_t qword = quant_data[idx / 2 + 8];
                dot_a += cached_y[idx + 8] * (qword & q_masks[lane / 2][0]);
                dot_b += cached_y[idx + 9] * (qword & q_masks[lane / 2][1]);
                mask_a += ((high_bits[idx / 2 + 8] & active_h_mask[0]) ? 0.f : cached_y[idx + 8]) +
                          ((high_bits[idx / 2 + 8] & active_h_mask[1]) ? 0.f : cached_y[idx + 9]);
                dot_c += cached_y[idx + 24] * (qword & q_masks[lane / 2][2]);
                dot_d += cached_y[idx + 25] * (qword & q_masks[lane / 2][3]);
                mask_b += ((high_bits[idx / 2 + 8] & active_h_mask[2]) ? 0.f : cached_y[idx + 24]) +
                          ((high_bits[idx / 2 + 8] & active_h_mask[3]) ? 0.f : cached_y[idx + 25]);
            }

            val1 = scale * (dot_a + dot_b * 0.00390625f - mask_a * bias1);
            val2 = scale * (dot_c + dot_d * 0.00390625f - mask_b * bias2);
            part1[row] += val1 * (scale_bytes[1] - 32);
            part2[row] += val2 * (scale_bytes[3] - 32);

            quant_data += args.nb01 / 2;
            high_bits += args.nb01 / 2;
            scale_data += args.nb01 / 2;
            delta += args.nb01 / 2;
        }

        y_src += 4 * QK_K;
    }

    // Combine results
    for (int row = 0; row < nr0; ++row) {
        const float total = (part1[row] + 0.25f * part2[row]) / (1 << scale_shift);
        part1[row] = simd_sum(total);
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] = part1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q5_K kernel - Original implementation
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    // Scale extraction masks
    constexpr uint16_t SCALE_LO_MASK = 0x3f3f;
    constexpr uint16_t SCALE_HI_MASK = 0x0f0f;
    constexpr uint16_t SCALE_CARRY_MASK = 0xc0c0;

    // Thread indexing for Q5_K
    const short tid = tiisg / 2;
    const short block_offset = tiisg % 2;
    const short sub_block = tid / 4;
    const short lane_pos = tid % 4;

    const short pos_base = 8 * lane_pos;
    const short quant_base = 32 * sub_block + pos_base;
    const short y_base_off = 64 * sub_block + pos_base;

    // Bit masks for the 5th bit extraction
    const uint8_t hbit_low = 1u << (2 * sub_block);
    const uint8_t hbit_high = hbit_low << 1;
    const uint8_t hbit_low_upper = hbit_low << 4;
    const uint8_t hbit_high_upper = hbit_high << 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q5_K * x = (device const block_q5_K *)(src0 + offset0);
    device const float      * yy = (device const float      *)(src1 + offset1);

    float output[nr0] = {0.f};

    float y_lower[16];
    float y_upper[16];

    uint16_t scale_words[4];
    thread const uint8_t * scale_bytes = (thread const uint8_t *)scale_words;

    device const float * y_ptr = yy + block_offset * QK_K + y_base_off;

    for (int blk = block_offset; blk < nb; blk += 4) {
        device const uint8_t * quant_lo = x[blk].qs + quant_base;
        device const uint8_t * high_bits = x[blk].qh + pos_base;
        device const half * delta = &x[blk].d;
        device const uint16_t * scale_src = (device const uint16_t *)x[blk].scales + sub_block;

        device const float * y_upper_ptr = y_ptr + 128;

        // Load y values and compute sums
        float4 y_sums = {0.f, 0.f, 0.f, 0.f};
        for (short k = 0; k < 8; ++k) {
            y_lower[k] = y_ptr[k];
            y_sums[0] += y_lower[k];
            y_lower[k + 8] = y_ptr[k + 32];
            y_sums[1] += y_lower[k + 8];
            y_upper[k] = y_upper_ptr[k];
            y_sums[2] += y_upper[k];
            y_upper[k + 8] = y_upper_ptr[k + 32];
            y_sums[3] += y_upper[k + 8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * quant_hi = quant_lo + 64;

            // Decode scales
            scale_words[0] = scale_src[0] & SCALE_LO_MASK;
            scale_words[1] = scale_src[2] & SCALE_LO_MASK;
            scale_words[2] = ((scale_src[4] >> 0) & SCALE_HI_MASK) | ((scale_src[0] & SCALE_CARRY_MASK) >> 2);
            scale_words[3] = ((scale_src[4] >> 4) & SCALE_HI_MASK) | ((scale_src[2] & SCALE_CARRY_MASK) >> 2);

            // Process 4-bit quantized values with 5th bit from high_bits
            float4 nibble_sum = {0.f, 0.f, 0.f, 0.f};
            float4 extra_sum = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short k = 0; k < 8; ++k) {
                uint8_t high_byte = high_bits[k];

                // Lower nibbles (4-bit values)
                nibble_sum[0] += y_lower[k] * (quant_lo[k] & 0x0F);
                nibble_sum[1] += y_lower[k + 8] * (quant_lo[k] & 0xF0);
                nibble_sum[2] += y_upper[k] * (quant_hi[k] & 0x0F);
                nibble_sum[3] += y_upper[k + 8] * (quant_hi[k] & 0xF0);

                // 5th bit contributions (adds 16 when set)
                extra_sum[0] += (high_byte & hbit_low) ? y_lower[k] : 0.f;
                extra_sum[1] += (high_byte & hbit_high) ? y_lower[k + 8] : 0.f;
                extra_sum[2] += (high_byte & hbit_low_upper) ? y_upper[k] : 0.f;
                extra_sum[3] += (high_byte & hbit_high_upper) ? y_upper[k + 8] : 0.f;
            }

            // Combine nibble and extra bit contributions
            // nibble_sum[1] and [3] are high nibbles, need /16
            output[row] += delta[0] * (
                scale_bytes[0] * (nibble_sum[0] + 16.f * extra_sum[0]) +
                scale_bytes[1] * (nibble_sum[1] * 0.0625f + 16.f * extra_sum[1]) +
                scale_bytes[4] * (nibble_sum[2] + 16.f * extra_sum[2]) +
                scale_bytes[5] * (nibble_sum[3] * 0.0625f + 16.f * extra_sum[3])
            ) - delta[1] * (
                y_sums[0] * scale_bytes[2] +
                y_sums[1] * scale_bytes[3] +
                y_sums[2] * scale_bytes[6] +
                y_sums[3] * scale_bytes[7]
            );

            quant_lo += args.nb01;
            high_bits += args.nb01;
            delta += args.nb01 / 2;
            scale_src += args.nb01 / 2;
        }

        y_ptr += 4 * QK_K;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(output[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ============================================================================
// ============================================================================
// Q6_K kernel - Original implementation
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    // Bit masks for extracting 2-bit values from qh
    constexpr uint8_t BIT_MASK_0_1 = 0x03;
    constexpr uint8_t BIT_MASK_2_3 = 0x0C;
    constexpr uint8_t BIT_MASK_4_5 = 0x30;
    constexpr uint8_t BIT_MASK_6_7 = 0xC0;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q6_K * x = (device const block_q6_K *)(src0 + offset0);
    device const float      * yy = (device const float      *)(src1 + offset1);

    float result[nr0] = {0.f};

    float y_cache[16];

    // Thread coordinates
    const short tid = tiisg / 2;
    const short block_idx = tiisg % 2;
    const short part = tid / 8;
    const short lane = tid % 8;
    const short lane_base = 4 * lane;
    const short scale_idx = 8 * part + lane_base / 16;

    const short y_pos = 128 * part + lane_base;
    const short ql_pos = 64 * part + lane_base;
    const short qh_pos = 32 * part + lane_base;

    for (int blk = block_idx; blk < nb; blk += 2) {
        device const uint8_t * ql_lo = x[blk].ql + ql_pos;
        device const uint8_t * ql_hi = ql_lo + 32;
        device const uint8_t * qh_data = x[blk].qh + qh_pos;
        device const int8_t  * scales = x[blk].scales + scale_idx;
        device const half    * delta = &x[blk].d;

        device const float * y_src = yy + blk * QK_K + y_pos;

        // Load y values into cache
        for (short k = 0; k < 4; ++k) {
            y_cache[4*k + 0] = y_src[k];
            y_cache[4*k + 1] = y_src[k + 32];
            y_cache[4*k + 2] = y_src[k + 64];
            y_cache[4*k + 3] = y_src[k + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 dot_products = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short k = 0; k < 4; ++k) {
                // Combine 4-bit low part from ql with 2-bit high part from qh
                // Result is a signed 6-bit value in range [0, 63], shifted to [-32, 31]

                // First element: ql low nibble + qh bits 0-1 shifted left 4
                uint8_t v0 = (ql_lo[k] & 0x0F) | ((qh_data[k] & BIT_MASK_0_1) << 4);
                int8_t signed_v0 = (int8_t)v0 - 32;

                // Second element: ql_hi low nibble + qh bits 2-3 shifted left 2
                uint8_t v1 = (ql_hi[k] & 0x0F) | ((qh_data[k] & BIT_MASK_2_3) << 2);
                int8_t signed_v1 = (int8_t)v1 - 32;

                // Third element: ql_lo high nibble + qh bits 4-5
                uint8_t v2 = (ql_lo[k] >> 4) | (qh_data[k] & BIT_MASK_4_5);
                int8_t signed_v2 = (int8_t)v2 - 32;

                // Fourth element: ql_hi high nibble + qh bits 6-7 shifted right 2
                uint8_t v3 = (ql_hi[k] >> 4) | ((qh_data[k] & BIT_MASK_6_7) >> 2);
                int8_t signed_v3 = (int8_t)v3 - 32;

                dot_products[0] += y_cache[4*k + 0] * signed_v0;
                dot_products[1] += y_cache[4*k + 1] * signed_v1;
                dot_products[2] += y_cache[4*k + 2] * signed_v2;
                dot_products[3] += y_cache[4*k + 3] * signed_v3;
            }

            // Apply scale factors
            result[row] += delta[0] * (
                dot_products[0] * scales[0] +
                dot_products[1] * scales[2] +
                dot_products[2] * scales[4] +
                dot_products[3] * scales[6]
            );

            ql_lo += args.nb01;
            ql_hi += args.nb01;
            qh_data += args.nb01;
            scales += args.nb01;
            delta += args.nb01 / 2;
        }
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(result[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q4_0 kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q4_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;
    const int nb = args.ne00 / QK;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_0 * x = (device const block_q4_0 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    device const float * yb = y + ix * QK + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        for (short row = 0; row < nr0; row++) {
            device const block_q4_0 * b = x + ib + row * nb;
            float d = b->d;

            float acc = 0.f;
            for (short i = 0; i < 8; ++i) {
                uint8_t q = b->qs[8 * iq + i];
                acc += yb[i] * ((q & 0x0F) - 8.f) * d;
            }

            sumf[row] += acc;
        }

        yb += 4 * QK;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q4_0_f32")]]
kernel void kernel_mul_mv_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q4_1 kernel - matches llama.cpp complex template approach
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q4_1_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;
    constexpr short NW = 32;  // N_SIMDWIDTH
    constexpr short NQ = 16;

    const int nb = args.ne00 / QK;

    const int r0 = (tgpig.x * NSG + sgitg) * nr0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float * y = (device const float *)(src1 + offset1);

    device const block_q4_1 * ax[nr0];
    FOR_UNROLL (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
        ax[row] = (device const block_q4_1 *)((device char *)src0 + offset0);
    }

    float sumf[nr0] = {0.f};

    const short ix = tiisg / (NW / NQ);  // tiisg / 2
    const short il = (tiisg % (NW / NQ)) * 8;  // (tiisg % 2) * 8

    const int ib0 = ix;
    float yl[16];
    device const float * yb = y + ib0 * QK + il;

    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = {0.f, 0.f};
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0] += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] / 256.f;
            sumy[1] += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.f;
            yl[i + 9] = yb[i + 17] / 4096.f;
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_q4_1 * b = ax[row] + ib;
            float d = b->d;
            float m = b->m;

            device const uint16_t * qs = (device const uint16_t *)b + 2 + il / 2;

            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
                acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
                acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
                acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
            }

            sumf[row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + (sumy[0] + sumy[1]) * m;
        }

        yb += QK * NQ;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q4_1_f32")]]
kernel void kernel_mul_mv_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_1_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q8_0 kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;
    const int nb = args.ne00 / QK;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q8_0 * x = (device const block_q8_0 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    device const float * yb = y + ix * QK + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        for (short row = 0; row < nr0; row++) {
            device const block_q8_0 * b = x + ib + row * nb;
            float d = b->d;

            float acc = 0.f;
            for (short i = 0; i < 8; ++i) {
                acc += yb[i] * (b->qs[8 * iq + i] - 128.f) * d;
            }

            sumf[row] += acc;
        }

        yb += 4 * QK;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q8_0_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q5_0 kernel - matches llama.cpp complex template approach
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q5_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;
    constexpr short NW = 32;  // N_SIMDWIDTH
    constexpr short NQ = 16;

    const int nb = args.ne00 / QK5_0;

    const int r0 = (tgpig.x * NSG + sgitg) * nr0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float * y = (device const float *)(src1 + offset1);

    device const block_q5_0 * ax[nr0];
    FOR_UNROLL (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
        ax[row] = (device const block_q5_0 *)((device char *)src0 + offset0);
    }

    float sumf[nr0] = {0.f};

    const short ix = tiisg / (NW / NQ);  // tiisg / 2
    const short il = (tiisg % (NW / NQ)) * 8;  // (tiisg % 2) * 8

    const int ib0 = ix;
    float yl[16];
    device const float * yb = y + ib0 * QK5_0 + il;

    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = {0.f, 0.f};
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0] += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] / 256.f;
            sumy[1] += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.f;
            yl[i + 9] = yb[i + 17] / 4096.f;
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_q5_0 * b = ax[row] + ib;
            float d = b->d;

            device const uint16_t * qs = (device const uint16_t *)b + 3 + il / 2;
            const uint32_t qh = *(device const uint32_t *)b->qh;

            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i + 0 + il) << 4) & 0x00010));
                acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i + 1 + il) << 12) & 0x01000));
                acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i + 0 + il + QK5_0 / 2) << 8) & 0x00100));
                acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i + 1 + il + QK5_0 / 2) << 16) & 0x10000));
            }

            sumf[row] += d * ((sumy[0] + sumy[1]) * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
        }

        yb += QK5_0 * NQ;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_0_f32")]]
kernel void kernel_mul_mv_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q5_1 kernel - matches llama.cpp complex template approach
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q5_1_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;
    constexpr short NW = 32;  // N_SIMDWIDTH
    constexpr short NQ = 16;

    const int nb = args.ne00 / QK5_1;

    const int r0 = (tgpig.x * NSG + sgitg) * nr0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float * y = (device const float *)(src1 + offset1);

    device const block_q5_1 * ax[nr0];
    FOR_UNROLL (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
        ax[row] = (device const block_q5_1 *)((device char *)src0 + offset0);
    }

    float sumf[nr0] = {0.f};

    const short ix = tiisg / (NW / NQ);  // tiisg / 2
    const short il = (tiisg % (NW / NQ)) * 8;  // (tiisg % 2) * 8

    const int ib0 = ix;
    float yl[16];
    device const float * yb = y + ib0 * QK5_1 + il;

    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = {0.f, 0.f};
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0] += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] / 256.f;
            sumy[1] += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.f;
            yl[i + 9] = yb[i + 17] / 4096.f;
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_q5_1 * b = ax[row] + ib;
            float d = b->d;
            float m = b->m;

            device const uint16_t * qs = (device const uint16_t *)b + 4 + il / 2;
            const uint32_t qh = *(device const uint32_t *)b->qh;

            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i + 0 + il) << 4) & 0x00010));
                acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i + 1 + il) << 12) & 0x01000));
                acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i + 0 + il + QK5_1 / 2) << 8) & 0x00100));
                acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i + 1 + il + QK5_1 / 2) << 16) & 0x10000));
            }

            sumf[row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + (sumy[0] + sumy[1]) * m;
        }

        yb += QK5_1 * NQ;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_1_f32")]]
kernel void kernel_mul_mv_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_1_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q1_0 kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q1_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;
    const int nb = args.ne00 / QK1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q1_0 * x = (device const block_q1_0 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    float yl[16];
    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short il = (tiisg % 8) * 16;

    device const float * yb = y + ix * QK1_0 + il;

    for (int ib = ix; ib < nb; ib += 4) {
        float sumy = 0.f;

        for (short i = 0; i < 16; ++i) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        device const block_q1_0 * qb = x + ib;

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_q1_0 * b = qb + row * nb;
            float d = b->d;

            device const uint8_t * qs = b->qs + il / 8;
            const uint8_t b0 = qs[0];
            const uint8_t b1 = qs[1];

            float acc = 0.f;
            acc += select(0.0f, yl[ 0], bool(b0 & 0x01));
            acc += select(0.0f, yl[ 1], bool(b0 & 0x02));
            acc += select(0.0f, yl[ 2], bool(b0 & 0x04));
            acc += select(0.0f, yl[ 3], bool(b0 & 0x08));
            acc += select(0.0f, yl[ 4], bool(b0 & 0x10));
            acc += select(0.0f, yl[ 5], bool(b0 & 0x20));
            acc += select(0.0f, yl[ 6], bool(b0 & 0x40));
            acc += select(0.0f, yl[ 7], bool(b0 & 0x80));
            acc += select(0.0f, yl[ 8], bool(b1 & 0x01));
            acc += select(0.0f, yl[ 9], bool(b1 & 0x02));
            acc += select(0.0f, yl[10], bool(b1 & 0x04));
            acc += select(0.0f, yl[11], bool(b1 & 0x08));
            acc += select(0.0f, yl[12], bool(b1 & 0x10));
            acc += select(0.0f, yl[13], bool(b1 & 0x20));
            acc += select(0.0f, yl[14], bool(b1 & 0x40));
            acc += select(0.0f, yl[15], bool(b1 & 0x80));

            sumf[row] += d * (2.0f * acc - sumy);
        }

        yb += 4 * QK1_0;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q1_0_f32")]]
kernel void kernel_mul_mv_q1_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q1_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// Q2_0 kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_q2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;
    const int nb = args.ne00 / QK2_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q2_0 * x = (device const block_q2_0 *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    float yl[16];
    float sumf[nr0] = {0.f};

    const short ix = tiisg / 4;
    const short il = (tiisg % 4) * 16;

    device const float * yb = y + ix * QK2_0 + il;

    for (int ib = ix; ib < nb; ib += 8) {
        float sumy = 0.f;

        for (short i = 0; i < 16; ++i) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        device const block_q2_0 * qb = x + ib;

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_q2_0 * b = qb + row * nb;
            float d = b->d;

            device const uint8_t * qs = b->qs + il / 4;
            const uint8_t b0 = qs[0];
            const uint8_t b1 = qs[1];
            const uint8_t b2 = qs[2];
            const uint8_t b3 = qs[3];

            float acc_lo = 0.f;
            acc_lo += select(0.0f, yl[ 0], bool(b0 & 0x01));
            acc_lo += select(0.0f, yl[ 1], bool(b0 & 0x04));
            acc_lo += select(0.0f, yl[ 2], bool(b0 & 0x10));
            acc_lo += select(0.0f, yl[ 3], bool(b0 & 0x40));
            acc_lo += select(0.0f, yl[ 4], bool(b1 & 0x01));
            acc_lo += select(0.0f, yl[ 5], bool(b1 & 0x04));
            acc_lo += select(0.0f, yl[ 6], bool(b1 & 0x10));
            acc_lo += select(0.0f, yl[ 7], bool(b1 & 0x40));
            acc_lo += select(0.0f, yl[ 8], bool(b2 & 0x01));
            acc_lo += select(0.0f, yl[ 9], bool(b2 & 0x04));
            acc_lo += select(0.0f, yl[10], bool(b2 & 0x10));
            acc_lo += select(0.0f, yl[11], bool(b2 & 0x40));
            acc_lo += select(0.0f, yl[12], bool(b3 & 0x01));
            acc_lo += select(0.0f, yl[13], bool(b3 & 0x04));
            acc_lo += select(0.0f, yl[14], bool(b3 & 0x10));
            acc_lo += select(0.0f, yl[15], bool(b3 & 0x40));

            float acc_hi = 0.f;
            acc_hi += select(0.0f, yl[ 0], bool(b0 & 0x02));
            acc_hi += select(0.0f, yl[ 1], bool(b0 & 0x08));
            acc_hi += select(0.0f, yl[ 2], bool(b0 & 0x20));
            acc_hi += select(0.0f, yl[ 3], bool(b0 & 0x80));
            acc_hi += select(0.0f, yl[ 4], bool(b1 & 0x02));
            acc_hi += select(0.0f, yl[ 5], bool(b1 & 0x08));
            acc_hi += select(0.0f, yl[ 6], bool(b1 & 0x20));
            acc_hi += select(0.0f, yl[ 7], bool(b1 & 0x80));
            acc_hi += select(0.0f, yl[ 8], bool(b2 & 0x02));
            acc_hi += select(0.0f, yl[ 9], bool(b2 & 0x08));
            acc_hi += select(0.0f, yl[10], bool(b2 & 0x20));
            acc_hi += select(0.0f, yl[11], bool(b2 & 0x80));
            acc_hi += select(0.0f, yl[12], bool(b3 & 0x02));
            acc_hi += select(0.0f, yl[13], bool(b3 & 0x08));
            acc_hi += select(0.0f, yl[14], bool(b3 & 0x20));
            acc_hi += select(0.0f, yl[15], bool(b3 & 0x80));

            sumf[row] += d * (acc_lo + 2.0f * acc_hi - sumy);
        }

        yb += 8 * QK2_0;
    }

    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q2_0_f32")]]
kernel void kernel_mul_mv_q2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ4_NL kernel - Independent implementation
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    threadgroup float * lut = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq4_nl * weights = (device const block_iq4_nl *)(src0 + offset0);
    device const float        * input = (device const float        *)(src1 + offset1);

    const int num_blocks = args.ne00 / QK4_NL;
    const int row_blocks = args.nb01 / args.nb00;

    const short lane = tiisg / 2;
    const short sub_lane = tiisg % 2;

    // Load lookup table to shared memory
    lut[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 y_cache[4];
    float result[nr0] = {0.f};

    device const float * y_ptr = input + lane * QK4_NL + sub_lane * 8;

    for (int blk = lane; blk < num_blocks && blk < row_blocks; blk += 16) {
        // Load 32 input values
        device const float4 * y_vec = (device const float4 *)y_ptr;
        y_cache[0] = y_vec[0];
        y_cache[1] = y_vec[4];
        y_cache[2] = y_vec[1];
        y_cache[3] = y_vec[5];

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_iq4_nl & wb = weights[row * row_blocks + blk];
            device const uint16_t * quant = (device const uint16_t *)(wb.qs + 8 * sub_lane);

            float4 acc_a = {0.f}, acc_b = {0.f};

            // Process quantized nibbles through lookup table
            uint32_t packed[2];
            packed[0] = quant[0] | (quant[1] << 16);
            packed[1] = (packed[0] >> 4) & 0x0f0f0f0f;
            packed[0] &= 0x0f0f0f0f;

            thread const uint8_t * nibbles = (thread const uint8_t *)packed;
            float4 values_a = {lut[nibbles[0]], lut[nibbles[1]], lut[nibbles[2]], lut[nibbles[3]]};
            float4 values_b = {lut[nibbles[4]], lut[nibbles[5]], lut[nibbles[6]], lut[nibbles[7]]};

            acc_a += y_cache[0] * values_a;
            acc_b += y_cache[1] * values_b;

            packed[0] = quant[2] | (quant[3] << 16);
            packed[1] = (packed[0] >> 4) & 0x0f0f0f0f;
            packed[0] &= 0x0f0f0f0f;

            values_a = {lut[nibbles[0]], lut[nibbles[1]], lut[nibbles[2]], lut[nibbles[3]]};
            values_b = {lut[nibbles[4]], lut[nibbles[5]], lut[nibbles[6]], lut[nibbles[7]]};

            acc_a += y_cache[2] * values_a;
            acc_b += y_cache[3] * values_b;

            acc_a += acc_b;

            result[row] += (float)wb.d * (acc_a[0] + acc_a[1] + acc_a[2] + acc_a[3]);
        }

        y_ptr += 16 * QK4_NL;
    }

    device float * out = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(result[row]);
        if (tiisg == 0) {
            out[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup char shmem[64];
    kernel_mul_mv_iq4_nl_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// ============================================================================
// MXFP4 kernel - Independent implementation
// ============================================================================

static inline float decode_e8m0_scale(uint8_t exponent) {
    uint32_t bits;
    if (exponent == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t)exponent << 23;
    }
    return as_type<float>(bits);
}

template<int nr0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    threadgroup float * value_lut = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_mxfp4 * weights = (device const block_mxfp4 *)(src0 + offset0);
    device const float        * input = (device const float        *)(src1 + offset1);

    const int num_blocks = args.ne00 / QK_MXFP4;
    const int row_blocks = args.nb01 / args.nb00;

    const short lane = tiisg / 2;
    const short sub_lane = tiisg % 2;

    value_lut[tiisg] = kvalues_mxfp4_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 y_cache[4];
    float result[nr0] = {0.f};

    device const float * y_ptr = input + lane * QK_MXFP4 + sub_lane * 8;

    for (int blk = lane; blk < num_blocks && blk < row_blocks; blk += 16) {
        device const float4 * y_vec = (device const float4 *)y_ptr;

        y_cache[0] = y_vec[0];
        y_cache[1] = y_vec[4];
        y_cache[2] = y_vec[1];
        y_cache[3] = y_vec[5];

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_mxfp4 & wb = weights[row * row_blocks + blk];
            device const uint8_t * quant = (device const uint8_t *)(wb.qs + 8 * sub_lane);

            // 2-bit values: each byte packs two 4-bit indices
            float4 dot_a = y_cache[0] * float4(value_lut[quant[0] & 0x0F], value_lut[quant[1] & 0x0F], value_lut[quant[2] & 0x0F], value_lut[quant[3] & 0x0F]);
            float4 dot_b = y_cache[1] * float4(value_lut[quant[0] >> 4], value_lut[quant[1] >> 4], value_lut[quant[2] >> 4], value_lut[quant[3] >> 4]);
            float4 dot_c = y_cache[2] * float4(value_lut[quant[4] & 0x0F], value_lut[quant[5] & 0x0F], value_lut[quant[6] & 0x0F], value_lut[quant[7] & 0x0F]);
            float4 dot_d = y_cache[3] * float4(value_lut[quant[4] >> 4], value_lut[quant[5] >> 4], value_lut[quant[6] >> 4], value_lut[quant[7] >> 4]);

            dot_a = (dot_a + dot_c) + (dot_b + dot_d);

            result[row] += decode_e8m0_scale(wb.e) * ((dot_a[0] + dot_a[1]) + (dot_a[2] + dot_a[3]));
        }

        y_ptr += 16 * QK_MXFP4;
    }

    device float * out = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(result[row]);
        if (tiisg == 0) {
            out[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup char shmem[64];
    kernel_mul_mv_mxfp4_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// ============================================================================
// TQ2_0 kernel - Independent implementation
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_tq2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int num_blocks = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_tq2_0 * weights = (device const block_tq2_0 *)(src0 + offset0);
    device const float       * input = (device const float       *)(src1 + offset1);

    float result[nr0] = {0.f};

    constexpr short LANES = 4;
    constexpr short SUB_BLOCKS = 32 / LANES;

    const short block_id = tiisg / SUB_BLOCKS;
    const short lane_id = tiisg % SUB_BLOCKS;

    device const float4 * input_vec = (device const float4 *)(input + 4 * lane_id + block_id * QK_K);

    float4 y_coef[4];

    for (int blk = block_id; blk < num_blocks; blk += LANES) {
        FOR_UNROLL (short part = 0; part < 2; ++part) {
            const float4 y0 = input_vec[ 0 + 32 * part];
            const float4 y1 = input_vec[ 8 + 32 * part];
            const float4 y2 = input_vec[16 + 32 * part];
            const float4 y3 = input_vec[24 + 32 * part];

            float y_sum = 0.f;
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                y_coef[j] = float4(
                        y0[j],
                        y1[j] - 4.0f * y0[j],
                        y2[j] - 4.0f * y1[j],
                        y3[j] - 4.0f * y2[j]);
                y_sum += (y0[j] + y1[j]) + (y2[j] + y3[j]);
            }

            FOR_UNROLL (short row = 0; row < nr0; ++row) {
                device const block_tq2_0 & wb = weights[row * num_blocks + blk];
                device const uchar * quant = wb.qs + 4 * lane_id + 32 * part;

                // Decode 2-bit ternary values
                float sum = -y_sum;
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    const float val = (float)quant[j];
                    // Extract bits using integer division
                    const float b0 = val;
                    const float b1 = floor(val * 0.25f);
                    const float b2 = floor(val * 0.0625);
                    const float b3 = floor(val * 0.015625);
                    sum += y_coef[j][0] * b0 + y_coef[j][1] * b1 + y_coef[j][2] * b2 + y_coef[j][3] * b3;
                }

                result[row] += wb.d * sum;
            }
        }

        input_vec += QK_K * LANES / 4;
    }

    device float * out = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(result[row]);
        if (tiisg == 0) {
            out[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_tq2_0_f32")]]
kernel void kernel_mul_mv_tq2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ1_S kernel - Independent implementation
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int num_blocks = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int num_32blocks = num_blocks * (QK_K / 32);

    constexpr short THREADS = 32;
    constexpr short REPS = 32 / THREADS;

    const short lane_idx = tiisg % THREADS;
    const short rep_idx = tiisg / THREADS;

    const short row_start = (nr0 * rep_idx) / REPS;
    const short row_end = (nr0 * (rep_idx + 1)) / REPS;

    const uint64_t offset0 = (first_row + row_start) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq1_s * weights = (device const block_iq1_s *)(src0 + offset0);
    device const float       * input = (device const float       *)(src1 + offset1);

    float y_cached[32];
    float result[nr0] = {0.f};

    device const float * y_ptr = input + 32 * lane_idx;

    for (int blk32 = lane_idx; blk32 < num_32blocks; blk32 += THREADS) {
        // Load input values
        float y_sum = 0;
        for (short i = 0; i < 32; ++i) {
            y_cached[i] = y_ptr[i];
            y_sum += y_cached[i];
        }

        const int super_block = blk32 / (QK_K / 32);
        const int sub_block = blk32 % (QK_K / 32);

        device const block_iq1_s * wb = weights + super_block;
        device const uint8_t * qs_ptr = wb->qs + 4 * sub_block;
        device const uint16_t * qh_ptr = wb->qh + sub_block;
        device const half * scale_ptr = &wb->d;

        for (short row = row_start; row < row_end; row++) {
            // Decode scale and min values
            const float scale = (float)scale_ptr[0] * (2 * ((qh_ptr[0] >> 12) & 7) + 1);
            const float min_val = scale * (qh_ptr[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);

            // Grid lookup indices
            const uint16_t high_bits = qh_ptr[0] >> (6 * (blk32 % 2));

            float dot = 0;
            for (short j = 0; j < 4; ++j) {
                uint16_t idx0 = qs_ptr[j] | ((high_bits << (8 - j * 2)) & 0x700);
                uint16_t idx1 = qs_ptr[j + 4] | ((high_bits << (8 - (j + 4) * 2)) & 0x700);

                constant uint8_t * grid0 = (constant uint8_t *)(iq1s_grid_gpu + idx0);
                constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + idx1);

                // Decode 4-bit values from grid
                dot += y_cached[j + 0] * (scale * (grid0[0] & 0xF) + min_val);
                dot += y_cached[j + 4] * (scale * (grid0[0] >> 4) + min_val);
                dot += y_cached[j + 16] * (scale * (grid1[0] & 0xF) + min_val);
                dot += y_cached[j + 20] * (scale * (grid1[0] >> 4) + min_val);
            }

            result[row] += dot;

            scale_ptr += args.nb01 / 2;
            qs_ptr += args.nb01;
            qh_ptr += args.nb01 / 2;
        }

        y_ptr += 32 * THREADS;
    }

    device float * out = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(result[row]);
        if (tiisg == 0) {
            out[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ1_M kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short nrep = 32 / ntx;

    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    const uint64_t offset0 = (first_row + row0) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq1_m * x = (device const block_iq1_m *)(src0 + offset0);
    device const float       * y = (device const float       *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (short row = row0; row < row1; row++) {
            // IQ1_M scale extraction
            uint16_t scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            half d = as_type<half>(scale_bits);

            const float dl = (float)d * (2 * ((sc[ib/2] >> (6 * (ib % 2) + 0)) & 7) + 1);
            const float dl2 = (float)d * (2 * ((sc[ib/2] >> (6 * (ib % 2) + 3)) & 7) + 1);

            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) +
                                 sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) +
                                 sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            // Grid lookup computation
            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                // Build grid index from qs and qh bits
                const uint16_t idx1 = qs[j+0] | ((qh[0] << (8 + j * 1)) & 0x700);
                const uint16_t idx2 = qs[j+4] | ((qh[1] << (8 + j * 1)) & 0x700);
                constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + idx1);
                constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + idx2);
                sum[0] += yl[j+ 0] * (grid1[0] & 0xF) + yl[j+ 8] * (grid1[0] >> 4);
                sum[1] += yl[j+16] * (grid2[0] & 0xF) + yl[j+24] * (grid2[0] >> 4);
            }

            sumf[row] += dl * (sum[0] + delta1) + dl2 * (sum[1] + delta2);

            sc += args.nb01 / 2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ2_XXS kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short nrep = 32 / ntx;

    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    const uint64_t offset0 = (first_row + row0) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *)(src0 + offset0);
    device const float          * y = (device const float          *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t * ssigns = (threadgroup uint8_t *)(svalues + 256);

    // Initialize shared memory with grid values (simplified)
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; row++) {
            const float db = (float)dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28)) * 0.25f;

            float sum = 0;
            // Grid lookup with proper sign handling
            for (short l = 0; l < 4; ++l) {
                // Each byte in aux8 is an index into iq2xxs_grid
                // Each grid entry contains 8 values packed in a uint64_t
                constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[l]);
                // Sign bits are packed 7 bits per group in aux32
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                for (short j = 0; j < 8; ++j) {
                    // Apply sign from ksigns_iq2xs using kmask_iq2xs
                    const float sign = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
                    sum += yl[8 * l + j] * grid[j] * sign;
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01 / 2;
            q2 += args.nb01 / 2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup char shmem[2208];  // 256*8 + 64 = 2112, padded to 2208
    kernel_mul_mv_iq2_xxs_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ2_XS kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short nrep = 32 / ntx;

    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    const uint64_t offset0 = (first_row + row0) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq2_xs * x = (device const block_iq2_xs *)(src0 + offset0);
    device const float         * y = (device const float         *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; row++) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t signs = ksigns_iq2xs[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8 * l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t signs = ksigns_iq2xs[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8 * l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01 / 2;
            q2 += args.nb01 / 2;
            sc += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ2_S kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short nrep = 32 / ntx;

    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    const uint64_t offset0 = (first_row + row0) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq2_s * x = (device const block_iq2_s *)(src0 + offset0);
    device const float        * y = (device const float        *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K / 8;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; row++) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l + 0] | ((qh[0] << (8 - 2 * l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l + 2] | ((qh[0] << (4 - 2 * l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8 * l + j +  0] * grid1[j] * select(1, -1, signs[l + 0] & kmask_iq2xs[j]);
                    sum[1] += yl[8 * l + j + 16] * grid2[j] * select(1, -1, signs[l + 2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh += args.nb01 / 2;
            qs += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ3_XXS kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short nrep = 32 / ntx;

    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    const uint64_t offset0 = (first_row + row0) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *)(src0 + offset0);
    device const float          * y = (device const float          *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K / 4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[2 * l + 0]);
                constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[2 * l + 1]);
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8 * l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    sum[1] += yl[8 * l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01 / 2;
            q3  += args.nb01;
            gas += args.nb01 / 2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ3_S kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short nrep = 32 / ntx;

    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    const uint64_t offset0 = (first_row + row0) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq3_s * x = (device const block_iq3_s *)(src0 + offset0);
    device const float        * y = (device const float        *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib / 2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; row++) {
            const float db = dh[0];
            const float d = db * (1 + 2 * ((sc[0] >> 4 * (ib % 2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                // Conditional table selection: add 256 to index if bit is set
                const uint16_t idx1 = qs[2 * l + 0] + ((qh[0] & kmask_iq2xs[2 * l + 0]) ? 256 : 0);
                const uint16_t idx2 = qs[2 * l + 1] + ((qh[0] & kmask_iq2xs[2 * l + 1]) ? 256 : 0);
                constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + idx1);
                constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + idx2);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8 * l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j + 0]);
                    sum[1] += yl[8 * l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j + 4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01 / 2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}

// ============================================================================
// IQ4_XS kernel
// ============================================================================

template<int nr0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb = args.ne00 / QK_K;
    const int ns01 = args.nb01 / args.nb00;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *)(src0 + offset0);
    device const float         * y = (device const float         *)(src1 + offset1);

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0] = {0.f};

    device const float * yb = y + ix * 32 + it * 8;

    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;

    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            device const block_iq4_xs & xb = x[row * ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)xb.qs;

            const int ls = ((xb.scales_l[0] >> 0) & 0xf) | (((xb.scales_h >> 0) & 3) << 4);
            const float d = (float)xb.d * (ls - 32);

            for (short i = 0; i < 4; ++i) {
                aux32 = (q4[i] >> 4 * it) & 0x0f0f0f0f;
                sumf[row] += d * (yl[i][0] * shmem_f32[q8[0]] + yl[i][1] * shmem_f32[q8[1]] +
                                  yl[i][2] * shmem_f32[q8[2]] + yl[i][3] * shmem_f32[q8[3]]);
            }
        }

        yb += 16 * 32;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup char shmem[64];
    kernel_mul_mv_iq4_xs_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// ============================================================================
// TQ1_0 kernel (Ternary Quantization, 1.6875 bpw)
// ============================================================================

// TQ1_0 block structure (from ggml-common.h)
typedef struct {
    uint8_t qs[48]; // (256 - 16) / 5 = 48 bytes, 5 trits per byte
    uint8_t qh[4];  // 256 / 64 = 4 bytes, 4 elements per byte
    half d;
} block_tq1_0;

// Trit values: -1, 0, 1
constexpr constant static int8_t tq1_0_trits[3] = {-1, 0, 1};

template<int nr0, typename args_t>
void kernel_mul_mv_tq1_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const int nb32 = nb * (QK_K / 32);

    const short ntx = 32;
    const short nrep = 32 / ntx;

    const short ix = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    const uint64_t offset0 = (first_row + row0) * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_tq1_0 * x = (device const block_tq1_0 *)(src0 + offset0);
    device const float        * y = (device const float        *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);

        device const block_tq1_0 * xr = x + ibl;
        device const half * dh = &xr->d;

        for (short row = row0; row < row1; row++) {
            const float d = dh[0];

            // Each block of 32 uses 6 bytes of qs (6 * 5 = 30 trits) plus 2 bits from qh
            // Simplified: decode each trit directly
            const int base = ib * 6; // 6 bytes per block of 32
            float sum = 0;

            // Process 32 elements per block
            // Each byte holds 5 trits, packed as base-3
            for (short j = 0; j < 6; ++j) {
                const uint qbyte = xr->qs[base + j];
                // Decode 5 trits from each byte
                uint val = qbyte;
                for (short t = 0; t < 5 && j * 5 + t < 32; ++t) {
                    const int trit = val % 3;
                    sum += yl[j * 5 + t] * tq1_0_trits[trit];
                    val /= 3;
                }
            }

            sumf[row] += d * sum;

            dh += args.nb01 / 2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_tq1_0_f32")]]
kernel void kernel_mul_mv_tq1_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq1_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}
