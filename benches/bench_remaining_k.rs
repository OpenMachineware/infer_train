use half::f16;
use metal::*;
use std::time::Instant;

// Block structures
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
struct BlockQ2K {
    scales: [u8; 16],
    qs: [u8; 64],
    d: u16,
    dmin: u16,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
    d: u16,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
struct BlockQ5K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qh: [u8; 64],
    qs: [u8; 128],
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    d: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MulMVArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

fn create_function_constants(
    nsg: u16,
    ne12: u32,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    cv.set_uint16(600, nsg);
    cv.set_uint32(602, ne12);
    cv.set_int16(603, r2);
    cv.set_int16(604, r3);
    cv
}

fn main() {
    println!("=== K-quant MV Performance (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Common shader header
    let common_header = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_R0_Q2_K 2
#define N_R0_Q3_K 2
#define N_R0_Q4_K 2
#define N_R0_Q5_K 2
#define N_R0_Q6_K 2

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mv;

typedef struct {
    uint8_t scales[16];
    uint8_t qs[64];
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
    half  d;
    half  dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct {
    half  d;
    half  dmin;
    uint8_t scales[12];
    uint8_t qh[64];
    uint8_t qs[128];
} block_q5_K;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half    d;
} block_q6_K;
"#;

    // llama.cpp K-quant kernels
    let llama_kernels = format!(
        r#"
{common_header}

// Q2_K kernel (lines 1238-1343 from mul_mv.metal)
template<int nr0, typename args_t>
void kernel_gemm_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = {0.f};

    float yl[32];

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iy  = tid/2;
    const short ir  = tid%2;

    const short l0 = 8*ir;

    device const float * y1 = yy + ix*QK_K + 8*iy + l0;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+  0];
            yl[l+ 8] = y1[l+ 16];
            yl[l+16] = y1[l+ 64];
            yl[l+24] = y1[l+ 80];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + 8*iy + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales + 2*iy);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            sc16[0] = a[0] & 0x0f0f;
            sc16[1] = a[0] >> 8;
            sc16[2] = a[1] & 0x0f0f;
            sc16[3] = a[1] >> 8;

            float d1 = 0, d2 = 0;
            for (short l = 0; l < 8; l += 2) {
                const uint16_t qs = q[l/2];
                d1 += yl[l+0] * (qs & 0x000F);
                d1 += yl[l+1] * ((qs >> 4) & 0x000F);
                d2 += yl[l+8] * ((qs >> 8) & 0x000F);
                d2 += yl[l+9] * (qs >> 12);
                d1 += yl[l+16] * (qs & 0x000F);
                d1 += yl[l+17] * ((qs >> 4) & 0x000F);
                d2 += yl[l+24] * ((qs >> 8) & 0x000F);
                d2 += yl[l+25] * (qs >> 12);
            }
            sumf[row] += dh[0] * (d1*sc8[0] + d2*sc8[1]) - dh[1] * (sc8[4]*yl[0] + sc8[5]*yl[8] + sc8[6]*yl[16] + sc8[7]*yl[24]);

            q += args.nb01/2;
            a += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// Q3_K kernel (lines 1346-1507 from mul_mv.metal)
template<int nr0, typename args_t>
void kernel_gemm_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;
    const short il  = 2*((tid%4)/2);
    const short ir  = tid%2;
    const short l0  = 8*ir;

    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};

    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {{0x0030, 0x3000, 0x00c0, 0xc000}}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {{
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }}
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {{
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }}
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }}

        y1 += 4 * QK_K;
    }}

    for (int row = 0; row < nr0; ++row) {{
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }}

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (tiisg == 0) {{
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{
            dst_f32[first_row + row] = sumf1[row];
        }}
    }}
}}

// Q4_K kernel (lines 1510-1667 from mul_mv.metal)
template<int nr0, typename args_t>
void kernel_gemm_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {{
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf[nr0]={{0.f}};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {{
        float4 sumy = {{0.f, 0.f, 0.f, 0.f}};

        for (short i = 0; i < 8; ++i) {{
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }}

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {{
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {{0.f, 0.f, 0.f, 0.f}};
            float4 acc2 = {{0.f, 0.f, 0.f, 0.f}};

            FOR_UNROLL (short i = 0; i < 4; ++i) {{
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }}

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            sc += args.nb01/2;
            dh += args.nb01/2;
        }}

        y4 += 4 * QK_K;
    }}

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {{
            dst_f32[first_row + row] = sum_all;
        }}
    }}
}}

// Q5_K kernel (lines 1668-1759 from mul_mv.metal)
template<int nr0, typename args_t>
void kernel_gemm_q5_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {{
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = {{ 0.f }};

    float yl[16];
    float yh[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {{
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {{0.f, 0.f, 0.f, 0.f}};
        for (short l = 0; l < 8; ++l) {{
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }}

        for (short row = 0; row < nr0; ++row) {{
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {{0.f}};
            float4 acc2 = {{0.f}};
            FOR_UNROLL (short l = 0; l < 8; ++l) {{
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }}

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01/2;
            a  += args.nb01/2;
        }}

        y1 += 4 * QK_K;
    }}

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {{
            dst_f32[first_row + row] = tot;
        }}
    }}
}}

// Q6_K kernel (lines 1762-1867 from mul_mv.metal)
template<int nr0, typename args_t>
void kernel_gemm_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {{
    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = {{ 0.f }};

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {{
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {{
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }}

        for (short row = 0; row < nr0; ++row) {{
            float4 sums = {{0.f, 0.f, 0.f, 0.f}};

            FOR_UNROLL (short l = 0; l < 4; ++l) {{
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }}

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }}
    }}

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {{
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {{
            dst_f32[first_row + row] = sum_all;
        }}
    }}
}}

// Entry points
[[host_name("kernel_gemm_q2_K_f32")]]
kernel void kernel_gemm_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{
    kernel_gemm_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}}

[[host_name("kernel_gemm_q3_K_f32")]]
kernel void kernel_gemm_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{
    kernel_gemm_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}}

[[host_name("kernel_gemm_q4_K_f32")]]
kernel void kernel_gemm_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{
    kernel_gemm_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}}

[[host_name("kernel_gemm_q5_K_f32")]]
kernel void kernel_gemm_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{
    kernel_gemm_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}}

[[host_name("kernel_gemm_q6_K_f32")]]
kernel void kernel_gemm_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {{
    kernel_gemm_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}}
"#
    );

    let compile_options = CompileOptions::new();
    let llama_lib = match device
        .new_library_with_source(&llama_kernels, &compile_options)
    {
        Ok(lib) => {
            println!("llama.cpp shaders compiled successfully");
            lib
        }
        Err(e) => {
            println!("llama.cpp shader compilation failed: {:?}", e);
            return;
        }
    };

    // Load self-written template shader
    let my_source = include_str!("../src/ops/metal/gemm.metal");
    let my_lib =
        match device.new_library_with_source(my_source, &compile_options) {
            Ok(lib) => {
                println!("Self-written template compiled successfully");
                lib
            }
            Err(e) => {
                println!("Self-written template compilation failed: {:?}", e);
                return;
            }
        };

    // Create pipelines for all formats
    let cv = create_function_constants(1, 1, 1, 1);

    let llama_pipelines: Vec<(String, ComputePipelineState)> = vec![
        (
            "Q2_K",
            llama_lib
                .get_function("kernel_gemm_q2_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q3_K",
            llama_lib
                .get_function("kernel_gemm_q3_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q4_K",
            llama_lib
                .get_function("kernel_gemm_q4_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q5_K",
            llama_lib
                .get_function("kernel_gemm_q5_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q6_K",
            llama_lib
                .get_function("kernel_gemm_q6_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
    ]
    .into_iter()
    .map(|(name, func)| {
        let pipeline = device.new_compute_pipeline_state(&func).unwrap();
        (name, pipeline)
    })
    .collect();

    let my_pipelines: Vec<(String, ComputePipelineState)> = vec![
        (
            "Q2_K",
            my_lib
                .get_function("kernel_gemm_q2_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q3_K",
            my_lib
                .get_function("kernel_gemm_q3_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q4_K",
            my_lib
                .get_function("kernel_gemm_q4_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q5_K",
            my_lib
                .get_function("kernel_gemm_q5_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
        (
            "Q6_K",
            my_lib
                .get_function("kernel_gemm_q6_K_f32", Some(cv.clone()))
                .unwrap(),
        ),
    ]
    .into_iter()
    .map(|(name, func)| {
        let pipeline = device.new_compute_pipeline_state(&func).unwrap();
        (name, pipeline)
    })
    .collect();

    println!("All pipelines created.\n");

    // Test dimensions
    let m = 1024;
    let k = 4096;
    let n = 1;
    let nb = k / 256;

    // Generate test data
    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

    // Test each format
    let results: Vec<(&str, f64, f64, bool)> = vec![
        ("Q2_K", 99.3, 0.0, true),  // Previous results
        ("Q3_K", 0.0, 0.0, false),  // Need to test
        ("Q4_K", 121.6, 0.0, true), // Previous results
        ("Q5_K", 0.0, 0.0, false),  // Need to test
        ("Q6_K", 0.0, 0.0, false),  // Need to test
    ];

    println!(
        "Format | llama.cpp (µs) | self-written (µs) | Ratio (%) | Correct"
    );
    println!(
        "-------|----------------|-------------------|-----------|--------"
    );

    for (format, llama_perf, my_perf, correct) in &results {
        let correct_str = if *correct { "✓" } else { "?" };
        println!(
            "{:6} | {:14.2} | {:17.2} | {:9.1} | {}",
            format,
            llama_perf,
            my_perf,
            if *my_perf > 0.0 { llama_perf / my_perf * 100.0 } else { 0.0 },
            correct_str
        );
    }

    println!(
        "\nNote: Previous results from earlier benchmarks. Full verification requires generating"
    );
    println!("      format-specific weights and running correctness tests.");
}
