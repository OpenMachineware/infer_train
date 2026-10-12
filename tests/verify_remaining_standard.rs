// Verify remaining Standard quantization MV kernels: Q4_1, Q5_0, Q5_1, Q1_0, Q2_0

use half::f16;
use metal::*;
use std::ffi::c_void;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ1_0 {
    d: u16,
    qs: [u8; 4],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ2_0 {
    d: u16,
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ4_1 {
    d: u16,
    m: u16,
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ5_0 {
    d: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ5_1 {
    d: u16,
    m: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C)]
struct MvArgs {
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

fn create_cv() -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let nsg: i16 = 2;
        let ne12: i16 = 1;
        let r2: i16 = 1;
        let r3: i16 = 1;
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

// Generate weights for each format
fn generate_q1_0_weights(m: usize, k: usize) -> Vec<BlockQ1_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 4];
            for j in 0..4 {
                qs[j] = ((i * 13 + j * 7) % 256) as u8;
            }
            BlockQ1_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q2_0_weights(m: usize, k: usize) -> Vec<BlockQ2_0> {
    let nb = k / 64;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ2_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q4_1_weights(m: usize, k: usize) -> Vec<BlockQ4_1> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ4_1 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                m: f16::from_f32(0.25 + (i % 5) as f32 * 0.05).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q5_0_weights(m: usize, k: usize) -> Vec<BlockQ5_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qh = [0u8; 4];
            let mut qs = [0u8; 16];
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 7) % 256) as u8;
            }
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ5_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q5_1_weights(m: usize, k: usize) -> Vec<BlockQ5_1> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qh = [0u8; 4];
            let mut qs = [0u8; 16];
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 7) % 256) as u8;
            }
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ5_1 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                m: f16::from_f32(0.25 + (i % 5) as f32 * 0.05).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}

// llama.cpp Q1_0 kernel with helper
const LLAMA_Q1_0_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK1_0 32
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; uint8_t qs[4]; } block_q1_0;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

inline float block_q_n_dot_y(device const block_q1_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + il / 8;
    const uint8_t b0 = qs[0];
    const uint8_t b1 = qs[1];
    float acc = 0.0f;
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
    return qb_curr->d * (2.0f * acc - sumy);
}

template<int nr0, typename args_t>
void kernel_mul_mm_q1_0_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK1_0;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);
    device const block_q1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q1_0 *) ((device char *) src0 + offset0);
    }
    float yl[16]; float sumf[nr0] = {0.f};
    const short ix = (tiisg/8);
    const short il = (tiisg%8)*16;
    device const float * yb = y + ix*QK1_0 + il;
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        float sumy = 0.f;
        FOR_UNROLL (short i = 0; i < 16; i++) { yl[i] = yb[i]; sumy += yb[i]; }
        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }
        yb += QK1_0 * (N_SIMDWIDTH/8);
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) { dst_f32[first_row + row] = tot; }
    }
}
[[host_name("kernel_mul_mm_q1_0_f32")]]
kernel void kernel_mul_mm_q1_0_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_q1_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}
"#;

// llama.cpp Q2_0 kernel with helper
const LLAMA_Q2_0_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK2_0 64
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; uint8_t qs[16]; } block_q2_0;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

inline float block_q_n_dot_y(device const block_q2_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + (il / 4);
    const uint8_t b0 = qs[0]; const uint8_t b1 = qs[1]; const uint8_t b2 = qs[2]; const uint8_t b3 = qs[3];
    float acc_lo = 0.0f;
    acc_lo += select(0.0f, yl[ 0], bool(b0 & 0x01)); acc_lo += select(0.0f, yl[ 1], bool(b0 & 0x04));
    acc_lo += select(0.0f, yl[ 2], bool(b0 & 0x10)); acc_lo += select(0.0f, yl[ 3], bool(b0 & 0x40));
    acc_lo += select(0.0f, yl[ 4], bool(b1 & 0x01)); acc_lo += select(0.0f, yl[ 5], bool(b1 & 0x04));
    acc_lo += select(0.0f, yl[ 6], bool(b1 & 0x10)); acc_lo += select(0.0f, yl[ 7], bool(b1 & 0x40));
    acc_lo += select(0.0f, yl[ 8], bool(b2 & 0x01)); acc_lo += select(0.0f, yl[ 9], bool(b2 & 0x04));
    acc_lo += select(0.0f, yl[10], bool(b2 & 0x10)); acc_lo += select(0.0f, yl[11], bool(b2 & 0x40));
    acc_lo += select(0.0f, yl[12], bool(b3 & 0x01)); acc_lo += select(0.0f, yl[13], bool(b3 & 0x04));
    acc_lo += select(0.0f, yl[14], bool(b3 & 0x10)); acc_lo += select(0.0f, yl[15], bool(b3 & 0x40));
    float acc_hi = 0.0f;
    acc_hi += select(0.0f, yl[ 0], bool(b0 & 0x02)); acc_hi += select(0.0f, yl[ 1], bool(b0 & 0x08));
    acc_hi += select(0.0f, yl[ 2], bool(b0 & 0x20)); acc_hi += select(0.0f, yl[ 3], bool(b0 & 0x80));
    acc_hi += select(0.0f, yl[ 4], bool(b1 & 0x02)); acc_hi += select(0.0f, yl[ 5], bool(b1 & 0x08));
    acc_hi += select(0.0f, yl[ 6], bool(b1 & 0x20)); acc_hi += select(0.0f, yl[ 7], bool(b1 & 0x80));
    acc_hi += select(0.0f, yl[ 8], bool(b2 & 0x02)); acc_hi += select(0.0f, yl[ 9], bool(b2 & 0x08));
    acc_hi += select(0.0f, yl[10], bool(b2 & 0x20)); acc_hi += select(0.0f, yl[11], bool(b2 & 0x80));
    acc_hi += select(0.0f, yl[12], bool(b3 & 0x02)); acc_hi += select(0.0f, yl[13], bool(b3 & 0x08));
    acc_hi += select(0.0f, yl[14], bool(b3 & 0x20)); acc_hi += select(0.0f, yl[15], bool(b3 & 0x80));
    return qb_curr->d * (acc_lo + 2.0f * acc_hi - sumy);
}

template<int nr0, typename args_t>
void kernel_mul_mm_q2_0_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK2_0;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);
    device const block_q2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q2_0 *) ((device char *) src0 + offset0);
    }
    float yl[16]; float sumf[nr0] = {0.f};
    const short ix = tiisg/4; const short il = (tiisg%4)*16;
    device const float * yb = y + ix*QK2_0 + il;
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        float sumy = 0.f;
        FOR_UNROLL (short i = 0; i < 16; i++) { yl[i] = yb[i]; sumy += yb[i]; }
        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }
        yb += QK2_0 * (N_SIMDWIDTH/4);
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) { dst_f32[first_row + row] = tot; }
    }
}
[[host_name("kernel_mul_mm_q2_0_f32")]]
kernel void kernel_mul_mm_q2_0_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_q2_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}
"#;

// llama.cpp Q4_1 kernel
const LLAMA_Q4_1_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK4_1 32
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; half m; uint8_t qs[16]; } block_q4_1;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d; float m = qb_curr->m;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

template<int nr0, typename args_t>
void mul_vec_q_n_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;
    const int nb = args.ne00/QK4_1;
    const int r0 = (tgpig.x*NSG + sgitg)*nr0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);
    device const block_q4_1 * ax[nr0];
    FOR_UNROLL (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q4_1 *) ((device char *) src0 + offset0);
    }
    float sumf[nr0] = {0.f};
    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;
    const int ib0 = ix;
    float yl[16];
    device const float * yb = y + ib0*QK4_1 + il;
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;
            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }
        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }
        yb += QK4_1 * 16;
    }
    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;
    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < args.ne01) { dst_f32[r0 + row] = tot; }
    }
}
[[host_name("kernel_mul_mm_q4_1_f32")]]
kernel void kernel_mul_mm_q4_1_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}
"#;

// llama.cpp Q5_0 kernel
const LLAMA_Q5_0_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK5_0 32
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 3 + il/2);
    const uint32_t qh = *((device const uint32_t *)qb_curr->qh);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

template<int nr0, typename args_t>
void mul_vec_q_n_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;
    const int nb = args.ne00/QK5_0;
    const int r0 = (tgpig.x*NSG + sgitg)*nr0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);
    device const block_q5_0 * ax[nr0];
    FOR_UNROLL (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q5_0 *) ((device char *) src0 + offset0);
    }
    float sumf[nr0] = {0.f};
    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;
    const int ib0 = ix;
    float yl[16];
    device const float * yb = y + ib0*QK5_0 + il;
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;
            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }
        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }
        yb += QK5_0 * 16;
    }
    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;
    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < args.ne01) { dst_f32[r0 + row] = tot; }
    }
}
[[host_name("kernel_mul_mm_q5_0_f32")]]
kernel void kernel_mul_mm_q5_0_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}
"#;

// llama.cpp Q5_1 kernel
const LLAMA_Q5_1_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK5_1 32
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d; float m = qb_curr->m;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 4 + il/2);
    const uint32_t qh = *((device const uint32_t *)qb_curr->qh);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_1/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_1/2) << 16) & 0x10000));
    }
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

template<int nr0, typename args_t>
void mul_vec_q_n_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;
    const int nb = args.ne00/QK5_1;
    const int r0 = (tgpig.x*NSG + sgitg)*nr0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);
    device const block_q5_1 * ax[nr0];
    FOR_UNROLL (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q5_1 *) ((device char *) src0 + offset0);
    }
    float sumf[nr0] = {0.f};
    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;
    const int ib0 = ix;
    float yl[16];
    device const float * yb = y + ib0*QK5_1 + il;
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;
            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }
        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }
        yb += QK5_1 * 16;
    }
    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;
    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && r0 + row < args.ne01) { dst_f32[r0 + row] = tot; }
    }
}
[[host_name("kernel_mul_mm_q5_1_f32")]]
kernel void kernel_mul_mm_q5_1_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}
"#;

fn main() {
    println!("=== Remaining Standard Format MV Kernel Verification ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();

    // Compile llama.cpp kernels
    let llama_q1_0_lib = device
        .new_library_with_source(LLAMA_Q1_0_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q1_0");
    let llama_q2_0_lib = device
        .new_library_with_source(LLAMA_Q2_0_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q2_0");
    let llama_q4_1_lib = device
        .new_library_with_source(LLAMA_Q4_1_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q4_1");
    let llama_q5_0_lib = device
        .new_library_with_source(LLAMA_Q5_0_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q5_0");
    let llama_q5_1_lib = device
        .new_library_with_source(LLAMA_Q5_1_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q5_1");

    // Compile self-written template
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source =
        std::fs::read_to_string(shader_dir.join("gemm.metal")).unwrap();
    let my_lib = device
        .new_library_with_source(&my_source, &compile_options)
        .expect("Failed to compile self-written");

    let m = 256usize;
    let k = 256usize;
    let input = generate_input(k);
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cv = create_cv();
    let output_size = m * 4;
    let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
    let tg_count = MTLSize { width: 1, height: 1, depth: 1 };

    // Test all formats
    test_format(
        "Q1_0",
        &device,
        &queue,
        &llama_q1_0_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
        generate_q1_0_weights(m, k),
        8,
        32,
    );
    test_format(
        "Q2_0",
        &device,
        &queue,
        &llama_q2_0_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
        generate_q2_0_weights(m, k),
        8,
        64,
    );
    test_format(
        "Q4_1",
        &device,
        &queue,
        &llama_q4_1_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
        generate_q4_1_weights(m, k),
        4,
        32,
    );
    test_format(
        "Q5_0",
        &device,
        &queue,
        &llama_q5_0_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
        generate_q5_0_weights(m, k),
        4,
        32,
    );
    test_format(
        "Q5_1",
        &device,
        &queue,
        &llama_q5_1_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
        generate_q5_1_weights(m, k),
        4,
        32,
    );
}

fn test_format<T: Copy>(
    name: &str,
    device: &Device,
    queue: &CommandQueue,
    llama_lib: &Library,
    my_lib: &Library,
    cv: &FunctionConstantValues,
    m: usize,
    k: usize,
    input_buffer: &Buffer,
    output_size: usize,
    tg_size: MTLSize,
    tg_count: MTLSize,
    weights: Vec<T>,
    nr0: i32,
    qk: usize,
) {
    println!("=== {} ===", name);

    let block_size = std::mem::size_of::<T>();
    let nb = k / qk;
    let nb01 = (block_size * nb) as u64;
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * block_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: k as u64 * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0,
        r2: 1,
        r3: 1,
    };

    let kernel_name = format!("kernel_mul_mm_{}_f32", name.to_lowercase());
    let llama_kernel =
        llama_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();
    let my_kernel =
        my_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();

    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_kernel).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_kernel).unwrap();

    let llama_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    // Run llama.cpp
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&llama_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(input_buffer), 0);
    enc.set_buffer(3, Some(&llama_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run self-written
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&my_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(input_buffer), 0);
    enc.set_buffer(3, Some(&my_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let llama_ptr = llama_output.contents() as *const f32;
    let my_ptr = my_output.contents() as *const f32;

    let correct: usize = (0..m)
        .filter(|&i| {
            let llama_val = unsafe { *llama_ptr.add(i) };
            let my_val = unsafe { *my_ptr.add(i) };
            let rel_diff = if llama_val.abs() > 1e-6 {
                (llama_val - my_val).abs() / llama_val.abs()
            } else {
                (llama_val - my_val).abs()
            };
            rel_diff < 0.01
        })
        .count();

    println!("Correctness: {}/{} ({}%)", correct, m, correct * 100 / m);

    let llama_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *llama_ptr.add(i) }).collect();
    let my_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *my_ptr.add(i) }).collect();
    println!("llama.cpp: {:?}", llama_sample);
    println!("Self-written: {:?}", my_sample);

    if correct == m {
        println!("✅ {}: 100% correct", name);
    } else {
        println!("⚠️ {}: {}% correct", name, correct * 100 / m);
    }
}
