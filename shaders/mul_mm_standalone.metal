// Self-contained GEMM kernel for Q4_K
// Adapted from llama.cpp (MIT license)
// No C++ header dependencies - everything defined in Metal

#include <metal_stdlib>
using namespace metal;

// Constants
#define QK_K 256
#define QK_NL 16
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Block structures (matching Rust BlockQ4K exactly)
struct block_q4_K {
    half d;            // Super-block scale
    half dmin;         // Super-block scale for mins
    uint8_t scales[12]; // Scales and mins
    uint8_t qs[QK_K/2]; // 4-bit quants (128 bytes)
};

struct block_q3_K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};

struct block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

// Standard Q block structures (32 elements per block)
struct block_q4_0 {
    half d;            // Scale
    uint8_t qs[16];    // 4-bit quants (32 elements packed)
};

struct block_q8_0 {
    half d;            // Scale
    int8_t qs[32];     // 8-bit quants
};

struct block_q4_1 {
    half d;            // Scale
    half m;            // Min
    uint8_t qs[16];    // 4-bit quants
};

struct block_q5_0 {
    half d;            // Scale
    uint8_t qh[4];     // High bits for 32 elements
    uint8_t qs[16];    // Low 4 bits
};

struct block_q5_1 {
    half d;            // Scale
    half m;            // Min
    uint8_t qh[4];     // High bits
    uint8_t qs[16];    // Low 4 bits
};

// MXFP4 block structure
struct block_mxfp4 {
    uint8_t e;         // E8M0 scale byte
    uint8_t qs[16];    // 4-bit indices into kvalues table
};

// IQ4_NL block structure (similar to Q4_0 but with nonlinear lookup)
struct block_iq4_nl {
    half d;            // Scale
    uint8_t qs[16];    // 4-bit indices into kvalues_iq4nl table
};

// Kernel argument structure (matching llama.cpp's ggml_metal_kargs_mul_mm)
struct ggml_metal_kargs_mul_mm {
    int32_t  ne00;   // K dimension
    int32_t  ne02;   // batch dimension
    uint64_t nb01;   // row stride for A (bytes between rows)
    uint64_t nb02;   // batch stride for A
    uint64_t nb03;   // batch3 stride for A
    int32_t  ne12;   // batch dimension for B
    uint64_t nb10;   // element stride for B (bytes between elements)
    uint64_t nb11;   // row stride for B (bytes between rows)
    uint64_t nb12;   // batch stride for B
    uint64_t nb13;   // batch3 stride for B
    int32_t  ne0;    // M dimension (output rows)
    int32_t  ne1;    // N dimension (output cols)
    int16_t  r2;     // broadcast ratio for batch dimension
    int16_t  r3;     // broadcast ratio for batch3 dimension
};

// Function constants for specialization
// These allow the Metal compiler to optimize at pipeline creation time
constant bool  FC_mul_mm_bc_inp [[function_constant(0)]];
constant bool  FC_mul_mm_bc_out [[function_constant(1)]];
constant short FC_mul_mm_ne12  [[function_constant(2)]];
constant short FC_mul_mm_ne13  [[function_constant(3)]];
constant short FC_mul_mm_r2    [[function_constant(4)]];
constant short FC_mul_mm_r3    [[function_constant(5)]];

// Lookup tables for IQ and MXFP formats
constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_mxfp4_f[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// E8M0 scale decode: exponent stored in byte, value = 2^(e-127)
static inline float decode_e8m0_scale(uint8_t e) {
    // E8M0 format: 8-bit exponent, mantissa is implicit 1
    // Value = 2^(exponent - 127)
    // For Metal, we compute this as exp2(e - 127)
    return exp2(float(e) - 127.f);
}

// Helper function for Q4_K/Q5_K
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Dequantize functions (from llama.cpp)
template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
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
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// Standard Q dequantize functions
// Q4_0: 4-bit values with scale, symmetric quantization
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    device const uint8_t * q = xb->qs + 16 * (il & 1);

    // 4-bit values: 0-15 mapped to -8 to 7
    for (int i = 0; i < 16; ++i) {
        uint8_t qval = (il & 1) ? (q[i] >> 4) : (q[i] & 0x0F);
        reg[i/4][i%4] = d * (qval - 8.f);
    }
}

// Q8_0: 8-bit values with scale
template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    device const int8_t * q = xb->qs + 16 * (il & 1);

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = d * q[i];
    }
}

// Q4_1: 4-bit values with scale and min
template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float m = xb->m;
    device const uint8_t * q = xb->qs + 16 * (il & 1);

    for (int i = 0; i < 16; ++i) {
        uint8_t qval = (il & 1) ? (q[i] >> 4) : (q[i] & 0x0F);
        reg[i/4][i%4] = d * qval + m;
    }
}

// Q5_0: 5-bit values with scale
template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    device const uint8_t * q = xb->qs + 16 * (il & 1);
    device const uint8_t * qh = xb->qh;

    // 5-bit: low 4 bits from qs, high 1 bit from qh
    uint32_t hmask = ((qh[0] | (qh[1] << 8) | (qh[2] << 16) | (qh[3] << 24)) >> (il * 16)) & 0xFFFF;

    for (int i = 0; i < 16; ++i) {
        uint8_t qval = (il & 1) ? (q[i] >> 4) : (q[i] & 0x0F);
        uint8_t hbit = (hmask >> i) & 1;
        // 5-bit: 0-31 mapped to -16 to 15
        reg[i/4][i%4] = d * ((qval | (hbit << 4)) - 16.f);
    }
}

// Q5_1: 5-bit values with scale and min
template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float m = xb->m;
    device const uint8_t * q = xb->qs + 16 * (il & 1);
    device const uint8_t * qh = xb->qh;

    uint32_t hmask = ((qh[0] | (qh[1] << 8) | (qh[2] << 16) | (qh[3] << 24)) >> (il * 16)) & 0xFFFF;

    for (int i = 0; i < 16; ++i) {
        uint8_t qval = (il & 1) ? (q[i] >> 4) : (q[i] & 0x0F);
        uint8_t hbit = (hmask >> i) & 1;
        reg[i/4][i%4] = d * (qval | (hbit << 4)) + m;
    }
}

// MXFP4: 4-bit indices into kvalues_mxfp4_f with E8M0 scale
template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 *xb, short il, thread type4x4 & reg) {
    const float scale = decode_e8m0_scale(xb->e);
    device const uint8_t * q = xb->qs + 16 * (il & 1);

    for (int i = 0; i < 16; ++i) {
        uint8_t idx = (il & 1) ? (q[i] >> 4) : (q[i] & 0x0F);
        reg[i/4][i%4] = scale * kvalues_mxfp4_f[idx];
    }
}

// IQ4_NL: 4-bit indices into kvalues_iq4nl_f with scale
template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    device const uint8_t * q = xb->qs + 16 * (il & 1);

    for (int i = 0; i < 16; ++i) {
        uint8_t idx = (il & 1) ? (q[i] >> 4) : (q[i] & 0x0F);
        reg[i/4][i%4] = d * kvalues_iq4nl_f[idx];
    }
}

// GEMM kernel template (from llama.cpp)
template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        // Full tile: write directly to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // Partial tile: write to threadgroup memory first, then copy valid data
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// Template instantiations for K-quant formats
typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>) mul_mm_t;

template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;

// Standard Q GEMM kernels
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;

// MXFP4 and IQ GEMM kernels
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
