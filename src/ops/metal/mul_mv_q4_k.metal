// Q4_K Matrix-Vector Multiplication Kernel
// Self-written implementation targeting 100% llama.cpp performance

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32

// Loop unrolling macro from llama.cpp
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Q4_K block structure: 144 bytes for 256 elements
typedef struct {
    half  d;           // Super-block scale (2 bytes)
    half  dmin;        // Super-block min scale (2 bytes)
    uint8_t scales[12]; // Scales and mins encoded (12 bytes)
    uint8_t qs[128];   // 4-bit quants, 2 elements per byte (128 bytes)
} block_q4_K;

// Kernel arguments matching llama.cpp layout
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

// Function constants for template specialization
constant short FC_nsg  [[function_constant(600)]];
constant short FC_ne12 [[function_constant(602)]];
constant short FC_r2   [[function_constant(603)]];
constant short FC_r3   [[function_constant(604)]];

// N_R0 for Q4_K: each simdgroup processes 2 rows
#define N_R0_Q4_K 2

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    // Scale extraction masks
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    // Thread indexing
    const short ix = tiisg / 8;  // 0..3: which quarter of K
    const short it = tiisg % 8; // 0..7
    const short iq = it / 4;    // 0 or 1
    const short ir = it % 4;    // 0..3

    const int nb = args.ne00 / QK_K;  // Number of blocks per row

    // Row indices
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    // Memory offsets with batch/repeat factors
    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K * x = (device const block_q4_K *)(src0 + offset0);
    device const float      * y = (device const float      *)(src1 + offset1);

    // Accumulator for each row this thread contributes to
    float sumf[nr0] = {0.f};

    // Load y values for this thread's portion
    // Each thread handles 8 consecutive y values, split into yl (low) and yh (high)
    float yl[16];
    float yh[16];

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    // Temporary storage for decoded scales
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    // Loop over blocks, each thread handles 1/4 of blocks
    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        // Load y values: 8 values from each of 4 segments
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        // Pointers to quantized data
        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        // Process each row assigned to this simdgroup
        for (short row = 0; row < nr0; row++) {
            // Decode scales from packed 12-byte format
            // The scales are packed in a complex way to save space
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            // Dot product: quantized * float
            // 4-bit values extracted via bit masks
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

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

            // Final accumulation with scales and mins
            // The formula: d * (dot_product * scale - min * sum_y)
            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            // Advance to next row
            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    // Write results
    device float * dst_f32 = (device float *)dst + (int64_t)im * args.ne0 * args.ne1 + (int64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// Kernel entry point
[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg, sgitg);
}
