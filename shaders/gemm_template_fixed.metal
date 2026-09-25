#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define QK_NL 16
#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// K-quant block structures (matching llama.cpp and Rust)
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
    uint8_t scales[12];
};

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct block_q6_K {
    half d;
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
};

// Kernel argument structure (matching llama.cpp)
struct ggml_metal_kargs_mul_mm {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    int32_t ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    int32_t ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Helper function from llama.cpp
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Dequantize Q4_K to FP16 (from llama.cpp)
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

// Templated GEMM kernel - FIXED VERSION
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
    constant ggml_metal_kargs_mul_mm & args,
    device const char * src0,
    device const char * src1,
    device char * dst,
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]])
{
    threadgroup SA * sa = (threadgroup SA *)(shmem);
    threadgroup SB * sb = (threadgroup SB *)(shmem + 4096);

    constexpr int NR0 = 64;  // M dimension per threadgroup
    constexpr int NR1 = 32;  // N dimension per threadgroup

    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;  // 2
    constexpr int NL1 = NK / 8;   // 4

    const int K = args.ne00;
    const int M = args.ne0;  // Note: ne0 is M (rows)
    const int N = args.ne1;  // Note: ne1 is N (cols)

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = min(NR0, M - r0);
    const short nr1 = min(NR1, N - r1);

    simdgroup_SA_8x8 mdA[NL0];
    simdgroup_SB_8x8 mdB[NL1];

    for (short i = 0; i < NL0; ++i) {
        simdgroup_fill(mdA[i], 0);
    }
    for (short i = 0; i < NL1; ++i) {
        simdgroup_fill(mdB[i], 0);
    }

    // Number of quantized blocks in K dimension
    const int nb = K / (16 * nl);  // Each block_q contains 16*nl K values

    for (int ib = 0; ib < nb; ++ib) {
        const short load0 = tiitg % NL0;
        const short load1 = tiitg % NL1;
        const short load0l = tiitg / NL0;

        // === PHASE 1: Dequantize A tile ===
        for (short row = 0; row < nr0/NL0; ++row) {
            const short i = load0l + row * NL0;
            if (i < nr0) {
                // FIX: Use args.nb00 (block size in bytes) instead of hardcoded value
                device const block_q * x = (device const block_q *)(src0 + args.nb01 * (r0 + i) + ib * args.nb00);

                SA_4x4 temp;
                dequantize_func(x, load0, temp);
                for (short j = 0; j < 16; ++j) {
                    sa[load0 * 16 + j] = temp[j/4][j%4];
                }
            }
        }

        // === PHASE 2: Load B tile ===
        for (short col = 0; col < nr1/NL1; ++col) {
            const short i = load1 + col * NL1;
            if (i < nr1) {
                // B matrix is FP16 (T1 type), load 16 values
                device const T1 * y = (device const T1 *)(src1 + args.nb11 * (r1 + i) + ib * 16 * nl * sizeof(T1));
                for (short j = 0; j < 16; ++j) {
                    sb[load1 * 16 + j] = y[j];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 3: Load from threadgroup and multiply ===
        for (short i = 0; i < NL0; ++i) {
            simdgroup_load(mdA[i], sa + i * 16, 16);
        }
        for (short i = 0; i < NL1; ++i) {
            simdgroup_load(mdB[i], sb + i * 16, 16);
        }

        // Matrix multiplication (accumulate)
        for (short k = 0; k < NL0; ++k) {
            for (short j = 0; j < NL1; ++j) {
                simdgroup_multiply(mdB[j], mdA[k], mdB[j]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === PHASE 4: Store result ===
    device float * dst_f32 = (device float *)dst + r0 * N + r1;

    for (short i = 0; i < NL0; ++i) {
        for (short j = 0; j < NL1; ++j) {
            simdgroup_store(mdB[j], dst_f32 + i * 8 * N + j * 8, N);
        }
    }
}

// Explicit template instantiation for Q4_K
typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>) mul_mm_t;

template [[host_name("kernel_mul_mm_q4_K_f32")]]
kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;