#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define QK_NL 16
#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Q4_K block structure
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
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

// GEMM kernel using dequantize approach
// A(M,K) x B(K,N) -> C(M,N)
// A is Q4_K, B is FP16, C is FP32
kernel void kernel_gemm_q4_k_f32(
    device const char * src0 [[buffer(0)]],  // A matrix (Q4_K)
    device const half * src1 [[buffer(1)]],  // B matrix (FP16)
    device float * dst [[buffer(2)]],         // C matrix (FP32)
    constant uint32_t & ne00 [[buffer(3)]],  // K
    constant uint32_t & ne01 [[buffer(4)]],  // M
    constant uint64_t & nb01 [[buffer(5)]],  // stride for A
    constant uint64_t & nb11 [[buffer(6)]],  // stride for B
    constant uint32_t & ne11 [[buffer(7)]],  // N
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]])
{
    const int K = ne00;
    const int M = ne01;
    const int N = ne11;

    // Tile size: 64x32 (like llama.cpp)
    constexpr int NR0 = 64;  // M dimension per threadgroup
    constexpr int NR1 = 32;  // N dimension per threadgroup

    // Threadgroup memory for dequantized A tile
    threadgroup half * sa = (threadgroup half *)shmem;

    // Get tile position
    const int r0 = tgpig.y * NR0;  // Row offset
    const int r1 = tgpig.x * NR1;  // Column offset

    const short nr0 = min(NR0, M - r0);
    const short nr1 = min(NR1, N - r1);

    // Threadgroup memory size: 32 K values per row x 64 rows = 2048 half values = 4096 bytes
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;  // 2
    constexpr int NL1 = NK / 8;   // 4

    // Load B tile into registers (FP16)
    simdgroup_half8x8 mdB[NL1];
    for (short i = 0; i < NL1; ++i) {
        simdgroup_fill(mdB[i], 0);
    }

    // Process K in blocks of 32
    const int nb = K / QK_K;  // Number of Q4_K blocks in K dimension

    for (int ib = 0; ib < nb; ++ib) {
        // === PHASE 1: Dequantize A tile ===
        // Each thread dequantizes one block (16 values) for one row
        for (int load0l = tiitg / NL0; load0l < nr0; load0l += N_SIMDWIDTH) {
            const int i = load0l;
            const int load0 = tiitg % NL0;

            device const block_q4_K * x = (device const block_q4_K *)(src0 + nb01 * (r0 + i) + ib * QK_K);

            // Dequantize two blocks of 16 values each (total 32 values for this K slice)
            half4x4 temp;
            dequantize_q4_K(x, load0, temp);

            // Store in threadgroup memory
            for (short j = 0; j < 16; ++j) {
                sa[load0 * 16 + load0l * NK + j] = temp[j/4][j%4];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Load B and multiply ===
        // Load B tile (FP16, no dequantization needed)
        for (short col = 0; col < nr1; ++col) {
            device const half * y = src1 + r1 + col + ib * QK_K;

            // Load 32 values into 4 simdgroups
            for (short j = 0; j < NL1; ++j) {
                half8 vals;
                for (short k = 0; k < 8; ++k) {
                    vals[k] = y[j * 8 + k];
                }
                simdgroup_load(mdB[j], vals);
            }

            // Multiply with A tile
            simdgroup_half8x8 mdA[NL0];
            for (short i = 0; i < NL0; ++i) {
                simdgroup_load(mdA[i], sa + i * 16, NK);
            }

            // Matrix multiplication
            for (short k = 0; k < NL0; ++k) {
                for (short j = 0; j < NL1; ++j) {
                    simdgroup_multiply(mdB[j], mdA[k], mdB[j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === PHASE 3: Store result ===
    device float * dst_f32 = dst + r0 * N + r1;

    for (short i = 0; i < NL0; ++i) {
        for (short j = 0; j < NL1; ++j) {
            simdgroup_store(mdB[j], dst_f32 + i * 8 * N + j * 8, N);
        }
    }
}