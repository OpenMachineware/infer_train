#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define QK4_K 256
#define QK5_K 256
#define QK6_K 256
#define QK3_K 256
#define QK2_K 256
#define QK_NL 16

#define N_MM_NK 2
#define N_MM_NK_TOTAL (16 * N_MM_NK) // 32

template <typename T, typename T4x4, typename simdgroup_T8x8,
          typename TACC, typename TACC4x4, typename TACC2x4>
struct MulMM {
    void mul(const simdgroup_T8x8& A, const simdgroup_T8x8& B, simdgroup_T8x8& C) const {
        simdgroup_multiply(TACC(A), TACC(B), C);
    }
};

// Q4_K block structure
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

// Dequantize Q4_K block to 4x4 FP16 matrix
template <typename T4x4>
inline void dequantize_q4_K(device const block_q4_K& xb, threadgroup T4x4& reg, uint32_t i) {
    // i is in range 0..15 (16 blocks of 16 values each)
    const uint32_t q_offset = i * 8;  // 8 bytes per block of 16 values
    const uint32_t scale_idx = i / 8;  // 2 blocks share a scale
    
    // Get scales
    const uint8_t sc = xb.scales[scale_idx] & 0x3F;
    const uint8_t m = xb.scales[scale_idx + 1] & 0x3F;
    
    // Dequantize 16 values from 8 bytes
    for (uint32_t j = 0; j < 4; ++j) {
        const uint8_t q = xb.qs[q_offset + j];
        reg[j][0] = T((q & 0x0F) * sc - m) * xb.d;
        reg[j][1] = T(((q >> 4) & 0x0F) * sc - m) * xb.d;
    }
}

// Templated GEMM kernel
template <typename T, typename T4x4, typename simdgroup_T8x8,
          typename block_q, int qk, 
          void (*dequantize_func)(device const block_q&, threadgroup T4x4&, uint32_t),
          typename TACC, typename TACC4x4, typename TACC2x4>
kernel void kernel_mul_mm(
    device const char* src0 [[buffer(0)]],
    device const char* src1 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int32_t& ne00 [[buffer(3)]],
    constant int32_t& ne01 [[buffer(4)]],
    constant int32_t& ne02 [[buffer(5)]],
    constant uint64_t& nb00 [[buffer(6)]],
    constant uint64_t& nb01 [[buffer(7)]],
    constant int32_t& ne10 [[buffer(8)]],
    constant int32_t& ne11 [[buffer(9)]],
    constant int32_t& ne12 [[buffer(10)]],
    constant uint64_t& nb10 [[buffer(11)]],
    constant uint64_t& nb11 [[buffer(12)]],
    constant int32_t& ne0 [[buffer(13)]],
    constant int32_t& ne1 [[buffer(14)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int nb = ne00 / qk;
    const int nsg = 4;  // 4 simdgroups per threadgroup (like llama.cpp)
    const int nk = N_MM_NK_TOTAL;  // 32
    
    // Threadgroup shared memory for dequantized A tile
    threadgroup T4x4 A_tile[4][nk];  // 4 rows x 32 columns per row = 128 values
    threadgroup T4x4 B_tile[nsg][nk];  // 4 simdgroups x 32 columns
    
    // Get row/column indices
    const int r0 = tgpig.x;
    const int c0 = tgpig.y;
    
    device const block_q* x = (device const block_q*)(src0 + r0 * nb01);
    device const half* y = (device const half*)(src1 + c0 * nb11);
    
    simdgroup_T8x8 C_result;
    simdgroup_fill(C_result, 0);
    
    for (int k0 = 0; k0 < nb; k0 += nk / 4) {  // Each iteration processes 8 blocks (32 values per row)
        // Dequantize A tile into threadgroup memory
        for (int k = tiisg; k < nk; k += 32) {
            const int k_block = k0 + k / 4;  // Block index
            if (k_block < nb) {
                dequantize_func(x[k_block], A_tile[sgitg][k], k % 4);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load B tile from src1 (FP16)
        for (int k = tiisg; k < nk; k += 32) {
            const int k_offset = (k0 + k / 4) * qk + (k % 4) * 8;
            // Load 8 FP16 values into B_tile
            device const half* y_ptr = y + k_offset;
            // ... load into B_tile ...
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Matrix multiplication using Metal tensor operations
        // Each simdgroup multiplies 8x8 tile of A x 8x8 tile of B
        simdgroup_T8x8 A_sg;
        simdgroup_T8x8 B_sg;
        
        // Load from threadgroup tiles into simdgroup registers
        simdgroup_load(A_sg, A_tile[sgitg], nk);
        simdgroup_load(B_sg, B_tile[sgitg], nk);
        
        simdgroup_multiply(C_result, A_sg, B_sg);
    }
    
    // Store result
    if (r0 < ne01 && c0 < ne11) {
        simdgroup_store(C_result, dst + r0 * ne11 + c0, ne11);
    }
}

// Explicit instantiation for Q4_K
template [[host_name("kernel_mul_mm_q4_K_f32")]]
kernel void kernel_mul_mm<half, half4x4, simdgroup_half8x8,
                         block_q4_K, QK_K, dequantize_q4_K,
                         float, float4x4, float2x4>;