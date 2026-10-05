// GEMM kernel using MetalPerformancePrimitives (MPP) tensor API
// Self-written implementation, NOT copied from llama.cpp

#include "common.h"
#include "dequantize.h"

// Function constants for kernel configuration
constant bool FC_BC_INP [[function_constant(700)]];
constant bool FC_BC_OUT [[function_constant(701)]];
constant short FC_NE12  [[function_constant(702)]];
constant short FC_NE13  [[function_constant(703)]];
constant short FC_R2    [[function_constant(704)]];
constant short FC_R3    [[function_constant(705)]];

// Tile dimensions
constant int TILE_K = 256;  // K tile size (N_MM_NK_TOTAL in llama.cpp)
constant int TILE_M = 64;   // M tile size (NRA in llama.cpp)
constant int TILE_N = 64;   // N tile size (NRB in llama.cpp)
constant int SIMD_SIZE = 32;
constant int SIMD_GROUPS_X = 2;
constant int SIMD_GROUPS_Y = 2;

// MPP-based GEMM kernel template
#ifdef GGML_METAL_HAS_TENSOR

template<
    typename AccumType,      // Type for dequantized weights in threadgroup
    typename AccumType4x4,
    typename BlockType,      // Quantized block type
    short NL,                // Number of 16-element chunks per block
    void (*DequantFunc)(device const BlockType *, short, thread AccumType4x4 &),
    typename InputType,      // Input matrix element type (half or float)
    typename InputType2x4>
kernel void gemm_mpp(
    constant ggml_metal_kargs_mul_mm & args,
    device const char * weights,      // srcA: quantized weight matrix (M, K)
    device const char * input,        // srcB: input matrix (K, N)
    device       char * output,       // dst: output matrix (M, N)
    threadgroup  char * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],     // tile position
    ushort tiitg [[thread_index_in_threadgroup]],      // thread index
    ushort sgitg [[simdgroup_index_in_threadgroup]]) { // simdgroup index

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch handling
    const int batch = tgpig.z;
    const int i12 = batch % FC_NE12;
    const int i13 = batch / FC_NE12;
    const uint64_t batch_offset = (i12 / FC_R2) * args.nb02 + (i13 / FC_R3) * args.nb03;

    // Tile position in output matrix
    const int tile_m = tgpig.y * TILE_M;
    const int tile_n = tgpig.x * TILE_N;

    // Threadgroup memory for dequantized weights
    threadgroup AccumType * weight_tile = (threadgroup AccumType *)(shmem);

    // Work distribution for weight dequantization
    constexpr int WORK_ITEMS = TILE_M * (TILE_K / 16);  // Each work item processes 16 elements
    constexpr int NUM_THREADS = SIMD_SIZE * SIMD_GROUPS_X * SIMD_GROUPS_Y;

    // Wrap threadgroup memory as a tensor
    auto weight_tensor = tensor(weight_tile, dextents<int32_t, 2>(TILE_K, TILE_M));

    // Wrap input matrix as a tensor (device memory)
    device InputType * input_ptr = (device InputType *)(input + args.nb12 * i12 + args.nb13 * i13);
    const int input_stride = args.nb11 / sizeof(InputType);
    auto input_tensor = tensor(input_ptr, dextents<int32_t, 2>(K, N), array<int, 2>({1, input_stride}));

    // Configure MPP matrix multiply
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            TILE_N, TILE_M, static_cast<int>(dynamic_extent),
            false, true, true,  // transpose flags
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<SIMD_GROUPS_X * SIMD_GROUPS_Y>> matmul_op;

    // Get destination cooperative tensor for accumulating results
    auto result_tensor = matmul_op.get_destination_cooperative_tensor<decltype(input_tensor), decltype(weight_tensor), float>();

    // Iterate over K in tiles
    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // === Phase 1: Dequantize weight tile into threadgroup memory ===
        for (int work = tiitg; work < WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / (TILE_K / 16);
            const int k_chunk = work % (TILE_K / 16);
            const int k_pos = k_start + k_chunk * 16;
            const int k_base = k_chunk * 16;

            // Bounds check for M dimension
            if (tile_m + row < M) {
                const int block_idx = k_pos / (16 * NL);
                const short il = (k_pos / 16) % NL;

                device const BlockType * row_ptr = (device const BlockType *)(weights + args.nb01 * (tile_m + row) + batch_offset);

                AccumType4x4 temp;
                DequantFunc(row_ptr + block_idx, il, temp);

                // Store dequantized values, with zero-padding for partial K
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    weight_tile[row * TILE_K + (k_base + i)] = (k_pos + i < K) ? temp[i / 4][i % 4] : (AccumType)0;
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    weight_tile[row * TILE_K + (k_base + i)] = (AccumType)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Phase 2: Execute matrix multiply using MPP ===
        const int k_extent = min(TILE_K, K - k_start);

        auto weight_view = tensor(weight_tile, dextents<int32_t, 2>(k_extent, TILE_M), array<int, 2>({1, TILE_K}));
        auto input_view = tensor(input_ptr + k_start + tile_n * input_stride, dextents<int32_t, 2>(k_extent, N - tile_n), array<int, 2>({1, input_stride}));

        matmul_op.run(input_view, weight_view, result_tensor);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Phase 3: Store result to output matrix ===
    device float * output_ptr = (device float *)output + batch * N * M;
    auto output_tensor = tensor(output_ptr, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    result_tensor.store(output_tensor.slice(tile_m, tile_n));
}

#else
// Fallback: empty stub for non-MPP compilation
template<
    typename AccumType, typename AccumType4x4,
    typename BlockType, short NL,
    void (*DequantFunc)(device const BlockType *, short, thread AccumType4x4 &),
    typename InputType, typename InputType2x4>
kernel void gemm_mpp(
    constant ggml_metal_kargs_mul_mm & args,
    device const char * weights,
    device const char * input,
    device       char * output,
    threadgroup  char * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    // MPP not available
}
#endif // GGML_METAL_HAS_TENSOR

// Kernel instantiations
// Typedef for function type (needed for proper template instantiation syntax)
typedef decltype(gemm_mpp<half, half4x4, block_iq1_s, 16, dequantize_iq1_s, half, half2x4>) gemm_mpp_t;

// IQ formats with F16 input
template [[host_name("gemm_mpp_iq1_s_f16")]]   kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq1_s,   16, dequantize_iq1_s,   half, half2x4>;
template [[host_name("gemm_mpp_iq1_m_f16")]]   kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq1_m,   16, dequantize_iq1_m,   half, half2x4>;
template [[host_name("gemm_mpp_iq2_xxs_f16")]] kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq2_xxs, 16, dequantize_iq2_xxs, half, half2x4>;
template [[host_name("gemm_mpp_iq2_xs_f16")]]  kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq2_xs,  16, dequantize_iq2_xs,  half, half2x4>;
template [[host_name("gemm_mpp_iq2_s_f16")]]   kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq2_s,   16, dequantize_iq2_s,   half, half2x4>;
template [[host_name("gemm_mpp_iq3_xxs_f16")]] kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq3_xxs, 16, dequantize_iq3_xxs, half, half2x4>;
template [[host_name("gemm_mpp_iq3_s_f16")]]   kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq3_s,   16, dequantize_iq3_s,   half, half2x4>;
template [[host_name("gemm_mpp_iq4_nl_f16")]]  kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq4_nl,  2,  dequantize_iq4_nl,  half, half2x4>;
template [[host_name("gemm_mpp_iq4_xs_f16")]]  kernel gemm_mpp_t gemm_mpp<half, half4x4, block_iq4_xs,  16, dequantize_iq4_xs,  half, half2x4>;

// K-quant and standard Q formats
template [[host_name("gemm_mpp_q2_k_f16")]]    kernel gemm_mpp_t gemm_mpp<half, half4x4, block_q2_K,  16, dequantize_q2_K,  half, half2x4>;
template [[host_name("gemm_mpp_q3_k_f16")]]    kernel gemm_mpp_t gemm_mpp<half, half4x4, block_q3_K,  16, dequantize_q3_K,  half, half2x4>;
template [[host_name("gemm_mpp_q4_k_f16")]]    kernel gemm_mpp_t gemm_mpp<half, half4x4, block_q4_K,  16, dequantize_q4_K,  half, half2x4>;
template [[host_name("gemm_mpp_q5_k_f16")]]    kernel gemm_mpp_t gemm_mpp<half, half4x4, block_q5_K,  16, dequantize_q5_K,  half, half2x4>;
template [[host_name("gemm_mpp_q6_k_f16")]]    kernel gemm_mpp_t gemm_mpp<half, half4x4, block_q6_K,  16, dequantize_q6_K,  half, half2x4>;
template [[host_name("gemm_mpp_q4_0_f16")]]    kernel gemm_mpp_t gemm_mpp<half, half4x4, block_q4_0,   2,  dequantize_q4_0,  half, half2x4>;
template [[host_name("gemm_mpp_q8_0_f16")]]    kernel gemm_mpp_t gemm_mpp<half, half4x4, block_q8_0,   2,  dequantize_q8_0,  half, half2x4>;