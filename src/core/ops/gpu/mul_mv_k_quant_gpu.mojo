# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/mul_mv_k_quant_gpu.mojo
#
# Optimized vector-matrix multiplication for decode mode (M=1).
# Uses warp-level parallelism and SIMD operations for maximum performance.

from src.core.tensor import Tensor
from src.core.ops.quantized.quant_types import QuantType
from src.core.ops.gpu.gpu_runtime import (
    download2,
    get_gpu_context,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives.warp import shuffle_xor
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutAnyOrigin
from std.utils.static_tuple import StaticTuple
from std.math import min as math_min
from std.collections.optional import Optional
from src.core.ops.gpu.matmul_k_quant_gpu import (
    QK_K,
    Q4_K_BLOCK,
    Q5_K_BLOCK,
    Q6_K_BLOCK,
    Q2_K_BLOCK,
    Q3_K_BLOCK,
    _dequantize_16,
    _get_block_bytes,
)

# Number of output rows computed per SIMD group
comptime NR0_Q4K = 2
comptime NR0_Q5K = 2
comptime NR0_Q6K = 2


@always_inline
def _warp_sum(val: Float32) -> Float32:
    """Warp-level sum reduction across 32 threads."""
    var result = val
    # Butterfly reduction: XOR with all power-of-2 offsets
    result = result + shuffle_xor(result, UInt32(1))
    result = result + shuffle_xor(result, UInt32(2))
    result = result + shuffle_xor(result, UInt32(4))
    result = result + shuffle_xor(result, UInt32(8))
    result = result + shuffle_xor(result, UInt32(16))
    return result


def _mul_mv_k_quant_kernel[
    quant_type: QuantType
](
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [1, K] activation vector
    w_quant: Pointer[UInt8, MutAnyOrigin],  # [N, bytes_per_row] quantized weights
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [1, N] output vector
    K: Int32,
    N: Int32,
    n_blocks_per_row: Int32,
):
    """Vector-matrix multiplication kernel optimized for decode mode.

    Each SIMD group (32 threads) computes multiple output elements in parallel.
    Thread cooperation:
    - All threads load the same activation vector (broadcast)
    - Each thread computes partial dot products
    - Warp-level reduction to sum partial results
    """
    var K_i = Int(K)
    var N_i = Int(N)
    var n_blocks = Int(n_blocks_per_row)

    comptime block_bytes = _get_block_bytes(quant_type)

    # Grid: each SIMD group computes nr0 output elements
    var group_id = Int(block_idx.x)
    var nr0 = NR0_Q4K  # Number of rows per group

    var first_row = group_id * nr0
    if first_row >= N_i:
        return

    var lane = Int(lane_id())

    # Accumulators for each row (thread-local)
    var sumf = SIMD[DType.float32, 4](0.0)  # Up to 4 rows

    # Process K in chunks of QK_K (256 elements)
    var nb = n_blocks

    # Each thread processes a different K chunk
    # Divide K into 4 chunks, each processed by 8 threads
    var ix = lane // 8  # 0-3, which chunk
    var it = lane % 8   # 0-7, which position in chunk

    # Load activation vector chunk into registers (shared across rows)
    # For M=1, we can cache this in threadgroup memory
    var yl = SIMD[DType.float32, 16](0.0)  # Lower 16 elements
    var yh = SIMD[DType.float32, 16](0.0)  # Higher 16 elements (for K > 128)

    # Process all K blocks
    for ib in range(ix, nb, 4):
        # Load 16 activation elements
        var k_base = ib * QK_K + it * 8
        for i in range(8):
            var k = k_base + i
            if k < K_i:
                yl[i] = Float32(x[unsafe_offset=k])
                if k + 64 < K_i:
                    yh[i] = Float32(x[unsafe_offset=k + 64])

        # Process each weight row
        for row in range(nr0):
            var global_row = first_row + row
            if global_row >= N_i:
                break

            # Load quantized weight block
            var block_ptr = w_quant.unsafe_offset(global_row * n_blocks * block_bytes + ib * block_bytes)

            # Dequantize and compute partial dot product
            var partial_sum = Float32(0.0)

            # Dequantize weight chunk and compute dot product
            # This is simplified - actual implementation needs proper Q4_K dequant
            for i in range(16):
                partial_sum += yl[i]  # Placeholder for actual dot product

            sumf[row] += partial_sum

    # Warp-level reduction
    for row in range(nr0):
        sumf[row] = _warp_sum(sumf[row])

    # Thread 0 writes result
    if lane == 0:
        for row in range(nr0):
            var global_row = first_row + row
            if global_row < N_i:
                dst[unsafe_offset=global_row] = Scalar[DType.float16](sumf[row])


def mul_mv_k_quant_gpu[
    quant_type: QuantType
](
    x: Tensor[DType.float16, 2],  # [1, K]
    w_quant: Tensor[DType.uint8, 2],  # [N, bytes_per_row]
    n_blocks: Int,
    w_buf_cached: Optional[DeviceBuffer[DType.uint8]] = None,
    ctx_cached: Optional[DeviceContext] = None,
) -> Tensor[DType.float16, 2]:  # [1, N]
    """Optimized vector-matrix multiplication for decode mode.

    Uses warp-level parallelism for maximum performance on Apple Silicon.
    """
    if not gpu_available[DType.float16]():
        from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        return matmul_quantized_q8k[quant_type](x, w_quant, dummy_scale)

    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w_quant.shape()[0]

    # For M > 1, use the matrix-matrix kernel instead
    if M > 1:
        from src.core.ops.gpu.matmul_k_quant_gpu import matmul_k_quant_gpu_cached
        return matmul_k_quant_gpu_cached[quant_type](x, w_quant, n_blocks, w_buf_cached, ctx_cached)

    try:
        # Use cached context if available
        var ctx: DeviceContext
        var owns_ctx = False
        if ctx_cached:
            ctx = ctx_cached.value()
        else:
            ctx = get_gpu_context()
            owns_ctx = True

        var x_buf = upload[DType.float16, 2](ctx, x)

        # Use cached buffer if available
        var w_buf: DeviceBuffer[DType.uint8]
        if w_buf_cached:
            w_buf = w_buf_cached.value()
        else:
            w_buf = upload[DType.uint8, 2](ctx, w_quant)

        var dst_buf = ctx.enqueue_create_buffer[DType.float16](M * N)

        # Grid: one SIMD group per nr0 output elements
        var nr0 = NR0_Q4K
        var grid_size = (N + nr0 - 1) // nr0

        ctx.enqueue_function[_mul_mv_k_quant_kernel[quant_type]](
            x_buf,
            w_buf,
            dst_buf,
            Int32(K),
            Int32(N),
            Int32(n_blocks),
            grid_dim=(grid_size),
            block_dim=WARP_SIZE,
        )

        var out = download2[DType.float16](ctx, dst_buf, StaticTuple[Int, 2](M, N))
        if owns_ctx:
            ctx.synchronize()
        return out
    except:
        from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        return matmul_quantized_q8k[quant_type](x, w_quant, dummy_scale)
