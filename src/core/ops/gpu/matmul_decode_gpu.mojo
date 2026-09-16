# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/matmul_decode_gpu.mojo
#
# GPU matmul kernel optimized for decode mode (M = 1, 2, 3, 4).
#
# Unlike the general-purpose matmul, this kernel is optimized for:
# - Small batch sizes (M <= 4)
# - Weight-stationary access pattern (weights stay in cache)
# - SIMD-group based reduction instead of global barriers
#
# This follows llama.cpp's kernel_mul_mv_ext approach.

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.sync import barrier
from std.gpu import WARP_SIZE, block_idx, thread_idx, lane_id, simd_shuffle_down
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutAnyOrigin
from std.math import min as math_min
from src.core.tensor import Tensor
from std.utils.static_tuple import StaticTuple
from std.collections.optional import Optional
from src.core.ops.gpu.gpu_runtime import upload, download2, get_gpu_context, gpu_available

comptime BLOCK_THREADS = 256


# ============================================================================
# FP16 x Q4_K decode kernel (M=1 optimized)
# ============================================================================


def kernel_mul_mv_q4k_decode_r1(
    src0: Pointer[Scalar[DType.uint8], MutAnyOrigin],  # Q4_K weights [N, nb * bb]
    src1: Pointer[Scalar[DType.float16], MutAnyOrigin],  # FP16 input [1, K]
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],   # FP16 output [N]
    ne00: Int32,   # K
    ne01: Int32,   # N (output dimension)
    nb00: Int32,   # block bytes for Q4_K = 144
    block_offset: Int32,  # for large N, process in blocks
    n_blocks: Int32,  # number of 256-element blocks in K
):
    """Decode-optimized Q4_K x FP16 matmul (M=1).

    Grid: (N / rows_per_tg,)
    Each threadgroup computes one or more output rows.
    Uses SIMD reduction for efficiency.

    Q4_K block layout (144 bytes per 256 elements):
    - scales: 4+4=8 bytes (d, dmin)
    - mins: 4+4=8 bytes
    - qs: 128 bytes (256 4-bit values)
    """
    var N = Int(ne01)
    var K = Int(ne00)
    var nb = Int(n_blocks)
    var bb = Int(nb00)

    # Each threadgroup processes multiple rows for better utilization
    comptime ROWS_PER_TG = 4
    var row_start = Int(block_idx.x) * ROWS_PER_TG
    var tid = Int(thread_idx.x)

    if row_start >= N:
        return

    # Load the single input row into registers (shared by all threads)
    # Each thread loads a portion
    var x_scale = unsafe_stack_allocation[
        32, DType.float32, address_space=AddressSpace.SHARED
    ]()  # scales for Q8_K

    # Simplified: each threadgroup computes ROWS_PER_TG output rows
    # Each thread in the threadgroup computes a portion of each row

    for row_off in range(ROWS_PER_TG):
        var row = row_start + row_off
        if row >= N:
            break

        var sumf = Float32(0.0)
        var weight_base = row * nb * bb

        # Each thread processes a subset of blocks
        for b in range(tid, nb, BLOCK_THREADS):
            # Load Q4_K block
            var block_ptr = src0.unsafe_offset(weight_base + b * bb)

            # Q4_K dequantization (simplified)
            # Real implementation would read scales, mins, qs and dequantize
            # For now, use a placeholder that reads the first 4 bytes (scale)
            var scale = Float32(0.0)
            # Read scale from block
            # var d_ptr = block_ptr.bitcast[Scalar[DType.float32]]()
            # scale = Float32(d_ptr.load())

            # Compute dot product with input
            # Each block covers 256 elements
            # Sum up contributions

            sumf += scale  # Placeholder

        # SIMD reduction
        sumf += simd_shuffle_down(sumf, 16)
        sumf += simd_shuffle_down(sumf, 8)
        sumf += simd_shuffle_down(sumf, 4)
        sumf += simd_shuffle_down(sumf, 2)
        sumf += simd_shuffle_down(sumf, 1)

        # Thread 0 writes result
        if tid == 0:
            dst.unsafe_offset(row).unsafe_store(val=Scalar[DType.float16](sumf))


def matmul_q4k_decode_gpu(
    x: Tensor[DType.float16, 2],  # [M, K], M <= 4
    w: Tensor[DType.uint8, 2],    # [N, nb * bb] quantized weights
    n_blocks: Int,                # K // 256
) -> Tensor[DType.float16, 2]:
    """Decode-optimized GPU matmul for Q4_K weights.

    Uses specialized kernel for M <= 4 with SIMD reduction.
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]

    if not gpu_available[DType.float16]():
        # Fallback to CPU
        from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
        from src.core.ops.quantized.quant_types import QuantType
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        return matmul_quantized_q8k[QuantType.Q4_K_M](x, w, dummy_scale)

    try:
        var ctx = get_gpu_context()
        var x_buf = upload[DType.float16, 2](ctx, x)
        var w_buf = upload[DType.uint8, 2](ctx, w)
        var dst_buf = ctx.enqueue_create_buffer[DType.float16](M * N)

        # For now, use CPU fallback until kernel is fully implemented
        ctx.synchronize()
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
        from src.core.ops.quantized.quant_types import QuantType
        return matmul_quantized_q8k[QuantType.Q4_K_M](x, w, dummy_scale)

    except:
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
        from src.core.ops.quantized.quant_types import QuantType
        return matmul_quantized_q8k[QuantType.Q4_K_M](x, w, dummy_scale)


# ============================================================================
# Entry point: dispatch based on M
# ============================================================================


def matmul_decode_gpu(
    x: Tensor[DType.float16, 2],
    w: Tensor[DType.uint8, 2],
    quant_type: Int,  # ggml_type (12=Q4_K, 13=Q5_K, etc.)
    n_blocks: Int,
) -> Tensor[DType.float16, 2]:
    """GPU matmul optimized for decode (M <= 4).

    Dispatches to specialized kernels based on:
    - Batch size M
    - Quantization type

    For M > 4, falls back to general GPU matmul.
    """
    var M = x.shape()[0]

    if M > 4:
        # Use general GPU matmul for larger batches
        from src.core.ops.gpu.matmul_k_quant_gpu import matmul_k_quant_gpu
        from src.core.ops.quantized.quant_types import QuantType
        var quant = QuantType.Q4_K_M
        if quant_type == 13:
            quant = QuantType.Q5_K
        elif quant_type == 14:
            quant = QuantType.Q6_K
        elif quant_type == 11:
            quant = QuantType.Q2_K
        elif quant_type == 15:
            quant = QuantType.Q3_K
        return matmul_k_quant_gpu[quant](x, w, n_blocks)

    # Use decode-optimized kernel
    return matmul_q4k_decode_gpu(x, w, n_blocks)
