# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/matmul_decode_gpu.mojo
#
# GPU matmul kernel optimized for decode mode (M = 1, 2, 3, 4).
#
# Unlike the general-purpose matmul, this kernel is optimized for:
# - Small batch sizes (M <= 4)
# - Weight-stationary access pattern
# - SIMD-group based reduction (shuffle_down)
#
# This follows llama.cpp's kernel_mul_mv_ext approach.

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.sync import barrier
from std.gpu import WARP_SIZE, block_idx, thread_idx, lane_id
from std.gpu.primitives.warp import sum as warp_sum
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutAnyOrigin
from std.math import min as math_min
from src.core.tensor import Tensor
from std.utils.static_tuple import StaticTuple
from std.collections.optional import Optional
from src.core.ops.gpu.gpu_runtime import upload, download2, get_gpu_context, gpu_available
from src.core.ops.gpu.matmul_k_quant_gpu import (
    dequantize_q4_k_16,
    dequantize_q5_k_16,
    dequantize_q6_k_16,
    dequantize_q2_k_16,
    dequantize_q3_k_16,
    _get_scale_min_k4_just2,
    matmul_k_quant_gpu,
)
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.quantized.quant_types import QuantType

comptime BLOCK_THREADS = 256
comptime ROWS_PER_TG = 4  # Each threadgroup processes 4 output rows
comptime BB_Q4_K = 144   # Q4_K block bytes
comptime CHUNKS_PER_BLOCK = 16  # 16 x 16-element chunks per 256-element block


# ============================================================================
# Q4_K decode kernel (M=1-4 optimized) - following llama.cpp's kernel_mul_mv_ext
# ============================================================================


def kernel_mul_mv_q4k_decode[
    M: Int,  # Batch size (1-4)
](
    src0: Pointer[Scalar[DType.uint8], MutAnyOrigin],  # Q4_K weights [N, nb * 144]
    src1: Pointer[Scalar[DType.float16], MutAnyOrigin],  # FP16 input [M, K]
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],   # FP16 output [M, N]
    ne00: Int32,   # K
    ne01: Int32,   # N (output dimension)
    n_blocks: Int32,  # number of 256-element blocks in K
):
    """Decode-optimized Q4_K x FP16 matmul.

    Following llama.cpp's kernel_mul_mv_ext pattern:
    - 32 threads per SIMD group (one warp)
    - Each thread processes chunks with stride = 32
    - Warp shuffle reduction across threads

    Grid: (N,)
    Each threadgroup (256 threads = 8 warps) computes 4 output rows.
    """
    var N = Int(ne01)
    var K = Int(ne00)
    var nb = Int(n_blocks)

    var row_start = Int(block_idx.x) * ROWS_PER_TG
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())  # 0-31 within warp

    if row_start >= N:
        return

    # Warp-based parallelism: 8 warps per threadgroup
    # Each warp processes one output row
    var warp_id = tid // 32  # 0-7
    var row_off = warp_id

    var weight_row = row_start + row_off
    if weight_row >= N:
        return

    var weight_base = weight_row * nb * BB_Q4_K  # 144 bytes per Q4_K block

    # Process each input row (M rows)
    for m in range(M):
        var sumf = Float32(0.0)

        # Each thread processes chunks with stride = 32 (warp size)
        # This parallelizes K dimension across threads in the warp
        for block_idx in range(nb):
            var block_ptr = src0.unsafe_offset(weight_base + block_idx * BB_Q4_K)
            var input_block_offset = m * K + block_idx * 256

            # Each thread processes chunks: il = lane + 32*i
            # lane 0-31 each take different chunks within each block
            for chunk_offset in range(0, CHUNKS_PER_BLOCK, 32):
                var il = lane + chunk_offset
                if il >= CHUNKS_PER_BLOCK:
                    break

                # Dequantize 16 elements
                var deq_buf = unsafe_stack_allocation[16, DType.float16]()

                dequantize_q4_k_16(block_ptr, il, deq_buf)

                # Load input values and compute dot product
                var input_offset = input_block_offset + il * 16
                for i in range(16):
                    var w_val = Float32(deq_buf[unsafe_offset=i])
                    var x_val = Float32(src1.unsafe_load[width=1](offset=input_offset + i))
                    sumf += w_val * x_val

        # Warp shuffle reduction - sum across all lanes
        # Use warp.sum for efficient warp-wide reduction
        var sum_vec = SIMD[DType.float32, 1](sumf)
        sumf = warp_sum(sum_vec)

        # Lane 0 writes result
        if lane == 0:
            var out_offset = m * N + weight_row
            dst.unsafe_offset(out_offset).unsafe_store(val=Scalar[DType.float16](sumf))


# ============================================================================
# Dispatch function
# ============================================================================


def matmul_q4k_decode_gpu[
    M: Int,
](
    x: Tensor[DType.float16, 2],  # [M, K], M <= 4
    w: Tensor[DType.uint8, 2],    # [N, nb * 144] quantized weights
    n_blocks: Int,                # K // 256
) -> Tensor[DType.float16, 2]:
    """Decode-optimized GPU matmul for Q4_K weights.

    Uses specialized kernel for M <= 4 with warp shuffle reduction.
    """
    var K = x.shape()[1]
    var N = w.shape()[0]

    if not gpu_available[DType.float16]():
        # Fallback to CPU
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        return matmul_quantized_q8k[QuantType.Q4_K_M](x, w, dummy_scale)

    try:
        var ctx = get_gpu_context()
        var x_buf = upload[DType.float16, 2](ctx, x)
        var w_buf = upload[DType.uint8, 2](ctx, w)
        var dst_buf = ctx.enqueue_create_buffer[DType.float16](M * N)

        var grid_x = (N + ROWS_PER_TG - 1) // ROWS_PER_TG

        ctx.enqueue_function[kernel_mul_mv_q4k_decode[M]](
            w_buf,
            x_buf,
            dst_buf,
            Int32(K),
            Int32(N),
            Int32(n_blocks),
            grid_dim=(grid_x,),
            block_dim=BLOCK_THREADS,
        )

        var out = download2[DType.float16](ctx, dst_buf, StaticTuple[Int, 2](M, N))
        ctx.synchronize()
        return out

    except:
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        return matmul_quantized_q8k[QuantType.Q4_K_M](x, w, dummy_scale)


# ============================================================================
# Generic dispatch for all K-quant types
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
    # Currently only Q4_K is implemented with shuffle reduction
    if quant_type == 12:  # Q4_K
        if M == 1:
            return matmul_q4k_decode_gpu[1](x, w, n_blocks)
        elif M == 2:
            return matmul_q4k_decode_gpu[2](x, w, n_blocks)
        elif M == 3:
            return matmul_q4k_decode_gpu[3](x, w, n_blocks)
        else:
            return matmul_q4k_decode_gpu[4](x, w, n_blocks)

    # Fallback to general GPU matmul for other types
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
