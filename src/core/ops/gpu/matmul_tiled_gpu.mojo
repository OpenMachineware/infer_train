# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/matmul_tiled_gpu.mojo
#
# Tiled GPU matrix multiplication using threadgroup memory.
# Phase 2: Each thread computes multiple outputs for better utilization.

from src.core.tensor import Tensor
from src.core.utils import unimplemented
from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu
from src.core.ops.gpu.gpu_runtime import (
    download2,
    get_gpu_context,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutAnyOrigin
from std.utils.static_tuple import StaticTuple
from std.math import min as math_min

# Tile sizes
comptime BLOCK_M = 32
comptime BLOCK_N = 64
comptime BLOCK_K = 32
comptime BLOCK_THREADS = 256

# Each thread computes ROWS_PER_THREAD x COLS_PER_THREAD outputs
comptime ROWS_PER_THREAD = 4
comptime COLS_PER_THREAD = 2


def _matmul_weight_tiled_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    w: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    """Tiled matmul kernel: y = x @ w^T where w is [N, K].

    Each thread computes ROWS_PER_THREAD x COLS_PER_THREAD outputs.
    Threadgroup memory caches K tiles of A and B.
    """
    var M_i = Int(M)
    var K_i = Int(K)
    var N_i = Int(N)

    # Block and thread indices
    var block_row = Int(block_idx.y) * BLOCK_M
    var block_col = Int(block_idx.x) * BLOCK_N
    var tid = Int(thread_idx.x)

    # Thread position within the tile
    var thread_row_in_tile = tid // (BLOCK_N // COLS_PER_THREAD)  # 0..7
    var thread_col_in_tile = tid % (BLOCK_N // COLS_PER_THREAD)   # 0..31

    # Threadgroup memory for A and B tiles
    var tile_a = unsafe_stack_allocation[
        BLOCK_M * BLOCK_K,
        DType.float16,
        address_space=AddressSpace.SHARED,
    ]()
    var tile_b = unsafe_stack_allocation[
        BLOCK_K * BLOCK_N,
        DType.float16,
        address_space=AddressSpace.SHARED,
    ]()

    # Accumulators for this thread's outputs (ROWS_PER_THREAD x COLS_PER_THREAD)
    var acc = SIMD[DType.float32, ROWS_PER_THREAD * COLS_PER_THREAD](0.0)

    # Loop over K dimension in tiles
    var k_tiles = (K_i + BLOCK_K - 1) // BLOCK_K
    for k_tile in range(k_tiles):
        var k_start = k_tile * BLOCK_K
        var k_end = math_min(k_start + BLOCK_K, K_i)
        var k_size = k_end - k_start

        # Cooperatively load A tile into threadgroup memory
        var total_a = BLOCK_M * k_size
        for work in range(tid, total_a, BLOCK_THREADS):
            var a_row = work // k_size
            var a_k = work % k_size
            var src_row = block_row + a_row
            var src_k = k_start + a_k
            if src_row < M_i and src_k < K_i:
                tile_a[unsafe_offset=a_row * BLOCK_K + a_k] = x[
                    unsafe_offset=src_row * K_i + src_k
                ]
            else:
                tile_a[unsafe_offset=a_row * BLOCK_K + a_k] = Scalar[DType.float16](0.0)

        # Cooperatively load B tile into threadgroup memory
        var total_b = BLOCK_N * k_size
        for work in range(tid, total_b, BLOCK_THREADS):
            var b_col = work // k_size
            var b_k = work % k_size
            var src_col = block_col + b_col
            var src_k = k_start + b_k
            if src_col < N_i and src_k < K_i:
                tile_b[unsafe_offset=b_col * BLOCK_K + b_k] = w[
                    unsafe_offset=src_col * K_i + src_k
                ]
            else:
                tile_b[unsafe_offset=b_col * BLOCK_K + b_k] = Scalar[DType.float16](0.0)

        # Barrier to ensure all loads complete
        barrier()

        # Compute partial dot products for all this thread's outputs
        for k in range(k_size):
            # For each output this thread handles
            for r in range(ROWS_PER_THREAD):
                var row_in_tile = thread_row_in_tile * ROWS_PER_THREAD + r
                var a_val = Float32(tile_a[unsafe_offset=row_in_tile * BLOCK_K + k])

                for c in range(COLS_PER_THREAD):
                    var col_in_tile = thread_col_in_tile * COLS_PER_THREAD + c
                    var b_val = Float32(tile_b[unsafe_offset=col_in_tile * BLOCK_K + k])
                    acc[r * COLS_PER_THREAD + c] += a_val * b_val

        # Barrier before next K tile
        barrier()

    # Store results to output
    for r in range(ROWS_PER_THREAD):
        var row = block_row + thread_row_in_tile * ROWS_PER_THREAD + r
        for c in range(COLS_PER_THREAD):
            var col = block_col + thread_col_in_tile * COLS_PER_THREAD + c
            if row < M_i and col < N_i:
                dst[unsafe_offset=row * N_i + col] = Scalar[DType.float16](
                    acc[r * COLS_PER_THREAD + c]
                )


def _matmul_weight_tiled_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext, x: Tensor[dtype, 2], w: Tensor[dtype, 2]
) raises -> Tensor[dtype, 2]:
    """Launch tiled matmul kernel: y = x @ w^T."""
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]  # w is [N, K]
    if K != w.shape()[1]:
        unimplemented("matmul_weight_tiled_gpu: K mismatch between x and w")

    var x_buf = upload[dtype, 2](ctx, x)
    var w_buf = upload[dtype, 2](ctx, w)
    var dst_buf = ctx.enqueue_create_buffer[dtype](M * N)

    # Grid dimensions
    var grid_x = (N + BLOCK_N - 1) // BLOCK_N
    var grid_y = (M + BLOCK_M - 1) // BLOCK_M

    ctx.enqueue_function[_matmul_weight_tiled_kernel_f16](
        x_buf,
        w_buf,
        dst_buf,
        Int32(M),
        Int32(K),
        Int32(N),
        grid_dim=(grid_x, grid_y),
        block_dim=BLOCK_THREADS,
    )

    var out = download2[dtype](ctx, dst_buf, StaticTuple[Int, 2](M, N))
    ctx.synchronize()
    return out


def matmul_weight_tiled_gpu[
    dtype: DType
](x: Tensor[dtype, 2], w: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Tiled GPU matmul: y = x @ w^T where w is [N, K] (GGUF layout).

    Uses threadgroup memory for 10-20x speedup over the naive kernel.

    Falls back to CPU on any GPU error.
    """
    if not gpu_available[dtype]():
        return matmul_weight_cpu[dtype](x, w)
    try:
        var ctx = get_gpu_context()
        return _matmul_weight_tiled_gpu_launch[dtype](ctx, x, w)
    except:
        return matmul_weight_cpu[dtype](x, w)
